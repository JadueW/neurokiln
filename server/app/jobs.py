from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

JOB_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    result_json TEXT,
    error TEXT,
    error_code TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""


class JobCancelled(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


Runner = Callable[[str, Callable[[float, str], None], Callable[[], bool]], dict]
RunnerFactory = Callable[[dict[str, Any]], Runner]


class JobManager:
    """SQLite job board with a small in-process worker pool.

    Job payloads are persisted so a crashed or restarted API can rebuild the
    runner from its kind registry and automatically requeue unfinished work.
    Runners with database side effects must be idempotent per job id; the
    arena and explanation jobs implement that contract at their call sites.
    """

    def __init__(
        self,
        db_path: str | Path,
        max_workers: int = 1,
        max_attempts: int = 3,
    ):
        self.path = Path(db_path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_attempts = max(1, int(max_attempts))
        self._factories: dict[str, RunnerFactory] = {}
        self._events: dict[str, threading.Event] = {}
        self._scheduled: set[str] = set()
        self._closed = False
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)), thread_name_prefix="neurokiln-job"
        )
        with self._connect() as conn:
            conn.executescript(JOB_SCHEMA)
            columns = {
                str(row["name"]) for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
            }
            if "attempts" not in columns:
                conn.execute(
                    "ALTER TABLE jobs ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0"
                )
            if "error_code" not in columns:
                conn.execute("ALTER TABLE jobs ADD COLUMN error_code TEXT")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def register(self, kind: str, factory: RunnerFactory) -> None:
        self._factories[str(kind)] = factory

    def recover(self) -> list[dict[str, Any]]:
        """Requeue jobs left queued/running by a previous process.

        Gracefully cancelled jobs are skipped; a job that has already been
        retried ``max_attempts`` times is failed instead of looping forever.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status IN ('queued', 'running') "
                "ORDER BY created_at, rowid"
            ).fetchall()
        recovered: list[dict[str, Any]] = []
        for row in rows:
            job_id = str(row["id"])
            if self._is_scheduled(job_id):
                continue
            if row["cancel_requested"]:
                self._update(
                    job_id,
                    status="cancelled",
                    completed_at=_now(),
                    progress=0.0,
                    message="已取消",
                    error=None,
                    error_code=None,
                )
                continue
            attempts = int(row["attempts"] or 0) + 1
            kind = str(row["kind"])
            if attempts > self.max_attempts:
                self._update(
                    job_id,
                    status="failed",
                    completed_at=_now(),
                    attempts=attempts,
                    message="运行失败",
                    error=f"任务连续中断 {attempts - 1} 次，已停止自动恢复",
                    error_code="job.retry_limit",
                )
                continue
            factory = self._factories.get(kind)
            if factory is None:
                self._update(
                    job_id,
                    status="failed",
                    completed_at=_now(),
                    attempts=attempts,
                    message="运行失败",
                    error=f"服务重启后找不到任务类型 {kind} 的恢复入口",
                    error_code="job.no_recovery_factory",
                )
                continue
            payload = _loads(row["payload_json"], {})
            try:
                runner = factory(payload)
            except Exception as exc:  # noqa: BLE001 - reported through the job record
                self._update(
                    job_id,
                    status="failed",
                    completed_at=_now(),
                    attempts=attempts,
                    message="运行失败",
                    error=f"{type(exc).__name__}: {exc}",
                    error_code="job.recovery_failed",
                )
                continue
            self._update(
                job_id,
                status="queued",
                progress=0.0,
                message=f"服务重启后自动恢复（第 {attempts} 次尝试）",
                result_json=None,
                error=None,
                error_code=None,
                attempts=attempts,
                cancel_requested=0,
                started_at=None,
                completed_at=None,
            )
            if not self._schedule(job_id, runner):
                self._update(
                    job_id,
                    status="failed",
                    completed_at=_now(),
                    message="运行失败",
                    error="任务调度器已关闭，无法恢复任务",
                    error_code="job.scheduler_closed",
                )
                continue
            item = self.get(job_id)
            if item is not None:
                recovered.append(item)
        return recovered

    def recover_stale(self) -> int:
        """Backward-compatible alias for :meth:`recover`."""
        return len(self.recover())

    def submit(
        self,
        kind: str,
        payload: dict[str, Any],
        runner: Runner | None = None,
    ) -> dict[str, Any]:
        if runner is None:
            factory = self._factories.get(str(kind))
            if factory is None:
                raise ValueError(f"任务类型 {kind} 没有注册 runner")
            runner = factory(payload)
        if self._closed:
            raise RuntimeError("任务调度器已关闭")
        job_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs(
                    id, kind, status, payload_json, progress, message,
                    attempts, created_at
                ) VALUES(?, ?, 'queued', ?, 0, '排队中', 0, ?)
                """,
                (job_id, kind, _dumps(payload), _now()),
            )
        if not self._schedule(job_id, runner):
            self._update(
                job_id,
                status="failed",
                completed_at=_now(),
                message="运行失败",
                error="任务调度器已关闭",
                error_code="job.scheduler_closed",
            )
        return self.get(job_id) or {"id": job_id}

    def _schedule(self, job_id: str, runner: Runner) -> bool:
        event = threading.Event()
        with self._lock:
            if self._closed:
                return False
            self._events[job_id] = event
            self._scheduled.add(job_id)
        try:
            future = self._executor.submit(self._run, job_id, runner, event)
        except RuntimeError:
            with self._lock:
                self._events.pop(job_id, None)
                self._scheduled.discard(job_id)
            return False
        future.add_done_callback(lambda _future: self._forget(job_id))
        return True

    def _is_scheduled(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._scheduled

    def _forget(self, job_id: str) -> None:
        with self._lock:
            self._events.pop(job_id, None)
            self._scheduled.discard(job_id)

    def _cancel_requested(self, job_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return bool(row and row["cancel_requested"])

    def _update(self, job_id: str, **fields: Any) -> None:
        if not fields:
            return
        columns = ", ".join(f"{name} = ?" for name in fields)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE jobs SET {columns} WHERE id = ?",
                [*fields.values(), job_id],
            )

    def _run(self, job_id: str, runner: Runner, event: threading.Event) -> None:
        if event.is_set() or self._cancel_requested(job_id):
            self._update(
                job_id, status="cancelled", completed_at=_now(),
                message="已取消", progress=0.0, error_code=None,
            )
            return
        self._update(
            job_id, status="running", started_at=_now(),
            message="正在运行", progress=0.0, error=None, error_code=None,
        )

        def on_progress(fraction: float, message: str) -> None:
            self._update(
                job_id,
                progress=max(0.0, min(1.0, float(fraction))),
                message=str(message)[:200],
            )

        def should_cancel() -> bool:
            return event.is_set() or self._cancel_requested(job_id)

        try:
            result = runner(job_id, on_progress, should_cancel)
        except JobCancelled:
            self._update(
                job_id, status="cancelled", completed_at=_now(),
                message="已取消", progress=0.0, error_code=None,
            )
            return
        except Exception as exc:  # noqa: BLE001 - surfaced through the job record
            self._update(
                job_id, status="failed", completed_at=_now(),
                message="运行失败", error=f"{type(exc).__name__}: {exc}",
                error_code=_error_code(exc),
            )
            return
        self._update(
            job_id, status="completed", completed_at=_now(),
            message="已完成", progress=1.0, result_json=_dumps(result),
            error=None, error_code=None,
        )

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _job_item(row, include_result=True) if row is not None else None

    def list(self, limit: int = 20, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
                (int(limit), int(offset)),
            ).fetchall()
        return [_job_item(row, include_result=False) for row in rows], int(total)

    def active_ids(self) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id FROM jobs WHERE status IN ('queued', 'running')"
            ).fetchall()
        return {str(row["id"]) for row in rows}

    def cancel(self, job_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None or row["status"] not in {"queued", "running"}:
                return False
            conn.execute(
                "UPDATE jobs SET cancel_requested = 1, message = '取消已请求' WHERE id = ?",
                (job_id,),
            )
        with self._lock:
            event = self._events.get(job_id)
            if event is not None:
                event.set()
        return True

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
            events = list(self._events.values())
            active = list(self._scheduled)
        for event in events:
            event.set()
        # Persist cancellation before the process can die so recovery does not
        # resurrect work the user or the service explicitly stopped.
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET cancel_requested = 1, status = 'cancelled',
                    completed_at = ?, message = '服务已停止', progress = 0
                WHERE status IN ('queued', 'running')
                """,
                (_now(),),
            )
        self._executor.shutdown(wait=False, cancel_futures=True)
        with self._lock:
            for job_id in active:
                self._scheduled.discard(job_id)


def _error_code(exc: Exception) -> str:
    if isinstance(exc, ValueError):
        return "job.invalid_request"
    if isinstance(exc, TimeoutError):
        return "job.timeout"
    if isinstance(exc, MemoryError):
        return "job.out_of_memory"
    return "job.failed"


def _job_item(row: sqlite3.Row, include_result: bool) -> dict[str, Any]:
    item = {
        "id": row["id"],
        "kind": row["kind"],
        "status": row["status"],
        "progress": float(row["progress"]),
        "message": row["message"],
        "attempts": int(row["attempts"] or 0),
        "cancel_requested": bool(row["cancel_requested"]),
        "request": _loads(row["payload_json"], {}),
        "error": row["error"],
        "error_code": row["error_code"],
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "completed_at": row["completed_at"],
    }
    if include_result:
        item["result"] = _loads(row["result_json"], None)
    return item


__all__ = ["JobCancelled", "JobManager"]
