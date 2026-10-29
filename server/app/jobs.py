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
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
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


class JobManager:
    """SQLite job board with a small in-process worker pool.

    NeuroKiln is a local tool, so jobs share the API process. Job state is
    persisted so the UI can reconnect after a page reload; on restart any
    interrupted job is marked failed rather than silently resumed.
    """

    def __init__(self, db_path: str | Path, max_workers: int = 1):
        self.path = Path(db_path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)), thread_name_prefix="neurokiln-job"
        )
        with self._connect() as conn:
            conn.executescript(JOB_SCHEMA)
        self.recover_stale()

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

    def recover_stale(self) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE jobs
                SET status = 'failed', completed_at = ?,
                    error = '服务重启导致任务中断，请重新提交'
                WHERE status IN ('queued', 'running')
                """,
                (_now(),),
            )
        return int(cursor.rowcount)

    def submit(self, kind: str, payload: dict[str, Any],
               runner: Callable[[str, Callable[[float, str], None], Callable[[], bool]], dict]) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        event = threading.Event()
        with self._lock:
            self._events[job_id] = event
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs(id, kind, status, payload_json, progress, message, created_at)
                VALUES(?, ?, 'queued', ?, 0, '排队中', ?)
                """,
                (job_id, kind, _dumps(payload), _now()),
            )
        future = self._executor.submit(self._run, job_id, runner, event)
        future.add_done_callback(lambda _future: self._forget(job_id))
        return self.get(job_id) or {"id": job_id}

    def _forget(self, job_id: str) -> None:
        with self._lock:
            self._events.pop(job_id, None)

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

    def _run(self, job_id: str, runner, event: threading.Event) -> None:
        if event.is_set() or self._cancel_requested(job_id):
            self._update(
                job_id, status="cancelled", completed_at=_now(),
                message="已取消", progress=0.0,
            )
            return
        self._update(job_id, status="running", started_at=_now(), message="正在运行", progress=0.0)

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
                message="已取消", progress=0.0,
            )
            return
        except Exception as exc:  # noqa: BLE001 - surfaced through the job record
            self._update(
                job_id, status="failed", completed_at=_now(),
                message="运行失败", error=f"{type(exc).__name__}: {exc}",
            )
            return
        self._update(
            job_id, status="completed", completed_at=_now(),
            message="已完成", progress=1.0, result_json=_dumps(result),
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
            events = list(self._events.values())
        for event in events:
            event.set()
        self._executor.shutdown(wait=False, cancel_futures=True)


def _job_item(row: sqlite3.Row, include_result: bool) -> dict[str, Any]:
    item = {
        "id": row["id"],
        "kind": row["kind"],
        "status": row["status"],
        "progress": float(row["progress"]),
        "message": row["message"],
        "cancel_requested": bool(row["cancel_requested"]),
        "request": _loads(row["payload_json"], {}),
        "error": row["error"],
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "completed_at": row["completed_at"],
    }
    if include_result:
        item["result"] = _loads(row["result_json"], None)
    return item


__all__ = ["JobCancelled", "JobManager"]
