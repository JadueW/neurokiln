import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))

from app.jobs import JobManager


def _insert_job(
    db,
    job_id,
    *,
    kind="fake",
    status="running",
    payload=None,
    attempts=0,
    cancel_requested=0,
):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            INSERT INTO jobs(
                id, kind, status, payload_json, progress, message,
                attempts, cancel_requested, created_at
            ) VALUES(?, ?, ?, ?, 0.25, ?, ?, ?, ?)
            """,
            (
                job_id,
                kind,
                status,
                json.dumps(payload or {"value": job_id}),
                "运行中",
                attempts,
                cancel_requested,
                now,
            ),
        )


def _wait(manager, job_id, statuses=("completed",), timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        item = manager.get(job_id)
        if item is not None and item["status"] in statuses:
            return item
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not reach {statuses}: {manager.get(job_id)}")


def test_recover_requeues_queued_and_running_jobs(tmp_path):
    manager = JobManager(tmp_path / "jobs.db", max_workers=1, max_attempts=3)
    calls = []

    def factory(payload):
        def runner(job_id, on_progress, should_cancel):
            calls.append((job_id, payload["value"]))
            on_progress(0.5, "half")
            return {"value": payload["value"]}

        return runner

    manager.register("fake", factory)
    _insert_job(manager.path, "stale-queued", status="queued", payload={"value": "q"})
    _insert_job(manager.path, "stale-running", status="running", payload={"value": "r"})

    recovered = manager.recover()
    assert {item["id"] for item in recovered} == {"stale-queued", "stale-running"}

    queued = _wait(manager, "stale-queued")
    running = _wait(manager, "stale-running")
    assert queued["result"] == {"value": "q"}
    assert running["result"] == {"value": "r"}
    assert queued["attempts"] == 1
    assert running["attempts"] == 1
    assert sorted(calls) == [("stale-queued", "q"), ("stale-running", "r")]
    manager.shutdown()


def test_recover_stops_after_retry_limit(tmp_path):
    manager = JobManager(tmp_path / "jobs.db", max_workers=1, max_attempts=2)
    _insert_job(manager.path, "too-many", status="running", attempts=2)

    assert manager.recover() == []
    item = manager.get("too-many")
    assert item["status"] == "failed"
    assert item["error_code"] == "job.retry_limit"
    assert item["attempts"] == 3
    manager.shutdown()


def test_recover_fails_without_registered_factory(tmp_path):
    manager = JobManager(tmp_path / "jobs.db", max_workers=1, max_attempts=3)
    _insert_job(manager.path, "unknown-kind", kind="mystery", status="queued")

    assert manager.recover() == []
    item = manager.get("unknown-kind")
    assert item["status"] == "failed"
    assert item["error_code"] == "job.no_recovery_factory"
    manager.shutdown()


def test_recover_skips_cancelled_and_factory_errors(tmp_path):
    manager = JobManager(tmp_path / "jobs.db", max_workers=1, max_attempts=3)

    def broken(payload):
        raise ValueError("pipeline removed")

    manager.register("broken", broken)
    _insert_job(manager.path, "already-stopped", status="running", cancel_requested=1)
    _insert_job(manager.path, "broken-factory", kind="broken", status="running")

    assert manager.recover() == []
    stopped = manager.get("already-stopped")
    assert stopped["status"] == "cancelled"
    failed = manager.get("broken-factory")
    assert failed["status"] == "failed"
    assert failed["error_code"] == "job.recovery_failed"
    assert "pipeline removed" in failed["error"]
    manager.shutdown()


def test_submit_can_resolve_runner_from_registry(tmp_path):
    manager = JobManager(tmp_path / "jobs.db", max_workers=1)
    manager.register(
        "fake",
        lambda payload: (
            lambda job_id, on_progress, should_cancel: {"echo": payload["value"]}
        ),
    )
    job = manager.submit("fake", {"value": 7})
    assert job["status"] == "queued"
    assert job["attempts"] == 0
    done = _wait(manager, job["id"])
    assert done["result"] == {"echo": 7}
    assert done["error_code"] is None
    manager.shutdown()
