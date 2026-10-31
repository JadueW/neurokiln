"""SQLite experiment store.

Every arena run can be persisted with its dataset fingerprint, protocol,
split signatures and per-trial predictions. The database is intentionally
plain SQLite so NeuroKiln remains a single-process local tool; all tables
are created forward-compatibly and reads are tolerant of older rows.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from . import __version__ as engine_version
from .stats import compare_pipelines

SCHEMA_VERSION = 6

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    adapter TEXT NOT NULL,
    signature TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT,
    name TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    dataset_version TEXT NOT NULL DEFAULT '',
    dataset_signature TEXT NOT NULL,
    dataset_manifest_json TEXT NOT NULL,
    protocol TEXT NOT NULL,
    protocol_params_json TEXT NOT NULL,
    pipelines_json TEXT NOT NULL,
    seeds_json TEXT NOT NULL,
    engine_version TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'running',
    created_at TEXT NOT NULL,
    completed_at TEXT,
    duration_ms REAL,
    summary_json TEXT,
    n_runs INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    pipeline TEXT NOT NULL,
    pipeline_name TEXT NOT NULL,
    fold INTEGER NOT NULL,
    repeat INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    split_signature TEXT NOT NULL,
    n_train INTEGER NOT NULL,
    n_test INTEGER NOT NULL,
    balanced_accuracy REAL NOT NULL,
    cohen_kappa REAL NOT NULL,
    duration_ms REAL NOT NULL,
    split_json TEXT NOT NULL,
    y_true_json TEXT NOT NULL,
    y_pred_json TEXT NOT NULL,
    y_proba_json TEXT,
    proba_classes_json TEXT,
    calibration_json TEXT
);

CREATE TABLE IF NOT EXISTS custom_pipelines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    spec_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS custom_pipeline_spaces (
    custom_pipeline_id INTEGER PRIMARY KEY,
    space_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(custom_pipeline_id) REFERENCES custom_pipelines(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS explanations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER REFERENCES experiments(id) ON DELETE CASCADE,
    job_id TEXT,
    dataset_id TEXT NOT NULL DEFAULT '',
    pipeline TEXT NOT NULL,
    pipeline_name TEXT NOT NULL,
    fold INTEGER NOT NULL DEFAULT 0,
    repeat INTEGER NOT NULL DEFAULT 0,
    seed INTEGER,
    status TEXT NOT NULL DEFAULT 'pending',
    directory TEXT NOT NULL DEFAULT '',
    summary_json TEXT,
    warnings_json TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    duration_ms REAL
);

CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiments(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_experiments_dataset ON experiments(dataset_id);
CREATE INDEX IF NOT EXISTS idx_custom_pipelines_updated ON custom_pipelines(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_explanations_experiment ON explanations(experiment_id);
CREATE INDEX IF NOT EXISTS idx_explanations_created ON explanations(created_at DESC);
"""


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


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _summarize_rows(
    arena_rows: Sequence[dict], pipeline_names: dict[str, str]
) -> tuple[dict[str, Any], int]:
    summaries = []
    n_runs = 0
    statistics = compare_pipelines(arena_rows)
    ci_pipelines = (statistics.get("ci") or {}).get("pipelines") or {}
    for row in arena_rows:
        splits = row.get("splits", [])
        n_runs += len(splits)
        durations = [float(item.get("duration_ms", 0.0)) for item in splits]
        interval = ci_pipelines.get(row["pipeline"]) or {}
        summaries.append(
            {
                "pipeline": row["pipeline"],
                "pipeline_name": pipeline_names.get(row["pipeline"], row["pipeline"]),
                "n_splits": len(splits),
                "balanced_accuracy_mean": float(row.get("balanced_accuracy_mean", 0.0)),
                "balanced_accuracy_std": float(row.get("balanced_accuracy_std", 0.0)),
                "balanced_accuracy_ci95": interval.get("ci95"),
                "ci_unit": (statistics.get("ci") or {}).get("unit"),
                "ci_n_units": interval.get("n_units"),
                "cohen_kappa_mean": float(row.get("cohen_kappa_mean", 0.0)),
                "calibration": row.get("calibration") or {"available": False},
                "duration_ms_mean": float(sum(durations) / len(durations)) if durations else 0.0,
                "duration_ms_total": float(sum(durations)),
            }
        )
    best = max(summaries, key=lambda item: item["balanced_accuracy_mean"], default=None)
    summary = {
        "n_pipelines": len(summaries),
        "n_runs": n_runs,
        "best_pipeline": best["pipeline"] if best else None,
        "best_balanced_accuracy": best["balanced_accuracy_mean"] if best else None,
        "pipelines": summaries,
        "statistics": statistics,
    }
    return summary, n_runs


class ExperimentStore:
    """Persistent experiment audit log backed by SQLite (WAL mode)."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.executescript(_SCHEMA)
            columns = {
                str(row["name"]) for row in conn.execute("PRAGMA table_info(runs)").fetchall()
            }
            for name in ("y_proba_json", "proba_classes_json", "calibration_json"):
                if name not in columns:
                    conn.execute(f"ALTER TABLE runs ADD COLUMN {name} TEXT")
            experiment_columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(experiments)").fetchall()
            }
            if "job_id" not in experiment_columns:
                conn.execute("ALTER TABLE experiments ADD COLUMN job_id TEXT")
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_experiments_job "
                "ON experiments(job_id) WHERE job_id IS NOT NULL"
            )
            conn.execute(
                "INSERT INTO schema_meta(key, value) VALUES('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(SCHEMA_VERSION),),
            )

    def save_dataset(
        self,
        dataset_id: str,
        name: str,
        adapter: str,
        signature: str,
        manifest: dict[str, Any],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO datasets(id, name, adapter, signature, manifest_json, updated_at)
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    adapter = excluded.adapter,
                    signature = excluded.signature,
                    manifest_json = excluded.manifest_json,
                    updated_at = excluded.updated_at
                """,
                (dataset_id, name, adapter, signature, _dumps(manifest), _now()),
            )

    def record_experiment(
        self,
        *,
        name: str,
        dataset_id: str,
        dataset_name: str,
        dataset_version: str,
        dataset_signature: str,
        dataset_manifest: dict[str, Any],
        protocol: str,
        protocol_params: dict[str, Any],
        pipelines: Sequence[dict[str, Any]],
        seeds: Sequence[int],
        arena_rows: Sequence[dict],
        duration_ms: float,
        notes: str = "",
        status: str = "completed",
        job_id: str | None = None,
    ) -> int:
        """Insert an experiment and all per-split runs in one transaction.

        ``job_id`` makes retries idempotent: a job that was interrupted after
        the experiment was committed reuses the existing row instead of
        creating a duplicate on recovery.
        """
        pipeline_names = {
            str(item.get("id")): str(item.get("name") or item.get("id"))
            for item in pipelines
        }
        summary, n_runs = _summarize_rows(arena_rows, pipeline_names)
        now = _now()
        with self._connect() as conn:
            if job_id:
                existing = conn.execute(
                    "SELECT id FROM experiments WHERE job_id = ?", (str(job_id),)
                ).fetchone()
                if existing is not None:
                    return int(existing["id"])
            cursor = conn.execute(
                """
                INSERT INTO experiments(
                    job_id, name, dataset_id, dataset_name, dataset_version,
                    dataset_signature, dataset_manifest_json, protocol,
                    protocol_params_json, pipelines_json, seeds_json,
                    engine_version, notes, status, created_at, completed_at,
                    duration_ms, summary_json, n_runs
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    name,
                    dataset_id,
                    dataset_name,
                    dataset_version,
                    dataset_signature,
                    _dumps(dataset_manifest),
                    protocol,
                    _dumps(protocol_params),
                    _dumps(list(pipelines)),
                    _dumps(list(seeds)),
                    engine_version,
                    notes,
                    status,
                    now,
                    now if status == "completed" else None,
                    float(duration_ms),
                    _dumps(summary),
                    n_runs,
                ),
            )
            experiment_id = int(cursor.lastrowid)
            run_values = []
            for row in arena_rows:
                pipeline = str(row["pipeline"])
                for split in row.get("splits", []):
                    info = split["split"]
                    run_values.append(
                        (
                            experiment_id,
                            pipeline,
                            pipeline_names.get(pipeline, pipeline),
                            int(info["fold"]),
                            int(info["repeat"]),
                            int(info["seed"]),
                            str(info.get("signature", "")),
                            int(info["n_train"]),
                            int(info["n_test"]),
                            float(split["balanced_accuracy"]),
                            float(split["cohen_kappa"]),
                            float(split.get("duration_ms", 0.0)),
                            _dumps(info),
                            _dumps(split.get("y_true", [])),
                            _dumps(split.get("y_pred", [])),
                            (
                                _dumps(split["y_proba"])
                                if split.get("y_proba") is not None else None
                            ),
                            (
                                _dumps(split["proba_classes"])
                                if split.get("proba_classes") is not None else None
                            ),
                            (
                                _dumps(split["calibration"])
                                if split.get("calibration") is not None else None
                            ),
                        )
                    )
            conn.executemany(
                """
                INSERT INTO runs(
                    experiment_id, pipeline, pipeline_name, fold, repeat,
                    seed, split_signature, n_train, n_test, balanced_accuracy,
                    cohen_kappa, duration_ms, split_json, y_true_json, y_pred_json,
                    y_proba_json, proba_classes_json, calibration_json
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                run_values,
            )
        return experiment_id

    def mark_failed(self, experiment_id: int, message: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE experiments SET status = 'failed', completed_at = ?, notes = ? WHERE id = ?",
                (_now(), message, experiment_id),
            )

    def list_experiments(
        self,
        limit: int = 50,
        offset: int = 0,
        dataset_id: str | None = None,
        status: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        where, params = [], []
        if dataset_id:
            where.append("dataset_id = ?")
            params.append(dataset_id)
        if status:
            where.append("status = ?")
            params.append(status)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        with self._connect() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM experiments {clause}", params
            ).fetchone()[0]
            rows = conn.execute(
                f"""
                SELECT * FROM experiments {clause}
                ORDER BY id DESC LIMIT ? OFFSET ?
                """,
                [*params, int(limit), int(offset)],
            ).fetchall()
        return [_list_item(row) for row in rows], int(total)

    def get_experiment(self, experiment_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
            if row is None:
                return None
            runs = conn.execute(
                "SELECT * FROM runs WHERE experiment_id = ? ORDER BY pipeline, repeat, fold",
                (experiment_id,),
            ).fetchall()
        detail = _list_item(row)
        summary = detail.pop("summary", None) or {}
        by_pipeline: dict[str, dict[str, Any]] = {}
        for run in runs:
            item = by_pipeline.setdefault(
                run["pipeline"],
                {
                    "pipeline": run["pipeline"],
                    "pipeline_name": run["pipeline_name"],
                    "splits": [],
                },
            )
            item["splits"].append(
                {
                    "balanced_accuracy": run["balanced_accuracy"],
                    "cohen_kappa": run["cohen_kappa"],
                    "duration_ms": run["duration_ms"],
                    "split": _loads(run["split_json"], {}),
                    "y_true": _loads(run["y_true_json"], []),
                    "y_pred": _loads(run["y_pred_json"], []),
                    "y_proba": _loads(run["y_proba_json"], None),
                    "proba_classes": _loads(run["proba_classes_json"], None),
                    "calibration": _loads(run["calibration_json"], None),
                }
            )
        detail["summary"] = summary
        detail["rows"] = [
            {
                **next(
                    (p for p in summary.get("pipelines", []) if p["pipeline"] == pipeline),
                    {"pipeline": pipeline, "pipeline_name": item["pipeline_name"]},
                ),
                "splits": item["splits"],
            }
            for pipeline, item in by_pipeline.items()
        ]
        detail["statistics"] = summary.get("statistics") or compare_pipelines(detail["rows"])
        return detail

    def delete_experiment(self, experiment_id: int) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
        return cursor.rowcount > 0

    def create_explanation(
        self,
        *,
        pipeline: str,
        pipeline_name: str,
        folder: str,
        experiment_id: int | None = None,
        dataset_id: str = "",
        job_id: str | None = None,
        fold: int = 0,
        repeat: int = 0,
        seed: int | None = None,
    ) -> int:
        """Register a pending explanation before the job starts computing."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO explanations(
                    experiment_id, job_id, dataset_id, pipeline, pipeline_name,
                    fold, repeat, seed, status, directory, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, ?)
                """,
                (
                    experiment_id, job_id, dataset_id, pipeline, pipeline_name,
                    int(fold), int(repeat), seed, folder, _now(),
                ),
            )
            return int(cursor.lastrowid)

    def get_explanation_by_job_id(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM explanations WHERE job_id = ? ORDER BY id DESC LIMIT 1",
                (str(job_id),),
            ).fetchone()
        return _explanation_item(row) if row is not None else None

    def restart_explanation(self, explanation_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE explanations
                SET status = 'running', error = NULL, completed_at = NULL,
                    duration_ms = NULL, summary_json = NULL, warnings_json = NULL
                WHERE id = ?
                """,
                (int(explanation_id),),
            )

    def fail_explanations_for_jobs(
        self, job_ids: Sequence[str], message: str
    ) -> int:
        if not job_ids:
            return 0
        placeholders = ", ".join("?" for _ in job_ids)
        with self._connect() as conn:
            cursor = conn.execute(
                f"""
                UPDATE explanations
                SET status = 'failed', error = ?, completed_at = ?
                WHERE status = 'running' AND job_id IN ({placeholders})
                """,
                (str(message)[:2000], _now(), *[str(item) for item in job_ids]),
            )
        return int(cursor.rowcount)

    def finish_explanation(
        self,
        explanation_id: int,
        *,
        folder: str,
        summary: dict[str, Any],
        warnings: Sequence[str],
        duration_ms: float,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE explanations
                SET status = 'completed', directory = ?, summary_json = ?,
                    warnings_json = ?, error = NULL, completed_at = ?, duration_ms = ?
                WHERE id = ?
                """,
                (
                    folder, _dumps(summary), _dumps(list(warnings)), _now(),
                    float(duration_ms), int(explanation_id),
                ),
            )

    def fail_explanation(self, explanation_id: int, message: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE explanations
                SET status = 'failed', error = ?, completed_at = ?
                WHERE id = ?
                """,
                (str(message)[:2000], _now(), int(explanation_id)),
            )

    def get_explanation(self, explanation_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM explanations WHERE id = ?", (int(explanation_id),)
            ).fetchone()
        return _explanation_item(row) if row is not None else None

    def list_explanations(
        self,
        experiment_id: int | None = None,
        pipeline: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        where, params = [], []
        if experiment_id is not None:
            where.append("experiment_id = ?")
            params.append(int(experiment_id))
        if pipeline:
            where.append("pipeline = ?")
            params.append(str(pipeline))
        if status:
            where.append("status = ?")
            params.append(str(status))
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        with self._connect() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM explanations {clause}", params
            ).fetchone()[0]
            rows = conn.execute(
                f"SELECT * FROM explanations {clause} ORDER BY id DESC LIMIT ? OFFSET ?",
                [*params, int(limit), int(offset)],
            ).fetchall()
        return [_explanation_item(row) for row in rows], int(total)

    def delete_explanation(self, explanation_id: int) -> dict[str, Any] | None:
        """Remove the registry row and return it so the caller can drop files."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM explanations WHERE id = ?", (int(explanation_id),)
            ).fetchone()
            if row is None:
                return None
            conn.execute("DELETE FROM explanations WHERE id = ?", (int(explanation_id),))
        return _explanation_item(row)

    def stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            experiments = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
            completed = conn.execute(
                "SELECT COUNT(*) FROM experiments WHERE status = 'completed'"
            ).fetchone()[0]
            failed = conn.execute(
                "SELECT COUNT(*) FROM experiments WHERE status = 'failed'"
            ).fetchone()[0]
            runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            datasets = conn.execute(
                "SELECT COUNT(DISTINCT dataset_id) FROM experiments"
            ).fetchone()[0]
            # 最佳结果按实验级折均值比较，和报告中心口径一致；
            # 单折最高分会高估表现，不能当作平台能力展示。
            best: dict[str, Any] | None = None
            for row in conn.execute(
                "SELECT id, name, summary_json FROM experiments "
                "WHERE status = 'completed' AND summary_json IS NOT NULL"
            ).fetchall():
                summary = _loads(row["summary_json"], {})
                score = summary.get("best_balanced_accuracy")
                if score is None:
                    continue
                pipeline = summary.get("best_pipeline")
                label = next(
                    (
                        item.get("pipeline_name")
                        for item in summary.get("pipelines") or []
                        if item.get("pipeline") == pipeline
                    ),
                    pipeline,
                )
                if best is None or float(score) > float(best["balanced_accuracy"]):
                    best = {
                        "id": row["id"],
                        "name": row["name"],
                        "pipeline": pipeline,
                        "pipeline_name": label,
                        "balanced_accuracy": float(score),
                    }
            latest = conn.execute(
                "SELECT id, name, dataset_name, status, created_at FROM experiments "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
            custom = conn.execute("SELECT COUNT(*) FROM custom_pipelines").fetchone()[0]
        return {
            "experiments": int(experiments),
            "completed": int(completed),
            "failed": int(failed),
            "runs": int(runs),
            "datasets_used": int(datasets),
            "custom_pipelines": int(custom),
            "best_run": best,
            "latest_experiment": _row_to_dict(latest) if latest else None,
            "storage": str(self.path),
        }

    def datasets(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM datasets ORDER BY updated_at DESC").fetchall()
        return [
            {
                **{key: row[key] for key in ("id", "name", "adapter", "signature", "updated_at")},
                "manifest": _loads(row["manifest_json"], {}),
            }
            for row in rows
        ]

    def save_custom_pipeline(
        self,
        name: str,
        description: str,
        spec: dict[str, Any],
        pipeline_id: int | None = None,
    ) -> int:
        now = _now()
        payload = _dumps(spec)
        with self._connect() as conn:
            if pipeline_id is None:
                cursor = conn.execute(
                    """
                    INSERT INTO custom_pipelines(name, description, spec_json, created_at, updated_at)
                    VALUES(?, ?, ?, ?, ?)
                    """,
                    (name, description, payload, now, now),
                )
                return int(cursor.lastrowid)
            existing = conn.execute(
                "SELECT id FROM custom_pipelines WHERE id = ?", (int(pipeline_id),)
            ).fetchone()
            if existing is None:
                raise KeyError(f"custom pipeline {pipeline_id} not found")
            conn.execute(
                """
                UPDATE custom_pipelines
                SET name = ?, description = ?, spec_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (name, description, payload, now, int(pipeline_id)),
            )
            return int(pipeline_id)

    def list_custom_pipelines(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM custom_pipelines ORDER BY updated_at DESC, id DESC"
            ).fetchall()
        return [_custom_item(row) for row in rows]

    def get_custom_pipeline(self, pipeline_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM custom_pipelines WHERE id = ?", (int(pipeline_id),)
            ).fetchone()
        return _custom_item(row) if row is not None else None

    def save_custom_pipeline_space(
        self, pipeline_id: int, dims: Sequence[dict[str, Any]]
    ) -> None:
        payload = _dumps(list(dims))
        now = _now()
        with self._connect() as conn:
            exists = conn.execute(
                "SELECT id FROM custom_pipelines WHERE id = ?", (int(pipeline_id),)
            ).fetchone()
            if exists is None:
                raise KeyError(f"custom pipeline {pipeline_id} not found")
            conn.execute(
                """
                INSERT INTO custom_pipeline_spaces(custom_pipeline_id, space_json, updated_at)
                VALUES(?, ?, ?)
                ON CONFLICT(custom_pipeline_id) DO UPDATE SET
                    space_json = excluded.space_json,
                    updated_at = excluded.updated_at
                """,
                (int(pipeline_id), payload, now),
            )

    def get_custom_pipeline_space(
        self, pipeline_id: int
    ) -> list[dict[str, Any]] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT space_json FROM custom_pipeline_spaces WHERE custom_pipeline_id = ?",
                (int(pipeline_id),),
            ).fetchone()
        if row is None:
            return None
        return _loads(row["space_json"], [])

    def list_custom_pipeline_spaces(self) -> dict[int, list[dict[str, Any]]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT custom_pipeline_id, space_json FROM custom_pipeline_spaces"
            ).fetchall()
        return {
            int(row["custom_pipeline_id"]): _loads(row["space_json"], [])
            for row in rows
        }

    def delete_custom_pipeline_space(self, pipeline_id: int) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM custom_pipeline_spaces WHERE custom_pipeline_id = ?",
                (int(pipeline_id),),
            )
        return cursor.rowcount > 0

    def delete_custom_pipeline(self, pipeline_id: int) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM custom_pipelines WHERE id = ?", (int(pipeline_id),)
            )
        return cursor.rowcount > 0


def _list_item(row: sqlite3.Row) -> dict[str, Any]:
    summary = _loads(row["summary_json"], None)
    best = None
    if summary:
        best = {
            "pipeline": summary.get("best_pipeline"),
            "balanced_accuracy": summary.get("best_balanced_accuracy"),
        }
    return {
        "id": row["id"],
        "job_id": row["job_id"],
        "name": row["name"],
        "dataset": {
            "id": row["dataset_id"],
            "name": row["dataset_name"],
            "version": row["dataset_version"],
            "signature": row["dataset_signature"],
            "manifest": _loads(row["dataset_manifest_json"], {}),
        },
        "protocol": row["protocol"],
        "protocol_params": _loads(row["protocol_params_json"], {}),
        "pipelines": _loads(row["pipelines_json"], []),
        "seeds": _loads(row["seeds_json"], []),
        "engine_version": row["engine_version"],
        "notes": row["notes"],
        "status": row["status"],
        "created_at": row["created_at"],
        "completed_at": row["completed_at"],
        "duration_ms": row["duration_ms"],
        "n_runs": row["n_runs"],
        "summary": summary,
        "best": best,
    }


def _custom_item(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"],
        "spec": _loads(row["spec_json"], {}),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _explanation_item(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "experiment_id": row["experiment_id"],
        "job_id": row["job_id"],
        "dataset_id": row["dataset_id"],
        "pipeline": row["pipeline"],
        "pipeline_name": row["pipeline_name"],
        "fold": row["fold"],
        "repeat": row["repeat"],
        "seed": row["seed"],
        "status": row["status"],
        "directory": row["directory"],
        "summary": _loads(row["summary_json"], None),
        "warnings": _loads(row["warnings_json"], []),
        "error": row["error"],
        "created_at": row["created_at"],
        "completed_at": row["completed_at"],
        "duration_ms": row["duration_ms"],
    }


__all__ = ["ExperimentStore", "SCHEMA_VERSION"]
