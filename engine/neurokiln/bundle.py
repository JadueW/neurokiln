from __future__ import annotations

import csv
import importlib.metadata
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .data.adapters import DatasetRegistry
from .evaluate import run_arena
from .protocol import build_protocol
from .report import build_html_report, build_workbook
from .runner import resolve_pipeline_definitions
from .store import ExperimentStore

_REQUIRED_BUNDLE_FILES = ("experiment.json", "protocol.json", "data_manifest.json")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment() -> dict[str, Any]:
    packages = [
        "neurokiln-engine", "numpy", "scipy", "scikit-learn", "pyriemann",
        "mne", "moabb", "torch", "pyarrow", "openpyxl",
    ]
    return {
        "created_at": _now(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {name: _version(name) for name in packages},
    }


def _safe_name(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in ("-", "_", ".") else "_"
        for character in str(value)
    ).strip("._")
    return cleaned[:80] or "pipeline"


def _split_ids(split: dict[str, Any], key: str, length: int) -> list[str]:
    values = split.get(key)
    if isinstance(values, list) and len(values) == length:
        return [str(value) for value in values]
    if key == "test_subject_ids":
        subjects = split.get("test_subjects") or []
        if len(subjects) == 1:
            return [str(subjects[0])] * length
    return [f"{key}-{index}" for index in range(length)]


def _prediction_rows(detail: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in detail.get("rows") or []:
        for split in row.get("splits") or []:
            info = split.get("split") or {}
            truth = split.get("y_true") or []
            predicted = split.get("y_pred") or []
            if not truth or len(truth) != len(predicted):
                continue
            trial_ids = _split_ids(info, "test_trial_ids", len(truth))
            subject_ids = _split_ids(info, "test_subject_ids", len(truth))
            session_ids = _split_ids(info, "test_session_ids", len(truth))
            proba = split.get("y_proba")
            proba_classes = split.get("proba_classes")
            valid_proba = (
                isinstance(proba, list)
                and isinstance(proba_classes, list)
                and len(proba) == len(truth)
            )
            for index, (actual, guess) in enumerate(zip(truth, predicted)):
                row_proba = proba[index] if valid_proba else None
                confidence = max(row_proba) if row_proba else None
                rows.append(
                    {
                        "experiment_id": detail.get("id"),
                        "pipeline": row.get("pipeline"),
                        "pipeline_name": row.get("pipeline_name") or row.get("pipeline"),
                        "fold": info.get("fold"),
                        "repeat": info.get("repeat"),
                        "seed": info.get("seed"),
                        "split_signature": info.get("signature"),
                        "trial_id": trial_ids[index],
                        "subject_id": subject_ids[index],
                        "session_id": session_ids[index],
                        "y_true": int(actual),
                        "y_pred": int(guess),
                        "correct": int(actual) == int(guess),
                        "confidence": confidence,
                        "proba_classes": (
                            json.dumps(proba_classes) if valid_proba else None
                        ),
                        "proba": json.dumps(row_proba) if row_proba is not None else None,
                        "duration_ms": split.get("duration_ms"),
                    }
                )
    return rows


def _write_predictions(detail: dict[str, Any], directory: Path) -> str:
    rows = _prediction_rows(detail)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        path = directory / "predictions.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)
        return path.name
    path = directory / "predictions.parquet"
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd")
    return path.name


def _write_metrics(detail: dict[str, Any], directory: Path) -> str:
    path = directory / "metrics.csv"
    fields = [
        "pipeline", "pipeline_name", "fold", "repeat", "seed", "n_train",
        "n_test", "balanced_accuracy", "cohen_kappa", "ece", "brier", "nll",
        "duration_ms", "split_signature",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in detail.get("rows") or []:
            for split in row.get("splits") or []:
                info = split.get("split") or {}
                calibration = split.get("calibration") or {}
                writer.writerow(
                    {
                        "pipeline": row.get("pipeline"),
                        "pipeline_name": row.get("pipeline_name") or row.get("pipeline"),
                        "fold": info.get("fold"),
                        "repeat": info.get("repeat"),
                        "seed": info.get("seed"),
                        "n_train": info.get("n_train"),
                        "n_test": info.get("n_test"),
                        "balanced_accuracy": split.get("balanced_accuracy"),
                        "cohen_kappa": split.get("cohen_kappa"),
                        "ece": calibration.get("ece"),
                        "brier": calibration.get("brier"),
                        "nll": calibration.get("nll"),
                        "duration_ms": split.get("duration_ms"),
                        "split_signature": info.get("signature"),
                    }
                )
    return path.name


def _experiment_config(detail: dict[str, Any]) -> dict[str, Any]:
    dataset = detail.get("dataset") or {}
    manifest = dataset.get("manifest") or {}
    selection = manifest.get("selection") or {}
    protocol_params = detail.get("protocol_params") or {}
    return {
        "format": "neurokiln-bundle/v1",
        "created_at": _now(),
        "experiment": {
            "id": detail.get("id"),
            "name": detail.get("name"),
            "engine_version": detail.get("engine_version"),
            "notes": detail.get("notes"),
            "status": detail.get("status"),
            "created_at": detail.get("created_at"),
            "completed_at": detail.get("completed_at"),
            "duration_ms": detail.get("duration_ms"),
            "n_runs": detail.get("n_runs"),
        },
        "dataset": {
            "id": dataset.get("id"),
            "name": dataset.get("name"),
            "version": dataset.get("version") or manifest.get("dataset_version"),
            "signature": dataset.get("signature"),
            "manifest": manifest,
            "subject_ids": selection.get("subject_ids") or [],
            "classes": selection.get("classes"),
            "options": selection.get("options") or {},
        },
        "protocol": {
            "name": detail.get("protocol"),
            "params": protocol_params,
            "seeds": detail.get("seeds") or protocol_params.get("seeds") or [0],
        },
        "pipelines": detail.get("pipelines") or [],
        "expected": {
            "summary": detail.get("summary") or {},
            "rows": [
                {
                    "pipeline": row.get("pipeline"),
                    "pipeline_name": row.get("pipeline_name"),
                    "balanced_accuracy_mean": row.get("balanced_accuracy_mean"),
                    "balanced_accuracy_std": row.get("balanced_accuracy_std"),
                    "balanced_accuracy_ci95": row.get("balanced_accuracy_ci95"),
                    "n_splits": row.get("n_splits"),
                }
                for row in detail.get("rows") or []
            ],
        },
    }


def _read_config(path: Path) -> dict[str, Any]:
    config_path = path / "experiment.json" if path.is_dir() else path
    if not config_path.exists():
        raise FileNotFoundError(f"找不到实验配置: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("format") != "neurokiln-bundle/v1":
        raise ValueError("不是 neurokiln bundle v1 目录")
    return config


def export_bundle(
    store: ExperimentStore,
    experiment_id: int,
    output_dir: str | Path,
    include_report: bool = True,
) -> dict[str, Any]:
    detail = store.get_experiment(int(experiment_id))
    if detail is None:
        raise KeyError(f"experiment {experiment_id} not found")
    directory = Path(output_dir).expanduser().resolve()
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"输出目录非空: {directory}")
    directory.mkdir(parents=True, exist_ok=True)

    config = _experiment_config(detail)
    _json_dump(directory / "experiment.json", config)
    _json_dump(directory / "protocol.json", config["protocol"])
    _json_dump(directory / "data_manifest.json", config["dataset"])
    _json_dump(directory / "environment.json", _environment())
    pipelines_dir = directory / "pipelines"
    pipelines_dir.mkdir(exist_ok=True)
    for index, definition in enumerate(config["pipelines"]):
        identifier = _safe_name(str(definition.get("id") or f"pipeline-{index}"))
        _json_dump(pipelines_dir / f"{identifier}.json", definition)

    metrics_file = _write_metrics(detail, directory)
    predictions_file = _write_predictions(detail, directory)

    explanations: list[dict[str, Any]] = []
    explanation_items, _ = store.list_explanations(
        experiment_id=int(experiment_id), limit=200
    )
    for item in explanation_items:
        if item.get("status") != "completed" or not item.get("directory"):
            continue
        source = Path(item["directory"])
        if not (source / "global.json").is_file():
            continue
        target = directory / "explanations" / str(item["id"])
        shutil.copytree(source, target, dirs_exist_ok=True)
        explanations.append(
            {
                "id": item["id"],
                "pipeline": item["pipeline"],
                "folder": str(target.relative_to(directory)),
                "summary": item.get("summary"),
            }
        )

    report_files = {}
    if include_report:
        html_path = directory / "report.html"
        html_path.write_text(build_html_report(detail), encoding="utf-8")
        xlsx_path = directory / "report.xlsx"
        xlsx_path.write_bytes(build_workbook(detail))
        report_files = {"html": html_path.name, "xlsx": xlsx_path.name}

    script = (
        "@echo off\r\n"
        "setlocal\r\n"
        "neurokiln reproduce \"%~dp0\" %*\r\n"
    )
    (directory / "reproduce.bat").write_text(script, encoding="utf-8")
    (directory / "reproduce.sh").write_text(
        "#!/usr/bin/env sh\nset -eu\nneurokiln reproduce \"$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\" \"$@\"\n",
        encoding="utf-8",
    )
    (directory / "README.md").write_text(
        "\n".join(
            [
                f"# 实验 #{detail.get('id')} 可复现包",
                "",
                f"- 名称：{detail.get('name')}",
                f"- 数据集：{(detail.get('dataset') or {}).get('name')}",
                f"- 数据签名：{(detail.get('dataset') or {}).get('signature')}",
                f"- 协议：{detail.get('protocol')} {detail.get('protocol_params')}",
                "",
                "目录内容：",
                "- `experiment.json`：完整配置、丹方定义与基准指标",
                "- `protocol.json` / `data_manifest.json` / `environment.json`",
                "- `pipelines/`：每条丹方的定义（自定义 DAG 含完整图）",
                f"- `{metrics_file}`：逐折指标",
                f"- `{predictions_file}`：逐试次预测",
                "- `report.html` / `report.xlsx`：报告",
                *(
                    ["- `explanations/`：可解释性分析产物（global.json、逐试次与 topomap）"]
                    if explanations else []
                ),
                "",
                "复现：",
                "```bash",
                "neurokiln reproduce .",
                "```",
                "",
                "复现会校验数据签名，并用相同划分、种子和丹方重跑；默认容差 0.02 BA。",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "directory": str(directory),
        "files": sorted(
            str(path.relative_to(directory))
            for path in directory.rglob("*")
            if path.is_file()
        ),
        "metrics": metrics_file,
        "predictions": predictions_file,
        "reports": report_files,
        "explanations": explanations,
    }


def _expected_means(config: dict[str, Any]) -> dict[str, float]:
    expected = {}
    for row in (config.get("expected") or {}).get("rows") or []:
        if row.get("pipeline") is not None and row.get("balanced_accuracy_mean") is not None:
            expected[str(row["pipeline"])] = float(row["balanced_accuracy_mean"])
    return expected


def _reproduce_config(
    registry: DatasetRegistry,
    config: dict[str, Any],
    store: ExperimentStore | None,
    save: bool,
    tolerance: float,
    allow_data_mismatch: bool,
    should_cancel: Callable[[], bool] | None = None,
    plugins: Any | None = None,
) -> dict[str, Any]:
    dataset = config.get("dataset") or {}
    protocol = config.get("protocol") or {}
    params = protocol.get("params") or {}
    subject_ids = dataset.get("subject_ids") or []
    epochs = registry.load(
        str(dataset.get("id")),
        subject_ids=subject_ids or None,
        classes=dataset.get("classes"),
        options=dataset.get("options") or {},
    )
    actual_signature = epochs.signature()
    expected_signature = dataset.get("signature")
    if expected_signature and actual_signature != expected_signature and not allow_data_mismatch:
        raise ValueError(
            "数据签名不匹配：当前加载的数据与复现包记录不同。"
            f"\n  bundle: {expected_signature}\n  local:  {actual_signature}"
            "\n确认数据版本后可用 --allow-data-mismatch 跳过校验。"
        )
    manifest = registry.manifest(str(dataset.get("id")), epochs)
    manifest["dataset_version"] = epochs.dataset_version
    manifest["selection"] = {
        "subject_ids": subject_ids,
        "classes": dataset.get("classes"),
        "options": dataset.get("options") or {},
    }
    seeds = protocol.get("seeds") or params.get("seeds") or [0]
    folds = int(params.get("folds", 5))
    repeats = int(params.get("repeats", 1))
    splits = build_protocol(
        str(protocol.get("name") or "within_subject"),
        folds=folds,
        repeats=repeats,
        seeds=tuple(int(seed) for seed in seeds),
    ).split(epochs)
    definitions, pipelines = resolve_pipeline_definitions(
        config.get("pipelines") or [], plugins=plugins
    )
    rows = run_arena(epochs, pipelines, splits, should_cancel=should_cancel)
    expected = _expected_means(config)
    comparisons = []
    for row in rows:
        pipeline = str(row["pipeline"])
        actual = float(row["balanced_accuracy_mean"])
        reference = expected.get(pipeline)
        difference = None if reference is None else actual - reference
        comparisons.append(
            {
                "pipeline": pipeline,
                "pipeline_name": row.get("pipeline_name") or pipeline,
                "expected_balanced_accuracy": reference,
                "actual_balanced_accuracy": actual,
                "difference": difference,
                "matched": reference is not None and abs(difference) <= tolerance,
            }
        )
    matched = all(item["matched"] for item in comparisons) if comparisons else False
    experiment_id = None
    if save and store is not None:
        experiment_id = store.record_experiment(
            name=f"reproduce: {config.get('experiment', {}).get('name') or 'experiment'}",
            dataset_id=str(dataset.get("id")),
            dataset_name=str(manifest.get("name", dataset.get("id"))),
            dataset_version=str(epochs.dataset_version),
            dataset_signature=actual_signature,
            dataset_manifest=manifest,
            protocol=str(protocol.get("name") or "within_subject"),
            protocol_params={"folds": folds, "repeats": repeats, "seeds": list(seeds)},
            pipelines=definitions,
            seeds=list(seeds),
            arena_rows=rows,
            duration_ms=float(sum(split.get("duration_ms", 0.0) for row in rows for split in row.get("splits", []))),
            notes=f"复现 experiment {config.get('experiment', {}).get('id')}",
        )
    return {
        "matched": matched,
        "tolerance": tolerance,
        "data_signature": actual_signature,
        "expected_signature": expected_signature,
        "comparisons": comparisons,
        "rows": rows,
        "experiment_id": experiment_id,
        "saved": experiment_id is not None,
        "note": "复现使用复现包中的划分种子与丹方定义，并逐项比较平均 BA。",
    }


def reproduce_bundle(
    registry: DatasetRegistry,
    bundle_path: str | Path,
    store: ExperimentStore | None = None,
    save: bool = False,
    tolerance: float = 0.02,
    allow_data_mismatch: bool = False,
    should_cancel: Callable[[], bool] | None = None,
    plugins: Any | None = None,
) -> dict[str, Any]:
    path = Path(bundle_path).expanduser().resolve()
    config = _read_config(path)
    return _reproduce_config(
        registry, config, store, save, tolerance, allow_data_mismatch,
        should_cancel, plugins,
    )


def reproduce_experiment(
    registry: DatasetRegistry,
    store: ExperimentStore,
    experiment_id: int,
    save: bool = False,
    tolerance: float = 0.02,
    allow_data_mismatch: bool = False,
    should_cancel: Callable[[], bool] | None = None,
    plugins: Any | None = None,
) -> dict[str, Any]:
    detail = store.get_experiment(int(experiment_id))
    if detail is None:
        raise KeyError(f"experiment {experiment_id} not found")
    config = _experiment_config(detail)
    return _reproduce_config(
        registry, config, store, save, tolerance, allow_data_mismatch,
        should_cancel, plugins,
    )


__all__ = [
    "export_bundle",
    "reproduce_bundle",
    "reproduce_experiment",
]
