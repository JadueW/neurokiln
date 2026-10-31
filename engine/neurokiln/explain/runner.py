"""Orchestrate every explanation method for one pipeline and one split.

Output layout (self-describing, consumed by the API, reports and the web UI)::

    explanations/<id>/global.json
    explanations/<id>/per_trial.json
    explanations/<id>/per_trial.parquet
    explanations/<id>/topomaps/<entry-id>.png

``provenance.background`` is always ``train_fold_only``: the occlusion
reference signal, the student model and the surrogate are all fitted on the
training fold, never on the test trials they explain.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .. import __version__ as engine_version
from ..data.model import Epochs
from ..errors import RunCancelled
from ..evaluate import balanced_accuracy, cohen_kappa
from ..pipeline.nodes import Pipeline
from ..protocol.types import Split
from .attribution import (
    DEFAULT_BANDS,
    band_ablation,
    gradient_attribution,
    occlusion_blocks,
    weight_randomization_check,
)
from .audit import knowledge_audit
from .faithfulness import faithfulness
from .intrinsic import extract_intrinsic
from .montage import topomap_png

FORMAT = "neurokiln.explain"
FORMAT_VERSION = 1


@dataclass
class ExplanationConfig:
    pipeline_id: str
    fold: int = 0
    repeat: int = 0
    window_ms: int = 250
    max_blocks: int = 120
    gradient_method: str = "integrated_gradients"
    include_audit: bool = True
    include_sanity: bool = True
    faithfulness_steps: int = 16
    bands: tuple[tuple[float, float], ...] = field(default_factory=lambda: DEFAULT_BANDS)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _plain(value: Any) -> Any:
    """Recursively convert NumPy containers so the payload is JSON-safe."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _progress(callback, start: float, end: float, label: str):
    if callback is None:
        return None
    span = max(end - start, 1e-6)

    def report(done: int, total: int) -> None:
        fraction = start + span * min(1.0, done / max(total, 1))
        callback(fraction, label)

    return report


def _write_topomap(
    directory: Path, identifier: str, values, channels, *, signed: bool
) -> dict:
    payload, meta = topomap_png(values, channels, signed=signed)
    path = directory / "topomaps" / f"{identifier}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "asset": f"topomaps/{identifier}.png",
        "meta": meta,
    }


def _attach_topomaps(
    entries: list[dict], channels: list[str], directory: Path
) -> dict[str, dict]:
    assets: dict[str, dict] = {}
    for entry in entries:
        values = entry.get("values") or []
        if len(values) != len(channels):
            continue
        rendered = _write_topomap(
            directory, entry["id"], values, channels, signed=bool(entry.get("signed"))
        )
        entry["topomap"] = rendered["asset"]
        entry["topomap_meta"] = rendered["meta"]
        assets[entry["id"]] = {"asset": rendered["asset"], "meta": rendered["meta"]}
        for per_class in entry.get("per_class") or []:
            class_id = f"{entry['id']}__class_{per_class['class']}"
            rendered_class = _write_topomap(
                directory, class_id, per_class["values"], channels,
                signed=bool(entry.get("signed")),
            )
            per_class["topomap"] = rendered_class["asset"]
            assets[class_id] = rendered_class
    return assets


def _per_trial_rows(fitted, occlusion: dict, classes, label_names) -> list[dict]:
    epochs = fitted.test_epochs
    predicted = np.asarray(occlusion["target"], dtype=int)
    predicted_labels = np.asarray(classes)[predicted]
    confidence = np.asarray(occlusion["base_confidence"], dtype=float)
    drops = np.asarray(occlusion["per_trial_channel"], dtype=float)
    channels = [str(name) for name in epochs.channel_names]
    order = np.argsort(-np.abs(drops), axis=1)[:, :5]
    rows = []
    for index in range(len(epochs.labels)):
        label = int(epochs.labels[index])
        rows.append(
            {
                "trial_id": str(epochs.trial_ids[index]),
                "subject": str(epochs.subject_ids[index]),
                "session": str(epochs.session_ids[index]),
                "label": label,
                "label_name": (
                    str(label_names[label]) if label < len(label_names) else str(label)
                ),
                "predicted": int(predicted_labels[index]),
                "predicted_name": (
                    str(label_names[int(predicted_labels[index])])
                    if int(predicted_labels[index]) < len(label_names)
                    else str(int(predicted_labels[index]))
                ),
                "confidence": round(float(confidence[index]), 5),
                "correct": bool(int(predicted_labels[index]) == label),
                "channel_drops": [round(float(value), 6) for value in drops[index]],
                "top_channels": [channels[int(item)] for item in order[index]],
            }
        )
    return rows


def _write_per_trial(directory: Path, rows: list[dict], channels: list[str]) -> dict:
    payload = {
        "channels": channels,
        "rows": rows,
    }
    (directory / "per_trial.json").write_text(_dumps(payload), encoding="utf-8")
    written = {"json": "per_trial.json", "parquet": None}
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:  # pragma: no cover - pyarrow is a hard dependency
        return written
    columns = {
        "trial_id": [row["trial_id"] for row in rows],
        "subject": [row["subject"] for row in rows],
        "session": [row["session"] for row in rows],
        "label": [row["label"] for row in rows],
        "predicted": [row["predicted"] for row in rows],
        "confidence": [row["confidence"] for row in rows],
        "correct": [row["correct"] for row in rows],
    }
    for index, channel in enumerate(channels):
        columns[f"drop_{channel}"] = [row["channel_drops"][index] for row in rows]
    table = pa.Table.from_pydict(columns)
    pq.write_table(table, directory / "per_trial.parquet")
    written["parquet"] = "per_trial.parquet"
    return written


def _warnings(payload: dict) -> list[str]:
    warnings = []
    faith = payload.get("faithfulness") or {}
    norm = faith.get("normalization") or {}
    usable = bool(norm.get("usable", True))
    if not usable:
        warnings.append(
            "模型在输入被完全遮挡后决策分数反而更高，删除/插入曲线不可解释；"
            "该模型很可能没有学到任务相关证据。"
        )
    deletion = faith.get("deletion_auc")
    random_auc = faith.get("random_deletion_auc")
    if usable and deletion is not None and random_auc is not None and deletion >= random_auc - 0.02:
        warnings.append(
            "遮挡归因的删除曲线与随机删除几乎相同，说明该解释对该模型不可信；"
            "请检查模型是否过拟合或测试折过小。"
        )
    stability = (faith.get("stability") or {}).get("half_split_spearman")
    if stability is not None and stability < 0.3:
        warnings.append("归因在测试集两半之间不一致（Spearman < 0.3），结论不稳定。")
    agreements = faith.get("agreement") or {}
    intrinsic = (agreements.get("intrinsic") or {}).get("spearman")
    if intrinsic is not None and intrinsic < 0.2:
        warnings.append(
            "遮挡归因与模型内在参数（CSP/xDAWN/卷积核）不一致，"
            "可能说明模型依赖的是分布式或非空间证据。"
        )
    sanity = payload.get("sanity") or {}
    rho = sanity.get("spearman")
    if rho is not None and abs(rho) > 0.3:
        warnings.append(
            "权重随机化后归因仍与原模型相关（|ρ| > 0.3），"
            "归因可能主要由输入统计而非模型权重驱动。"
        )
    audit = payload.get("audit") or {}
    closure = audit.get("closure_ratio")
    if closure is not None and closure < 0.6:
        warnings.append(
            f"已知频带/Hjorth 特征只能解释约 {closure * 100:.0f}% 的模型表现，"
            "模型的增益需要额外证据支撑。"
        )
    performance = payload.get("performance") or {}
    n_classes = int((payload.get("dataset") or {}).get("n_classes") or 0)
    balanced = performance.get("balanced_accuracy")
    if n_classes > 1 and balanced is not None:
        chance = 1.0 / n_classes
        if balanced <= chance + 0.05:
            warnings.append(
                f"该模型平衡准确率 {balanced:.3f} 接近随机水平（1/{n_classes} = {chance:.3f}），"
                "解释只描述了一个未学会任务的模型。"
            )
    return warnings


def run_explanation(
    epochs: Epochs,
    pipeline: Pipeline,
    split: Split,
    output_dir: str | Path,
    config: ExplanationConfig,
    *,
    explanation_id: str | None = None,
    experiment_id: int | None = None,
    pipeline_name: str | None = None,
    should_cancel: Callable[[], bool] | None = None,
    on_progress: Callable[[float, str], None] | None = None,
) -> dict:
    if not isinstance(pipeline, Pipeline):
        raise ValueError(
            "可解释性分析目前支持内置线性算法；图 DAG / 插件算法请先导出为等效的"
            "内置算法后再分析。"
        )
    started = time.perf_counter()
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    channels = [str(name) for name in epochs.channel_names]
    label_names = list(epochs.label_names)

    def note(fraction: float, message: str) -> None:
        if on_progress is not None:
            on_progress(fraction, message)

    note(0.02, "拟合训练折")
    result, fitted = pipeline.fit_split(
        epochs, split, should_cancel=should_cancel
    )
    note(0.14, "提取内在解释")

    intrinsic = extract_intrinsic(fitted)
    assets = _attach_topomaps(intrinsic, channels, directory)
    note(0.18, "遮挡归因（通道 × 时间）")

    occlusion = occlusion_blocks(
        fitted,
        window_ms=config.window_ms,
        max_blocks=config.max_blocks,
        should_cancel=should_cancel,
        on_step=_progress(on_progress, 0.18, 0.55, "遮挡归因（通道 × 时间）"),
    )
    occlusion_public = _plain(
        {key: value for key, value in occlusion.items() if key != "per_trial_channel"}
    )
    rendered = _write_topomap(
        directory, "occlusion_channel", occlusion["channel_importance"], channels,
        signed=True,
    )
    occlusion_public["topomap"] = rendered["asset"]
    assets["occlusion_channel"] = rendered
    note(0.58, "频带消融")

    bands = band_ablation(
        fitted,
        should_cancel=should_cancel,
        on_step=_progress(on_progress, 0.58, 0.72, "频带消融"),
    )
    if bands.get("available"):
        drops = np.asarray(bands["drop"], dtype=float)
        band_index = int(np.argmax(np.abs(drops))) if drops.size else 0
        low, high = bands["bands"][band_index]
        rendered = _write_topomap(
            directory, f"band_channel_{low:g}_{high:g}",
            bands["channel_importance"][f"{low:g}-{high:g}"], channels, signed=True,
        )
        bands["topomap"] = rendered["asset"]
        bands["topomap_band"] = [low, high]
        assets[f"band_channel_{low:g}_{high:g}"] = rendered
    note(0.74, "梯度归因")

    gradient = {"available": False, "reason": "未启用"}
    if config.gradient_method and config.gradient_method != "none":
        gradient = gradient_attribution(
            fitted, method=config.gradient_method
        )
        if gradient.get("available"):
            rendered = _write_topomap(
                directory, "gradient_channel", gradient["channel_importance"],
                channels, signed=True,
            )
            gradient["topomap"] = rendered["asset"]
            assets["gradient_channel"] = rendered
    note(0.82, "忠实性检验")

    faith = faithfulness(
        fitted,
        occlusion,
        gradient=gradient,
        max_steps=config.faithfulness_steps,
        should_cancel=should_cancel,
    )
    sanity = {"available": False, "reason": "未启用"}
    if config.include_sanity:
        sanity = weight_randomization_check(fitted, occlusion["channel_importance"])
    note(0.9, "知识审计")

    audit = {"available": False, "reason": "未启用"}
    if config.include_audit:
        audit = knowledge_audit(
            fitted, occlusion_channel_importance=occlusion["channel_importance"]
        )
        if audit.get("available"):
            rendered = _write_topomap(
                directory, "audit_bandpower", audit["channel_importance"], channels,
                signed=False,
            )
            audit["topomap"] = rendered["asset"]
            assets["audit_bandpower"] = rendered
    note(0.95, "写入产物")

    rows = _per_trial_rows(fitted, occlusion, occlusion["predicted_classes"], label_names)
    per_trial_files = _write_per_trial(directory, rows, channels)
    duration_ms = (time.perf_counter() - started) * 1000.0
    payload = {
        "format": FORMAT,
        "version": FORMAT_VERSION,
        "id": explanation_id,
        "experiment_id": experiment_id,
        "pipeline": {"id": config.pipeline_id, "name": pipeline_name or config.pipeline_id},
        "dataset": {
            "name": epochs.dataset,
            "version": epochs.dataset_version,
            "paradigm": epochs.paradigm,
            "signature": epochs.signature()[:16],
            "n_trials": int(epochs.n_trials),
            "n_channels": len(channels),
            "n_classes": int(epochs.n_classes),
            "n_times": int(epochs.data.shape[-1]),
            "sfreq": float(epochs.sfreq),
            "tmin": float(epochs.tmin),
            "tmax": float(epochs.tmax),
            "channel_names": channels,
            "label_names": label_names,
        },
        "split": split.summary(epochs),
        "provenance": {
            "background": "train_fold_only",
            "n_background_trials": int(fitted.train_epochs.n_trials),
            "n_test_trials": int(fitted.test_epochs.n_trials),
            "engine_version": engine_version,
            "generated_at": _now(),
            "duration_ms": round(duration_ms, 2),
            "methods": [
                "intrinsic_parameters",
                "occlusion_channel_time",
                "band_ablation",
                f"gradient_{config.gradient_method}" if gradient.get("available") else "gradient_unavailable",
                "deletion_insertion",
                "knowledge_audit",
            ],
            "notes": [
                "遮挡基线为训练折逐通道均值；测试折不参与任何拟合。",
                "归因目标是模型自身预测类别的 log-odds（对目标类 vs 其余类），"
                "对 softmax 的常数偏移不敏感且不会像概率那样饱和。",
                "topomap 使用 10-10 电极坐标与反距离插值；未知电极会标注为环形布局。",
            ],
        },
        "performance": {
            "balanced_accuracy": round(float(balanced_accuracy(result["y_true"], result["y_pred"])), 5),
            "cohen_kappa": round(float(cohen_kappa(result["y_true"], result["y_pred"])), 5),
            "n_test": int(len(result["y_true"])),
        },
        "intrinsic": intrinsic,
        "occlusion": occlusion_public,
        "bands": bands,
        "gradient": gradient,
        "faithfulness": faith,
        "sanity": sanity,
        "audit": audit,
        "assets": assets,
        "per_trial": per_trial_files,
        "config": {
            "window_ms": config.window_ms,
            "max_blocks": config.max_blocks,
            "gradient_method": config.gradient_method,
            "bands": [list(item) for item in config.bands],
        },
    }
    payload["warnings"] = _warnings(payload)
    payload = _plain(payload)
    (directory / "global.json").write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=1),
        encoding="utf-8",
    )
    note(1.0, "完成")
    return {
        "global": payload,
        "directory": str(directory),
        "rows": rows,
        "duration_ms": duration_ms,
        "summary": {
            "pipeline": payload["pipeline"],
            "performance": payload["performance"],
            "n_intrinsic": len(intrinsic),
            "has_gradient": bool(gradient.get("available")),
            "has_audit": bool(audit.get("available")),
            "deletion_auc": faith.get("deletion_auc"),
            "insertion_auc": faith.get("insertion_auc"),
            "random_deletion_auc": faith.get("random_deletion_auc"),
            "closure_ratio": audit.get("closure_ratio"),
            "warnings": payload["warnings"],
        },
    }


__all__ = ["ExplanationConfig", "FORMAT", "FORMAT_VERSION", "run_explanation"]
