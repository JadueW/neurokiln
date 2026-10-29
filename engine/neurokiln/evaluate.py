from __future__ import annotations

import time
from typing import Callable

import numpy as np

from .cache import ArtifactCache
from .calibration import aggregate_calibration, calibration_metrics
from .data.model import Epochs
from .errors import RunCancelled
from .pipeline.nodes import Pipeline
from .protocol.types import Split


__all__ = [
    "RunCancelled",
    "balanced_accuracy",
    "cohen_kappa",
    "confusion_matrix",
    "run_arena",
]


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    k = n_classes or int(max(y_true.max(), y_pred.max())) + 1
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = []
    for cls in np.unique(y_true):
        mask = y_true == cls
        recalls.append(float((y_pred[mask] == cls).mean()))
    return float(np.mean(recalls))


def cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = confusion_matrix(y_true, y_pred)
    total = cm.sum()
    observed = np.trace(cm) / total
    expected = (cm.sum(axis=0) * cm.sum(axis=1)).sum() / (total * total)
    if expected == 1.0:
        return 0.0
    return float((observed - expected) / (1.0 - expected))


def run_arena(
    epochs: Epochs,
    pipelines: list[Pipeline],
    splits: list[Split],
    on_progress: Callable[[int, int, dict], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    cache: ArtifactCache | None = None,
    data_signature: str | None = None,
) -> list[dict]:
    rows = []
    total = max(1, len(pipelines) * len(splits))
    done = 0
    signature = data_signature or epochs.signature()
    for pipeline in pipelines:
        per_split = []
        for split_index, split in enumerate(splits):
            if should_cancel is not None and should_cancel():
                raise RunCancelled("arena run cancelled")
            started = time.perf_counter()
            result = pipeline.run_split(
                epochs,
                split,
                should_cancel=should_cancel,
                cache=cache,
                data_signature=signature,
            )
            duration_ms = (time.perf_counter() - started) * 1000.0
            proba = result.get("y_proba")
            proba_classes = result.get("proba_classes")
            calibration = calibration_metrics(
                result["y_true"], proba, proba_classes
            )
            per_split.append({
                "split": {
                    **split.summary(epochs),
                    "test_trial_ids": np.asarray(result["test_trial_ids"]).astype(str).tolist(),
                    "test_subject_ids": np.asarray(result["test_subject_ids"]).astype(str).tolist(),
                    "test_session_ids": np.asarray(result["test_session_ids"]).astype(str).tolist(),
                },
                "balanced_accuracy": balanced_accuracy(result["y_true"], result["y_pred"]),
                "cohen_kappa": cohen_kappa(result["y_true"], result["y_pred"]),
                "duration_ms": duration_ms,
                "y_true": result["y_true"].tolist(),
                "y_pred": result["y_pred"].tolist(),
                "y_proba": None if proba is None else np.asarray(proba, dtype=float).tolist(),
                "proba_classes": (
                    None if proba_classes is None else np.asarray(proba_classes).astype(int).tolist()
                ),
                "calibration": calibration,
            })
            done += 1
            if on_progress is not None:
                on_progress(done, total, {
                    "pipeline": pipeline.name,
                    "split_index": split_index,
                    "n_splits": len(splits),
                })
        rows.append({
            "pipeline": pipeline.name,
            "n_splits": len(splits),
            "balanced_accuracy_mean": float(np.mean([r["balanced_accuracy"] for r in per_split])),
            "balanced_accuracy_std": float(np.std([r["balanced_accuracy"] for r in per_split])),
            "cohen_kappa_mean": float(np.mean([r["cohen_kappa"] for r in per_split])),
            "calibration": aggregate_calibration(
                [r["calibration"] for r in per_split]
            ),
            "splits": per_split,
        })
    rows.sort(key=lambda r: r["balanced_accuracy_mean"], reverse=True)
    return rows
