"""Probability calibration diagnostics for held-out predictions.

All metrics are computed from concatenated per-fold test predictions. They
describe the reliability of the reported probabilities, not a re-fit of the
decoder; calibration on the test folds is therefore an evaluation only.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

_DEFAULT_BINS = 10


def _as_probabilities(y_proba: Any, classes: Any) -> tuple[np.ndarray, np.ndarray] | None:
    if y_proba is None or classes is None:
        return None
    proba = np.asarray(y_proba, dtype=float)
    labels = np.asarray(classes)
    if proba.ndim != 2 or labels.ndim != 1 or proba.shape[1] != labels.size:
        return None
    if proba.shape[0] == 0 or not np.all(np.isfinite(proba)):
        return None
    row_sums = proba.sum(axis=1)
    if np.any(row_sums <= 0) or not np.allclose(row_sums, 1.0, atol=1e-6):
        return None
    return proba, labels


def calibration_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_proba: Any,
    classes: Any,
    n_bins: int = _DEFAULT_BINS,
) -> dict[str, Any]:
    """ECE / MCE / Brier / NLL plus a reliability curve for one prediction set."""
    parsed = _as_probabilities(y_proba, classes)
    if parsed is None:
        return {"available": False, "reason": "该算法未提供有效的 predict_proba 输出"}
    proba, labels = parsed
    truth = np.asarray(y_true, dtype=int)
    if truth.shape != (proba.shape[0],):
        return {"available": False, "reason": "概率与真实标签长度不一致"}
    lookup = {int(value): index for index, value in enumerate(labels)}
    known = np.asarray([int(value) in lookup for value in truth])
    if not bool(known.any()):
        return {"available": False, "reason": "测试标签不在训练类别中"}
    proba = proba[known]
    truth = truth[known]
    true_index = np.asarray([lookup[int(value)] for value in truth], dtype=int)

    confidence = proba.max(axis=1)
    predicted = labels[proba.argmax(axis=1)]
    correct = predicted == truth
    n_bins = max(2, int(n_bins))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(confidence, edges[1:-1], right=False), 0, n_bins - 1)

    reliability: list[dict[str, float]] = []
    ece = 0.0
    mce = 0.0
    for index in range(n_bins):
        members = bins == index
        count = int(members.sum())
        lower, upper = float(edges[index]), float(edges[index + 1])
        if count == 0:
            continue
        avg_confidence = float(confidence[members].mean())
        accuracy = float(correct[members].mean())
        gap = abs(avg_confidence - accuracy)
        ece += gap * count / confidence.size
        mce = max(mce, gap)
        reliability.append(
            {
                "bin_lower": round(lower, 4),
                "bin_upper": round(upper, 4),
                "confidence": round(avg_confidence, 6),
                "accuracy": round(accuracy, 6),
                "count": count,
            }
        )

    one_hot = np.zeros_like(proba)
    one_hot[np.arange(proba.shape[0]), true_index] = 1.0
    brier = float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))
    nll = float(-np.mean(np.log(np.clip(proba[np.arange(proba.shape[0]), true_index], 1e-12, 1.0))))
    return {
        "available": True,
        "n_evaluated": int(confidence.size),
        "n_bins": n_bins,
        "ece": round(float(ece), 6),
        "mce": round(float(mce), 6),
        "brier": round(brier, 6),
        "nll": round(nll, 6),
        "accuracy": round(float(correct.mean()), 6),
        "confidence_mean": round(float(confidence.mean()), 6),
        "reliability": reliability,
    }


def aggregate_calibration(
    items: Iterable[dict[str, Any]], n_bins: int = _DEFAULT_BINS
) -> dict[str, Any]:
    """Combine per-split calibration summaries, weighting by evaluated trials."""
    usable = [item for item in items if item.get("available")]
    if not usable:
        return {"available": False, "reason": "没有可用的概率输出（算法未实现 predict_proba）"}
    weights = np.asarray([max(1, int(item.get("n_evaluated", 0))) for item in usable], dtype=float)
    total = float(weights.sum())
    payload: dict[str, Any] = {
        "available": True,
        "n_evaluated": int(sum(int(item.get("n_evaluated", 0)) for item in usable)),
        "n_bins": max(2, int(n_bins)),
    }
    for key in ("ece", "mce", "brier", "nll", "accuracy", "confidence_mean"):
        payload[key] = round(
            float(sum(float(item.get(key, 0.0)) * weight for item, weight in zip(usable, weights)) / total),
            6,
        )
    bins: dict[int, dict[str, float]] = {}
    for item, weight in zip(usable, weights):
        for curve in item.get("reliability", []):
            key = int(round(float(curve.get("bin_lower", 0.0)) * 1000))
            entry = bins.setdefault(
                key,
                {"bin_lower": curve.get("bin_lower", 0.0), "bin_upper": curve.get("bin_upper", 0.0),
                 "confidence": 0.0, "accuracy": 0.0, "count": 0.0, "weight": 0.0},
            )
            count = float(curve.get("count", 0))
            entry["confidence"] += float(curve.get("confidence", 0.0)) * count
            entry["accuracy"] += float(curve.get("accuracy", 0.0)) * count
            entry["count"] += count
            entry["weight"] += count
    payload["reliability"] = [
        {
            "bin_lower": round(float(entry["bin_lower"]), 4),
            "bin_upper": round(float(entry["bin_upper"]), 4),
            "confidence": round(entry["confidence"] / max(entry["weight"], 1e-9), 6),
            "accuracy": round(entry["accuracy"] / max(entry["weight"], 1e-9), 6),
            "count": int(entry["count"]),
        }
        for _, entry in sorted(bins.items())
    ]
    return payload


__all__ = ["aggregate_calibration", "calibration_metrics"]
