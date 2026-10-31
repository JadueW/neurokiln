"""Helpers shared by the explainability methods.

Everything here operates on a :class:`FittedSplit`: the same fitted objects
and the same train/test arrays that produced the reported metrics. No method
in this package ever calls ``fit`` again, so attribution can never see data
the model did not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from ..pipeline.nodes import FittedSplit


@dataclass(frozen=True)
class DecisionProbs:
    proba: np.ndarray
    classes: np.ndarray
    predicted_index: np.ndarray
    confidence: np.ndarray

    @property
    def n_trials(self) -> int:
        return self.proba.shape[0]


def decision_probs(fitted: FittedSplit, X: np.ndarray | None = None) -> DecisionProbs:
    """Probability of the model's own decision for every trial.

    Attributions explain the model's decision, so the target is the
    probability it assigned to its argmax class rather than the ground-truth
    class (which would describe the data, not the model).
    """
    data = fitted.test_epochs.data if X is None else np.asarray(X)
    y_pred, proba, classes = fitted.predict_from_raw(data)
    if proba is None or classes is None:
        raise RuntimeError("该算法没有概率输出，无法计算基于置信度的归因")
    classes = np.asarray(classes)
    indices = np.searchsorted(classes, y_pred)
    indices = np.clip(indices, 0, classes.size - 1)
    confidence = proba[np.arange(len(indices)), indices]
    return DecisionProbs(
        proba=np.asarray(proba, dtype=float),
        classes=classes,
        predicted_index=indices,
        confidence=np.asarray(confidence, dtype=float),
    )


def true_class_confidence(fitted: FittedSplit, X: np.ndarray | None = None) -> np.ndarray:
    data = fitted.test_epochs.data if X is None else np.asarray(X)
    labels = fitted.test_epochs.labels
    decision = decision_probs(fitted, data)
    indices = np.searchsorted(decision.classes, labels)
    indices = np.clip(indices, 0, decision.classes.size - 1)
    return decision.proba[np.arange(len(labels)), indices]


def train_baseline(fitted: FittedSplit) -> np.ndarray:
    """Per-channel training-fold mean, the occlusion reference signal."""
    return np.asarray(fitted.train_epochs.data, dtype=float).mean(axis=(0, 2))


def decoder_scores(decoder, features: np.ndarray) -> np.ndarray | None:
    """Per-class decision scores ``(n_trials, n_classes)`` for any decoder.

    Probabilities saturate: a confident LDA returns exactly ``1.0`` for every
    trial, so probability deltas collapse to zero and attribution becomes
    useless. Decision scores keep the same ordering with an unbounded scale,
    which is what occlusion and deletion curves need.
    """
    from ..pipeline.deep import TorchDecoder
    from ..pipeline.riemann import RiemannMDM
    from ..pipeline.ssvep import SSVEPCCA

    if isinstance(decoder, SSVEPCCA):
        return np.asarray(decoder._scores(features), dtype=float)
    if isinstance(decoder, TorchDecoder):
        try:
            import torch
        except ImportError:  # pragma: no cover - torch is optional
            return None
        prepared = np.asarray(decoder.prepare(features), dtype=np.float32)
        with torch.no_grad():
            logits = decoder.model_(torch.from_numpy(prepared).unsqueeze(1))
        return logits.detach().cpu().numpy().astype(float)
    if isinstance(decoder, RiemannMDM):
        distances = getattr(decoder.model_, "_predict_distances", None)
        if callable(distances):
            return -np.asarray(distances(features), dtype=float)
        return None
    model = getattr(decoder, "model_", None)
    for candidate in (decoder, model):
        function = getattr(candidate, "decision_function", None)
        if callable(function):
            values = np.asarray(function(features), dtype=float)
            if values.ndim == 1:
                return np.stack([-values, values], axis=1)
            return values
    return None


def decision_scores(fitted: FittedSplit, X: np.ndarray | None = None):
    """Raw per-class decision scores, preferring the model's own margin."""
    data = fitted.test_epochs.data if X is None else np.asarray(X)
    values = fitted.transform_from_raw(data)
    decoder = fitted.traces[-1].node
    scores = decoder_scores(decoder, values)
    if scores is None:
        decision = decision_probs(fitted, data)
        scores = np.log(np.clip(decision.proba, 1e-12, 1.0))
    return np.asarray(scores, dtype=float)


def log_odds(scores: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Target class against every other class, in log-odds.

    Shift-invariant (softmax does not care about a constant offset) and
    unbounded (a saturated model keeps moving), which is what perturbation
    oracles need. Positive values mean the model still prefers the target.
    """
    scores = np.asarray(scores, dtype=float)
    target = np.asarray(target, dtype=int).reshape(-1)
    if scores.ndim != 2 or scores.shape[1] < 2:
        return np.zeros(scores.shape[0] if scores.ndim else 0)
    rows = np.arange(scores.shape[0])
    index = np.clip(target, 0, scores.shape[1] - 1)
    chosen = scores[rows, index]
    masked = scores.copy()
    masked[rows, index] = -np.inf
    return chosen - np.logaddexp.reduce(masked, axis=1)


def target_log_odds(fitted: FittedSplit, X: np.ndarray, target: np.ndarray) -> np.ndarray:
    return log_odds(decision_scores(fitted, X), target)


SCORE_UNIT = "log_odds"


def score_source(fitted: FittedSplit) -> str:
    """Where the log-odds came from, reported verbatim in artifacts."""
    from ..pipeline.deep import TorchDecoder
    from ..pipeline.riemann import RiemannMDM
    from ..pipeline.ssvep import SSVEPCCA

    decoder = fitted.traces[-1].node
    if isinstance(decoder, (SSVEPCCA, TorchDecoder, RiemannMDM)):
        return "decision_function"
    for candidate in (decoder, getattr(decoder, "model_", None)):
        if callable(getattr(candidate, "decision_function", None)):
            return "decision_function"
    return "log_probability"


def channel_groups(n_channels: int, max_groups: int = 24) -> list[np.ndarray]:
    if n_channels <= max_groups:
        return [np.array([index]) for index in range(n_channels)]
    edges = np.linspace(0, n_channels, max_groups + 1).astype(int)
    return [np.arange(edges[i], edges[i + 1]) for i in range(max_groups) if edges[i + 1] > edges[i]]


def window_slices(n_times: int, n_windows: int) -> list[slice]:
    n_windows = max(1, min(int(n_windows), int(n_times)))
    edges = np.linspace(0, n_times, n_windows + 1).astype(int)
    return [slice(int(edges[i]), int(edges[i + 1])) for i in range(n_windows) if edges[i + 1] > edges[i]]


def normalize_magnitude(values, limit: float | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scale = float(limit) if limit is not None else float(np.nanmax(np.abs(values)) or 1.0)
    return values / max(scale, 1e-12)


def spearman(left, right) -> float | None:
    left = np.asarray(left, dtype=float).ravel()
    right = np.asarray(right, dtype=float).ravel()
    if left.size < 3 or np.allclose(left, left[0]) or np.allclose(right, right[0]):
        return None
    rho = spearmanr(left, right).statistic
    return None if not np.isfinite(rho) else float(rho)


def top_k_agreement(left, right, k: int = 5) -> float | None:
    """Jaccard overlap of the top-k channels between two rankings."""
    left = np.asarray(left, dtype=float).ravel()
    right = np.asarray(right, dtype=float).ravel()
    if left.size < k or right.size < k:
        return None
    a = set(np.argsort(-np.abs(left))[:k].tolist())
    b = set(np.argsort(-np.abs(right))[:k].tolist())
    return len(a & b) / len(a | b)


__all__ = [
    "SCORE_UNIT",
    "DecisionProbs",
    "channel_groups",
    "decision_probs",
    "decision_scores",
    "decoder_scores",
    "log_odds",
    "normalize_magnitude",
    "score_source",
    "spearman",
    "target_log_odds",
    "top_k_agreement",
    "train_baseline",
    "true_class_confidence",
    "window_slices",
]
