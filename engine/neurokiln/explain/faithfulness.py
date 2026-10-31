"""Validity checks for attributions.

An attribution map is only useful if it is faithful to the model. These
checks are deliberately behavioural: deleting the most important blocks must
hurt the model faster than deleting random ones, and the ranking must survive
a half-split of the test trials and (when available) agree with an
independent method.

Curves are measured on the model's target log-odds (see
:func:`~neurokiln.explain.trace.log_odds`) rather than on probability,
because a saturated classifier returns exactly ``1.0`` for every trial and a
probability curve would be flat, and log-odds are shift-invariant unlike raw
logits. Each curve is then rescaled with the shared endpoints - full input
and all-baseline input - so "retained evidence" is comparable across models:
1.0 means the model is as confident as it was on the untouched epoch, 0.0
means the input carries no evidence left.
"""

from __future__ import annotations

import numpy as np

from ..errors import RunCancelled
from ..pipeline.nodes import FittedSplit
from .attribution import probe_scores
from .intrinsic import channel_reference
from .trace import SCORE_UNIT, decision_probs, score_source, spearman, train_baseline


def _check_cancel(should_cancel) -> None:
    if should_cancel is not None and should_cancel():
        raise RunCancelled("explanation cancelled")


def _blocks_probe(
    X: np.ndarray,
    template: np.ndarray,
    baseline: np.ndarray,
    blocks: list[dict],
    windows: list[tuple[int, int]],
    selected,
    *,
    mode: str,
) -> np.ndarray:
    probe = X.copy() if mode == "delete" else template.copy()
    for index in selected:
        block = blocks[int(index)]
        channels = block["channels"]
        start, stop = windows[block["window"]]
        if mode == "delete":
            probe[:, channels, start:stop] = baseline[channels][None, :, None]
        else:
            probe[:, channels, start:stop] = X[:, channels, start:stop]
    return probe


def _curve(
    fitted: FittedSplit,
    X: np.ndarray,
    baseline: np.ndarray,
    blocks: list[dict],
    windows: list[tuple[int, int]],
    order: np.ndarray,
    target: np.ndarray,
    *,
    mode: str,
    max_steps: int,
    should_cancel=None,
) -> list[float]:
    n_blocks = len(blocks)
    step = max(1, int(np.ceil(n_blocks / max_steps)))
    template = np.broadcast_to(baseline[None, :, None], X.shape).copy()
    values = []
    base_probe = X if mode == "delete" else template
    values.append(float(np.mean(probe_scores(fitted, base_probe.copy(), target))))
    for count in range(step, n_blocks + 1, step):
        _check_cancel(should_cancel)
        probe = _blocks_probe(
            X, template, baseline, blocks, windows, order[:count], mode=mode
        )
        values.append(float(np.mean(probe_scores(fitted, probe, target))))
    return values


def _fractions(n_blocks: int, max_steps: int) -> list[float]:
    step = max(1, int(np.ceil(n_blocks / max_steps)))
    fractions = [0.0]
    for count in range(step, n_blocks + 1, step):
        fractions.append(count / n_blocks)
    return fractions


def _trapezoid(values: np.ndarray, grid: np.ndarray) -> float:
    # np.trapezoid only exists from NumPy 2.0; the engine floor is 1.26.
    function = getattr(np, "trapezoid", None) or np.trapz
    return float(function(values, grid))


def _auc(fractions: list[float], values: list[float]) -> float | None:
    if len(fractions) < 2:
        return None
    area = _trapezoid(np.asarray(values), np.asarray(fractions))
    span = max(fractions[-1] - fractions[0], 1e-9)
    return area / span


def _rescale(values: list[float], high: float, low: float) -> list[float]:
    span = high - low
    if abs(span) < 1e-12:
        return [0.5 for _ in values]
    return [float((value - low) / span) for value in values]


def faithfulness(
    fitted: FittedSplit,
    occlusion: dict,
    *,
    gradient: dict | None = None,
    max_steps: int = 16,
    random_draws: int = 3,
    seed: int = 2026,
    should_cancel=None,
) -> dict:
    X = np.asarray(fitted.test_epochs.data, dtype=float)
    baseline = train_baseline(fitted)
    blocks = occlusion["blocks"]
    windows = occlusion["windows"]
    scores = np.asarray(occlusion["block_values"], dtype=float)
    target = np.asarray(occlusion["target"], dtype=int)
    order = np.argsort(-scores)
    fractions = _fractions(len(blocks), max_steps)

    deletion = _curve(
        fitted, X, baseline, blocks, windows, order, target,
        mode="delete", max_steps=max_steps, should_cancel=should_cancel,
    )
    insertion = _curve(
        fitted, X, baseline, blocks, windows, order, target,
        mode="insert", max_steps=max_steps, should_cancel=should_cancel,
    )

    rng = np.random.default_rng(seed)
    random_curves = []
    for _ in range(max(1, random_draws)):
        shuffled = rng.permutation(len(blocks))
        random_curves.append(
            _curve(
                fitted, X, baseline, blocks, windows, shuffled, target,
                mode="delete", max_steps=max_steps, should_cancel=should_cancel,
            )
        )
    random_curve = np.mean(np.asarray(random_curves), axis=0).tolist()

    high = float(deletion[0])
    low = float(deletion[-1])
    deletion_scaled = _rescale(deletion, high, low)
    insertion_scaled = _rescale(insertion, high, low)
    random_scaled = _rescale(random_curve, high, low)
    usable = (high - low) > 1e-9

    aopc_steps = min(len(deletion_scaled) - 1, 8)
    aopc = float(np.mean(1.0 - np.asarray(deletion_scaled[1: aopc_steps + 1])))
    if not usable:
        aopc = None

    per_trial = np.asarray(occlusion["per_trial_channel"], dtype=float)
    half = per_trial.shape[0] // 2
    stability = None
    if half >= 2:
        left = per_trial[:half].mean(axis=0)
        right = per_trial[half:].mean(axis=0)
        stability = spearman(left, right)

    subjects = np.asarray(fitted.test_epochs.subject_ids)
    subject_scores = []
    for subject in np.unique(subjects):
        mask = subjects == subject
        if mask.sum() >= 3:
            subject_scores.append(per_trial[mask].mean(axis=0))
    subject_consistency = None
    if len(subject_scores) >= 2:
        values = []
        for index in range(len(subject_scores)):
            for other in range(index + 1, len(subject_scores)):
                rho = spearman(subject_scores[index], subject_scores[other])
                if rho is not None:
                    values.append(rho)
        if values:
            subject_consistency = float(np.mean(values))

    reference = channel_reference(fitted)
    agreement = {}
    occlusion_magnitude = np.abs(np.asarray(occlusion["channel_importance"], dtype=float))
    if reference is not None:
        # Compare magnitudes: the sign of a CSP pattern encodes class
        # direction, while an occlusion drop is positive for every channel
        # that supports the decision.
        agreement["intrinsic"] = {
            "spearman": spearman(occlusion_magnitude, np.abs(reference)),
            "note": "遮挡通道重要性（|Δ决策分数|）与模型内在参数（CSP/xDAWN/卷积核等）的一致性",
        }
    if gradient and gradient.get("available"):
        agreement["gradient"] = {
            "spearman": spearman(occlusion_magnitude, gradient["channel_magnitude"]),
            "method": gradient.get("method"),
            "note": "遮挡与梯度归因的一致性；两种方法独立，越高越可信",
        }

    decision = decision_probs(fitted)
    base_probability = float(np.mean(decision.confidence))
    return {
        "deletion_auc": _auc(fractions, deletion_scaled) if usable else None,
        "insertion_auc": _auc(fractions, insertion_scaled) if usable else None,
        "random_deletion_auc": _auc(fractions, random_scaled) if usable else None,
        "aopc": aopc,
        "n_steps": len(fractions) - 1,
        "base_probability": round(base_probability, 5),
        "score_unit": SCORE_UNIT,
        "score_source": score_source(fitted),
        "normalization": {
            "high": round(high, 5),
            "low": round(low, 5),
            "usable": bool(usable),
            "note": (
                "曲线按完整输入（high）与全部替换为训练折均值（low）归一化；"
                "usable=False 表示模型对输入完全遮挡后分数反而更高，曲线不可解释。"
            ),
        },
        "deletion_curve": {
            "fraction": [round(value, 4) for value in fractions],
            "retained": (
                [round(float(value), 5) for value in deletion_scaled] if usable else None
            ),
            "score": [round(float(value), 5) for value in deletion],
            "random": (
                [round(float(value), 5) for value in random_scaled] if usable else None
            ),
        },
        "insertion_curve": {
            "fraction": [round(value, 4) for value in fractions],
            "retained": (
                [round(float(value), 5) for value in insertion_scaled] if usable else None
            ),
            "score": [round(float(value), 5) for value in insertion],
        },
        "stability": {
            "half_split_spearman": None if stability is None else round(stability, 4),
            "n_trials": int(per_trial.shape[0]),
        },
        "subject_consistency": {
            "spearman": None if subject_consistency is None else round(subject_consistency, 4),
            "n_subjects": len(subject_scores),
        },
        "agreement": agreement,
        "interpretation": (
            "曲线使用决策分数（概率在饱和模型上恒为 1.0，无法反映扰动），"
            "并按「完整输入 / 全部替换为训练折均值」两个端点归一化为保留证据比例。"
            "删除曲线下降越快（AUC 越小）、插入曲线上升越快（AUC 越大）越忠实；"
            "与随机删除的差距是核心证据。"
        ),
    }


__all__ = ["faithfulness"]
