from __future__ import annotations

import math
import warnings
from collections import defaultdict
from typing import Any, Sequence

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from .evaluate import balanced_accuracy

# Two-tailed Nemenyi critical values q_alpha for alpha=0.05,
# Demsar (2006), "Statistical Comparisons of Classifiers over Multiple Data Sets".
_NEMENYI_Q = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
    6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
}

_BOOTSTRAP_SEED = 20261029


def holm_correction(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, preserving the input order."""
    values = [1.0 if not math.isfinite(float(p)) else float(p) for p in p_values]
    order = sorted(range(len(values)), key=lambda index: values[index])
    adjusted = [1.0] * len(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, values[index] * (total - rank))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = _BOOTSTRAP_SEED,
) -> list[float | None]:
    """Percentile bootstrap CI for the mean of independent units."""
    finite = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=float)
    if finite.size == 0:
        return [None, None]
    if finite.size == 1:
        return [float(finite[0]), float(finite[0])]
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, finite.size, size=(max(200, int(n_boot)), finite.size))
    means = finite[samples].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return [float(low), float(high)]


def _score_matrix(rows: Sequence[dict[str, Any]]) -> tuple[list[str], np.ndarray, int]:
    usable = [row for row in rows if row.get("splits")]
    names = [str(row["pipeline"]) for row in usable]
    lengths = {len(row["splits"]) for row in usable}
    if not usable or len(lengths) != 1:
        return names, np.empty((0, len(names))), 0
    matrix = np.asarray(
        [
            [float(split["balanced_accuracy"]) for split in row["splits"]]
            for row in usable
        ],
        dtype=float,
    ).T
    return names, matrix, matrix.shape[0]


def _split_subject_ids(split: dict[str, Any], n_trials: int) -> list[str] | None:
    meta = split.get("split") or {}
    ids = meta.get("test_subject_ids")
    if isinstance(ids, list) and len(ids) == n_trials:
        return [str(value) for value in ids]
    subjects = meta.get("test_subjects") or []
    if len(subjects) == 1:
        return [str(subjects[0])] * n_trials
    return None


def _subject_matrix(
    rows: Sequence[dict[str, Any]],
) -> tuple[bool, str, list[str], np.ndarray, dict[str, list[dict[str, Any]]]]:
    """Build a pipeline x subject matrix from per-trial predictions.

    A subject's score is the mean balanced accuracy over all held-out splits in
    which that subject appears. Only subjects present for every pipeline enter
    the inferential matrix, so the Wilcoxon/Friedman blocks stay paired.
    """
    names = [str(row["pipeline"]) for row in rows]
    per_pipeline: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        name = str(row["pipeline"])
        values: dict[str, list[float]] = defaultdict(list)
        trials: dict[str, int] = defaultdict(int)
        usable = True
        for split in row.get("splits") or []:
            y_true = split.get("y_true")
            y_pred = split.get("y_pred")
            if y_true is None or y_pred is None or len(y_true) != len(y_pred) or not y_true:
                usable = False
                break
            ids = _split_subject_ids(split, len(y_true))
            if ids is None:
                usable = False
                break
            truth = np.asarray(y_true, dtype=int)
            predicted = np.asarray(y_pred, dtype=int)
            id_array = np.asarray(ids, dtype=str)
            for subject in np.unique(id_array):
                mask = id_array == subject
                values[str(subject)].append(
                    balanced_accuracy(truth[mask], predicted[mask])
                )
                trials[str(subject)] += int(mask.sum())
        if not usable or not values:
            per_pipeline[name] = {}
            continue
        per_pipeline[name] = {
            subject: {
                "balanced_accuracy": float(np.mean(scores)),
                "n_splits": len(scores),
                "n_trials": trials[subject],
            }
            for subject, scores in values.items()
        }

    if not names or any(not per_pipeline.get(name) for name in names):
        return False, "逐试次被试信息不完整，无法进行被试级检验", names, np.empty((0, len(names))), {}

    common = sorted(set.intersection(*(set(per_pipeline[name]) for name in names)))
    scores = {
        name: [
            {"subject": subject, **per_pipeline[name][subject]}
            for subject in sorted(per_pipeline[name])
        ]
        for name in names
    }
    if len(common) < 2:
        return (
            False,
            "仅有一个被试（或没有共同被试），无法进行被试级推断；已回退到折级检验",
            names,
            np.empty((0, len(names))),
            scores,
        )
    matrix = np.asarray(
        [
            [per_pipeline[name][subject]["balanced_accuracy"] for name in names]
            for subject in common
        ],
        dtype=float,
    )
    return True, "", names, matrix, scores


def _pairwise_wilcoxon(
    names: Sequence[str], matrix: np.ndarray, alpha: float
) -> list[dict[str, Any]]:
    if matrix.shape[0] < 2 or len(names) < 2:
        return []
    pairs = []
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            a = matrix[:, left]
            b = matrix[:, right]
            statistic, p_value = float("nan"), 1.0
            if np.any(np.abs(a - b) > 1e-12):
                try:
                    result = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
                    statistic, p_value = float(result.statistic), float(result.pvalue)
                except ValueError:
                    statistic, p_value = float("nan"), 1.0
            else:
                statistic = 0.0
            pairs.append(
                {
                    "pipeline_a": names[left],
                    "pipeline_b": names[right],
                    "statistic": statistic,
                    "p_value": p_value,
                    "mean_diff": float(np.mean(a) - np.mean(b)),
                }
            )
    adjusted = holm_correction([pair["p_value"] for pair in pairs])
    for pair, corrected in zip(pairs, adjusted):
        pair["p_value_adjusted"] = float(corrected)
        pair["significant"] = bool(corrected < alpha)
    pairs.sort(key=lambda item: item["p_value_adjusted"])
    return pairs


def _friedman(names: Sequence[str], matrix: np.ndarray) -> dict[str, Any]:
    if len(names) < 3 or matrix.shape[0] < 2:
        return {
            "available": False,
            "reason": "Friedman 检验需要至少 3 个算法与 2 个检验单位",
            "statistic": None,
            "p_value": None,
            "significant": False,
        }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = friedmanchisquare(*[matrix[:, index] for index in range(len(names))])
        statistic, p_value = float(result.statistic), float(result.pvalue)
    except ValueError as exc:
        return {
            "available": False,
            "reason": str(exc),
            "statistic": None,
            "p_value": None,
            "significant": False,
        }
    if not (math.isfinite(statistic) and math.isfinite(p_value)):
        return {
            "available": False,
            "reason": "所有算法在全部检验单位上的得分完全相同，无法计算 Friedman 统计量",
            "statistic": None,
            "p_value": None,
            "significant": False,
        }
    return {
        "available": True,
        "reason": "",
        "statistic": statistic,
        "p_value": p_value,
        "significant": bool(p_value < 0.05),
    }


def _average_ranks(
    names: Sequence[str], matrix: np.ndarray
) -> dict[str, Any]:
    if matrix.size == 0:
        return {"available": False, "ranks": []}
    ranks = np.stack(
        [rankdata(-matrix[:, index], method="average") for index in range(len(names))],
        axis=1,
    )
    values = [
        {
            "pipeline": names[index],
            "mean_rank": float(ranks[:, index].mean()),
            "mean_balanced_accuracy": float(matrix[:, index].mean()),
        }
        for index in range(len(names))
    ]
    values.sort(key=lambda item: item["mean_rank"])
    return {"available": True, "ranks": values}


def _nemenyi(
    names: Sequence[str], matrix: np.ndarray, alpha: float
) -> dict[str, Any]:
    k = len(names)
    if k < 2 or matrix.shape[0] < 2 or k not in _NEMENYI_Q:
        return {
            "available": False,
            "reason": "Nemenyi 需要 2–10 个算法与至少 2 个检验单位",
            "critical_difference": None,
            "pairs": [],
        }
    q_alpha = _NEMENYI_Q[k]
    critical_difference = q_alpha * math.sqrt(k * (k + 1) / (6.0 * matrix.shape[0]))
    ranks = _average_ranks(names, matrix)["ranks"]
    rank_by_name = {item["pipeline"]: item["mean_rank"] for item in ranks}
    pairs = []
    for left in range(k):
        for right in range(left + 1, k):
            difference = abs(rank_by_name[names[left]] - rank_by_name[names[right]])
            pairs.append(
                {
                    "pipeline_a": names[left],
                    "pipeline_b": names[right],
                    "rank_diff": float(difference),
                    "significant": bool(difference > critical_difference),
                }
            )
    pairs.sort(key=lambda item: item["rank_diff"], reverse=True)
    return {
        "available": True,
        "reason": "",
        "q_alpha": q_alpha,
        "critical_difference": float(critical_difference),
        "pairs": pairs,
    }


def _level_result(
    names: Sequence[str], matrix: np.ndarray, alpha: float, unit: str
) -> dict[str, Any]:
    ci = {}
    if matrix.size:
        for index, name in enumerate(names):
            values = matrix[:, index]
            ci[name] = {
                "balanced_accuracy_mean": float(values.mean()),
                "ci95": bootstrap_ci(values, alpha=alpha),
                "n_units": int(values.size),
            }
    return {
        "unit": unit,
        "n_units": int(matrix.shape[0]),
        "pairwise": _pairwise_wilcoxon(names, matrix, alpha),
        "friedman": _friedman(names, matrix),
        "average_ranks": _average_ranks(names, matrix),
        "nemenyi": _nemenyi(names, matrix, alpha),
        "ci": {"unit": unit, "n_units": int(matrix.shape[0]), "pipelines": ci},
    }


def compare_pipelines(
    rows: Sequence[dict[str, Any]], alpha: float = 0.05
) -> dict[str, Any]:
    """Compare arena pipelines at the subject level when possible.

    Subject-level inference is preferred because folds are not independent.
    With a single subject (typical within-subject experiment) the function
    falls back to paired fold-level tests and says so explicitly, so a
    single-subject result is never presented as population-level evidence.
    """
    fold_names, fold_matrix, n_splits = _score_matrix(rows)
    subject_available, subject_reason, subject_names, subject_matrix, subject_scores = (
        _subject_matrix(rows)
    )
    fold = _level_result(fold_names, fold_matrix, alpha, "fold")
    subject = {
        **_level_result(subject_names, subject_matrix, alpha, "subject"),
        "available": bool(subject_available),
        "reason": subject_reason,
        "n_subjects": int(subject_matrix.shape[0]),
        "scores": subject_scores,
    }
    active = subject if subject_available else fold
    all_names = [str(row["pipeline"]) for row in rows]
    max_splits = max((len(row.get("splits") or []) for row in rows), default=0)
    unit_note = (
        f"以被试为检验单位（n={subject['n_units']}）；每个被试的分数先在其全部测试折上取平均，"
        "配对 Wilcoxon p 值经 Holm 校正。"
        if subject_available
        else f"以折为检验单位（n={fold['n_units'] or max_splits}）；{subject_reason}。"
    )
    return {
        **active,
        "alpha": alpha,
        "n_pipelines": len(all_names),
        "n_splits": n_splits or max_splits,
        "n_subjects": int(subject.get("n_subjects", 0)),
        "unit": active["unit"],
        "n_units": active["n_units"],
        "note": unit_note,
        "fold_level": fold,
        "subject_level": subject,
    }


__all__ = ["bootstrap_ci", "compare_pipelines", "holm_correction"]
