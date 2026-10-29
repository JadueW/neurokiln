import numpy as np
import pytest

from neurokiln.evaluate import balanced_accuracy
from neurokiln.stats import bootstrap_ci, compare_pipelines, holm_correction


def _rows(scores: dict[str, list[float]]) -> list[dict]:
    return [
        {
            "pipeline": name,
            "splits": [{"balanced_accuracy": value} for value in values],
        }
        for name, values in scores.items()
    ]


def _subject_rows(subjects: list[str], correct: dict[str, int]) -> list[dict]:
    rows = []
    for pipeline, hits in correct.items():
        splits = []
        for subject in subjects:
            truth = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
            predicted = list(truth)
            for index in range(len(truth) - hits):
                predicted[index] = 1 - truth[index]
            splits.append(
                {
                    "balanced_accuracy": balanced_accuracy(
                        np.asarray(truth), np.asarray(predicted)
                    ),
                    "y_true": truth,
                    "y_pred": predicted,
                    "split": {
                        "test_subjects": [subject],
                        "test_subject_ids": [subject] * len(truth),
                    },
                }
            )
        rows.append({"pipeline": pipeline, "splits": splits})
    return rows


def test_holm_correction_preserves_order_and_monotonicity():
    adjusted = holm_correction([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])
    assert holm_correction([float("nan"), 0.5]) == [1.0, 1.0]


def test_compare_pipelines_detects_consistent_differences():
    base = np.linspace(0.70, 0.78, 8)
    rows = _rows(
        {
            "a": list(base + 0.10),
            "b": list(base + 0.03),
            "c": list(base - 0.02),
        }
    )
    result = compare_pipelines(rows)
    assert result["n_splits"] == 8
    assert result["friedman"]["available"]
    assert result["friedman"]["p_value"] < 0.05
    ranks = [item["pipeline"] for item in result["average_ranks"]["ranks"]]
    assert ranks == ["a", "b", "c"]
    assert all(pair["significant"] for pair in result["pairwise"])
    assert any(
        {pair["pipeline_a"], pair["pipeline_b"]} == {"a", "c"}
        for pair in result["pairwise"]
    )
    top = result["pairwise"][0]
    assert top["p_value_adjusted"] < 0.05
    assert 0.0 < top["p_value_adjusted"] <= 1.0
    assert result["nemenyi"]["critical_difference"] > 0


def test_identical_pipelines_produce_no_significant_pairs():
    values = [0.81, 0.79, 0.83, 0.80, 0.82]
    result = compare_pipelines(_rows({"a": values, "b": values, "c": values}))
    assert all(not pair["significant"] for pair in result["pairwise"])
    assert not result["friedman"]["available"]


def test_single_split_only_returns_summary():
    result = compare_pipelines(_rows({"a": [0.8], "b": [0.7]}))
    assert result["n_splits"] == 1
    assert result["pairwise"] == []
    assert not result["nemenyi"]["available"]


def test_subject_level_inference_uses_subjects_as_units():
    rows = _subject_rows(["S01", "S02", "S03"], {"a": 9, "b": 6})
    result = compare_pipelines(rows)
    assert result["unit"] == "subject"
    assert result["n_units"] == 3
    assert result["n_subjects"] == 3
    assert result["subject_level"]["available"]
    assert result["pairwise"]
    assert result["pairwise"][0]["pipeline_a"] == "a"
    assert result["pairwise"][0]["mean_diff"] == pytest.approx(0.3)
    ci = result["ci"]["pipelines"]["a"]["ci95"]
    assert ci[0] <= ci[1]
    scores = {item["subject"] for item in result["subject_level"]["scores"]["a"]}
    assert scores == {"S01", "S02", "S03"}


def test_single_subject_falls_back_to_fold_level_with_explicit_note():
    rows = _subject_rows(["S01"], {"a": 9, "b": 6})
    result = compare_pipelines(rows)
    assert result["unit"] == "fold"
    assert not result["subject_level"]["available"]
    assert "回退" in result["note"]
    assert result["fold_level"]["ci"]["pipelines"]["a"]["ci95"] == [0.9, 0.9]


def test_missing_trial_subject_metadata_falls_back_to_fold_level():
    rows = _rows({"a": [0.8, 0.9], "b": [0.7, 0.75]})
    result = compare_pipelines(rows)
    assert result["unit"] == "fold"
    assert "逐试次被试信息不完整" in result["note"]


def test_bootstrap_ci_is_deterministic_and_handles_small_samples():
    values = [0.6, 0.7, 0.8, 0.9]
    first = bootstrap_ci(values, seed=7)
    second = bootstrap_ci(values, seed=7)
    assert first == second
    assert first[0] <= np.mean(values) <= first[1]
    assert bootstrap_ci([0.75]) == [0.75, 0.75]
    assert bootstrap_ci([]) == [None, None]
