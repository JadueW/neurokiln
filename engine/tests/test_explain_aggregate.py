import copy

import numpy as np
import pytest

from neurokiln.data import Epochs
from neurokiln.explain import ExplanationConfig, aggregate_explanations, run_explanation
from neurokiln.pipeline import build_baseline
from neurokiln.protocol import WithinSubjectProtocol


def make_oscillatory_epochs(
    n_trials_per_class: int = 24,
    n_classes: int = 2,
    n_channels: int = 8,
    n_times: int = 192,
    sfreq: float = 128.0,
) -> Epochs:
    rng = np.random.default_rng(7)
    labels = np.repeat(np.arange(n_classes), n_trials_per_class)
    times = np.arange(n_times) / sfreq
    patterns = rng.normal(size=(n_classes, n_channels))
    data = rng.normal(0.0, 2.5, size=(labels.size, n_channels, n_times))
    for index, label in enumerate(labels):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        source = 1.2 * np.sin(2.0 * np.pi * (10.0 + 8.0 * label) * times + phase)
        data[index] += patterns[label][:, None] * source
    order = rng.permutation(labels.size)
    return Epochs(
        data=data[order],
        labels=labels[order],
        sfreq=sfreq,
        channel_names=tuple(f"C{index + 1}" for index in range(n_channels)),
        tmax=n_times / sfreq,
        dataset="oscillatory",
        dataset_version="test",
    )


def _run_fold(tmp_path, fold=0):
    epochs = make_oscillatory_epochs()
    split = WithinSubjectProtocol(folds=4, seeds=(2,)).split(epochs)[fold]
    pipeline = build_baseline("csp_lda")
    result = run_explanation(
        epochs,
        pipeline,
        split,
        tmp_path / f"explanation-{fold}",
        ExplanationConfig(
            pipeline_id="csp_lda",
            fold=fold,
            window_ms=250,
            max_blocks=16,
            gradient_method="none",
            include_audit=True,
            include_sanity=False,
            faithfulness_steps=8,
        ),
        explanation_id=str(fold + 1),
    )
    return result["global"]


def test_aggregate_cross_fold_documents(tmp_path):
    first = _run_fold(tmp_path, 0)
    second = _run_fold(tmp_path, 1)
    aggregate = aggregate_explanations([first, second])

    assert aggregate["format"] == "neurokiln.explain.aggregate"
    assert aggregate["pipeline"]["id"] == "csp_lda"
    assert aggregate["provenance"]["n_explanations"] == 2
    assert aggregate["provenance"]["n_unique_splits"] == 2
    assert aggregate["provenance"]["backgrounds"] == ["train_fold_only"]

    channels = aggregate["dataset"]["channel_names"]
    assert channels == first["dataset"]["channel_names"]
    occlusion = aggregate["occlusion"]
    assert occlusion["available_count"] == 2
    assert len(occlusion["channel_importance"]["mean"]) == len(channels)
    assert occlusion["channel_importance"]["n"] == [2] * len(channels)
    expected = np.mean(
        [
            first["occlusion"]["channel_importance"],
            second["occlusion"]["channel_importance"],
        ],
        axis=0,
    )
    assert np.allclose(occlusion["channel_importance"]["mean"], expected, atol=1e-5)

    intrinsic = {entry["id"]: entry for entry in aggregate["intrinsic"]}
    assert "csp_patterns" in intrinsic
    assert intrinsic["csp_patterns"]["n"] == [2] * len(channels)

    assert len(aggregate["bands"]) == len(first["bands"]["bands"])
    for band in aggregate["bands"]:
        assert band["drop"]["n"] == 2
        assert len(band["channel_importance"]["mean"]) == len(channels)

    expected_ba = (
        first["performance"]["balanced_accuracy"]
        + second["performance"]["balanced_accuracy"]
    ) / 2
    assert aggregate["performance"]["balanced_accuracy"]["mean"] == pytest.approx(
        expected_ba, abs=1e-5
    )
    assert aggregate["performance"]["n_test_total"] == 2 * first["performance"]["n_test"]
    assert aggregate["faithfulness"]["available_count"] == 2
    assert aggregate["audit"]["available_count"] == 2
    assert len(aggregate["top_channels"]) == min(10, len(channels))
    assert aggregate["splits"][0]["explanation_id"] in {"1", "2"}


def test_aggregate_aligns_channels_by_name(tmp_path):
    first = _run_fold(tmp_path, 0)
    second = _run_fold(tmp_path, 1)
    shuffled = copy.deepcopy(second)
    names = list(shuffled["dataset"]["channel_names"])
    order = list(reversed(range(len(names))))
    shuffled["dataset"]["channel_names"] = [names[index] for index in order]
    for key in (
        "channel_importance",
        "time_importance",
    ):
        values = shuffled["occlusion"].get(key)
        if isinstance(values, list) and len(values) == len(order):
            shuffled["occlusion"][key] = [values[index] for index in order]

    aggregate = aggregate_explanations([first, shuffled])
    expected = np.mean(
        [
            first["occlusion"]["channel_importance"],
            second["occlusion"]["channel_importance"],
        ],
        axis=0,
    )
    assert np.allclose(
        aggregate["occlusion"]["channel_importance"]["mean"], expected, atol=1e-5
    )


def test_aggregate_rejects_mixed_pipeline_or_dataset(tmp_path):
    first = _run_fold(tmp_path, 0)
    mixed_pipeline = copy.deepcopy(first)
    mixed_pipeline["pipeline"]["id"] = "riemann_mdm"
    with pytest.raises(ValueError, match="同一算法"):
        aggregate_explanations([first, mixed_pipeline])

    mixed_dataset = copy.deepcopy(first)
    mixed_dataset["dataset"]["signature"] = "different"
    with pytest.raises(ValueError, match="同一数据签名"):
        aggregate_explanations([first, mixed_dataset])

    with pytest.raises(ValueError, match="没有可聚合"):
        aggregate_explanations([])
