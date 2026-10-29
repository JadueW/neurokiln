from dataclasses import replace

import numpy as np

from neurokiln.data import Epochs
from neurokiln.evaluate import run_arena
from neurokiln.pipeline import (
    PIPELINE_DEFINITIONS,
    CSP,
    LDA,
    Bandpass,
    Pipeline,
    Standardize,
    build_baseline,
    torch_available,
)
from neurokiln.protocol import WithinSubjectProtocol


def make_oscillatory_epochs(
    n_trials_per_class: int = 30,
    n_classes: int = 2,
    n_channels: int = 8,
    n_times: int = 192,
    sfreq: float = 128.0,
    noise: float = 2.5,
    amplitude: float = 1.2,
    seed: int = 7,
) -> Epochs:
    """Class-specific oscillations projected through spatial patterns."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(n_classes), n_trials_per_class)
    times = np.arange(n_times) / sfreq
    freqs = (10.0, 18.0, 26.0, 14.0)
    patterns = rng.normal(size=(n_classes, n_channels))
    data = rng.normal(0.0, noise, size=(labels.size, n_channels, n_times))
    for index, label in enumerate(labels):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        source = amplitude * np.sin(2.0 * np.pi * freqs[label] * times + phase)
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


def test_pipeline_registry_matches_factory():
    ids = [item["id"] for item in PIPELINE_DEFINITIONS]
    assert len(ids) == len(set(ids))
    assert {"csp_lda", "xdawn_lda", "ssvep_cca", "ssvep_fbcca"} <= set(ids)
    for pipeline_id in ids:
        pipeline = build_baseline(pipeline_id)
        assert pipeline.name == pipeline_id
        assert pipeline.nodes


def test_csp_fbcsp_and_riemann_beat_chance():
    epochs = make_oscillatory_epochs()
    splits = WithinSubjectProtocol(folds=5, seeds=(0,)).split(epochs)
    names = ["csp_lda", "fbcsp_lda", "riemann_mdm", "ts_lr"]
    rows = run_arena(epochs, [build_baseline(name) for name in names], splits)
    assert {row["pipeline"] for row in rows} == set(names)
    for row in rows:
        assert row["balanced_accuracy_mean"] > 0.8, row
        assert row["balanced_accuracy_mean"] >= row["balanced_accuracy_std"]


def test_csp_supports_multiclass_one_vs_rest():
    epochs = make_oscillatory_epochs(n_classes=4)
    splits = WithinSubjectProtocol(folds=5, seeds=(0,)).split(epochs)
    rows = run_arena(epochs, [build_baseline("csp_lda"), build_baseline("fbcsp_lda")], splits)
    for row in rows:
        assert row["balanced_accuracy_mean"] > 0.8, row

    rng = np.random.default_rng(0)
    X = epochs.data[:40]
    y = epochs.labels[:40]
    node = CSP(4)
    from neurokiln.pipeline import FitContext

    ctx = FitContext(
        trial_ids=tuple(epochs.trial_ids[:40]), indices=np.arange(40), seed=0,
        paradigm="MI", channel_names=epochs.channel_names, sfreq=epochs.sfreq,
    )
    node.fit(X, y, ctx)
    assert node.filters_.shape == (4 * 4, X.shape[1])
    assert node.transform(X).shape == (40, 16)


def test_csp_clamps_components_for_low_channel_montages():
    epochs = make_oscillatory_epochs(n_channels=3, n_trials_per_class=20)
    from neurokiln.pipeline import FitContext

    X = epochs.data
    y = epochs.labels
    ctx = FitContext(
        trial_ids=tuple(epochs.trial_ids), indices=np.arange(len(y)), seed=0,
        paradigm="MI", channel_names=epochs.channel_names, sfreq=epochs.sfreq,
    )
    node = CSP(6).fit(X, y, ctx)
    assert node.filters_.shape == (3, 3)
    assert node.transform(X).shape == (len(y), 3)


def test_csp_filters_are_learned_from_training_trials_only():
    epochs = make_oscillatory_epochs()
    split = WithinSubjectProtocol(folds=5, seeds=(1,)).split(epochs)[0]

    def fit_filters(source: Epochs):
        pipeline = Pipeline(
            "csp", [Bandpass(8.0, 30.0), CSP(6), Standardize(), LDA()]
        )
        pipeline.run_split(source, split)
        return pipeline.nodes[1].filters_

    filters_before = fit_filters(epochs)
    altered = epochs.data.copy()
    rng = np.random.default_rng(123)
    altered[split.test] += rng.normal(0.0, 25.0, size=altered[split.test].shape)
    filters_after = fit_filters(replace(epochs, data=altered))
    np.testing.assert_allclose(filters_before, filters_after)


def test_riemann_and_fbcsp_are_deterministic():
    epochs = make_oscillatory_epochs(n_trials_per_class=20)
    splits = WithinSubjectProtocol(folds=4, seeds=(3,)).split(epochs)
    names = ["fbcsp_lda", "riemann_mdm", "ts_lr"]
    first = run_arena(epochs, [build_baseline(name) for name in names], splits)
    second = run_arena(epochs, [build_baseline(name) for name in names], splits)
    assert [row["pipeline"] for row in first] == [row["pipeline"] for row in second]
    for left, right in zip(first, second):
        assert left["balanced_accuracy_mean"] == right["balanced_accuracy_mean"]
        assert left["cohen_kappa_mean"] == right["cohen_kappa_mean"]


def test_deep_baselines_fit_deterministically_on_training_fold():
    if not torch_available():
        import pytest

        pytest.skip("PyTorch is not installed")
    from neurokiln.pipeline import EEGNet, FitContext, ShallowConvNet

    epochs = make_oscillatory_epochs(n_trials_per_class=12, n_channels=6, n_times=160)
    X, y = epochs.data, epochs.labels
    ctx = FitContext(
        trial_ids=tuple(epochs.trial_ids), indices=np.arange(len(y)), seed=5,
        paradigm="MI", channel_names=epochs.channel_names, sfreq=epochs.sfreq,
    )
    for factory in (EEGNet, ShallowConvNet):
        first = factory(max_epochs=4, batch_size=16).fit(X, y, ctx)
        second = factory(max_epochs=4, batch_size=16).fit(X, y, ctx)
        pred_first = first.predict(X)
        pred_second = second.predict(X)
        assert pred_first.shape == y.shape
        assert set(np.unique(pred_first)).issubset(set(np.unique(y)))
        np.testing.assert_array_equal(pred_first, pred_second)
        np.testing.assert_allclose(first.mean_, second.mean_)
