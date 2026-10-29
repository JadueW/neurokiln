from dataclasses import replace

import numpy as np
import pytest

from neurokiln.cache import ArtifactCache
from neurokiln.data import make_synthetic_p300_epochs
from neurokiln.evaluate import run_arena
from neurokiln.graph import build_graph_pipeline, node_catalog
from neurokiln.pipeline import (
    Bandpower,
    Baseline,
    CommonAverageReference,
    CovarianceVector,
    Crop,
    ErpWindowMean,
    FitContext,
    Hjorth,
    Notch,
    Resample,
    SelectK,
    TangentSpaceFeature,
)
from neurokiln.protocol import WithinSubjectProtocol
from test_baselines import make_oscillatory_epochs


def _node(node_id, node_type, **params):
    return {"id": node_id, "type": node_type, "params": params}


def _ctx(epochs):
    indices = np.arange(epochs.n_trials, dtype=int)
    return FitContext(
        trial_ids=tuple(epochs.trial_ids.tolist()),
        indices=indices,
        seed=0,
        paradigm=epochs.paradigm,
        channel_names=epochs.channel_names,
        sfreq=float(epochs.sfreq),
        paradigm_meta=dict(epochs.paradigm_meta),
        tmin=float(epochs.tmin),
    )


def _graph(nodes, edges):
    return {"nodes": nodes, "edges": [
        {"source": source, "target": target} for source, target in edges
    ]}


def test_catalog_lists_new_preprocess_and_feature_nodes():
    catalog = {item["type"]: item for item in node_catalog()}
    expected = {
        "preprocess.notch", "preprocess.car", "preprocess.crop",
        "preprocess.resample", "preprocess.baseline", "preprocess.xdawn",
        "feature.bandpower", "feature.hjorth", "feature.erp_window",
        "feature.covariance", "feature.tangent_space", "feature.select_k",
    }
    assert expected <= set(catalog)
    for node_type in expected:
        for param in catalog[node_type]["params"]:
            assert {"name", "label", "type", "default"}.issubset(param)


def test_crop_and_erp_window_use_absolute_epoch_time():
    epochs = make_oscillatory_epochs(n_trials_per_class=4, n_times=192, sfreq=128.0)
    ctx = _ctx(epochs)
    crop = Crop(start=0.25, duration=0.5).fit(epochs.data, epochs.labels, ctx)
    cropped = crop.transform(epochs.data)
    assert cropped.shape == (epochs.n_trials, epochs.data.shape[1], 64)
    assert crop.output_timebase(128.0, 0.0) == (128.0, 0.25)
    np.testing.assert_allclose(cropped, epochs.data[..., 32:96])

    window = ErpWindowMean(start=0.25, end=0.75)
    window.fit(epochs.data, epochs.labels, ctx)
    np.testing.assert_allclose(
        window.transform(epochs.data), epochs.data[..., 32:96].mean(axis=2)
    )


def test_baseline_requires_prestimulus_samples_and_centers_window():
    epochs = make_oscillatory_epochs(n_trials_per_class=4, n_times=192, sfreq=128.0)
    with pytest.raises(ValueError, match="刺激前"):
        Baseline().fit(epochs.data, epochs.labels, _ctx(epochs))

    shifted = replace(epochs, tmin=-0.25)
    ctx = _ctx(shifted)
    baseline = Baseline(start=-0.2, end=0.0).fit(shifted.data, shifted.labels, ctx)
    transformed = baseline.transform(shifted.data)
    assert np.abs(transformed[..., 7:32].mean(axis=2)).max() < 1e-10


def test_notch_suppresses_line_noise_and_keeps_signal():
    sfreq = 256.0
    times = np.arange(1024) / sfreq
    signal = np.sin(2 * np.pi * 10.0 * times)
    line = np.sin(2 * np.pi * 50.0 * times)
    X = np.stack([line, signal])[:, None, :]
    epochs = make_oscillatory_epochs(n_trials_per_class=1, n_times=1024, sfreq=sfreq)
    ctx = _ctx(epochs)
    notch = Notch(freq=50.0, q=30.0).fit(X, None, ctx)
    filtered = notch.transform(X)
    interior = slice(160, -160)
    residual = np.sqrt(np.mean(filtered[0, 0, interior] ** 2))
    original = np.sqrt(np.mean(X[0, 0, interior] ** 2))
    assert residual < 0.01 * original
    preserved = np.sqrt(np.mean(filtered[1, 0, interior] ** 2))
    assert preserved > 0.95 * np.sqrt(np.mean(X[1, 0, interior] ** 2))


def test_car_removes_common_mode():
    rng = np.random.default_rng(3)
    common = rng.normal(size=(1, 1, 64))
    X = rng.normal(size=(1, 6, 64)) + common
    epoch = make_oscillatory_epochs(n_trials_per_class=1, n_times=64)
    ctx = replace(_ctx(epoch), trial_ids=("t00000",), indices=np.array([0]))
    output = CommonAverageReference().fit(X, None, ctx).transform(X)
    assert np.abs(output.mean(axis=1)).max() < 1e-12


def test_resample_updates_rate_and_length_for_downstream_nodes():
    epochs = make_oscillatory_epochs(n_trials_per_class=10, n_times=192, sfreq=128.0)
    resample = Resample(target_sfreq=64.0)
    resample.fit(epochs.data, epochs.labels, _ctx(epochs))
    output = resample.transform(epochs.data)
    assert output.shape == (epochs.n_trials, epochs.data.shape[1], 96)
    assert resample.output_timebase(128.0, -0.2) == (64.0, -0.2)

    spec = _graph(
        [
            _node("resample", "preprocess.resample", target_sfreq=64),
            _node("bandpass", "preprocess.bandpass", low=8, high=30),
            _node("mean", "feature.channel_mean"),
            _node("centroid", "decoder.nearest_centroid"),
        ],
        [("resample", "bandpass"), ("bandpass", "mean"), ("mean", "centroid")],
    )
    split = WithinSubjectProtocol(folds=2, seeds=(1,)).split(epochs)[0]
    result = build_graph_pipeline("resampled", spec).run_split(epochs, split)
    assert result["y_pred"].shape == result["y_true"].shape
    rows = run_arena(
        epochs,
        [build_graph_pipeline("resampled", spec)],
        WithinSubjectProtocol(folds=2, seeds=(1,)).split(epochs),
    )
    assert 0.0 <= rows[0]["balanced_accuracy_mean"] <= 1.0


def test_cached_pipeline_with_resample_returns_identical_predictions(tmp_path):
    epochs = make_oscillatory_epochs(n_trials_per_class=12, n_times=192, sfreq=128.0)
    spec = _graph(
        [
            _node("notch", "preprocess.notch", freq=50.0, q=30.0),
            _node("car", "preprocess.car"),
            _node("crop", "preprocess.crop", start=0.0, duration=0.0),
            _node("resample", "preprocess.resample", target_sfreq=64),
            _node("bandpass", "preprocess.bandpass", low=8, high=30),
            _node("mean", "feature.channel_mean"),
            _node("centroid", "decoder.nearest_centroid"),
        ],
        [
            ("notch", "car"),
            ("car", "crop"),
            ("crop", "resample"),
            ("resample", "bandpass"),
            ("bandpass", "mean"),
            ("mean", "centroid"),
        ],
    )
    graph = build_graph_pipeline("cached-new-nodes", spec)
    split = WithinSubjectProtocol(folds=3, seeds=(2,)).split(epochs)[0]
    cache = ArtifactCache(tmp_path / "cache")
    first = graph.run_split(epochs, split, cache=cache, data_signature=epochs.signature())
    second = graph.run_split(epochs, split, cache=cache, data_signature=epochs.signature())
    np.testing.assert_array_equal(first["y_pred"], second["y_pred"])
    assert second["y_proba"] is not None
    assert cache.hits > 0
    assert cache.writes == 10


def test_xdawn_node_composes_with_erp_features_and_classifier():
    epochs = make_synthetic_p300_epochs(n_trials_per_class=30, seed=5)
    spec = _graph(
        [
            _node("bandpass", "preprocess.bandpass", low=1, high=16),
            _node("xdawn", "preprocess.xdawn", n_filters=2),
            _node("window", "feature.erp_window", start=0.2, end=0.5),
            _node("standardize", "feature.standardize"),
            _node("lda", "decoder.lda"),
        ],
        [
            ("bandpass", "xdawn"),
            ("xdawn", "window"),
            ("window", "standardize"),
            ("standardize", "lda"),
        ],
    )
    splits = WithinSubjectProtocol(folds=4, seeds=(3,)).split(epochs)
    rows = run_arena(epochs, [build_graph_pipeline("p300-xdawn-features", spec)], splits)
    assert rows[0]["balanced_accuracy_mean"] > 0.8


def test_feature_nodes_have_expected_shapes():
    epochs = make_oscillatory_epochs(n_trials_per_class=6)
    ctx = _ctx(epochs)
    n_channels = epochs.data.shape[1]
    power = Bandpower(low=8.0, high=30.0).fit(epochs.data, epochs.labels, ctx)
    assert power.transform(epochs.data).shape == (epochs.n_trials, n_channels)

    hjorth = Hjorth().fit(epochs.data, epochs.labels, ctx)
    features = hjorth.transform(epochs.data)
    assert features.shape == (epochs.n_trials, 3 * n_channels)
    assert np.all(np.isfinite(features))

    covariance = CovarianceVector(estimator="oas")
    vectors = covariance.transform(epochs.data)
    assert vectors.shape == (
        epochs.n_trials, n_channels * (n_channels + 1) // 2
    )

    tangent = TangentSpaceFeature().fit(epochs.data, epochs.labels, ctx)
    assert tangent.transform(epochs.data).shape == vectors.shape


def test_select_k_fits_on_training_features_and_discriminates():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(60, 20))
    y = np.repeat([0, 1], 30)
    X[y == 1, :3] += 2.5
    epochs = make_oscillatory_epochs(n_trials_per_class=1)
    ctx = _ctx(epochs)
    selector = SelectK(k=5, method="anova").fit(X, y, ctx)
    transformed = selector.transform(X)
    assert transformed.shape == (60, 5)
    assert {0, 1, 2} <= set(selector.indices_.tolist())

    mutual = SelectK(k=4, method="mutual_info").fit(X, y, ctx)
    assert mutual.transform(X).shape == (60, 4)


def test_bandpower_branches_merge_and_select_k_in_graph():
    epochs = make_oscillatory_epochs(n_trials_per_class=30)
    spec = _graph(
        [
            _node("bandpass", "preprocess.bandpass", low=8, high=30),
            _node("low", "feature.bandpower", low=8, high=14),
            _node("high", "feature.bandpower", low=14, high=30),
            _node("merge", "merge.concat"),
            _node("select", "feature.select_k", k=8, method="anova"),
            _node("standardize", "feature.standardize"),
            _node("lda", "decoder.lda"),
        ],
        [
            ("bandpass", "low"),
            ("bandpass", "high"),
            ("low", "merge"),
            ("high", "merge"),
            ("merge", "select"),
            ("select", "standardize"),
            ("standardize", "lda"),
        ],
    )
    splits = WithinSubjectProtocol(folds=3, seeds=(4,)).split(epochs)
    rows = run_arena(epochs, [build_graph_pipeline("bandpower-merge", spec)], splits)
    assert rows[0]["balanced_accuracy_mean"] > 0.8


def test_boolean_parameters_are_coerced_for_graph_nodes():
    graph = build_graph_pipeline(
        "bool-params",
        _graph(
            [
                _node(
                    "power", "feature.bandpower",
                    low=8, high=30, relative="true", log_power="false",
                ),
                _node("centroid", "decoder.nearest_centroid"),
            ],
            [("power", "centroid")],
        ),
    )
    node = graph.nodes["power"]
    assert node.relative is True
    assert node.log_power is False
