import json

import numpy as np
import pytest

from neurokiln.cache import ArtifactCache
from neurokiln.data import DatasetRegistry, make_synthetic_epochs
from neurokiln.evaluate import run_arena
from neurokiln.graph import build_graph_pipeline
from neurokiln.pipeline import build_baseline
from neurokiln.protocol import WithinSubjectProtocol
from neurokiln.runner import ExperimentRequest, run_experiment
from neurokiln.store import ExperimentStore


def test_cache_roundtrip_and_clear(tmp_path):
    cache = ArtifactCache(tmp_path / "cache")
    array = np.arange(24, dtype=np.float64).reshape(4, 3, 2)
    key = "a" * 64
    cache.put(key, array, {"tag": "unit"})
    loaded = cache.get(key)
    np.testing.assert_array_equal(loaded, array)
    loaded[0, 0, 0] = -1.0
    np.testing.assert_array_equal(cache.get(key), array)

    stats = cache.stats()
    assert stats["entries"] == 1
    assert stats["hits"] == 2
    assert stats["misses"] == 0
    assert stats["writes"] == 1
    assert "parquet" in json.loads(
        (next((tmp_path / "cache").glob("*/*")) / "meta.json").read_text(encoding="utf-8")
    )["format"]

    assert cache.get("b" * 64) is None
    cleared = cache.clear()
    assert cleared["entries"] == 0
    assert cleared["hits"] == 0


def test_cache_keys_bind_split_role_and_params():
    base = dict(
        data_signature="dataset",
        seed=0,
        parents=["parent"],
        descriptor={"class": "x", "params": {}},
        context={},
    )
    train = ArtifactCache.node_key(split_signature="split-a", role="train", **base)
    test = ArtifactCache.node_key(split_signature="split-a", role="test", **base)
    other_split = ArtifactCache.node_key(split_signature="split-b", role="train", **base)
    other_params = ArtifactCache.node_key(
        split_signature="split-a",
        role="train",
        **{**base, "descriptor": {"class": "x", "params": {"low": 4.0}}},
    )
    assert len({train, test, other_split, other_params}) == 4


def test_arena_cache_is_output_identical_and_reuses_entries(tmp_path):
    epochs = make_synthetic_epochs(n_trials_per_class=20, seed=7)
    splits = WithinSubjectProtocol(folds=4, seeds=(1,)).split(epochs)
    cold = run_arena(epochs, [build_baseline("csp_lda")], splits)

    cache = ArtifactCache(tmp_path / "cache")
    first = run_arena(
        epochs,
        [build_baseline("csp_lda")],
        splits,
        cache=cache,
        data_signature=epochs.signature(),
    )
    first_stats = cache.stats()
    second = run_arena(
        epochs,
        [build_baseline("csp_lda")],
        splits,
        cache=cache,
        data_signature=epochs.signature(),
    )
    second_stats = cache.stats()

    assert first_stats["hits"] == 0
    assert first_stats["writes"] == len(splits) * 2
    assert second_stats["hits"] == len(splits) * 2
    assert second_stats["writes"] == first_stats["writes"]
    np.testing.assert_array_equal(
        [row["balanced_accuracy_mean"] for row in cold],
        [row["balanced_accuracy_mean"] for row in first],
    )
    np.testing.assert_array_equal(
        [row["balanced_accuracy_mean"] for row in first],
        [row["balanced_accuracy_mean"] for row in second],
    )
    for row_cold, row_cached in zip(cold[0]["splits"], second[0]["splits"]):
        assert row_cold["y_pred"] == row_cached["y_pred"]


def test_cache_never_reuses_artifacts_across_splits_or_datasets(tmp_path):
    epochs = make_synthetic_epochs(n_trials_per_class=20, seed=7)
    splits = WithinSubjectProtocol(folds=4, seeds=(1,)).split(epochs)
    cache = ArtifactCache(tmp_path / "cache")
    pipeline = build_baseline("csp_lda")

    pipeline.run_split(epochs, splits[0], cache=cache)
    stats_after_first = cache.stats()
    build_baseline("csp_lda").run_split(epochs, splits[1], cache=cache)
    stats_after_second = cache.stats()
    assert stats_after_first["writes"] == 2
    assert stats_after_second["hits"] == 0
    assert stats_after_second["writes"] == 4

    changed = make_synthetic_epochs(n_trials_per_class=20, seed=8)
    build_baseline("csp_lda").run_split(changed, splits[0], cache=cache)
    stats_after_changed = cache.stats()
    assert stats_after_changed["hits"] == 0
    assert stats_after_changed["writes"] == 6


def test_graph_pipeline_uses_cache_without_behavior_change(tmp_path):
    epochs = make_synthetic_epochs(n_trials_per_class=16, seed=3)
    split = WithinSubjectProtocol(folds=4, seeds=(2,)).split(epochs)[0]
    spec = {
        "nodes": [
            {"id": "n1", "type": "preprocess.bandpass", "params": {"low": 8, "high": 30}},
            {"id": "n2", "type": "feature.csp", "params": {"n_components": 4}},
            {"id": "n3", "type": "feature.standardize", "params": {}},
            {"id": "n4", "type": "decoder.lda", "params": {}},
        ],
        "edges": [
            {"source": "n1", "target": "n2"},
            {"source": "n2", "target": "n3"},
            {"source": "n3", "target": "n4"},
        ],
    }
    baseline = build_graph_pipeline("graph", spec).run_split(epochs, split)
    cache = ArtifactCache(tmp_path / "cache")
    first = build_graph_pipeline("graph", spec).run_split(
        epochs, split, cache=cache, data_signature=epochs.signature()
    )
    second = build_graph_pipeline("graph", spec).run_split(
        epochs, split, cache=cache, data_signature=epochs.signature()
    )
    np.testing.assert_array_equal(baseline["y_pred"], first["y_pred"])
    np.testing.assert_array_equal(first["y_pred"], second["y_pred"])
    assert cache.stats()["hits"] == 2
    assert cache.stats()["entries"] == 2


def test_run_experiment_reports_cache_activity(tmp_path):
    registry = DatasetRegistry(tmp_path / "data")
    store = ExperimentStore(tmp_path / "neurokiln.db")
    request = ExperimentRequest(
        dataset_id="synthetic",
        subject_ids=["S01", "S02"],
        options={"n_trials_per_class": 10, "n_times": 40},
        protocol="loso",
        folds=2,
        seeds=[1],
        pipelines=["csp_lda"],
        name="cache test",
    )
    first = run_experiment(registry, store, request)
    second = run_experiment(registry, store, request)
    assert first["cache"]["writes"] > 0
    assert first["cache"]["hits"] == 0
    assert second["cache"]["hits"] == first["cache"]["writes"]
    assert first["rows"][0]["balanced_accuracy_mean"] == pytest.approx(
        second["rows"][0]["balanced_accuracy_mean"]
    )
