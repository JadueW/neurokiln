from dataclasses import replace

import numpy as np
import pytest

from neurokiln.data import (
    make_synthetic_p300_epochs,
    make_synthetic_ssvep_epochs,
)
from neurokiln.evaluate import run_arena
from neurokiln.graph import build_graph_pipeline, node_catalog, normalize_graph
from neurokiln.pipeline import build_baseline
from neurokiln.protocol import WithinSubjectProtocol
from neurokiln.runner import ExperimentRequest, ensure_paradigm_compatible, run_experiment
from neurokiln.data import DatasetRegistry
from neurokiln.store import ExperimentStore


def _splits(epochs, folds=4):
    return WithinSubjectProtocol(folds=folds, seeds=(3,)).split(epochs)


def test_p300_xdawn_beats_chance_and_reports_probabilities():
    epochs = make_synthetic_p300_epochs(n_trials_per_class=30, seed=5)
    rows = run_arena(
        epochs,
        [build_baseline("xdawn_lda"), build_baseline("raw_centroid")],
        _splits(epochs),
    )
    by_name = {row["pipeline"]: row for row in rows}
    assert by_name["xdawn_lda"]["balanced_accuracy_mean"] > 0.85
    for row in rows:
        calibration = row["calibration"]
        assert calibration["available"] is True
        assert 0.0 <= calibration["ece"] <= 1.0
        assert calibration["reliability"]
        for split in row["splits"]:
            proba = np.asarray(split["y_proba"], dtype=float)
            assert proba.shape[0] == len(split["y_true"])
            np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
            np.testing.assert_array_equal(
                np.asarray(split["proba_classes"]),
                np.unique(np.asarray(split["y_true"])),
            )


def test_ssvep_cca_and_fbcca_use_frequency_metadata():
    epochs = make_synthetic_ssvep_epochs(n_trials_per_class=8, seed=7)
    assert epochs.paradigm == "SSVEP"
    assert len(epochs.paradigm_meta["stimulus_frequencies"]) == epochs.n_classes
    rows = run_arena(
        epochs,
        [build_baseline("ssvep_cca"), build_baseline("ssvep_fbcca")],
        _splits(epochs),
    )
    for row in rows:
        assert row["balanced_accuracy_mean"] > 0.9, row
        assert row["calibration"]["available"] is True


def test_ssvep_without_frequencies_fails_clearly():
    epochs = make_synthetic_ssvep_epochs(n_trials_per_class=4, seed=1)
    bare = replace(epochs, paradigm_meta={})
    split = _splits(bare, folds=2)[0]
    with pytest.raises(ValueError, match="stimulus_frequencies"):
        build_baseline("ssvep_cca").run_split(bare, split)


def test_paradigm_metadata_changes_content_signature():
    epochs = make_synthetic_ssvep_epochs(n_trials_per_class=2, seed=2)
    changed = replace(
        epochs,
        paradigm_meta={**epochs.paradigm_meta, "window_seconds": 99.0},
    )
    assert epochs.signature() != changed.signature()


def test_pipeline_paradigm_compatibility_is_enforced():
    compatible = {"id": "xdawn_lda", "paradigm": "P300"}
    ensure_paradigm_compatible(compatible, "P300")
    with pytest.raises(ValueError, match="仅支持 P300"):
        ensure_paradigm_compatible(compatible, "MI")
    generic = {"id": "raw_centroid", "paradigm": "MI/P300/SSVEP"}
    for paradigm in ("MI", "P300", "SSVEP"):
        ensure_paradigm_compatible(generic, paradigm)


def test_runner_rejects_mismatched_paradigm(tmp_path):
    registry = DatasetRegistry(tmp_path / "data")
    store = ExperimentStore(tmp_path / "neurokiln.db")
    with pytest.raises(ValueError, match="仅支持 MI"):
        run_experiment(
            registry,
            store,
            ExperimentRequest(
                dataset_id="synthetic_ssvep",
                protocol="within_subject",
                folds=3,
                pipelines=["csp_lda"],
                save=False,
                options={"n_trials_per_class": 4},
            ),
        )


def test_xdawn_graph_node_is_registered_and_runs():
    types = {item["type"] for item in node_catalog()}
    assert {"decoder.xdawn_lda", "decoder.ssvep_cca", "decoder.ssvep_fbcca"} <= types
    spec = normalize_graph(
        {
            "nodes": [
                {"id": "bp", "type": "preprocess.bandpass", "params": {"low": 1, "high": 16}},
                {"id": "xdawn", "type": "decoder.xdawn_lda", "params": {"n_filters": 3}},
            ],
            "edges": [{"source": "bp", "target": "xdawn"}],
        }
    )
    epochs = make_synthetic_p300_epochs(n_trials_per_class=18, seed=4)
    split = _splits(epochs, folds=3)[0]
    result = build_graph_pipeline("p300-graph", spec).run_split(epochs, split)
    assert result["y_proba"] is not None
    assert result["y_proba"].shape[1] == 2
    assert result["y_pred"].shape == result["y_true"].shape


def test_store_persists_probabilities_and_calibration(tmp_path):
    registry = DatasetRegistry(tmp_path / "data")
    store = ExperimentStore(tmp_path / "neurokiln.db")
    result = run_experiment(
        registry,
        store,
        ExperimentRequest(
            dataset_id="synthetic_p300",
            protocol="within_subject",
            folds=3,
            seeds=[2],
            pipelines=["xdawn_lda"],
            save=True,
            options={"n_trials_per_class": 12},
        ),
    )
    detail = store.get_experiment(result["experiment_id"])
    split = detail["rows"][0]["splits"][0]
    assert split["y_proba"] is not None
    assert split["proba_classes"] == [0, 1]
    assert split["calibration"]["available"] is True
    assert detail["summary"]["pipelines"][0]["calibration"]["available"] is True
    assert detail["rows"][0]["calibration"]["available"] is True
