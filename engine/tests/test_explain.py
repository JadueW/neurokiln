import json
from pathlib import Path

import numpy as np

from neurokiln.data import Epochs
from neurokiln.explain import (
    ExplanationConfig,
    band_ablation,
    channel_positions,
    extract_intrinsic,
    faithfulness,
    gradient_attribution,
    knowledge_audit,
    normalize_name,
    occlusion_blocks,
    run_explanation,
    topomap_png,
    weight_randomization_check,
)
from neurokiln.pipeline import build_baseline
from neurokiln.protocol import WithinSubjectProtocol
from neurokiln.store import ExperimentStore, SCHEMA_VERSION


def make_oscillatory_epochs(
    n_trials_per_class: int = 30,
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


def fitted_split(epochs: Epochs | None = None, folds: int = 4):
    epochs = epochs or make_oscillatory_epochs()
    split = WithinSubjectProtocol(folds=folds, seeds=(2,)).split(epochs)[0]
    pipeline = build_baseline("csp_lda")
    _, fitted = pipeline.fit_split(epochs, split)
    return epochs, pipeline, split, fitted


def test_montage_normalizes_labels_and_places_unknown_channels():
    assert normalize_name("EEG C3-REF") == "C3"
    assert normalize_name("T3") == "T7"
    assert normalize_name("") is None

    coords, info = channel_positions(["C3", "C4", "Fz", "Pz", "O1", "O2"])
    assert coords.shape == (6, 2)
    assert np.isfinite(coords).all()
    assert (np.linalg.norm(coords, axis=1) <= 1.01).all()
    assert info["layout"] == "10-10"
    assert info["mapped"] == 6
    assert info["unknown"] == []

    ring, fallback = channel_positions(["C3", "X9"])
    assert ring.shape == (2, 2)
    assert fallback["layout"] == "ring"
    assert fallback["mapped"] == 0
    assert fallback["unknown"] == ["C3", "X9"]


def test_topomap_png_is_dependency_free_and_deterministic():
    values = np.array([0.4, -0.2, 1.0, -0.6, 0.1, 0.0])
    names = ["C3", "C4", "Fz", "Pz", "O1", "O2"]
    first, meta = topomap_png(values, names)
    second, _ = topomap_png(values, names)
    assert first[:8] == b"\x89PNG\r\n\x1a\n"
    assert first == second
    assert len(first) > 500
    assert meta["layout"] == "10-10"
    assert meta["signed"] is True


def test_intrinsic_parameters_explain_csp_lda():
    _, _, _, fitted = fitted_split()
    entries = extract_intrinsic(fitted)
    assert entries
    by_id = {entry["id"]: entry for entry in entries}
    assert "csp_patterns" in by_id
    entry = by_id["csp_patterns"]
    assert len(entry["values"]) == len(fitted.test_epochs.channel_names)
    assert entry["signed"] is True
    assert entry.get("per_class") in (None, [])


def test_occlusion_band_audit_and_faithfulness_are_consistent():
    epochs, _, _, fitted = fitted_split()
    channels = len(epochs.channel_names)
    occlusion = occlusion_blocks(
        fitted, window_ms=200, max_blocks=24, max_windows=4,
    )
    assert occlusion["score_unit"] == "log_odds"
    assert len(occlusion["channel_importance"]) == channels
    matrix = np.asarray(occlusion["matrix"], dtype=float)
    assert matrix.shape[1] == occlusion["n_windows"]
    assert len(occlusion["window_times"]) == occlusion["n_windows"]
    assert np.isfinite(matrix).all()

    bands = band_ablation(fitted)
    assert bands["available"] is True
    assert len(bands["bands"]) == len(bands["drop"])
    for low, high in bands["bands"]:
        assert f"{low:g}-{high:g}" in bands["channel_importance"]

    audit = knowledge_audit(
        fitted, occlusion_channel_importance=occlusion["channel_importance"]
    )
    assert audit["available"] is True
    assert audit["n_features"] > 0
    assert 0.0 <= audit["closure_ratio"] <= 2.0
    assert audit["channel_agreement"] is not None
    assert -1.0 <= audit["channel_agreement"] <= 1.0
    assert len(audit["top_features"]) <= 10
    assert all(item["name"] and item["weight"] >= 0 for item in audit["top_features"])

    faith = faithfulness(fitted, occlusion, max_steps=8)
    assert len(faith["deletion_curve"]["fraction"]) == len(
        faith["deletion_curve"]["retained"]
    )
    assert faith["score_unit"] == "log_odds"
    if faith["normalization"]["usable"]:
        assert faith["deletion_auc"] is not None
        assert faith["insertion_auc"] is not None
        assert faith["random_deletion_auc"] is not None
    if faith["stability"]["half_split_spearman"] is not None:
        assert -1.0 <= faith["stability"]["half_split_spearman"] <= 1.0


def test_gradient_and_weight_randomization_skip_linear_models():
    _, _, _, fitted = fitted_split()
    gradient = gradient_attribution(fitted, method="integrated_gradients")
    assert gradient["available"] is False
    sanity = weight_randomization_check(fitted, np.zeros(8))
    assert sanity["available"] is False


def test_run_explanation_writes_complete_bundle(tmp_path):
    epochs, pipeline, split, _ = fitted_split()
    config = ExplanationConfig(
        pipeline_id="csp_lda",
        window_ms=200,
        max_blocks=24,
        gradient_method="none",
        include_audit=True,
        include_sanity=False,
        faithfulness_steps=8,
    )
    result = run_explanation(
        epochs, pipeline, split, tmp_path / "explanation", config,
        explanation_id="abc123", pipeline_name="CSP + LDA",
    )
    document = result["global"]
    assert document["format"] == "neurokiln.explain"
    assert document["version"] == 1
    assert document["provenance"]["background"] == "train_fold_only"
    assert document["provenance"]["n_background_trials"] == len(split.train)
    assert document["performance"]["n_test"] == len(split.test)
    assert document["faithfulness"]["score_unit"] == "log_odds"
    assert document["occlusion"]["topomap"]
    assert document["bands"]["topomap"]
    assert document["audit"]["topomap"]
    assert isinstance(document["warnings"], list)

    directory = Path(result["directory"])
    on_disk = json.loads((directory / "global.json").read_text(encoding="utf-8"))
    assert on_disk["occlusion"]["matrix"] == document["occlusion"]["matrix"]
    for asset in (document["occlusion"]["topomap"], document["audit"]["topomap"]):
        payload = (directory / asset).read_bytes()
        assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    assert document["per_trial"]["json"] == "per_trial.json"
    assert (directory / "per_trial.json").is_file()


def test_store_schema_v5_roundtrips_explanations(tmp_path):
    assert SCHEMA_VERSION >= 5
    store = ExperimentStore(tmp_path / "neurokiln.db")
    explanation_id = store.create_explanation(
        pipeline="csp_lda",
        pipeline_name="CSP + LDA",
        folder=str(tmp_path / "folder"),
        dataset_id="synthetic",
        job_id="job0123456789ab",
        fold=1,
        repeat=0,
        seed=0,
    )
    pending = store.get_explanation(explanation_id)
    assert pending["status"] == "running"
    assert pending["fold"] == 1

    store.finish_explanation(
        explanation_id,
        folder=str(tmp_path / "folder"),
        summary={"deletion_auc": 0.9, "warnings": []},
        warnings=[],
        duration_ms=12.5,
    )
    ready = store.get_explanation(explanation_id)
    assert ready["status"] == "completed"
    assert ready["summary"]["deletion_auc"] == 0.9
    assert ready["duration_ms"] == 12.5

    items, total = store.list_explanations(experiment_id=None, limit=10, offset=0)
    assert total == 1 and items[0]["id"] == explanation_id

    removed = store.delete_explanation(explanation_id)
    assert removed["id"] == explanation_id
    assert store.get_explanation(explanation_id) is None
