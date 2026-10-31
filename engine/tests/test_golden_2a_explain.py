"""Golden explanation regression on cached BCI Competition IV 2a.

This complements ``test_golden_2a.py``: the accuracy regression checks the
decoder, this one checks that intrinsic maps, occlusion, band ablation,
faithfulness and cross-fold aggregation stay coherent on real EEG.

    NEUROKILN_GOLDEN=1 NEUROKILN_MNE_DATA=F:/algo_research/data/external/moabb \
        python -m pytest tests/test_golden_2a_explain.py -q
"""

import os

import numpy as np
import pytest

from neurokiln.data import DatasetRegistry
from neurokiln.explain import ExplanationConfig, aggregate_explanations, run_explanation
from neurokiln.pipeline import build_baseline
from neurokiln.protocol import WithinSubjectProtocol

pytestmark = pytest.mark.skipif(
    os.environ.get("NEUROKILN_GOLDEN") != "1",
    reason="set NEUROKILN_GOLDEN=1 to run the public-dataset explanation regression",
)


def test_bciciv2a_explanations_and_aggregation(tmp_path):
    mne_data = os.environ.get("NEUROKILN_MNE_DATA") or os.environ.get("MNE_DATA")
    registry = DatasetRegistry(tmp_path / "data", mne_data_dir=mne_data)
    epochs = registry.load("bciciv2a", subject_ids=["S01"], classes=2)
    assert epochs.n_trials == 288
    assert len(epochs.channel_names) >= 22
    assert any("C3" in name.upper() for name in epochs.channel_names)

    splits = WithinSubjectProtocol(folds=5, seeds=(0,)).split(epochs)
    documents = []
    for fold in (0, 1):
        directory = tmp_path / f"fold-{fold}"
        result = run_explanation(
            epochs,
            build_baseline("csp_lda"),
            splits[fold],
            directory,
            ExplanationConfig(
                pipeline_id="csp_lda",
                fold=fold,
                window_ms=250,
                max_blocks=32,
                gradient_method="none",
                include_audit=True,
                include_sanity=False,
                faithfulness_steps=12,
            ),
            explanation_id=str(fold + 1),
        )
        document = result["global"]
        documents.append(document)

        assert document["provenance"]["background"] == "train_fold_only"
        assert 0.6 <= document["performance"]["balanced_accuracy"] <= 1.0
        assert document["bands"]["available"] is True
        channel_values = np.asarray(
            document["occlusion"]["channel_importance"], dtype=float
        )
        assert channel_values.shape == (len(epochs.channel_names),)
        assert np.isfinite(channel_values).all()
        assert document["faithfulness"]["deletion_auc"] is not None
        assert (directory / "global.json").is_file()
        assert (directory / "per_trial.json").is_file()
        assert (directory / "topomaps").is_dir()

    aggregate = aggregate_explanations(documents)
    assert aggregate["provenance"]["n_explanations"] == 2
    assert aggregate["provenance"]["n_unique_splits"] == 2
    assert aggregate["dataset"]["channel_names"] == list(epochs.channel_names)
    assert aggregate["occlusion"]["channel_importance"]["n"] == [
        2
    ] * len(epochs.channel_names)
    assert aggregate["faithfulness"]["deletion_auc"]["n"] == 2
    assert aggregate["performance"]["balanced_accuracy"]["n"] == 2
    assert 0.6 <= aggregate["performance"]["balanced_accuracy"]["mean"] <= 1.0
    assert len(aggregate["top_channels"]) == min(10, len(epochs.channel_names))
    intrinsic = {entry["id"]: entry for entry in aggregate["intrinsic"]}
    assert "csp_patterns" in intrinsic
    assert intrinsic["csp_patterns"]["n"] == [2] * len(epochs.channel_names)
