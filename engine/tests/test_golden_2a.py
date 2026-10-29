"""Golden regression on cached BCI Competition IV 2a (subject 1).

Run explicitly after changing features/decoders:

    NEUROKILN_GOLDEN=1 NEUROKILN_MNE_DATA=D:/data/mne \
        python -m pytest tests/test_golden_2a.py -q

The reference means were recorded on 2026-10-29 with MOABB 1.7.2,
left_hand vs right_hand, 8-32 Hz, 0.5-3.5 s, 5-fold seed 0.
"""

import os

import pytest

from neurokiln.data import DatasetRegistry
from neurokiln.runner import ExperimentRequest, run_experiment

pytestmark = pytest.mark.skipif(
    os.environ.get("NEUROKILN_GOLDEN") != "1",
    reason="set NEUROKILN_GOLDEN=1 to run the public-dataset golden regression",
)

EXPECTED = {
    "csp_lda": 0.8819,
    "fbcsp_lda": 0.9341,
    "riemann_mdm": 0.8433,
    "ts_lr": 0.8888,
}
TOLERANCE = 0.03


def test_bciciv2a_subject1_matches_recorded_baseline(tmp_path):
    mne_data = os.environ.get("NEUROKILN_MNE_DATA") or os.environ.get("MNE_DATA")
    registry = DatasetRegistry(tmp_path / "data", mne_data_dir=mne_data)
    result = run_experiment(
        registry,
        None,
        ExperimentRequest(
            dataset_id="bciciv2a",
            subject_ids=["S01"],
            classes=2,
            protocol="within_subject",
            folds=5,
            repeats=1,
            seeds=[0],
            pipelines=list(EXPECTED),
            save=False,
        ),
    )
    assert result["dataset"]["loaded_n_trials"] == 288
    assert result["dataset"]["loaded_classes"] == 2
    assert set(result["dataset"]["label_names"]) == {"left_hand", "right_hand"}
    for row in result["rows"]:
        reference = EXPECTED[row["pipeline"]]
        assert abs(row["balanced_accuracy_mean"] - reference) <= TOLERANCE, (
            f"{row['pipeline']} drifted: {row['balanced_accuracy_mean']:.4f} "
            f"vs recorded {reference:.4f}"
        )
