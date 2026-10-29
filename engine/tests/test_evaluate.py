import numpy as np

from neurokiln.data import make_synthetic_epochs
from neurokiln.evaluate import balanced_accuracy, cohen_kappa, confusion_matrix, run_arena
from neurokiln.pipeline import build_baseline
from neurokiln.protocol import WithinSubjectProtocol


def test_metrics_basic_properties():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    cm = confusion_matrix(y_true, y_pred)
    assert cm.shape == (3, 3)
    assert cm.sum() == 6
    expected_ba = np.mean([0.5, 1.0, 0.5])
    assert np.isclose(balanced_accuracy(y_true, y_pred), expected_ba)
    assert balanced_accuracy(y_true, y_true) == 1.0
    assert cohen_kappa(y_true, y_true) == 1.0
    assert -1.0 <= cohen_kappa(y_true, y_pred) <= 1.0


def test_arena_variance_pipeline_wins_on_variance_signal():
    epochs = make_synthetic_epochs(
        n_trials_per_class=40, n_classes=2, seed=11,
        class_sep=0.0, class_noise_step=0.8,
    )
    splits = WithinSubjectProtocol(folds=5, seeds=(0,)).split(epochs)
    rows = run_arena(
        epochs,
        [build_baseline("raw_centroid"), build_baseline("var_centroid")],
        splits,
    )
    assert rows[0]["pipeline"] == "var_centroid"
    var_row = next(r for r in rows if r["pipeline"] == "var_centroid")
    raw_row = next(r for r in rows if r["pipeline"] == "raw_centroid")
    assert var_row["balanced_accuracy_mean"] > 0.85
    assert raw_row["balanced_accuracy_mean"] < var_row["balanced_accuracy_mean"]
    for row in rows:
        assert row["n_splits"] == 5
        assert len(row["splits"]) == 5


def test_arena_is_deterministic_for_same_seed():
    kwargs = dict(n_trials_per_class=20, seed=5, class_sep=0.1, class_noise_step=0.5)
    splits_a = WithinSubjectProtocol(folds=4, seeds=(3,)).split(make_synthetic_epochs(**kwargs))
    splits_b = WithinSubjectProtocol(folds=4, seeds=(3,)).split(make_synthetic_epochs(**kwargs))
    rows_a = run_arena(make_synthetic_epochs(**kwargs), [build_baseline("var_centroid")], splits_a)
    rows_b = run_arena(make_synthetic_epochs(**kwargs), [build_baseline("var_centroid")], splits_b)
    assert rows_a[0]["balanced_accuracy_mean"] == rows_b[0]["balanced_accuracy_mean"]
