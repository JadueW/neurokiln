import numpy as np
import pytest

from neurokiln.data import Epochs, make_synthetic_epochs


def test_epochs_defaults_and_counts():
    epochs = make_synthetic_epochs(n_trials_per_class=12, n_classes=3, subjects=("S01", "S02"))
    assert epochs.n_trials == 36
    assert epochs.n_classes == 3
    assert epochs.label_counts() == {0: 12, 1: 12, 2: 12}
    assert epochs.subjects == ["S01", "S02"]
    assert len(np.unique(epochs.trial_ids)) == 36


def test_select_preserves_metadata_and_order():
    epochs = make_synthetic_epochs(n_trials_per_class=10, sessions=("A", "B"))
    idx = np.array([7, 3, 11])
    sub = epochs.select(idx)
    assert sub.n_trials == 3
    np.testing.assert_array_equal(sub.labels, epochs.labels[idx])
    np.testing.assert_array_equal(sub.trial_ids, epochs.trial_ids[idx])
    np.testing.assert_array_equal(sub.session_ids, epochs.session_ids[idx])
    np.testing.assert_allclose(sub.data, epochs.data[idx])


def test_validation_rejects_bad_shapes():
    epochs = make_synthetic_epochs(n_trials_per_class=4)
    with pytest.raises(ValueError):
        Epochs(data=epochs.data, labels=epochs.labels[:-1], sfreq=128, channel_names=epochs.channel_names)
    with pytest.raises(ValueError):
        Epochs(data=epochs.data[0], labels=epochs.labels, sfreq=128, channel_names=epochs.channel_names)
    with pytest.raises(ValueError):
        Epochs(data=epochs.data, labels=epochs.labels, sfreq=128, channel_names=("C1",))


def test_signature_is_stable_and_content_sensitive():
    a = make_synthetic_epochs(seed=1)
    b = make_synthetic_epochs(seed=1)
    c = make_synthetic_epochs(seed=2)
    assert a.signature() == b.signature()
    assert a.signature() != c.signature()
