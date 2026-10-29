import numpy as np
import pytest

from neurokiln.data import make_synthetic_epochs
from neurokiln.pipeline import ChannelMean, MeanCenter, NearestCentroid, Node, Pipeline
from neurokiln.protocol import WithinSubjectProtocol


class SpyNode(Node):
    """Records everything it is allowed to see during fit."""

    name = "spy"

    def __init__(self):
        self.fit_calls = []

    def fit(self, X, y, ctx):
        self.fit_calls.append({
            "n_trials": X.shape[0],
            "y": None if y is None else y.copy(),
            "ctx_indices": ctx.indices.copy(),
            "ctx_trial_ids": tuple(ctx.trial_ids),
        })
        return self

    def transform(self, X):
        return X.mean(axis=2)


@pytest.fixture
def epochs():
    return make_synthetic_epochs(n_trials_per_class=20, n_classes=2, seed=3)


def test_fit_only_receives_training_trials(epochs):
    spy = SpyNode()
    pipeline = Pipeline("spy", [spy, NearestCentroid()])
    splits = WithinSubjectProtocol(folds=4, seeds=(5,)).split(epochs)
    for split in splits:
        result = pipeline.run_split(epochs, split)
        call = spy.fit_calls[-1]
        assert call["n_trials"] == split.train.size
        np.testing.assert_array_equal(call["ctx_indices"], split.train)
        assert set(call["ctx_trial_ids"]) == set(epochs.trial_ids[split.train])
        assert not set(call["ctx_trial_ids"]) & set(epochs.trial_ids[split.test])
        assert call["y"].shape == (split.train.size,)
        np.testing.assert_array_equal(result["y_true"], epochs.labels[split.test])
        assert result["y_pred"].shape == (split.test.size,)


def test_fitted_statistics_come_from_train_not_full_data(epochs):
    center = MeanCenter()
    pipeline = Pipeline("centered", [center, ChannelMean(), NearestCentroid()])
    split = WithinSubjectProtocol(folds=5, seeds=(9,)).split(epochs)[0]
    pipeline.run_split(epochs, split)
    expected = epochs.data[split.train].mean(axis=(0, 2), keepdims=True)
    full = epochs.data.mean(axis=(0, 2), keepdims=True)
    np.testing.assert_allclose(center.mean_, expected)
    assert not np.allclose(center.mean_, full)


def test_split_object_rejects_overlap(epochs):
    from neurokiln.protocol import Split

    with pytest.raises(ValueError):
        Split("within_subject", 0, 0, 0, np.array([0, 1, 2]), np.array([2, 3]))


def test_predictions_align_with_test_trials(epochs):
    pipeline = Pipeline("raw", [ChannelMean(), NearestCentroid()])
    split = WithinSubjectProtocol(folds=5, seeds=(1,)).split(epochs)[0]
    result = pipeline.run_split(epochs, split)
    np.testing.assert_array_equal(result["test_trial_ids"], epochs.trial_ids[split.test])
    np.testing.assert_array_equal(result["train_trial_ids"], epochs.trial_ids[split.train])
