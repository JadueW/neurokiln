import numpy as np
import pytest

from neurokiln.pipeline.fusion import (
    DecoderPrediction,
    average_predictions,
    vote_predictions,
)


def test_average_fusion_aligns_class_columns():
    first = DecoderPrediction(
        y_pred=np.array([0, 1]),
        proba=np.array([[0.8, 0.2], [0.3, 0.7]]),
        classes=np.array([0, 1]),
    )
    second = DecoderPrediction(
        y_pred=np.array([1, 2]),
        proba=np.array([[0.8, 0.2], [0.1, 0.9]]),
        classes=np.array([1, 2]),
    )
    fused = average_predictions([first, second])
    np.testing.assert_array_equal(fused.classes, [0, 1, 2])
    np.testing.assert_allclose(fused.proba[0], [0.4, 0.5, 0.1])
    np.testing.assert_allclose(fused.proba.sum(axis=1), 1.0)
    np.testing.assert_array_equal(fused.y_pred, [1, 2])


def test_average_fusion_requires_probabilities():
    missing = DecoderPrediction(
        y_pred=np.array([0, 1]), proba=None, classes=None
    )
    backed = DecoderPrediction(
        y_pred=np.array([0, 1]),
        proba=np.array([[0.9, 0.1], [0.2, 0.8]]),
        classes=np.array([0, 1]),
    )
    with pytest.raises(ValueError, match="概率"):
        average_predictions([missing, backed])


def test_vote_fusion_breaks_ties_with_probability_then_smallest_class():
    low = DecoderPrediction(
        y_pred=np.array([0]),
        proba=np.array([[0.6, 0.4]]),
        classes=np.array([0, 1]),
    )
    high = DecoderPrediction(
        y_pred=np.array([1]),
        proba=np.array([[0.2, 0.8]]),
        classes=np.array([0, 1]),
    )
    fused = vote_predictions([low, high])
    np.testing.assert_array_equal(fused.y_pred, [1])
    np.testing.assert_allclose(fused.proba, [[0.5, 0.5]])

    blind_low = DecoderPrediction(y_pred=np.array([0]), proba=None, classes=None)
    blind_high = DecoderPrediction(y_pred=np.array([1]), proba=None, classes=None)
    fallback = vote_predictions([blind_low, blind_high])
    np.testing.assert_array_equal(fallback.y_pred, [0])


def test_fusion_rejects_mismatched_trial_counts():
    first = DecoderPrediction(
        y_pred=np.array([0]), proba=None, classes=None
    )
    second = DecoderPrediction(
        y_pred=np.array([0, 1]), proba=None, classes=None
    )
    with pytest.raises(ValueError, match="试次数"):
        vote_predictions([first, second])
