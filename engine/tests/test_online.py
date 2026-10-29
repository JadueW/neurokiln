import numpy as np
import pytest

from neurokiln.data import make_synthetic_epochs
from neurokiln.online import (
    Prediction,
    ReplayDecoder,
    ReplaySource,
    StreamTrial,
    ThresholdPolicy,
    replay_from_split,
    run_closed_loop,
)
from neurokiln.pipeline import build_baseline
from neurokiln.protocol import WithinSubjectProtocol


def _trial(trial_id="t1", label=0):
    return StreamTrial(
        trial_id=trial_id,
        data=np.zeros((2, 8)),
        timestamp=1.5,
        true_label=label,
        subject_id="S01",
        session_id="0",
    )


def test_threshold_policy_abstains_below_confidence():
    trial = _trial()
    prediction = Prediction(label=1, probabilities=np.array([0.2, 0.8]), classes=(0, 1))
    assert prediction.confidence == pytest.approx(0.8)
    assert ThresholdPolicy(threshold=0.9).decide(prediction, trial) is None
    assert ThresholdPolicy(threshold=0.5).decide(prediction, trial) == "select:1"


def test_closed_loop_replays_held_out_predictions():
    epochs = make_synthetic_epochs(
        n_trials_per_class=12, n_classes=2, n_channels=4, n_times=24, seed=7,
        class_sep=0.4,
    )
    split = WithinSubjectProtocol(folds=3, seeds=(0,)).split(epochs)[0]
    result = build_baseline("raw_centroid").run_split(epochs, split)
    decoder = ReplayDecoder.from_split_results(epochs, split, result)
    source = ReplaySource(epochs=epochs, indices=split.test, hop_seconds=0.5)
    closed = run_closed_loop(
        decoder, source, ThresholdPolicy(threshold=0.0, action_template="move:{label}")
    )
    assert len(closed.events) == split.test.size
    assert closed.n_actions == split.test.size
    assert closed.n_abstained == 0
    assert closed.n_labeled == split.test.size
    assert closed.accuracy == pytest.approx(
        float(np.mean(result["y_pred"] == epochs.labels[split.test]))
    )
    assert all(event.action.startswith("move:") for event in closed.events)
    timestamps = [event.trial.timestamp for event in closed.events]
    assert timestamps == sorted(timestamps)
    assert timestamps[1] - timestamps[0] == pytest.approx(0.5)


def test_replay_from_split_respects_max_trials():
    epochs = make_synthetic_epochs(
        n_trials_per_class=8, n_classes=2, n_channels=4, n_times=16, seed=3,
    )
    split = WithinSubjectProtocol(folds=2, seeds=(0,)).split(epochs)[0]
    result = build_baseline("raw_centroid").run_split(epochs, split)
    decoder = ReplayDecoder.from_split_results(epochs, split, result)
    source = ReplaySource(epochs=epochs, indices=split.test)
    closed = run_closed_loop(decoder, source, ThresholdPolicy(), max_trials=3)
    assert len(closed.events) == 3
    assert closed.summary()["n_events"] == 3


def test_replay_decoder_rejects_unknown_trial():
    decoder = ReplayDecoder(predictions={})
    with pytest.raises(KeyError):
        decoder.predict(_trial("missing"))


def test_replay_from_split_helper_returns_summary():
    epochs = make_synthetic_epochs(
        n_trials_per_class=6, n_classes=2, n_channels=3, n_times=12, seed=5,
    )
    split = WithinSubjectProtocol(folds=2, seeds=(1,)).split(epochs)[0]
    result = build_baseline("raw_centroid").run_split(epochs, split)
    closed = replay_from_split(epochs, split, result)
    payload = closed.to_payload()
    assert payload["n_events"] == split.test.size
    assert {"trial_id", "predicted_label", "action"} <= set(payload["events"][0])
