from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from .data.model import Epochs
from .protocol.types import Split


@dataclass(frozen=True)
class StreamTrial:
    """One already-epoched trial delivered by a stream source."""

    trial_id: str
    data: np.ndarray
    timestamp: float = 0.0
    true_label: int | None = None
    subject_id: str = "unknown"
    session_id: str = "0"

    def __post_init__(self) -> None:
        data = np.asarray(self.data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("stream trial data must be 2D (n_channels, n_times)")
        if not np.all(np.isfinite(data)):
            raise ValueError("stream trial data contains NaN or inf")
        object.__setattr__(self, "data", data)


@dataclass(frozen=True)
class Prediction:
    label: int
    probabilities: np.ndarray | None = None
    classes: tuple[int, ...] = ()
    confidence: float | None = None

    def __post_init__(self) -> None:
        label = int(self.label)
        object.__setattr__(self, "label", label)
        if self.probabilities is not None:
            proba = np.asarray(self.probabilities, dtype=float)
            if proba.ndim != 1 or not np.all(np.isfinite(proba)):
                raise ValueError("probabilities must be a finite 1D array")
            if proba.size and (proba.min() < 0 or proba.max() > 1):
                raise ValueError("probabilities must be in [0, 1]")
            object.__setattr__(self, "probabilities", proba)
        classes = tuple(int(item) for item in self.classes)
        object.__setattr__(self, "classes", classes)
        confidence = self.confidence
        if confidence is None and self.probabilities is not None and self.probabilities.size:
            confidence = float(self.probabilities.max())
        if confidence is not None:
            confidence = float(confidence)
            if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be in [0, 1]")
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True)
class ClosedLoopEvent:
    trial: StreamTrial
    prediction: Prediction
    action: str | None
    decided_at: float


@runtime_checkable
class OnlineDecoder(Protocol):
    def predict(self, trial: StreamTrial) -> Prediction:
        """Decode one trial without using future trials."""


@runtime_checkable
class StreamSource(Protocol):
    def __iter__(self) -> Iterator[StreamTrial]:
        """Yield trials in acquisition order."""


@runtime_checkable
class ClosedLoopPolicy(Protocol):
    def decide(self, prediction: Prediction, trial: StreamTrial) -> str | None:
        """Map a prediction to an actuator command, or None to abstain."""


@dataclass(frozen=True)
class ReplaySource:
    """Iterate held-out epochs as a deterministic, offline stream."""

    epochs: Epochs
    indices: Sequence[int]
    start_time: float = 0.0
    hop_seconds: float = 0.0

    def __post_init__(self) -> None:
        indices = np.asarray(self.indices, dtype=int)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("replay indices must be a non-empty 1D sequence")
        if indices.min() < 0 or indices.max() >= self.epochs.n_trials:
            raise ValueError("replay indices out of range")
        if np.unique(indices).size != indices.size:
            raise ValueError("replay indices must be unique")
        object.__setattr__(self, "indices", indices)
        if self.hop_seconds < 0:
            raise ValueError("hop_seconds must be non-negative")

    def __iter__(self) -> Iterator[StreamTrial]:
        for order, index in enumerate(self.indices):
            yield StreamTrial(
                trial_id=str(self.epochs.trial_ids[index]),
                data=self.epochs.data[index],
                timestamp=float(self.start_time + order * self.hop_seconds),
                true_label=int(self.epochs.labels[index]),
                subject_id=str(self.epochs.subject_ids[index]),
                session_id=str(self.epochs.session_ids[index]),
            )


@dataclass
class ReplayDecoder:
    """Decoder backed by predictions already produced for held-out trials."""

    predictions: Mapping[str, Prediction]

    def predict(self, trial: StreamTrial) -> Prediction:
        try:
            return self.predictions[trial.trial_id]
        except KeyError as exc:
            raise KeyError(f"no replay prediction for trial {trial.trial_id}") from exc

    @classmethod
    def from_split_results(
        cls,
        epochs: Epochs,
        split: Split,
        result: Mapping[str, Any],
    ) -> "ReplayDecoder":
        test_ids = [str(value) for value in epochs.trial_ids[np.asarray(split.test, dtype=int)]]
        y_pred = np.asarray(result["y_pred"], dtype=int)
        if y_pred.shape != (len(test_ids),):
            raise ValueError("split result predictions do not match the test set")
        y_proba = result.get("y_proba")
        proba_classes = result.get("proba_classes")
        classes = (
            ()
            if proba_classes is None
            else tuple(int(item) for item in np.asarray(proba_classes).tolist())
        )
        proba = None if y_proba is None else np.asarray(y_proba, dtype=float)
        if proba is not None and proba.shape != (len(test_ids), len(classes)):
            raise ValueError("split result probabilities do not match the test set")
        predictions: dict[str, Prediction] = {}
        for position, trial_id in enumerate(test_ids):
            predictions[trial_id] = Prediction(
                label=int(y_pred[position]),
                probabilities=None if proba is None else proba[position],
                classes=classes,
            )
        return cls(predictions)


@dataclass(frozen=True)
class ThresholdPolicy:
    """Accept a prediction only when confidence reaches the threshold."""

    threshold: float = 0.6
    action_template: str = "select:{label}"
    abstain_action: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.threshold) <= 1.0:
            raise ValueError("threshold must be in [0, 1]")

    def decide(self, prediction: Prediction, trial: StreamTrial) -> str | None:
        confidence = prediction.confidence
        if confidence is not None and confidence < self.threshold:
            return self.abstain_action
        return self.action_template.format(label=prediction.label, trial=trial.trial_id)


@dataclass
class ClosedLoopResult:
    events: list[ClosedLoopEvent] = field(default_factory=list)
    n_labeled: int = 0
    n_correct: int = 0
    n_actions: int = 0
    n_abstained: int = 0

    @property
    def accuracy(self) -> float | None:
        if not self.n_labeled:
            return None
        return self.n_correct / self.n_labeled

    @property
    def action_rate(self) -> float | None:
        total = self.n_actions + self.n_abstained
        if not total:
            return None
        return self.n_actions / total

    def summary(self) -> dict[str, Any]:
        return {
            "n_events": len(self.events),
            "n_labeled": self.n_labeled,
            "n_correct": self.n_correct,
            "accuracy": self.accuracy,
            "n_actions": self.n_actions,
            "n_abstained": self.n_abstained,
            "action_rate": self.action_rate,
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "events": [
                {
                    "trial_id": event.trial.trial_id,
                    "timestamp": event.trial.timestamp,
                    "true_label": event.trial.true_label,
                    "predicted_label": event.prediction.label,
                    "confidence": event.prediction.confidence,
                    "action": event.action,
                    "decided_at": event.decided_at,
                }
                for event in self.events
            ],
        }


def run_closed_loop(
    decoder: OnlineDecoder,
    source: Iterable[StreamTrial],
    policy: ClosedLoopPolicy,
    *,
    on_event: Callable[[ClosedLoopEvent], None] | None = None,
    max_trials: int | None = None,
) -> ClosedLoopResult:
    """Consume a stream in order and apply a closed-loop policy.

    The decoder sees one trial at a time; no future trial is passed to the
    policy or the decoder. This function is transport-agnostic: a replay
    source, a live amplifier adapter, or a simulator can all satisfy
    ``StreamSource``.
    """
    if max_trials is not None and max_trials <= 0:
        raise ValueError("max_trials must be positive")
    result = ClosedLoopResult()
    for order, trial in enumerate(source):
        if max_trials is not None and order >= max_trials:
            break
        prediction = decoder.predict(trial)
        if not isinstance(prediction, Prediction):
            prediction = Prediction(label=int(prediction))
        action = policy.decide(prediction, trial)
        event = ClosedLoopEvent(
            trial=trial,
            prediction=prediction,
            action=action,
            decided_at=float(trial.timestamp),
        )
        result.events.append(event)
        if trial.true_label is not None:
            result.n_labeled += 1
            if int(trial.true_label) == prediction.label:
                result.n_correct += 1
        if action is None:
            result.n_abstained += 1
        else:
            result.n_actions += 1
        if on_event is not None:
            on_event(event)
    return result


def replay_from_split(
    epochs: Epochs,
    split: Split,
    result: Mapping[str, Any],
    *,
    policy: ClosedLoopPolicy | None = None,
    start_time: float = 0.0,
    hop_seconds: float = 0.0,
) -> ClosedLoopResult:
    """Replay a fitted pipeline's held-out predictions as a closed-loop stream."""
    decoder = ReplayDecoder.from_split_results(epochs, split, result)
    source = ReplaySource(
        epochs=epochs,
        indices=split.test,
        start_time=start_time,
        hop_seconds=hop_seconds,
    )
    return run_closed_loop(decoder, source, policy or ThresholdPolicy())


__all__ = [
    "ClosedLoopEvent",
    "ClosedLoopPolicy",
    "ClosedLoopResult",
    "OnlineDecoder",
    "Prediction",
    "ReplayDecoder",
    "ReplaySource",
    "StreamSource",
    "StreamTrial",
    "ThresholdPolicy",
    "replay_from_split",
    "run_closed_loop",
]
