from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace

import numpy as np


def _as_str_array(values, length: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=str)
    if arr.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {arr.shape}")
    return arr


@dataclass(frozen=True)
class Recording:
    """Continuous EEG recording. Signals are (n_channels, n_times) in volts."""

    signals: np.ndarray
    sfreq: float
    channel_names: tuple[str, ...]
    events: np.ndarray | None = None
    subject_id: str = "unknown"
    session_id: str = "0"
    run_id: str = "0"
    paradigm: str = "MI"
    dataset: str = "unknown"
    dataset_version: str = "0"

    def __post_init__(self):
        signals = np.asarray(self.signals, dtype=np.float64)
        if signals.ndim != 2:
            raise ValueError("signals must be 2D (n_channels, n_times)")
        if len(self.channel_names) != signals.shape[0]:
            raise ValueError("channel_names length must match signals.shape[0]")
        if self.sfreq <= 0:
            raise ValueError("sfreq must be positive")
        object.__setattr__(self, "signals", signals)

    @property
    def n_times(self) -> int:
        return self.signals.shape[1]


@dataclass(frozen=True)
class Epochs:
    """Epoched trials: data is (n_trials, n_channels, n_times).

    Trial-level subject/session arrays make within-subject, cross-session and
    LOSO protocols operate on one canonical structure.
    """

    data: np.ndarray
    labels: np.ndarray
    sfreq: float
    channel_names: tuple[str, ...]
    tmin: float = 0.0
    tmax: float = 1.0
    trial_ids: np.ndarray | None = None
    subject_ids: np.ndarray | None = None
    session_ids: np.ndarray | None = None
    paradigm: str = "MI"
    paradigm_meta: dict = field(default_factory=dict)
    dataset: str = "unknown"
    dataset_version: str = "0"
    label_names: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        data = np.asarray(self.data, dtype=np.float64)
        if data.ndim != 3:
            raise ValueError("data must be 3D (n_trials, n_channels, n_times)")
        n_trials = data.shape[0]
        if n_trials == 0:
            raise ValueError("data must contain at least one trial")
        labels = np.asarray(self.labels, dtype=int)
        if labels.shape != (n_trials,):
            raise ValueError(f"labels must have shape ({n_trials},)")
        if labels.min() < 0:
            raise ValueError("labels must be non-negative class indices")
        if not np.all(np.isfinite(data)):
            raise ValueError("data contains NaN or inf")
        if len(self.channel_names) != data.shape[1]:
            raise ValueError("channel_names length must match data.shape[1]")
        if self.sfreq <= 0:
            raise ValueError("sfreq must be positive")
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(
            self, "trial_ids", _as_str_array(
                self.trial_ids if self.trial_ids is not None else [f"t{i:05d}" for i in range(n_trials)],
                n_trials, "trial_ids"))
        object.__setattr__(
            self, "subject_ids", _as_str_array(
                self.subject_ids if self.subject_ids is not None else np.repeat("S01", n_trials),
                n_trials, "subject_ids"))
        object.__setattr__(
            self, "session_ids", _as_str_array(
                self.session_ids if self.session_ids is not None else np.repeat("0", n_trials),
                n_trials, "session_ids"))
        if len(np.unique(self.trial_ids)) != n_trials:
            raise ValueError("trial_ids must be unique")

    @property
    def n_trials(self) -> int:
        return self.data.shape[0]

    @property
    def n_classes(self) -> int:
        return int(self.labels.max()) + 1

    @property
    def subjects(self) -> list[str]:
        return sorted(np.unique(self.subject_ids).tolist())

    def label_counts(self, indices: np.ndarray | None = None) -> dict[int, int]:
        labels = self.labels if indices is None else self.labels[indices]
        values, counts = np.unique(labels, return_counts=True)
        return {int(v): int(c) for v, c in zip(values, counts)}

    def select(self, indices: np.ndarray) -> "Epochs":
        idx = np.asarray(indices, dtype=int)
        return replace(
            self,
            data=self.data[idx],
            labels=self.labels[idx],
            trial_ids=self.trial_ids[idx],
            subject_ids=self.subject_ids[idx],
            session_ids=self.session_ids[idx],
        )

    def signature(self) -> str:
        cached = getattr(self, "_signature_cache", None)
        if cached is not None:
            return cached
        payload = {
            "dataset": self.dataset,
            "dataset_version": self.dataset_version,
            "paradigm": self.paradigm,
            "paradigm_meta": self.paradigm_meta,
            "n_trials": self.n_trials,
            "n_channels": len(self.channel_names),
            "n_times": self.data.shape[2],
            "sfreq": self.sfreq,
            "tmin": self.tmin,
            "tmax": self.tmax,
            "trial_ids": self.trial_ids.tolist(),
            "subject_ids": self.subject_ids.tolist(),
            "session_ids": self.session_ids.tolist(),
            "labels": self.labels.tolist(),
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode())
        data = np.ascontiguousarray(self.data)
        digest.update(str(data.dtype).encode())
        digest.update(str(data.shape).encode())
        digest.update(memoryview(data).cast("B"))
        signature = digest.hexdigest()
        object.__setattr__(self, "_signature_cache", signature)
        return signature
