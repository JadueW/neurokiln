from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from ..data.model import Epochs


@dataclass(frozen=True)
class Split:
    """One train/test assignment over trial indices of a single Epochs object."""

    protocol: str
    fold: int
    repeat: int
    seed: int
    train: np.ndarray
    test: np.ndarray
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        train = np.asarray(self.train, dtype=int)
        test = np.asarray(self.test, dtype=int)
        if np.intersect1d(train, test).size:
            raise ValueError("train/test overlap detected")
        if train.size == 0 or test.size == 0:
            raise ValueError("train and test must both be non-empty")
        object.__setattr__(self, "train", train)
        object.__setattr__(self, "test", test)

    def signature(self) -> str:
        h = hashlib.sha256()
        h.update(f"{self.protocol}|{self.fold}|{self.repeat}|{self.seed}".encode())
        h.update(np.sort(self.train).tobytes())
        h.update(np.sort(self.test).tobytes())
        for key in sorted(self.meta):
            h.update(f"|{key}={self.meta[key]}".encode())
        return h.hexdigest()

    def summary(self, epochs: Epochs) -> dict:
        return {
            "protocol": self.protocol,
            "fold": self.fold,
            "repeat": self.repeat,
            "seed": self.seed,
            "n_train": int(self.train.size),
            "n_test": int(self.test.size),
            "train_labels": epochs.label_counts(self.train),
            "test_labels": epochs.label_counts(self.test),
            "train_subjects": sorted(np.unique(epochs.subject_ids[self.train]).tolist()),
            "test_subjects": sorted(np.unique(epochs.subject_ids[self.test]).tolist()),
            "signature": self.signature()[:16],
            **self.meta,
        }
