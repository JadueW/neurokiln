from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.model import Epochs
from .types import Split


def _stratified_folds(labels: np.ndarray, folds: int, rng: np.random.Generator) -> list[np.ndarray]:
    buckets: list[list[int]] = [[] for _ in range(folds)]
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        idx = idx[rng.permutation(idx.size)]
        for position, trial in enumerate(idx):
            buckets[position % folds].append(int(trial))
    return [np.sort(np.asarray(bucket, dtype=int)) for bucket in buckets]


@dataclass(frozen=True)
class WithinSubjectProtocol:
    """Stratified repeated k-fold over trials of exactly one subject."""

    folds: int = 5
    repeats: int = 1
    seeds: tuple[int, ...] = (0,)
    name: str = "within_subject"

    def __post_init__(self):
        if self.folds < 2:
            raise ValueError("folds must be >= 2")
        if self.repeats < 1:
            raise ValueError("repeats must be >= 1")
        if not self.seeds:
            raise ValueError("seeds must not be empty")

    def split(self, epochs: Epochs) -> list[Split]:
        if len(epochs.subjects) != 1:
            raise ValueError("within_subject requires epochs of exactly one subject")
        counts = epochs.label_counts()
        if min(counts.values()) < self.folds:
            raise ValueError("each class needs at least `folds` trials for stratified k-fold")
        splits: list[Split] = []
        for repeat in range(self.repeats):
            seed = int(self.seeds[repeat % len(self.seeds)])
            rng = np.random.default_rng(seed)
            test_folds = _stratified_folds(epochs.labels, self.folds, rng)
            all_idx = np.arange(epochs.n_trials)
            for fold, test in enumerate(test_folds):
                train = np.setdiff1d(all_idx, test)
                splits.append(Split(self.name, fold, repeat, seed, train, test))
        return splits


@dataclass(frozen=True)
class CrossSessionProtocol:
    """Train on one session, test on another, for every ordered session pair."""

    name: str = "cross_session"

    def split(self, epochs: Epochs) -> list[Split]:
        sessions = sorted(np.unique(epochs.session_ids).tolist())
        if len(sessions) < 2:
            raise ValueError("cross_session requires at least two sessions")
        splits: list[Split] = []
        fold = 0
        for train_session in sessions:
            for test_session in sessions:
                if train_session == test_session:
                    continue
                train = np.flatnonzero(epochs.session_ids == train_session)
                test = np.flatnonzero(epochs.session_ids == test_session)
                splits.append(Split(
                    self.name, fold, 0, 0, train, test,
                    {"train_session": train_session, "test_session": test_session},
                ))
                fold += 1
        return splits


@dataclass(frozen=True)
class LOSOProtocol:
    """Leave-one-subject-out; each held-out subject is one test fold."""

    name: str = "loso"

    def split(self, epochs: Epochs) -> list[Split]:
        subjects = epochs.subjects
        if len(subjects) < 2:
            raise ValueError("loso requires at least two subjects")
        splits: list[Split] = []
        for fold, subject in enumerate(subjects):
            test = np.flatnonzero(epochs.subject_ids == subject)
            train = np.flatnonzero(epochs.subject_ids != subject)
            splits.append(Split(self.name, fold, 0, 0, train, test, {"test_subject": subject}))
        return splits


def build_protocol(name: str, **params) -> WithinSubjectProtocol | CrossSessionProtocol | LOSOProtocol:
    if name == "within_subject":
        seeds = params.get("seeds", (0,))
        if isinstance(seeds, (list, tuple)) is False:
            seeds = (int(seeds),)
        return WithinSubjectProtocol(
            folds=int(params.get("folds", 5)),
            repeats=int(params.get("repeats", 1)),
            seeds=tuple(int(s) for s in seeds),
        )
    if name == "cross_session":
        return CrossSessionProtocol()
    if name == "loso":
        return LOSOProtocol()
    raise ValueError(f"unknown protocol: {name}")
