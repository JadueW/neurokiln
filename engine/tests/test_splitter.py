import numpy as np
import pytest

from neurokiln.data import make_synthetic_epochs
from neurokiln.protocol import (
    CrossSessionProtocol,
    LOSOProtocol,
    WithinSubjectProtocol,
    build_protocol,
)


def test_within_subject_disjoint_cover_and_stratified():
    epochs = make_synthetic_epochs(n_trials_per_class=25, n_classes=2)
    protocol = WithinSubjectProtocol(folds=5, repeats=2, seeds=(7, 8))
    splits = protocol.split(epochs)
    assert len(splits) == 10
    all_idx = np.arange(epochs.n_trials)
    for split in splits:
        assert np.intersect1d(split.train, split.test).size == 0
        np.testing.assert_array_equal(np.sort(np.concatenate([split.train, split.test])), all_idx)
        assert split.test.size == 10
        assert epochs.label_counts(split.test) == {0: 5, 1: 5}
        assert set(epochs.subject_ids[split.train]) == {"S01"}


def test_within_subject_reproducible_by_seed():
    epochs = make_synthetic_epochs(n_trials_per_class=20)
    a = WithinSubjectProtocol(folds=4, seeds=(42,)).split(epochs)
    b = WithinSubjectProtocol(folds=4, seeds=(42,)).split(epochs)
    c = WithinSubjectProtocol(folds=4, seeds=(43,)).split(epochs)
    for sa, sb in zip(a, b):
        np.testing.assert_array_equal(sa.train, sb.train)
        np.testing.assert_array_equal(sa.test, sb.test)
        assert sa.signature() == sb.signature()
    assert any(not np.array_equal(sa.test, sc.test) for sa, sc in zip(a, c))


def test_within_subject_guards():
    multi = make_synthetic_epochs(subjects=("S01", "S02"))
    with pytest.raises(ValueError):
        WithinSubjectProtocol().split(multi)
    small = make_synthetic_epochs(n_trials_per_class=3)
    with pytest.raises(ValueError):
        WithinSubjectProtocol(folds=5).split(small)


def test_loso_holds_out_each_subject():
    epochs = make_synthetic_epochs(n_trials_per_class=8, subjects=("S01", "S02", "S03"))
    splits = LOSOProtocol().split(epochs)
    assert len(splits) == 3
    for split, subject in zip(splits, ["S01", "S02", "S03"]):
        assert set(epochs.subject_ids[split.test]) == {subject}
        assert subject not in set(epochs.subject_ids[split.train])
        assert split.meta["test_subject"] == subject


def test_cross_session_ordered_pairs():
    epochs = make_synthetic_epochs(n_trials_per_class=6, sessions=("A", "B"))
    splits = CrossSessionProtocol().split(epochs)
    assert len(splits) == 2
    for split in splits:
        train_session = split.meta["train_session"]
        test_session = split.meta["test_session"]
        assert train_session != test_session
        assert set(epochs.session_ids[split.train]) == {train_session}
        assert set(epochs.session_ids[split.test]) == {test_session}


def test_build_protocol_factory():
    protocol = build_protocol("within_subject", folds=3, repeats=2, seeds=[1, 2])
    epochs = make_synthetic_epochs(n_trials_per_class=9)
    assert len(protocol.split(epochs)) == 6
    assert build_protocol("loso").name == "loso"
    with pytest.raises(ValueError):
        build_protocol("nope")
