import os
from pathlib import Path

import numpy as np
import pytest

from neurokiln.data import (
    DatasetRegistry,
    MOABBAdapter,
    load_npz_epochs,
    make_synthetic_epochs,
)
from neurokiln.data.adapters import BUILTIN_SPECS


def _write_npz(path: Path, epochs) -> Path:
    np.savez(
        path,
        data=epochs.data,
        labels=epochs.labels,
        sfreq=epochs.sfreq,
        channel_names=np.asarray(epochs.channel_names, dtype=str),
        tmin=epochs.tmin,
        tmax=epochs.tmax,
        trial_ids=epochs.trial_ids,
        subject_ids=epochs.subject_ids,
        session_ids=epochs.session_ids,
        label_names=np.asarray(epochs.label_names or ("left", "right"), dtype=str),
        dataset=np.asarray(epochs.dataset),
        dataset_version=np.asarray(epochs.dataset_version),
    )
    return path


def test_npz_roundtrip(tmp_path):
    epochs = make_synthetic_epochs(
        n_trials_per_class=6, subjects=("S01", "S02"), sessions=("T1", "T2"), seed=4
    )
    path = _write_npz(tmp_path / "demo.npz", epochs)
    loaded = load_npz_epochs(path)
    np.testing.assert_allclose(loaded.data, epochs.data)
    np.testing.assert_array_equal(loaded.labels, epochs.labels)
    np.testing.assert_array_equal(loaded.trial_ids, epochs.trial_ids)
    np.testing.assert_array_equal(loaded.subject_ids, epochs.subject_ids)
    assert loaded.sfreq == epochs.sfreq
    assert loaded.channel_names == epochs.channel_names
    assert loaded.dataset == "synthetic"
    assert loaded.label_names == ("left", "right")


def test_npz_rejects_missing_fields_and_wrong_suffix(tmp_path):
    bad = tmp_path / "bad.npz"
    np.savez(bad, data=np.zeros((2, 2, 4)), labels=np.array([0, 1]))
    with pytest.raises(ValueError, match="NPZ 缺少必需字段"):
        load_npz_epochs(bad)
    with pytest.raises(ValueError, match=r"\.npz"):
        load_npz_epochs(tmp_path / "bad.npy")


def test_import_npz_registers_local_dataset(tmp_path):
    epochs = make_synthetic_epochs(
        n_trials_per_class=5, subjects=("S01", "S02"), seed=9
    )
    source = _write_npz(tmp_path / "source.npz", epochs)
    registry = DatasetRegistry(tmp_path / "data", cache_entries=1)

    spec = registry.import_npz(
        source,
        dataset_id="my-mi",
        name="我的运动想象数据",
        license="内部使用",
        citation="内部采集",
        description="用于端到端测试",
    )
    assert spec.id == "my-mi"
    assert spec.adapter == "local_npz"
    assert spec.subject_count == 2
    assert spec.trials == epochs.n_trials
    assert spec.classes == 2

    listed = {item.id: item for item in registry.specs()}
    assert "my-mi" in listed
    assert listed["my-mi"].name == "我的运动想象数据"

    loaded = registry.load("my-mi")
    np.testing.assert_allclose(loaded.data, epochs.data)

    only_s2 = registry.load("my-mi", subject_ids=["S02"])
    assert set(only_s2.subjects) == {"S02"}
    assert only_s2.n_trials < epochs.n_trials

    with pytest.raises(ValueError, match="not present"):
        registry.load("my-mi", subject_ids=["S99"])


def test_import_npz_guards(tmp_path):
    epochs = make_synthetic_epochs(n_trials_per_class=4, seed=1)
    source = _write_npz(tmp_path / "source.npz", epochs)
    registry = DatasetRegistry(tmp_path / "data")

    with pytest.raises(ValueError, match="dataset id"):
        registry.import_npz(source, dataset_id="Bad ID", name="x")
    with pytest.raises(ValueError, match="built-in"):
        registry.import_npz(source, dataset_id="synthetic", name="x")

    registry.import_npz(source, dataset_id="local-a", name="A")
    with pytest.raises(FileExistsError):
        registry.import_npz(source, dataset_id="local-a", name="A")
    overwritten = registry.import_npz(source, dataset_id="local-a", name="A2", overwrite=True)
    assert overwritten.name == "A2"


def test_registry_cache_is_bounded(tmp_path):
    registry = DatasetRegistry(tmp_path / "data", cache_entries=1)
    first = registry.load("synthetic")
    assert registry.load("synthetic") is first
    other = registry.load("synthetic", classes=3, options={"n_trials_per_class": 10})
    assert registry.load("synthetic") is not first
    assert registry.load("synthetic") is not other
    registry.clear_cache()
    assert registry.describe("synthetic")[0]["ready"] is True


def test_moabb_conversion_uses_metadata():
    pytest.importorskip("mne")
    import mne
    import pandas as pd

    rng = np.random.default_rng(2)
    info = mne.create_info(["C3", "Cz", "C4"], sfreq=250, ch_types="eeg")
    data = rng.normal(size=(2, 3, 80))
    mne_epochs = mne.EpochsArray(data, info, tmin=0.0)
    meta = pd.DataFrame(
        {"subject": [1, 1], "session": ["0train", "0train"], "run": [1, 1]}
    )
    spec = next(item for item in BUILTIN_SPECS if item.id == "bciciv2a")
    converted = MOABBAdapter(spec)._to_epochs(
        mne_epochs, np.asarray(["left_hand", "right_hand"]), meta, "1.7.2"
    )
    assert converted.data.shape == (2, 3, 80)
    assert converted.sfreq == 250.0
    assert converted.channel_names == ("C3", "Cz", "C4")
    np.testing.assert_array_equal(converted.labels, [0, 1])
    assert converted.label_names == ("left_hand", "right_hand")
    assert converted.subjects == ["S01"]
    assert converted.session_ids.tolist() == ["0train", "0train"]
    assert len(set(converted.trial_ids)) == 2
    assert converted.dataset == "bciciv2a"
    assert converted.dataset_version == "moabb-1.7.2"


@pytest.mark.skipif(
    os.environ.get("NEUROKILN_REAL_DATA") != "1",
    reason="set NEUROKILN_REAL_DATA=1 to run cached MOABB integration",
)
def test_cached_moabb_dataset_integration(tmp_path):
    registry = DatasetRegistry(tmp_path / "data", mne_data_dir=os.environ.get("MNE_DATA"))
    epochs = registry.load("bciciv2a", subject_ids=[1])
    assert epochs.n_trials > 0
    assert epochs.n_classes == 4
    assert epochs.data.shape[1] == 22
    assert epochs.sfreq == 250.0
    assert epochs.subjects == ["S01"]
    assert len(epochs.signature()) == 64
