from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from neurokiln.data.importers import (
    build_csv_epochs,
    build_raw_epochs,
    inspect_import_file,
    write_epochs_npz,
)


def _write_csv(path: Path, rows: list[list[object]], headers: list[str]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)
    return path


def test_inspect_and_build_epoched_csv(tmp_path: Path):
    headers = ["time", "C3", "C4", "label", "trial", "subject"]
    rows = []
    timestamp = 0.0
    for trial, label in enumerate(["left", "right", "left", "right"]):
        for sample in range(10):
            rows.append([timestamp, float(sample + trial), float(-sample), label, trial, "S01"])
            timestamp += 0.01
    path = _write_csv(tmp_path / "epoched.csv", rows, headers)
    info = inspect_import_file(path, paradigm="MI")
    assert info["kind"] == "csv"
    assert info["channels"] == ["C3", "C4"]
    assert info["label_column"] == "label"
    assert info["trial_column"] == "trial"
    assert info["time_column"] == "time"
    assert info["sfreq"] == pytest.approx(100.0, rel=1e-3)
    assert set(info["event_counts"]) == {"left", "right"}
    epochs = build_csv_epochs(
        path,
        {
            "channels": ["C3", "C4"],
            "label_column": "label",
            "trial_column": "trial",
            "time_column": "time",
            "subject_column": "subject",
            "sfreq": 100.0,
            "resample": 100.0,
            "tmin": 0.0,
            "tmax": 0.09,
            "event_map": {"left": "left_hand", "right": "right_hand"},
            "paradigm": "MI",
        },
        dataset_id="csv-mi",
    )
    assert epochs.data.shape == (4, 2, 9)
    assert epochs.label_names == ("left_hand", "right_hand")
    assert epochs.labels.tolist() == [0, 1, 0, 1]
    assert epochs.subject_ids.tolist() == ["S01"] * 4

    npz_path = write_epochs_npz(epochs, tmp_path / "converted.npz")
    from neurokiln.data.adapters import load_npz_epochs

    loaded = load_npz_epochs(npz_path)
    assert loaded.n_trials == 4
    assert loaded.dataset == "csv-mi"


def test_csv_event_onsets_without_trial_column(tmp_path: Path):
    headers = ["time", "Cz", "event"]
    rows = []
    labels = ["a"] * 4 + [""] * 2 + ["b"] * 4 + ["a"] * 4
    for index, label in enumerate(labels):
        rows.append([index * 0.01, float(index), label])
    path = _write_csv(tmp_path / "onsets.csv", rows, headers)
    epochs = build_csv_epochs(
        path,
        {
            "channels": ["Cz"],
            "label_column": "event",
            "time_column": "time",
            "sfreq": 100.0,
            "resample": 100.0,
            "tmin": 0.0,
            "tmax": 0.03,
            "event_map": {"a": "A", "b": "B"},
            "paradigm": "MI",
        },
        dataset_id="onsets",
    )
    assert epochs.n_trials == 3
    assert epochs.labels.tolist() == [0, 1, 0]


def test_csv_rejects_missing_label_column(tmp_path: Path):
    path = _write_csv(
        tmp_path / "bad.csv",
        [[0.0, 1.0], [0.01, 2.0]],
        ["time", "Cz"],
    )
    with pytest.raises(ValueError, match="标签列"):
        inspect_import_file(path)


def test_raw_fif_inspect_and_build(tmp_path: Path):
    mne = pytest.importorskip("mne")
    mne.set_log_level("ERROR")
    info = mne.create_info(["C3", "C4"], sfreq=100.0, ch_types="eeg")
    data = np.zeros((2, 2000))
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_annotations(
        mne.Annotations(
            onset=[0.5, 8.0],
            duration=[0.0, 0.0],
            description=["left", "right"],
        )
    )
    path = tmp_path / "sample-raw.fif"
    raw.save(path, overwrite=True, verbose="ERROR")
    report = inspect_import_file(path, paradigm="MI")
    assert report["kind"] == "raw"
    assert report["channels"] == ["C3", "C4"]
    assert set(report["event_counts"]) == {"left", "right"}
    epochs = build_raw_epochs(
        path,
        {
            "channels": ["C3", "C4"],
            "event_map": {"left": "left_hand", "right": "right_hand"},
            "tmin": 0.0,
            "tmax": 0.1,
            "resample": 100.0,
            "paradigm": "MI",
            "subject_id": "S07",
        },
        dataset_id="fif-mi",
    )
    assert epochs.data.shape == (2, 2, 11)
    assert epochs.labels.tolist() == [0, 1]
    assert epochs.subject_ids.tolist() == ["S07", "S07"]
