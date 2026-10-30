"""Inspect and convert user-supplied EDF/FIF/CSV recordings into Epochs.

The importers keep the heavy lifting in MNE for raw formats, while CSV stays
dependency-light. Every path returns the same Epochs structure used by the
built-in adapters, so downstream protocols and pipelines need no special case.
"""

from __future__ import annotations

import csv
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .model import Epochs

SUPPORTED_SUFFIXES = (".npz", ".edf", ".fif", ".csv")
MAX_CSV_BYTES = 256 * 1024**2

_PARADIGM_WINDOWS = {
    "MI": (0.5, 3.5),
    "P300": (0.0, 0.8),
    "SSVEP": (0.15, 4.15),
}

_META_COLUMNS = {
    "label", "labels", "class", "classes", "event", "events", "marker",
    "markers", "trial_type", "trigger",
}
_TIME_COLUMNS = {"time", "times", "t", "timestamp", "seconds", "sec"}
_TRIAL_COLUMNS = {"trial", "trial_id", "epoch", "epoch_id", "run"}
_SUBJECT_COLUMNS = {"subject", "subject_id", "sub", "participant", "subjectid"}
_SESSION_COLUMNS = {"session", "session_id", "ses"}
_RESERVED_COLUMNS = _META_COLUMNS | _TIME_COLUMNS | _TRIAL_COLUMNS | _SUBJECT_COLUMNS | _SESSION_COLUMNS


def _mne():
    try:
        import mne
    except ImportError as exc:  # pragma: no cover - exercised when extras are absent
        raise ValueError(
            "读取 EDF/FIF 需要安装 MNE：uv pip install 'neurokiln-engine[datasets]'"
        ) from exc
    mne.set_log_level("ERROR")
    return mne


def _clean_name(value: str) -> str:
    text = re.sub(r"^(stimulus[/:_\s-]*|event[/:_\s-]*|trigger[/:_\s-]*)", "", str(value), flags=re.I)
    text = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_.-]+", "_", text).strip("_.-")
    return text or "class"


def _suggest_label(description: str, index: int) -> str:
    text = _clean_name(description)
    if text.isdigit():
        text = f"class_{text}"
    return text or f"class_{index + 1}"


def _default_window(paradigm: str, duration: float) -> tuple[float, float]:
    tmin, tmax = _PARADIGM_WINDOWS.get(paradigm.upper(), (0.0, 1.0))
    if duration <= 0:
        return tmin, max(tmax, tmin + 0.2)
    if tmin >= duration:
        tmin = 0.0
    tmax = min(tmax, max(tmin + 0.2, duration))
    return round(tmin, 3), round(tmax, 3)


def _suggested_events(keys: Sequence[str]) -> list[str]:
    blocked = {"bad", "edge", "boundary", "new segment", "rest", "resting"}
    return [key for key in keys if str(key).strip().lower() not in blocked]


def inspect_import_file(path: str | Path, paradigm: str = "MI") -> dict[str, Any]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return _inspect_npz(path)
    if suffix in {".edf", ".fif"}:
        return _inspect_raw(path, paradigm)
    if suffix == ".csv":
        return _inspect_csv(path, paradigm)
    raise ValueError("仅支持 .npz / .edf / .fif / .csv 文件")


def _inspect_npz(path: Path) -> dict[str, Any]:
    from .adapters import load_npz_epochs

    epochs = load_npz_epochs(path)
    counts = epochs.label_counts()
    return {
        "kind": "epochs",
        "format": "npz",
        "channels": list(epochs.channel_names),
        "all_channels": list(epochs.channel_names),
        "sfreq": float(epochs.sfreq),
        "n_trials": epochs.n_trials,
        "n_times": int(epochs.data.shape[2]),
        "duration": float(epochs.data.shape[2] / epochs.sfreq),
        "tmin": float(epochs.tmin),
        "tmax": float(epochs.tmax),
        "subjects": epochs.subjects,
        "label_names": list(epochs.label_names),
        "label_counts": {
            str(epochs.label_names[index] if index < len(epochs.label_names) else index): count
            for index, count in counts.items()
        },
        "paradigm": epochs.paradigm,
        "defaults": {
            "tmin": float(epochs.tmin),
            "tmax": float(epochs.tmax),
            "resample": float(epochs.sfreq),
            "baseline": None,
        },
        "event_counts": {},
        "default_mapping": {},
        "suggested_events": [],
        "ready": True,
    }


def _inspect_raw(path: Path, paradigm: str) -> dict[str, Any]:
    mne = _mne()
    if path.suffix.lower() == ".fif":
        raw = mne.io.read_raw_fif(str(path), preload=False, verbose="ERROR")
    else:
        raw = mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR")
    try:
        types = raw.get_channel_types()
        all_channels = list(raw.ch_names)
        eeg_channels = [
            name for name, kind in zip(all_channels, types) if str(kind).lower() == "eeg"
        ]
        channels = eeg_channels or all_channels
        sfreq = float(raw.info["sfreq"])
        n_times = int(raw.n_times)
        duration = n_times / sfreq if sfreq > 0 else 0.0
        event_counts: OrderedDict[str, int] = OrderedDict()
        event_source = "annotations"
        for description in raw.annotations.description:
            text = str(description)
            if text:
                event_counts[text] = event_counts.get(text, 0) + 1
        if not event_counts:
            event_source = "stim"
            try:
                events = mne.find_events(raw, verbose="ERROR")
            except (RuntimeError, ValueError):
                events = np.empty((0, 3), dtype=int)
            for code in events[:, 2] if len(events) else []:
                key = str(int(code))
                event_counts[key] = event_counts.get(key, 0) + 1
        if not event_counts:
            event_source = "none"
        suggested = _suggested_events(list(event_counts))
        tmin, tmax = _default_window(paradigm, duration)
        return {
            "kind": "raw",
            "format": path.suffix.lower().lstrip("."),
            "channels": channels,
            "all_channels": all_channels,
            "channel_types": dict(zip(all_channels, types)),
            "sfreq": sfreq,
            "n_times": n_times,
            "duration": duration,
            "event_counts": dict(event_counts),
            "event_source": event_source,
            "suggested_events": suggested,
            "default_mapping": {
                key: _suggest_label(key, index) for index, key in enumerate(suggested)
            },
            "defaults": {
                "tmin": tmin,
                "tmax": tmax,
                "resample": sfreq,
                "baseline": None,
            },
            "ready": bool(channels) and bool(suggested),
        }
    finally:
        del raw


def _detect_column(headers: Sequence[str], candidates: set[str]) -> str | None:
    lookup = {str(header).strip().lower(): header for header in headers}
    for candidate in sorted(candidates):
        if candidate in lookup:
            return lookup[candidate]
    return None


def _numeric(value: str) -> bool:
    try:
        float(str(value).strip())
        return True
    except (TypeError, ValueError):
        return False


def _inspect_csv(path: Path, paradigm: str) -> dict[str, Any]:
    if path.stat().st_size > MAX_CSV_BYTES:
        raise ValueError("CSV 文件超过 256 MiB 上限，请先转成 NPZ")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = [header for header in (reader.fieldnames or []) if header]
        if not headers:
            raise ValueError("CSV 缺少表头")
        sample_rows = []
        for index, row in enumerate(reader):
            sample_rows.append(row)
            if index >= 20:
                break
    numeric_columns = {
        header for header in headers
        if sample_rows and all(_numeric(row.get(header, "")) for row in sample_rows)
    }
    label_column = _detect_column(headers, _META_COLUMNS)
    time_column = _detect_column(headers, _TIME_COLUMNS)
    trial_column = _detect_column(headers, _TRIAL_COLUMNS)
    subject_column = _detect_column(headers, _SUBJECT_COLUMNS)
    session_column = _detect_column(headers, _SESSION_COLUMNS)
    channels = [
        header for header in headers
        if header in numeric_columns and header not in _RESERVED_COLUMNS
    ]
    if not channels:
        raise ValueError("CSV 中没有识别到数值通道列")
    if not label_column:
        raise ValueError("CSV 缺少标签列（label/class/event/marker 之一）")
    label_counts: OrderedDict[str, int] = OrderedDict()
    n_rows = 0
    first_time = None
    last_time = None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            n_rows += 1
            label = str(row.get(label_column, "")).strip()
            if label:
                label_counts[label] = label_counts.get(label, 0) + 1
            if time_column:
                value = row.get(time_column, "")
                if _numeric(value):
                    value = float(value)
                    if first_time is None:
                        first_time = value
                    last_time = value
    sfreq = 250.0
    if first_time is not None and last_time is not None and n_rows > 1:
        elapsed = last_time - first_time
        if elapsed > 0:
            sfreq = (n_rows - 1) / elapsed
    duration = n_rows / sfreq if sfreq > 0 else 0.0
    tmin, tmax = _default_window(paradigm, duration)
    suggested = _suggested_events(list(label_counts))
    return {
        "kind": "csv",
        "format": "csv",
        "channels": channels,
        "all_channels": headers,
        "sfreq": round(float(sfreq), 6),
        "n_times": n_rows,
        "duration": round(float(duration), 6),
        "event_counts": dict(label_counts),
        "event_source": "column",
        "suggested_events": suggested,
        "default_mapping": {
            key: _clean_name(key) for key in suggested
        },
        "label_column": label_column,
        "time_column": time_column,
        "trial_column": trial_column,
        "subject_column": subject_column,
        "session_column": session_column,
        "defaults": {
            "tmin": tmin,
            "tmax": tmax,
            "resample": round(float(sfreq), 6),
            "baseline": None,
        },
        "ready": bool(suggested),
    }


def _row_metadata(values: np.ndarray | None, rows: Sequence[int], fallback: str) -> np.ndarray:
    if values is None:
        return np.repeat(fallback, len(rows))
    return np.asarray(
        [str(values[row]).strip() or fallback for row in rows],
        dtype=str,
    )


def _paradigm_meta(paradigm: str, tmin: float, tmax: float, config: dict[str, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "task": paradigm.lower(),
        "window_seconds": round(float(tmax - tmin), 6),
    }
    if paradigm.upper() == "P300":
        meta["positive_label"] = str(config.get("positive_label") or "Target")
    if paradigm.upper() == "SSVEP" and config.get("stimulus_frequencies"):
        meta["stimulus_frequencies"] = [
            float(value) for value in config["stimulus_frequencies"]
        ]
    return meta


def _resample(data: np.ndarray, sfreq: float, target: float) -> tuple[np.ndarray, float]:
    if not target or abs(float(target) - float(sfreq)) < 1e-9:
        return data, float(sfreq)
    if target <= 0:
        raise ValueError("重采样目标必须大于 0")
    if target > 4096:
        raise ValueError("重采样目标不能超过 4096 Hz")
    from math import gcd

    from scipy.signal import resample_poly

    source = int(round(float(sfreq)))
    goal = int(round(float(target)))
    factor = gcd(source, goal)
    out = resample_poly(data, goal // factor, source // factor, axis=-1)
    return np.asarray(out, dtype=np.float64), float(target)


def _window_epochs(
    data: np.ndarray,
    labels: np.ndarray,
    row_subjects: np.ndarray,
    row_sessions: np.ndarray,
    sfreq: float,
    tmin: float,
    tmax: float,
    paradigm: str,
    config: dict[str, Any],
    dataset_id: str,
) -> Epochs:
    if tmax <= tmin:
        raise ValueError("tmax 必须大于 tmin")
    if tmax - tmin > 60:
        raise ValueError("单个试次窗口不能超过 60 秒")
    start_offset = int(round(tmin * sfreq))
    end_offset = int(round(tmax * sfreq))
    n_times = end_offset - start_offset
    if n_times < 2:
        raise ValueError("窗口太短，至少需要 2 个采样点")
    mapped = config.get("event_map") or {}
    clean_map = {
        str(key): _clean_name(str(value))
        for key, value in mapped.items()
        if str(key) in labels and str(value).strip()
    }
    if len(set(clean_map.values())) < 2:
        raise ValueError("请至少映射两个不同的事件类别")
    class_names = sorted(set(clean_map.values()))
    class_index = {name: index for index, name in enumerate(class_names)}
    if len(labels):
        changed = np.ones(len(labels), dtype=bool)
        changed[1:] = labels[1:] != labels[:-1]
        starts = np.flatnonzero(changed)
    else:
        starts = np.empty(0, dtype=int)
    kept: list[int] = []
    kept_labels: list[int] = []
    for start in starts:
        raw_label = labels[int(start)]
        display = clean_map.get(str(raw_label))
        if display is None:
            continue
        begin = int(start) + start_offset
        end = int(start) + end_offset
        if begin < 0 or end > data.shape[1]:
            continue
        kept.append(int(start))
        kept_labels.append(class_index[display])
    if not kept:
        raise ValueError("没有找到落在记录范围内的完整试次，请调整窗口或事件映射")
    windows = np.stack(
        [data[:, start + start_offset:start + end_offset] for start in kept],
        axis=0,
    ).astype(np.float64)
    n_epochs = len(kept)
    subject_ids = _row_metadata(
        row_subjects, kept, str(config.get("subject_id") or "S01")
    )
    session_ids = _row_metadata(
        row_sessions, kept, str(config.get("session_id") or "0")
    )
    return Epochs(
        data=windows,
        labels=np.asarray(kept_labels, dtype=int),
        sfreq=float(sfreq),
        channel_names=tuple(str(name) for name in config["channels"]),
        tmin=float(start_offset / sfreq),
        tmax=float((end_offset - 1) / sfreq),
        trial_ids=np.asarray([f"{dataset_id}-t{index:04d}" for index in range(n_epochs)], dtype=str),
        subject_ids=subject_ids,
        session_ids=session_ids,
        paradigm=paradigm.upper(),
        paradigm_meta=_paradigm_meta(paradigm, tmin, tmax, config),
        dataset=dataset_id,
        dataset_version="import-v1",
        label_names=tuple(class_names),
    )


def build_csv_epochs(path: str | Path, config: dict[str, Any], dataset_id: str) -> Epochs:
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError("CSV 导入器只接受 .csv 文件")
    if path.stat().st_size > MAX_CSV_BYTES:
        raise ValueError("CSV 文件超过 256 MiB 上限")
    channels = list(config.get("channels") or [])
    if not channels:
        raise ValueError("请至少选择一个通道")
    label_column = config.get("label_column")
    if not label_column:
        raise ValueError("请指定标签列")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = [header for header in (reader.fieldnames or []) if header]
        missing = [name for name in [*channels, label_column] if name not in headers]
        if missing:
            raise ValueError(f"CSV 缺少列：{', '.join(missing)}")
        time_column = config.get("time_column")
        trial_column = config.get("trial_column")
        subject_column = config.get("subject_column")
        session_column = config.get("session_column")
        values = {name: [] for name in channels}
        labels: list[str] = []
        times: list[float] = []
        subjects: list[str] = []
        sessions: list[str] = []
        for row in reader:
            for name in channels:
                raw = row.get(name, "")
                if not _numeric(raw):
                    raise ValueError(f"通道 {name} 含非数值：{raw!r}")
                values[name].append(float(raw))
            labels.append(str(row.get(label_column, "")).strip())
            if time_column:
                raw_time = row.get(time_column, "")
                if _numeric(raw_time):
                    times.append(float(raw_time))
            subjects.append(str(row.get(subject_column, "")).strip() if subject_column else "")
            sessions.append(str(row.get(session_column, "")).strip() if session_column else "")
    if not labels:
        raise ValueError("CSV 没有数据行")
    data = np.asarray([values[name] for name in channels], dtype=np.float64)
    sfreq = float(config.get("sfreq") or 0.0)
    if time_column and len(times) == len(labels) and len(times) > 1:
        elapsed = times[-1] - times[0]
        if elapsed > 0:
            sfreq = (len(times) - 1) / elapsed
    if not sfreq:
        sfreq = 250.0
    resample = float(config.get("resample") or sfreq)
    data, sfreq = _resample(data, sfreq, resample)
    label_array = np.asarray(labels, dtype=str)
    subject_array = np.asarray(subjects, dtype=str) if subject_column else None
    session_array = np.asarray(sessions, dtype=str) if session_column else None
    if trial_column:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            trial_values = [str(row.get(trial_column, "")).strip() for row in reader]
        if len(trial_values) != len(labels):
            raise ValueError("trial 列长度与数据行不一致")
        starts: list[int] = []
        previous = None
        for index, trial in enumerate(trial_values):
            if trial and trial != previous:
                starts.append(index)
            previous = trial
        mask = np.zeros(len(labels), dtype=bool)
        for start in starts:
            mask[start] = True
        base_labels = label_array
        labels = np.where(mask, base_labels, "")
        # A trial block uses the label of its first row; subsequent rows are
        # ignored by the onset scanner, so labels are only needed at starts.
        for start in starts:
            labels[start] = label_array[start]
        return _window_from_trials(
            data, labels, starts, subject_array, session_array, sfreq,
            config, dataset_id,
        )
    return _window_epochs(
        data, label_array, subject_array, session_array, sfreq,
        float(config.get("tmin", 0.0)), float(config.get("tmax", 1.0)),
        str(config.get("paradigm") or "MI"), config, dataset_id,
    )


def _window_from_trials(
    data: np.ndarray,
    labels: np.ndarray,
    starts: Sequence[int],
    subject_array: np.ndarray | None,
    session_array: np.ndarray | None,
    sfreq: float,
    config: dict[str, Any],
    dataset_id: str,
) -> Epochs:
    tmin = float(config.get("tmin", 0.0))
    tmax = float(config.get("tmax", 1.0))
    start_offset = int(round(tmin * sfreq))
    end_offset = int(round(tmax * sfreq))
    if end_offset - start_offset < 2:
        raise ValueError("窗口太短，至少需要 2 个采样点")
    mapped = config.get("event_map") or {}
    clean_map = {
        str(key): _clean_name(str(value))
        for key, value in mapped.items()
        if str(value).strip()
    }
    if len(set(clean_map.values())) < 2:
        raise ValueError("请至少映射两个不同的事件类别")
    class_names = sorted(set(clean_map.values()))
    class_index = {name: index for index, name in enumerate(class_names)}
    kept: list[int] = []
    kept_labels: list[int] = []
    for start in starts:
        raw_label = str(labels[int(start)])
        display = clean_map.get(raw_label)
        if display is None:
            continue
        begin = int(start) + start_offset
        end = int(start) + end_offset
        if begin < 0 or end > data.shape[1]:
            continue
        if not labels[int(start)]:
            continue
        kept.append(int(start))
        kept_labels.append(class_index[display])
    if not kept:
        raise ValueError("没有找到落在记录范围内的完整试次，请调整窗口或事件映射")
    windows = np.stack(
        [data[:, start + start_offset:start + end_offset] for start in kept],
        axis=0,
    ).astype(np.float64)
    subject_ids = _row_metadata(
        subject_array, kept, str(config.get("subject_id") or "S01")
    )
    session_ids = _row_metadata(
        session_array, kept, str(config.get("session_id") or "0")
    )
    return Epochs(
        data=windows,
        labels=np.asarray(kept_labels, dtype=int),
        sfreq=float(sfreq),
        channel_names=tuple(str(name) for name in config["channels"]),
        tmin=float(start_offset / sfreq),
        tmax=float((end_offset - 1) / sfreq),
        trial_ids=np.asarray([f"{dataset_id}-t{index:04d}" for index in range(len(kept))], dtype=str),
        subject_ids=subject_ids,
        session_ids=session_ids,
        paradigm=str(config.get("paradigm") or "MI").upper(),
        paradigm_meta=_paradigm_meta(str(config.get("paradigm") or "MI"), tmin, tmax, config),
        dataset=dataset_id,
        dataset_version="import-v1",
        label_names=tuple(class_names),
    )


def build_raw_epochs(path: str | Path, config: dict[str, Any], dataset_id: str) -> Epochs:
    path = Path(path)
    if path.suffix.lower() not in {".edf", ".fif"}:
        raise ValueError("原始数据导入器只接受 .edf / .fif 文件")
    mne = _mne()
    if path.suffix.lower() == ".fif":
        raw = mne.io.read_raw_fif(str(path), preload=False, verbose="ERROR")
    else:
        raw = mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR")
    try:
        channels = list(config.get("channels") or raw.ch_names)
        missing = [name for name in channels if name not in raw.ch_names]
        if missing:
            raise ValueError(f"记录缺少通道：{', '.join(missing)}")
        if not channels:
            raise ValueError("请至少选择一个通道")
        pick = [raw.ch_names.index(name) for name in channels]
        raw.pick(pick, verbose="ERROR")
        raw.load_data(verbose="ERROR")
        resample = float(config.get("resample") or raw.info["sfreq"])
        if resample > 0 and abs(resample - raw.info["sfreq"]) > 1e-9:
            raw.resample(resample, npad="auto", verbose="ERROR")
        event_map = {
            str(key): _clean_name(str(value))
            for key, value in (config.get("event_map") or {}).items()
            if str(value).strip()
        }
        if len(set(event_map.values())) < 2:
            raise ValueError("请至少映射两个不同的事件类别")
        events, code_by_desc, source = _extract_events(mne, raw)
        selected = {
            desc: code for desc, code in code_by_desc.items()
            if desc in event_map
        }
        if not selected:
            raise ValueError("选中的事件在记录中不存在")
        epochs = mne.Epochs(
            raw,
            events,
            event_id=selected,
            tmin=float(config.get("tmin", 0.0)),
            tmax=float(config.get("tmax", 1.0)),
            baseline=tuple(config["baseline"]) if config.get("baseline") else None,
            preload=True,
            reject_by_annotation=False,
            verbose="ERROR",
        )
        if len(epochs) == 0:
            raise ValueError("没有成功切出试次，请检查事件、窗口和被试标注")
        data = np.asarray(epochs.get_data(copy=True), dtype=np.float64)
        code_to_class = {code: event_map[desc] for desc, code in selected.items()}
        class_names = sorted(set(code_to_class.values()))
        class_index = {name: index for index, name in enumerate(class_names)}
        labels = np.asarray(
            [class_index[code_to_class[int(code)]] for code in epochs.events[:, 2]],
            dtype=int,
        )
        n_epochs = len(labels)
        subject_id = str(config.get("subject_id") or "S01")
        session_id = str(config.get("session_id") or "0")
        return Epochs(
            data=data,
            labels=labels,
            sfreq=float(epochs.info["sfreq"]),
            channel_names=tuple(epochs.ch_names),
            tmin=float(epochs.times[0]),
            tmax=float(epochs.times[-1]),
            trial_ids=np.asarray(
                [f"{dataset_id}-t{index:04d}" for index in range(n_epochs)], dtype=str
            ),
            subject_ids=np.repeat(subject_id, n_epochs),
            session_ids=np.repeat(session_id, n_epochs),
            paradigm=str(config.get("paradigm") or "MI").upper(),
            paradigm_meta={
                **_paradigm_meta(
                    str(config.get("paradigm") or "MI"),
                    float(epochs.times[0]),
                    float(epochs.times[-1]),
                    config,
                ),
                "event_source": source,
            },
            dataset=dataset_id,
            dataset_version="import-v1",
            label_names=tuple(class_names),
        )
    finally:
        del raw


def _extract_events(mne, raw) -> tuple[np.ndarray, dict[str, int], str]:
    descriptions = [str(value) for value in raw.annotations.description]
    if descriptions:
        events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
        return events, {str(key): int(value) for key, value in event_id.items()}, "annotations"
    events = mne.find_events(raw, verbose="ERROR")
    if not len(events):
        raise ValueError("记录中没有找到事件标注或 stim 通道")
    return events, {str(int(code)): int(code) for code in np.unique(events[:, 2])}, "stim"


def write_epochs_npz(epochs: Epochs, path: str | Path) -> Path:
    path = Path(path)
    np.savez_compressed(
        path,
        data=epochs.data,
        labels=epochs.labels,
        sfreq=np.asarray(epochs.sfreq),
        channel_names=np.asarray(epochs.channel_names, dtype=str),
        tmin=np.asarray(epochs.tmin),
        tmax=np.asarray(epochs.tmax),
        trial_ids=np.asarray(epochs.trial_ids, dtype=str),
        subject_ids=np.asarray(epochs.subject_ids, dtype=str),
        session_ids=np.asarray(epochs.session_ids, dtype=str),
        label_names=np.asarray(epochs.label_names, dtype=str),
        paradigm=np.asarray(epochs.paradigm),
        paradigm_meta_json=np.asarray(json.dumps(epochs.paradigm_meta, ensure_ascii=False)),
        dataset=np.asarray(epochs.dataset),
        dataset_version=np.asarray(epochs.dataset_version),
    )
    return path
