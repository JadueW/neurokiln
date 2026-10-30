from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..cache import ArtifactCache
from .model import Epochs
from .synthetic import (
    make_synthetic_epochs,
    make_synthetic_p300_epochs,
    make_synthetic_ssvep_epochs,
)


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    name: str
    adapter: str
    paradigm: str
    license: str
    citation: str
    description: str
    subject_count: int
    channels: int
    sfreq: float
    classes: int
    trials: int
    requires_download: bool = False
    homepage: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


BUILTIN_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        id="synthetic",
        name="合成演示数据（引擎自检）",
        adapter="builtin",
        paradigm="MI",
        license="CC0",
        citation="NeuroKiln synthetic generator",
        description="带类别均值与噪声差异的合成 EEG，用于零下载验证协议与管线。",
        subject_count=1,
        channels=8,
        sfreq=128,
        classes=2,
        trials=80,
        homepage="",
    ),
    DatasetSpec(
        id="synthetic_p300",
        name="合成 P300（引擎自检）",
        adapter="builtin",
        paradigm="P300",
        license="CC0",
        citation="NeuroKiln synthetic P300 generator",
        description="带 300 ms 正波与噪声的合成 ERP，用于零下载验证 xDAWN 与校准链路。",
        subject_count=1,
        channels=8,
        sfreq=128,
        classes=2,
        trials=80,
        homepage="",
    ),
    DatasetSpec(
        id="synthetic_ssvep",
        name="合成 SSVEP（引擎自检）",
        adapter="builtin",
        paradigm="SSVEP",
        license="CC0",
        citation="NeuroKiln synthetic SSVEP generator",
        description="8 个刺激频率的合成稳态视觉诱发电位，用于零下载验证 CCA / FBCCA。",
        subject_count=1,
        channels=8,
        sfreq=256,
        classes=8,
        trials=96,
        homepage="",
    ),
    DatasetSpec(
        id="bciciv2a",
        name="BCI Competition IV 2a",
        adapter="moabb",
        paradigm="MI",
        license="研究使用，需引用原始论文",
        citation="Brunner et al., BCI Competition 2008 Graz data set A",
        description="9 被试、22 通道、4 类运动想象；通过 MOABB 加载并缓存。",
        subject_count=9,
        channels=22,
        sfreq=250,
        classes=4,
        trials=576,
        requires_download=True,
        homepage="https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_001.html",
    ),
    DatasetSpec(
        id="bciciv2b",
        name="BCI Competition IV 2b",
        adapter="moabb",
        paradigm="MI",
        license="研究使用，需引用原始论文",
        citation="Leeb et al., BCI Competition 2008 Graz data set B",
        description="9 被试、3 通道、左右手二分类；通过 MOABB 加载并缓存。",
        subject_count=9,
        channels=3,
        sfreq=250,
        classes=2,
        trials=720,
        requires_download=True,
        homepage="https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_004.html",
    ),
    DatasetSpec(
        id="physionetmi",
        name="PhysioNet EEG Motor Imagery",
        adapter="moabb",
        paradigm="MI",
        license="ODC-BY 1.0",
        citation="Schalk et al., PhysioBank 2004",
        description="109 被试、64 通道、4 类运动想象；首次加载体积较大。",
        subject_count=109,
        channels=64,
        sfreq=160,
        classes=4,
        trials=0,
        requires_download=True,
        homepage="https://moabb.neurotechx.com/docs/generated/moabb.datasets.PhysionetMI.html",
    ),
    DatasetSpec(
        id="p300",
        name="BNCI2014_009 · P300 拼写",
        adapter="moabb",
        paradigm="P300",
        license="研究使用，需引用原始论文",
        citation="Krusienski et al., BCI Competition III dataset II / BNCI2014_009",
        description="10 被试 P300 拼写数据，Target/NonTarget 二分类；MOABB P300 范式 0–0.8 s。",
        subject_count=10,
        channels=16,
        sfreq=250,
        classes=2,
        trials=0,
        requires_download=True,
        homepage="https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_009.html",
    ),
    DatasetSpec(
        id="ssvep",
        name="Nakanishi 2015 SSVEP",
        adapter="moabb",
        paradigm="SSVEP",
        license="研究使用，需引用原始论文",
        citation="Nakanishi et al., 2015, SSVEP benchmark dataset",
        description="9 被试、12 个刺激频率（9.25–14.75 Hz）的 SSVEP 数据；CCA/FBCCA 使用频率元数据。",
        subject_count=9,
        channels=8,
        sfreq=256,
        classes=12,
        trials=0,
        requires_download=True,
        homepage="https://moabb.neurotechx.com/docs/generated/moabb.datasets.Nakanishi2015.html",
    ),
)

_MOABB_CONFIG = {
    "bciciv2a": {
        "dataset": "BNCI2014_001", "paradigm": "MotorImagery",
        "fmin": 8.0, "fmax": 32.0, "tmin": 0.5, "tmax": 3.5,
    },
    "bciciv2b": {
        "dataset": "BNCI2014_004", "paradigm": "MotorImagery",
        "fmin": 8.0, "fmax": 32.0, "tmin": 0.5, "tmax": 3.5,
    },
    "physionetmi": {
        "dataset": "PhysionetMI", "paradigm": "MotorImagery",
        "fmin": 8.0, "fmax": 32.0, "tmin": 0.5, "tmax": 3.5,
    },
    "p300": {
        "dataset": "BNCI2014_009", "paradigm": "P300",
        "fmin": 1.0, "fmax": 24.0, "tmin": 0.0, "tmax": 0.8,
    },
    "ssvep": {
        "dataset": "Nakanishi2015", "paradigm": "SSVEP",
        "fmin": 7.0, "fmax": 45.0, "tmin": 0.15, "tmax": 4.15,
    },
}

_MOABB_CLASSES = {
    key: value["dataset"] for key, value in _MOABB_CONFIG.items()
}

_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{1,62}$")


def _default_paradigm_meta(spec: DatasetSpec) -> dict[str, Any]:
    """Static paradigm metadata visible before a dataset is loaded."""
    if spec.id == "synthetic_p300":
        return {"task": "p300", "positive_label": "Target", "window_seconds": 1.0}
    if spec.id == "synthetic_ssvep":
        return {
            "task": "ssvep",
            "stimulus_frequencies": [round(8.0 + index, 2) for index in range(spec.classes)],
            "window_seconds": 1.0,
        }
    if spec.id == "p300":
        return {"task": "p300", "positive_label": "Target", "window_seconds": 0.8}
    if spec.id == "ssvep":
        return {
            "task": "ssvep",
            "stimulus_frequencies": [round(9.25 + 0.5 * index, 2) for index in range(12)],
            "window_seconds": 4.0,
        }
    if spec.paradigm == "MI":
        return {"task": "mi"}
    return {}


def _subject_number(value: int | str) -> int:
    text = str(value).strip()
    if text[:1].upper() == "S":
        text = text[1:]
    return int(text)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MOABBAdapter:
    """Load a MOABB dataset through its MotorImagery paradigm.

    The first load downloads into the configured MNE data directory if the
    dataset is not already cached. No dataset files are copied by this class.
    """

    def __init__(self, spec: DatasetSpec, mne_data_dir: str | Path | None = None):
        if spec.id not in _MOABB_CLASSES:
            raise ValueError(f"unsupported MOABB dataset: {spec.id}")
        self.spec = spec
        self.mne_data_dir = Path(mne_data_dir) if mne_data_dir else None

    def _configure_mne_path(self, mne) -> str:
        configured = (
            self.mne_data_dir
            or os.environ.get("NEUROKILN_MNE_DATA")
            or os.environ.get("MNE_DATA")
            or mne.get_config("MNE_DATA")
        )
        if configured:
            path = Path(configured).expanduser().resolve()
            path.mkdir(parents=True, exist_ok=True)
            mne.set_config("MNE_DATA", str(path), set_env=True)
            return str(path)
        return str(Path.home() / "mne_data")

    def load(
        self,
        subject_ids: Sequence[int | str] | None = None,
        classes: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Epochs:
        try:
            import mne
            import moabb
            from moabb.datasets import (
                BNCI2014_001,
                BNCI2014_004,
                BNCI2014_009,
                Nakanishi2015,
                PhysionetMI,
            )
            from moabb.paradigms import MotorImagery, P300, SSVEP
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "MOABB 适配器需要安装 mne 与 moabb：uv pip install 'neurokiln-engine[datasets]'"
            ) from exc

        dataset_classes = {
            "BNCI2014_001": BNCI2014_001,
            "BNCI2014_004": BNCI2014_004,
            "BNCI2014_009": BNCI2014_009,
            "Nakanishi2015": Nakanishi2015,
            "PhysionetMI": PhysionetMI,
        }
        config = _MOABB_CONFIG[self.spec.id]
        dataset_cls = dataset_classes[config["dataset"]]
        available = list(range(1, self.spec.subject_count + 1))
        selected = [_subject_number(s) for s in subject_ids] if subject_ids else [1]
        invalid = sorted(set(selected) - set(available))
        if invalid:
            raise ValueError(
                f"subjects {invalid} are outside the available range 1..{self.spec.subject_count}"
            )
        options = options or {}
        dataset = dataset_cls()
        paradigm_name = config["paradigm"]
        n_classes = int(classes or self.spec.classes)
        events = None
        requested = options.get("events")
        if requested:
            events = [str(value) for value in requested]
            n_classes = len(events)
        elif paradigm_name == "SSVEP" and classes is not None and int(classes) < len(dataset.event_id):
            events = sorted(dataset.event_id, key=float)[:n_classes]
        elif (
            paradigm_name == "MotorImagery"
            and classes is not None
            and int(classes) < len(dataset.event_id)
        ):
            available_events = list(dataset.event_id)
            preferred = [name for name in ("left_hand", "right_hand") if name in available_events]
            events = (
                preferred
                if n_classes == 2 and len(preferred) == 2
                else available_events[:n_classes]
            )
            n_classes = len(events)

        mne.set_log_level("ERROR")
        if hasattr(moabb, "set_log_level"):
            moabb.set_log_level("ERROR")
        self._configure_mne_path(mne)
        common = {
            "fmin": float(config["fmin"]),
            "fmax": min(float(config["fmax"]), self.spec.sfreq / 2.0 - 1.0),
            "events": events,
            "tmin": float(config["tmin"]),
            "tmax": float(config["tmax"]),
            "resample": float(self.spec.sfreq),
        }
        if paradigm_name == "MotorImagery":
            paradigm = MotorImagery(n_classes=n_classes, **common)
        elif paradigm_name == "P300":
            paradigm = P300(**common)
        elif paradigm_name == "SSVEP":
            paradigm = SSVEP(**common)
        else:  # pragma: no cover - guarded by config
            raise ValueError(f"unsupported MOABB paradigm: {paradigm_name}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            epochs, labels, meta = paradigm.get_data(
                dataset=dataset,
                subjects=selected,
                return_epochs=True,
                n_jobs=1,
            )
        return self._to_epochs(epochs, labels, meta, moabb.__version__)

    def _to_epochs(self, mne_epochs, labels, meta, moabb_version: str) -> Epochs:
        data = np.asarray(mne_epochs.get_data(copy=True), dtype=np.float64)
        meta = meta.reset_index(drop=True)
        labels = np.asarray(labels, dtype=str)
        raw_values = np.unique(labels).tolist()
        if self.spec.paradigm == "SSVEP":
            label_values = sorted(raw_values, key=lambda value: float(value))
        else:
            label_values = sorted(raw_values)
        label_lookup = {str(value): index for index, value in enumerate(label_values)}
        y = np.asarray([label_lookup[str(value)] for value in labels], dtype=int)

        subjects = (
            meta["subject"].astype(str).to_numpy()
            if "subject" in meta
            else np.repeat(str(self.spec.id), len(y))
        )
        sessions = (
            meta["session"].astype(str).to_numpy()
            if "session" in meta
            else np.repeat("0", len(y))
        )
        runs = (
            meta["run"].astype(str).to_numpy()
            if "run" in meta
            else np.repeat("0", len(y))
        )
        subject_ids = np.asarray([f"S{int(value):02d}" for value in subjects], dtype=str)
        trial_ids = np.asarray(
            [
                f"{self.spec.id}-s{int(subject):02d}-{session}-r{run}-{index:04d}"
                for index, (subject, session, run) in enumerate(zip(subjects, sessions, runs))
            ],
            dtype=str,
        )
        times = mne_epochs.times
        paradigm_meta: dict[str, Any] = {
            "task": self.spec.paradigm.lower(),
            "window_seconds": float(times[-1] - times[0]),
        }
        if self.spec.paradigm == "SSVEP":
            paradigm_meta["stimulus_frequencies"] = [float(value) for value in label_values]
        elif self.spec.paradigm == "P300":
            paradigm_meta["positive_label"] = "Target"
        return Epochs(
            data=data,
            labels=y,
            sfreq=float(mne_epochs.info["sfreq"]),
            channel_names=tuple(mne_epochs.ch_names),
            tmin=float(times[0]),
            tmax=float(times[-1]),
            trial_ids=trial_ids,
            subject_ids=subject_ids,
            session_ids=sessions,
            paradigm=self.spec.paradigm,
            paradigm_meta=paradigm_meta,
            dataset=self.spec.id,
            dataset_version=f"moabb-{moabb_version}",
            label_names=tuple(str(value) for value in label_values),
        )


class SyntheticAdapter:
    def __init__(self, spec: DatasetSpec):
        self.spec = spec

    def load(
        self,
        subject_ids: Sequence[int | str] | None = None,
        classes: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Epochs:
        options = options or {}
        n_subjects = int(options.get("n_subjects", 1))
        subjects = (
            tuple(str(value) for value in subject_ids)
            if subject_ids
            else tuple(f"S{index + 1:02d}" for index in range(n_subjects))
        )
        sessions = tuple(
            f"T{index + 1}" for index in range(int(options.get("n_sessions", 1)))
        )
        if self.spec.paradigm == "P300":
            return make_synthetic_p300_epochs(
                n_trials_per_class=int(options.get("n_trials_per_class", 40)),
                n_channels=int(options.get("n_channels", 8)),
                n_times=int(options.get("n_times", 128)),
                sfreq=float(options.get("sfreq", 128.0)),
                subjects=subjects,
                sessions=sessions,
                seed=int(options.get("seed", 2026)),
                noise=float(options.get("noise", 0.8)),
            )
        if self.spec.paradigm == "SSVEP":
            return make_synthetic_ssvep_epochs(
                n_trials_per_class=int(options.get("n_trials_per_class", 12)),
                n_classes=int(classes or self.spec.classes),
                n_channels=int(options.get("n_channels", 8)),
                n_times=int(options.get("n_times", 256)),
                sfreq=float(options.get("sfreq", 256.0)),
                subjects=subjects,
                sessions=sessions,
                seed=int(options.get("seed", 2026)),
                noise=float(options.get("noise", 0.55)),
            )
        return make_synthetic_epochs(
            n_trials_per_class=int(options.get("n_trials_per_class", 40)),
            n_classes=int(classes or 2),
            n_channels=int(options.get("n_channels", 8)),
            n_times=int(options.get("n_times", 64)),
            sfreq=float(options.get("sfreq", 128.0)),
            subjects=subjects,
            sessions=sessions,
            seed=int(options.get("seed", 2026)),
            class_sep=float(options.get("class_sep", 0.08)),
            class_noise_step=float(options.get("class_noise_step", 0.6)),
        )


def load_npz_epochs(path: str | Path) -> Epochs:
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("本地数据必须是 .npz 文件")
    with np.load(path, allow_pickle=False) as payload:
        required = {"data", "labels", "sfreq", "channel_names"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"NPZ 缺少必需字段: {', '.join(missing)}")
        data = np.asarray(payload["data"], dtype=np.float64)
        labels = np.asarray(payload["labels"], dtype=int)
        sfreq = float(payload["sfreq"])
        channel_names = tuple(str(value) for value in payload["channel_names"].tolist())
        tmin = float(payload["tmin"]) if "tmin" in payload.files else 0.0
        tmax = float(payload["tmax"]) if "tmax" in payload.files else data.shape[2] / sfreq
        trial_ids = payload["trial_ids"].astype(str) if "trial_ids" in payload.files else None
        subject_ids = payload["subject_ids"].astype(str) if "subject_ids" in payload.files else None
        session_ids = payload["session_ids"].astype(str) if "session_ids" in payload.files else None
        label_names = (
            tuple(str(value) for value in payload["label_names"].tolist())
            if "label_names" in payload.files
            else tuple(f"class_{index}" for index in range(int(labels.max()) + 1))
        )
        dataset = str(payload["dataset"].item()) if "dataset" in payload.files else path.stem
        dataset_version = (
            str(payload["dataset_version"].item()) if "dataset_version" in payload.files else "1"
        )
        paradigm = str(payload["paradigm"].item()) if "paradigm" in payload.files else "MI"
        paradigm_meta = {}
        if "paradigm_meta_json" in payload.files:
            try:
                paradigm_meta = json.loads(str(payload["paradigm_meta_json"].item()))
            except (TypeError, ValueError, json.JSONDecodeError):
                paradigm_meta = {}
    return Epochs(
        data=data,
        labels=labels,
        sfreq=sfreq,
        channel_names=channel_names,
        tmin=tmin,
        tmax=tmax,
        trial_ids=trial_ids,
        subject_ids=subject_ids,
        session_ids=session_ids,
        paradigm=paradigm,
        paradigm_meta=paradigm_meta,
        dataset=dataset,
        dataset_version=dataset_version,
        label_names=label_names,
    )


def _read_sidecar(npz_path: Path) -> dict[str, Any]:
    sidecar = npz_path.with_suffix(".json")
    if not sidecar.exists():
        return {}
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _local_spec(path: Path) -> DatasetSpec:
    metadata = _read_sidecar(path)
    if {"n_trials", "n_channels", "sfreq", "classes"}.issubset(metadata):
        return DatasetSpec(
            id=str(metadata.get("id") or path.stem),
            name=str(metadata.get("name") or path.stem),
            adapter="local_npz",
            paradigm=str(metadata.get("paradigm") or "MI"),
            license=str(metadata.get("license") or "未声明"),
            citation=str(metadata.get("citation") or f"本地数据：{path.name}"),
            description=str(metadata.get("description") or "用户导入的本地 NPZ 运动想象数据。"),
            subject_count=len(metadata.get("subjects") or ["S01"]),
            channels=int(metadata["n_channels"]),
            sfreq=float(metadata["sfreq"]),
            classes=int(metadata["classes"]),
            trials=int(metadata["n_trials"]),
            requires_download=False,
            homepage="",
        )
    epochs = load_npz_epochs(path)
    return DatasetSpec(
        id=str(metadata.get("id") or path.stem),
        name=str(metadata.get("name") or path.stem),
        adapter="local_npz",
        paradigm=str(metadata.get("paradigm") or epochs.paradigm),
        license=str(metadata.get("license") or "未声明"),
        citation=str(metadata.get("citation") or f"本地数据：{path.name}"),
        description=str(metadata.get("description") or "用户导入的本地 NPZ 运动想象数据。"),
        subject_count=len(epochs.subjects),
        channels=len(epochs.channel_names),
        sfreq=float(epochs.sfreq),
        classes=epochs.n_classes,
        trials=epochs.n_trials,
        requires_download=False,
        homepage="",
    )


class DatasetRegistry:
    """Dataset catalog, loader and a small LRU cache.

    The cache is deliberately bounded because a real EEG dataset can occupy
    hundreds of megabytes after epoching.
    """

    def __init__(
        self,
        data_root: str | Path,
        cache_entries: int = 2,
        mne_data_dir: str | Path | None = None,
        artifact_cache: ArtifactCache | None = None,
    ):
        self.data_root = Path(data_root).expanduser().resolve()
        self.local_dir = self.data_root / "datasets" / "local"
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.mne_data_dir = Path(mne_data_dir) if mne_data_dir else None
        self._cache: OrderedDict[tuple, Epochs] = OrderedDict()
        self._cache_entries = max(1, int(cache_entries))
        self._lock = threading.Lock()
        self._artifact_cache = artifact_cache

    @property
    def artifact_cache(self) -> ArtifactCache:
        if self._artifact_cache is None:
            self._artifact_cache = ArtifactCache.from_data_root(self.data_root)
        return self._artifact_cache

    @property
    def moabb_installed(self) -> bool:
        return find_spec("moabb") is not None and find_spec("mne") is not None

    def specs(self) -> list[DatasetSpec]:
        local_specs = []
        for path in sorted(self.local_dir.glob("*.npz")):
            try:
                local_specs.append(_local_spec(path))
            except (ValueError, OSError):
                continue
        return [*BUILTIN_SPECS, *local_specs]

    def get_spec(self, dataset_id: str) -> DatasetSpec:
        for spec in self.specs():
            if spec.id == dataset_id:
                return spec
        raise ValueError(f"unknown dataset: {dataset_id}")

    def describe(self, dataset_id: str | None = None) -> list[dict[str, Any]]:
        specs = self.specs() if dataset_id is None else [self.get_spec(dataset_id)]
        items = []
        for spec in specs:
            if spec.adapter == "moabb":
                status = "可加载 · 首次使用自动下载" if self.moabb_installed else "缺少 MOABB 依赖"
                ready = self.moabb_installed
            elif spec.adapter == "local_npz":
                status = "本地文件就绪"
                ready = True
            else:
                status = "可直接运行"
                ready = True
            item = spec.to_dict()
            item.update(
                {
                    "ready": ready,
                    "status": status,
                    "paradigm_meta": _default_paradigm_meta(spec),
                }
            )
            if spec.adapter == "local_npz":
                item["subject_ids"] = _read_sidecar(
                    self.local_dir / f"{spec.id}.npz"
                ).get("subjects", [])
            elif spec.adapter == "moabb":
                item["subject_ids"] = [
                    f"S{index:02d}" for index in range(1, spec.subject_count + 1)
                ]
            else:
                item["subject_ids"] = [f"S{index:02d}" for index in range(1, 10)]
            items.append(item)
        return items

    def load(
        self,
        dataset_id: str,
        subject_ids: Sequence[int | str] | None = None,
        classes: int | None = None,
        options: dict[str, Any] | None = None,
        force: bool = False,
    ) -> Epochs:
        options = options or {}
        key = (
            dataset_id,
            tuple(str(value) for value in (subject_ids or [])),
            int(classes) if classes else None,
            tuple(sorted((str(k), str(v)) for k, v in options.items())),
        )
        with self._lock:
            if not force and key in self._cache:
                cached = self._cache.pop(key)
                self._cache[key] = cached
                return cached
            spec = self.get_spec(dataset_id)
            if spec.adapter == "moabb":
                adapter = MOABBAdapter(spec, self.mne_data_dir)
                epochs = adapter.load(subject_ids, classes, options)
            elif spec.adapter == "builtin":
                epochs = SyntheticAdapter(spec).load(subject_ids, classes, options)
            elif spec.adapter == "local_npz":
                path = self.local_dir / f"{dataset_id}.npz"
                if not path.exists():
                    raise ValueError(f"local dataset file is missing: {path}")
                epochs = load_npz_epochs(path)
                if subject_ids:
                    wanted = {str(value) for value in subject_ids}
                    mask = np.isin(epochs.subject_ids, list(wanted))
                    if not mask.any():
                        raise ValueError(f"subjects {sorted(wanted)} not present in {dataset_id}")
                    epochs = epochs.select(np.flatnonzero(mask))
            else:  # pragma: no cover - guarded by specs
                raise ValueError(f"unsupported adapter: {spec.adapter}")
            self._cache[key] = epochs
            while len(self._cache) > self._cache_entries:
                self._cache.popitem(last=False)
            return epochs

    def import_npz(
        self,
        source: str | Path,
        dataset_id: str,
        name: str,
        license: str = "未声明",
        citation: str = "",
        description: str = "",
        overwrite: bool = False,
    ) -> DatasetSpec:
        source = Path(source).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"source file does not exist: {source}")
        if not _SAFE_ID.match(dataset_id):
            raise ValueError("dataset id must match [a-z0-9][a-z0-9_-]{1,62}")
        if dataset_id in {spec.id for spec in BUILTIN_SPECS}:
            raise ValueError("cannot overwrite a built-in dataset id")
        validate = load_npz_epochs(source)
        destination = self.local_dir / f"{dataset_id}.npz"
        if destination.exists() and not overwrite:
            raise FileExistsError(f"dataset '{dataset_id}' already exists")
        shutil.copy2(source, destination)
        sidecar = {
            "id": dataset_id,
            "name": name or dataset_id,
            "paradigm": validate.paradigm,
            "license": license,
            "citation": citation,
            "description": description,
            "sha256": _sha256_file(destination),
            "n_trials": validate.n_trials,
            "n_channels": len(validate.channel_names),
            "sfreq": validate.sfreq,
            "classes": validate.n_classes,
            "subjects": validate.subjects,
        }
        destination.with_suffix(".json").write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return self.get_spec(dataset_id)

    def manifest(self, dataset_id: str, epochs: Epochs) -> dict[str, Any]:
        spec = self.get_spec(dataset_id)
        item = spec.to_dict()
        item.update(
            {
                "data_signature": epochs.signature(),
                "loaded_n_trials": epochs.n_trials,
                "loaded_channels": len(epochs.channel_names),
                "loaded_sfreq": epochs.sfreq,
                "loaded_classes": epochs.n_classes,
                "paradigm_meta": dict(epochs.paradigm_meta),
                "label_names": list(epochs.label_names),
                "subjects": epochs.subjects,
                "sessions": sorted(np.unique(epochs.session_ids).tolist()),
                "tmin": epochs.tmin,
                "tmax": epochs.tmax,
            }
        )
        return item

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()
