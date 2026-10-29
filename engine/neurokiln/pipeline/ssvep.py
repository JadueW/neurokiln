from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .nodes import FitContext, Node


def _reference_signals(
    frequency: float, sfreq: float, n_times: int, n_harmonics: int
) -> np.ndarray:
    times = np.arange(n_times, dtype=float) / float(sfreq)
    rows: list[np.ndarray] = []
    for harmonic in range(1, n_harmonics + 1):
        angle = 2.0 * np.pi * frequency * harmonic * times
        rows.append(np.sin(angle))
        rows.append(np.cos(angle))
    return np.vstack(rows)


def _cca_score(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    qx, _ = np.linalg.qr(x.T, mode="reduced")
    qy, _ = np.linalg.qr(y.T, mode="reduced")
    if qx.size == 0 or qy.size == 0:
        return 0.0
    singular = np.linalg.svd(qx.T @ qy, compute_uv=False)
    return float(singular[0]) if singular.size else 0.0


class SSVEPCCA(Node):
    """CCA / filter-bank CCA decoder using stimulus frequencies from metadata.

    CCA has no supervised parameters, but ``fit`` still validates the frequency
    table and binds class indices to the training-fold label set.
    """

    name = "ssvep_cca"
    kind = "decoder"

    def __init__(
        self,
        n_harmonics: int = 5,
        n_bands: int = 0,
    ):
        self.n_harmonics = int(n_harmonics)
        self.n_bands = int(n_bands)

    def _frequencies_for_classes(self, ctx: FitContext, classes: np.ndarray) -> np.ndarray:
        raw = (ctx.paradigm_meta or {}).get("stimulus_frequencies") or []
        if not raw:
            raise ValueError(
                "SSVEP 解码器需要 paradigm_meta.stimulus_frequencies；"
                "请选择 SSVEP 数据集或导入带频率元数据的 NPZ。"
            )
        frequencies = np.asarray([float(value) for value in raw], dtype=float)
        if frequencies.size < int(classes.max()) + 1:
            raise ValueError("刺激频率数量与类别数不匹配")
        return frequencies[classes]

    def _bands(self, sfreq: float) -> list[np.ndarray]:
        if self.n_bands <= 0:
            return []
        nyquist = sfreq / 2.0
        edges = [(6.0, 90.0), (14.0, 90.0), (22.0, 90.0), (30.0, 90.0), (38.0, 90.0)]
        bands = []
        for low, high in edges[: self.n_bands]:
            high = min(high, nyquist * 0.99)
            if low >= high or low <= 0:
                continue
            bands.append(butter(4, [low / nyquist, high / nyquist], btype="bandpass", output="sos"))
        return bands

    def fit(self, X: np.ndarray, y: np.ndarray | None, ctx: FitContext):
        if y is None:
            raise ValueError("SSVEPCCA requires training labels for class mapping")
        if X.ndim != 3:
            raise ValueError("SSVEPCCA expects (n_trials, n_channels, n_times)")
        self.classes_ = np.unique(y)
        if self.classes_.size < 2:
            raise ValueError("SSVEPCCA requires at least two classes")
        self.frequencies_ = self._frequencies_for_classes(ctx, self.classes_)
        self.sfreq_ = float(ctx.sfreq)
        self.n_times_ = int(X.shape[-1])
        self.bands_ = self._bands(self.sfreq_)
        self.references_ = [
            _reference_signals(float(frequency), self.sfreq_, self.n_times_, self.n_harmonics)
            for frequency in self.frequencies_
        ]
        return self

    def _scores(self, X: np.ndarray) -> np.ndarray:
        scores = np.zeros((len(X), len(self.frequencies_)), dtype=float)
        for trial_index, trial in enumerate(X):
            if not self.bands_:
                for class_index, reference in enumerate(self.references_):
                    scores[trial_index, class_index] = _cca_score(trial, reference)
                continue
            for band_index, sos in enumerate(self.bands_):
                filtered = sosfiltfilt(sos, trial, axis=-1)
                weight = 1.0 / ((band_index + 1) ** 1.25) + 0.25
                for class_index, reference in enumerate(self.references_):
                    scores[trial_index, class_index] += weight * _cca_score(filtered, reference)
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self._scores(X)
        if scores.shape[1] != self.classes_.size:
            raise ValueError("SSVEP frequency table does not match classifier classes")
        return self.classes_[scores.argmax(axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self._scores(X)
        scaled = scores * 10.0
        scaled -= scaled.max(axis=1, keepdims=True)
        exponent = np.exp(scaled)
        return exponent / exponent.sum(axis=1, keepdims=True)
