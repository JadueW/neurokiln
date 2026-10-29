from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt

from .nodes import FitContext, Node


def design_bandpass(low: float, high: float, sfreq: float, order: int = 4) -> np.ndarray:
    nyquist = sfreq / 2.0
    if low <= 0 or high <= low:
        raise ValueError("bandpass requires 0 < low < high")
    high = min(high, nyquist * 0.99)
    if low >= high:
        raise ValueError("bandpass is outside the Nyquist range")
    return butter(order, [low / nyquist, high / nyquist], btype="bandpass", output="sos")


def filter_epochs(X: np.ndarray, sos: np.ndarray) -> np.ndarray:
    return sosfiltfilt(sos, X, axis=-1)


def _window_slice(
    n_times: int,
    sfreq: float,
    tmin: float,
    start: float,
    end: float,
) -> slice:
    """Map an absolute epoch-time window to a half-open sample slice."""
    if end <= start:
        raise ValueError("时间窗口的终点必须大于起点")
    tolerance = 0.5 / sfreq
    if start < tmin - tolerance:
        raise ValueError(
            f"时间窗口起点 {start:g}s 早于 epoch 起点 {tmin:g}s；"
            "基线校正需要包含刺激前采样点的 epoch"
        )
    i0 = max(0, int(np.ceil((start - tmin) * sfreq - 1e-9)))
    i1 = min(n_times, int(np.ceil((end - tmin) * sfreq - 1e-9)))
    if i0 >= n_times or i1 <= i0:
        raise ValueError(
            f"时间窗口 [{start:g}, {end:g})s 在当前 epoch "
            f"[{tmin:g}, {tmin + n_times / sfreq:g})s 内没有采样点"
        )
    return slice(i0, i1)


def _check_epochs(X: np.ndarray, name: str) -> None:
    if X.ndim != 3:
        raise ValueError(f"{name} expects (n_trials, n_channels, n_times)")


class Bandpass(Node):
    name = "bandpass"
    kind = "preprocess"
    cacheable = True

    def __init__(self, low: float = 8.0, high: float = 30.0, order: int = 4):
        self.low = float(low)
        self.high = float(high)
        self.order = int(order)

    def fit(self, X, y, ctx: FitContext):
        self.sos_ = design_bandpass(self.low, self.high, ctx.sfreq, self.order)
        return self

    def transform(self, X):
        return filter_epochs(X, self.sos_)


class FilterBankSplit(Node):
    """Split epochs into parallel adjacent bands exposed as named output ports."""

    name = "filter_bank_split"
    kind = "preprocess"
    cacheable = False

    def __init__(
        self, low: float = 4.0, high: float = 32.0, n_bands: int = 4, order: int = 4
    ):
        self.low = float(low)
        self.high = float(high)
        self.n_bands = int(n_bands)
        self.order = int(order)
        if not 2 <= self.n_bands <= 8:
            raise ValueError("滤波器组子带数必须在 2–8 之间")
        if self.low <= 0 or self.high <= self.low:
            raise ValueError("滤波器组需要 0 < low < high")

    def bands(self) -> list[tuple[float, float]]:
        edges = np.linspace(self.low, self.high, self.n_bands + 1)
        return [(float(edges[index]), float(edges[index + 1]))
                for index in range(self.n_bands)]

    def port_specs(self) -> list[dict[str, str]]:
        return [
            {
                "name": f"band_{index + 1}",
                "label": f"{low:g}–{high:g} Hz",
                "type": "epochs",
            }
            for index, (low, high) in enumerate(self.bands())
        ]

    def fit(self, X, y, ctx: FitContext):
        if self.high >= ctx.sfreq / 2.0:
            raise ValueError(
                f"滤波器组上限 {self.high:g} Hz 必须低于 Nyquist {ctx.sfreq / 2.0:g} Hz"
            )
        self.sos_ = [
            design_bandpass(low, high, ctx.sfreq, self.order)
            for low, high in self.bands()
        ]
        return self

    def transform(self, X):
        _check_epochs(X, "FilterBankSplit")
        return {
            f"band_{index + 1}": filter_epochs(X, sos)
            for index, sos in enumerate(self.sos_)
        }


class ChannelSelect(Node):
    """Keep the highest-variance channels; ranking uses training data only."""

    name = "channel_select"
    kind = "preprocess"
    cacheable = True

    def __init__(self, n_channels: int = 8):
        self.n_channels = int(n_channels)

    def fit(self, X, y, ctx: FitContext):
        if X.ndim != 3:
            raise ValueError("ChannelSelect expects (n_trials, n_channels, n_times)")
        count = min(max(1, self.n_channels), X.shape[1])
        variance = np.var(X, axis=(0, 2))
        self.indices_ = np.sort(np.argsort(variance)[::-1][:count])
        return self

    def transform(self, X):
        return X[:, self.indices_, :]


class Notch(Node):
    """Zero-phase IIR notch for mains or other narrowband interference."""

    name = "notch"
    kind = "preprocess"
    cacheable = True

    def __init__(self, freq: float = 50.0, q: float = 30.0):
        self.freq = float(freq)
        self.q = float(q)
        if self.freq <= 0:
            raise ValueError("陷波频率必须为正数")
        if self.q <= 0:
            raise ValueError("陷波 Q 值必须为正数")

    def fit(self, X, y, ctx: FitContext):
        _check_epochs(X, "Notch")
        if self.freq >= ctx.sfreq / 2.0:
            raise ValueError(
                f"陷波频率 {self.freq:g} Hz 必须低于 Nyquist ({ctx.sfreq / 2.0:g} Hz)"
            )
        self.b_, self.a_ = iirnotch(self.freq, self.q, fs=ctx.sfreq)
        return self

    def transform(self, X):
        return filtfilt(self.b_, self.a_, X, axis=-1)


class CommonAverageReference(Node):
    """Subtract the instantaneous channel average from every channel."""

    name = "car"
    kind = "preprocess"
    cacheable = True

    def transform(self, X):
        _check_epochs(X, "CommonAverageReference")
        return X - X.mean(axis=1, keepdims=True)


class Crop(Node):
    """Crop a time window using absolute epoch seconds (0 = stimulus onset)."""

    name = "crop"
    kind = "preprocess"
    cacheable = True

    def __init__(self, start: float = 0.0, duration: float = 0.0):
        self.start = float(start)
        self.duration = float(duration)
        if self.duration < 0:
            raise ValueError("裁剪时长不能为负数")

    def fit(self, X, y, ctx: FitContext):
        _check_epochs(X, "Crop")
        if self.duration <= 0:
            end = ctx.tmin + X.shape[-1] / ctx.sfreq
        else:
            end = self.start + self.duration
        self.slice_ = _window_slice(
            X.shape[-1], ctx.sfreq, ctx.tmin, self.start, end
        )
        return self

    def transform(self, X):
        return X[..., self.slice_]

    def output_timebase(self, sfreq: float, tmin: float) -> tuple[float, float]:
        return sfreq, self.start


class Resample(Node):
    """Polyphase resampling to a target sampling rate."""

    name = "resample"
    kind = "preprocess"
    cacheable = True

    def __init__(self, target_sfreq: float = 128.0):
        self.target_sfreq = float(target_sfreq)
        if self.target_sfreq <= 0:
            raise ValueError("目标采样率必须为正数")

    def _ratio(self, sfreq: float) -> tuple[int, int]:
        ratio = Fraction(self.target_sfreq / float(sfreq)).limit_denominator(1000)
        if ratio.numerator > 512 or ratio.denominator > 512:
            raise ValueError(
                f"重采样比例 {self.target_sfreq:g}/{sfreq:g} 过于极端"
            )
        return ratio.numerator, ratio.denominator

    def fit(self, X, y, ctx: FitContext):
        _check_epochs(X, "Resample")
        self.up_, self.down_ = self._ratio(ctx.sfreq)
        if X.shape[-1] * self.up_ < self.down_:
            raise ValueError("epoch 过短，无法重采样到目标采样率")
        return self

    def transform(self, X):
        if self.up_ == self.down_:
            return X
        return resample_poly(X, self.up_, self.down_, axis=-1)

    def output_timebase(self, sfreq: float, tmin: float) -> tuple[float, float]:
        up, down = self._ratio(sfreq)
        return float(sfreq) * up / down, tmin


class Baseline(Node):
    """Subtract the mean of an absolute epoch-time window from each channel."""

    name = "baseline"
    kind = "preprocess"
    cacheable = True

    def __init__(self, start: float = -0.2, end: float = 0.0):
        self.start = float(start)
        self.end = float(end)

    def fit(self, X, y, ctx: FitContext):
        _check_epochs(X, "Baseline")
        self.slice_ = _window_slice(
            X.shape[-1], ctx.sfreq, ctx.tmin, self.start, self.end
        )
        return self

    def transform(self, X):
        return X - X[..., self.slice_].mean(axis=2, keepdims=True)
