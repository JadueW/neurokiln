from __future__ import annotations

from functools import partial

import numpy as np
from scipy.signal import welch

from .filters import _check_epochs, _window_slice
from .nodes import FitContext, Node
from .riemann import estimate_covariances


class Bandpower(Node):
    """Welch band power per channel over one absolute epoch-time band."""

    name = "bandpower"
    kind = "feature"

    def __init__(
        self,
        low: float = 8.0,
        high: float = 30.0,
        relative: bool = False,
        log_power: bool = True,
    ):
        self.low = float(low)
        self.high = float(high)
        self.relative = bool(relative)
        self.log_power = bool(log_power)
        if self.low <= 0:
            raise ValueError("带通功率下界必须为正数")
        if self.high <= self.low:
            raise ValueError("带通功率上界必须大于下界")

    def fit(self, X, y, ctx: FitContext):
        _check_epochs(X, "Bandpower")
        nyquist = ctx.sfreq / 2.0
        if self.low >= nyquist:
            raise ValueError(
                f"带通功率下界 {self.low:g} Hz 高于 Nyquist ({nyquist:g} Hz)"
            )
        self.sfreq_ = float(ctx.sfreq)
        self.high_ = min(self.high, nyquist)
        n_times = X.shape[-1]
        self.nperseg_ = int(min(n_times, max(16, round(ctx.sfreq))))
        if self.nperseg_ < 8:
            raise ValueError("带通功率至少需要 8 个采样点")
        return self

    def transform(self, X):
        freqs, psd = welch(
            X, fs=self.sfreq_, nperseg=self.nperseg_, axis=-1
        )
        band_mask = (freqs >= self.low) & (freqs <= self.high_)
        if not np.any(band_mask):
            raise ValueError("带通功率频带内没有频率分辨率采样点")
        power = np.trapezoid(psd[..., band_mask], freqs[band_mask], axis=-1)
        if self.relative:
            total_mask = freqs > 0
            total = np.trapezoid(psd[..., total_mask], freqs[total_mask], axis=-1)
            power = power / np.maximum(total, 1e-30)
        if self.log_power:
            power = np.log10(np.maximum(power, 1e-30))
        return power


class Hjorth(Node):
    """Hjorth activity, mobility and complexity for every channel."""

    name = "hjorth"
    kind = "feature"

    def fit(self, X, y, ctx: FitContext):
        _check_epochs(X, "Hjorth")
        if X.shape[-1] < 3:
            raise ValueError("Hjorth 至少需要 3 个采样点")
        return self

    def transform(self, X):
        activity = np.var(X, axis=-1)
        diff = np.diff(X, axis=-1)
        diff_var = np.var(diff, axis=-1)
        mobility = np.sqrt(diff_var / np.maximum(activity, 1e-30))
        diff2_var = np.var(np.diff(diff, axis=-1), axis=-1)
        mobility_diff = np.sqrt(diff2_var / np.maximum(diff_var, 1e-30))
        complexity = mobility_diff / np.maximum(mobility, 1e-30)
        return np.concatenate([activity, mobility, complexity], axis=1)


class ErpWindowMean(Node):
    """Average amplitude in an absolute epoch-time window, per channel."""

    name = "erp_window"
    kind = "feature"

    def __init__(self, start: float = 0.2, end: float = 0.5):
        self.start = float(start)
        self.end = float(end)

    def fit(self, X, y, ctx: FitContext):
        _check_epochs(X, "ErpWindowMean")
        self.slice_ = _window_slice(
            X.shape[-1], ctx.sfreq, ctx.tmin, self.start, self.end
        )
        return self

    def transform(self, X):
        return X[..., self.slice_].mean(axis=2)


def _vectorize_covariances(covs: np.ndarray) -> np.ndarray:
    indices = np.triu_indices(covs.shape[1])
    upper = covs[:, indices[0], indices[1]]
    off_diagonal = indices[0] != indices[1]
    upper[:, off_diagonal] *= np.sqrt(2.0)
    return upper


class CovarianceVector(Node):
    """Vectorized per-trial covariance (upper triangle, sqrt(2) off-diagonal)."""

    name = "covariance"
    kind = "feature"

    def __init__(self, estimator: str = "oas"):
        self.estimator = str(estimator)
        if self.estimator not in {"oas", "lwf", "scm"}:
            raise ValueError("协方差估计必须是 oas、lwf 或 scm")
        self.covariance = self.estimator

    def transform(self, X):
        return _vectorize_covariances(estimate_covariances(X, self.covariance))


class TangentSpaceFeature(Node):
    """Riemannian tangent-space features with the reference fitted on training data."""

    name = "tangent_space"
    kind = "feature"

    def __init__(self, metric: str = "riemann", covariance: str = "oas"):
        self.metric = str(metric)
        if self.metric not in {"riemann", "logeuclid"}:
            raise ValueError("切空间度量必须是 riemann 或 logeuclid")
        self.covariance = str(covariance)
        if self.covariance not in {"oas", "lwf", "scm"}:
            raise ValueError("协方差估计必须是 oas、lwf 或 scm")

    def fit(self, X, y, ctx: FitContext):
        try:
            from pyriemann.tangentspace import TangentSpace
        except ImportError as exc:  # pragma: no cover - dependency contract
            raise RuntimeError("切空间特征需要 pyriemann") from exc
        _check_epochs(X, "TangentSpaceFeature")
        covs = estimate_covariances(X, self.covariance)
        self.tangent_ = TangentSpace(metric=self.metric).fit(covs)
        return self

    def transform(self, X):
        covs = estimate_covariances(X, self.covariance)
        return np.asarray(self.tangent_.transform(covs), dtype=float)


class SelectK(Node):
    """Train-fold-only ANOVA or mutual-information feature selection."""

    name = "select_k"
    kind = "feature"

    def __init__(self, k: int = 10, method: str = "anova"):
        self.k = int(k)
        self.method = str(method)
        if self.k < 1:
            raise ValueError("保留特征数必须为正数")
        if self.method not in {"anova", "mutual_info"}:
            raise ValueError("特征选择方法必须是 anova 或 mutual_info")

    def fit(self, X, y, ctx: FitContext):
        try:
            from sklearn.feature_selection import (
                SelectKBest,
                f_classif,
                mutual_info_classif,
            )
        except ImportError as exc:  # pragma: no cover - dependency contract
            raise RuntimeError("特征选择需要 scikit-learn") from exc
        if y is None:
            raise ValueError("SelectK 需要训练标签")
        if X.ndim != 2:
            raise ValueError("SelectK 期望二维特征 (n_trials, n_features)")
        if self.method == "anova":
            score_func = f_classif
        else:
            score_func = partial(mutual_info_classif, random_state=ctx.seed)
        self.selector_ = SelectKBest(
            score_func=score_func, k=min(self.k, X.shape[1])
        )
        self.selector_.fit(X, y)
        self.indices_ = np.flatnonzero(self.selector_.get_support())
        return self

    def transform(self, X):
        return self.selector_.transform(X)
