from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

from .filters import design_bandpass, filter_epochs
from .nodes import FitContext, Node


class Standardize(Node):
    name = "standardize"
    kind = "feature"

    def fit(self, X, y, ctx: FitContext):
        if X.ndim != 2:
            raise ValueError("Standardize expects 2D features")
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale < 1e-12] = 1.0
        self.scale_ = scale
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_


def _fit_csp_filters(
    X: np.ndarray, y: np.ndarray, n_components: int, reg: float
) -> np.ndarray:
    if X.ndim != 3:
        raise ValueError("CSP expects (n_trials, n_channels, n_times)")
    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError("CSP pair fit requires exactly two classes")
    n_trials, n_channels, n_times = X.shape
    if n_times < 2:
        raise ValueError("CSP requires at least two time samples")
    n_components = int(n_components)
    if n_components < 2 or n_components > n_channels:
        raise ValueError("n_components must be between 2 and n_channels")

    centered = X - X.mean(axis=2, keepdims=True)
    covs = np.einsum("nct,ndt->ncd", centered, centered) / float(n_times - 1)
    traces = np.trace(covs, axis1=1, axis2=2)
    covs = covs / np.maximum(traces[:, None, None], 1e-12)
    covs += reg * np.eye(n_channels)[None, :, :]

    class_covs = [covs[y == cls].mean(axis=0) for cls in classes]
    composite = np.sum(class_covs, axis=0)
    eigenvalues, eigenvectors = eigh(class_covs[0], composite)
    order = np.argsort(eigenvalues)[::-1]
    n_first = (n_components + 1) // 2
    n_last = n_components - n_first
    selected = np.concatenate([order[:n_first], order[len(order) - n_last:]])
    return eigenvectors[:, selected].T


def _fit_csp_multiclass(
    X: np.ndarray, y: np.ndarray, n_components: int, reg: float
) -> np.ndarray:
    """One-vs-rest CSP for more than two classes."""
    classes = np.unique(y)
    blocks = [
        _fit_csp_filters(X, (y == cls).astype(int), n_components, reg)
        for cls in classes
    ]
    return np.concatenate(blocks, axis=0)


def _csp_features(X: np.ndarray, filters: np.ndarray) -> np.ndarray:
    projected = np.einsum("nct,kc->nkt", X, filters, optimize=True)
    var = np.var(projected, axis=2)
    return np.log(np.maximum(var, 1e-12))


class CSP(Node):
    """CSP log-variance features fitted on training trials.

    Binary problems use the classic class-1 vs class-2 generalized
    eigenproblem. Multiclass problems use one-vs-rest filters so every class
    contributes a discriminative subspace; ``n_components`` is per class in
    that case.
    """

    name = "csp"
    kind = "feature"

    def __init__(self, n_components: int = 6, reg: float = 1e-6):
        self.n_components = int(n_components)
        self.reg = float(reg)

    def fit(self, X, y, ctx: FitContext):
        if y is None:
            raise ValueError("CSP requires training labels")
        if X.ndim != 3:
            raise ValueError("CSP expects (n_trials, n_channels, n_times)")
        classes = np.unique(y)
        if classes.size < 2:
            raise ValueError("CSP requires at least two classes")
        n_components = min(self.n_components, X.shape[1])
        if classes.size == 2:
            self.filters_ = _fit_csp_filters(X, y, n_components, self.reg)
        else:
            self.filters_ = _fit_csp_multiclass(X, y, n_components, self.reg)
        # Spatial patterns are the columns of the pseudo-inverse: filter k's
        # pattern tells which scalp locations project onto component k.
        self.patterns_ = np.linalg.pinv(self.filters_)
        return self

    def transform(self, X):
        return _csp_features(X, self.filters_)


class FilterBankCSP(Node):
    """Filter-bank CSP with a fixed number of components per band.

    Feature selection is intentionally fixed here: selecting features using
    held-out labels would leak information. A nested mutual-information
    selector can be added later, as long as it is fitted on training folds only.
    """

    name = "fbcsp"
    kind = "feature"

    def __init__(
        self,
        n_components: int = 2,
        bands: tuple[tuple[float, float], ...] = (
            (4.0, 8.0),
            (8.0, 12.0),
            (12.0, 16.0),
            (16.0, 20.0),
            (20.0, 24.0),
            (24.0, 28.0),
            (28.0, 32.0),
        ),
        reg: float = 1e-6,
    ):
        self.n_components = int(n_components)
        self.bands = tuple((float(low), float(high)) for low, high in bands)
        self.reg = float(reg)

    def fit(self, X, y, ctx: FitContext):
        if y is None:
            raise ValueError("FilterBankCSP requires training labels")
        if X.ndim != 3:
            raise ValueError("FilterBankCSP expects (n_trials, n_channels, n_times)")
        classes = np.unique(y)
        if classes.size < 2:
            raise ValueError("FilterBankCSP requires at least two classes")
        fit_filters = _fit_csp_filters if classes.size == 2 else _fit_csp_multiclass
        n_components = min(self.n_components, X.shape[1])
        self.filters_: list[np.ndarray] = []
        self.patterns_: list[np.ndarray] = []
        self.sos_: list[np.ndarray] = []
        for low, high in self.bands:
            if low >= ctx.sfreq / 2.0:
                continue
            sos = design_bandpass(low, high, ctx.sfreq)
            self.sos_.append(sos)
            filters = fit_filters(filter_epochs(X, sos), y, n_components, self.reg)
            self.filters_.append(filters)
            self.patterns_.append(np.linalg.pinv(filters))
        if not self.filters_:
            raise ValueError("no filter-bank band is below the Nyquist frequency")
        return self

    def transform(self, X):
        features = [
            _csp_features(filter_epochs(X, sos), filters)
            for sos, filters in zip(self.sos_, self.filters_)
        ]
        return np.concatenate(features, axis=1)


class LDA(Node):
    name = "lda"
    kind = "decoder"

    def __init__(self, shrinkage: str | float = "auto"):
        self.shrinkage = shrinkage

    def fit(self, X, y, ctx: FitContext):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        if y is None:
            raise ValueError("LDA requires training labels")
        if X.ndim != 2:
            raise ValueError("LDA expects 2D features")
        shrinkage = self.shrinkage
        if isinstance(shrinkage, str) and shrinkage.strip().lower() != "auto":
            shrinkage = float(shrinkage)
        self.classes_ = np.unique(y)
        self.model_ = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(X), dtype=int)

    def predict_proba(self, X):
        return np.asarray(self.model_.predict_proba(X), dtype=float)
