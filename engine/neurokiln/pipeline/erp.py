from __future__ import annotations

import numpy as np

from .nodes import FitContext, Node


def _log_variance(X: np.ndarray) -> np.ndarray:
    variance = np.var(X, axis=2)
    return np.log(np.maximum(variance, 1e-12))


class Xdawn(Node):
    """xDAWN spatial filtering fitted on training-fold ERP labels."""

    name = "xdawn"
    kind = "preprocess"

    def __init__(self, n_filters: int = 4):
        self.n_filters = int(n_filters)
        if self.n_filters < 1:
            raise ValueError("xDAWN 空间滤波器数必须为正数")

    def fit(self, X: np.ndarray, y: np.ndarray | None, ctx: FitContext):
        try:
            from pyriemann.spatialfilters import Xdawn as RiemannXdawn
        except ImportError as exc:  # pragma: no cover - dependency contract
            raise RuntimeError("xDAWN requires pyriemann") from exc
        if y is None:
            raise ValueError("xDAWN requires training labels")
        if X.ndim != 3:
            raise ValueError("xDAWN expects (n_trials, n_channels, n_times)")
        classes = np.unique(y)
        if classes.size < 2:
            raise ValueError("xDAWN requires at least two ERP classes")
        n_filters = max(1, min(self.n_filters, X.shape[1]))
        self.xdawn_ = RiemannXdawn(nfilter=n_filters, estimator="scm")
        self.xdawn_.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.xdawn_.transform(X), dtype=float)


class XdawnLDA(Node):
    """xDAWN spatial filtering + log-variance + shrinkage LDA for ERP data."""

    name = "xdawn_lda"
    kind = "decoder"

    def __init__(self, n_filters: int = 4, shrinkage: str | float = "auto"):
        self.n_filters = int(n_filters)
        self.shrinkage = shrinkage
        self._xdawn = Xdawn(n_filters)

    def fit(self, X: np.ndarray, y: np.ndarray | None, ctx: FitContext):
        try:
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        except ImportError as exc:  # pragma: no cover - dependency contract
            raise RuntimeError("xDAWN requires scikit-learn") from exc
        if y is None:
            raise ValueError("XdawnLDA requires training labels")
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError("XdawnLDA currently supports exactly two ERP classes")
        self._xdawn.fit(X, y, ctx)
        features = self.transform(X)
        shrinkage = self.shrinkage
        if isinstance(shrinkage, str) and shrinkage.strip().lower() != "auto":
            shrinkage = float(shrinkage)
        self.model_ = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage)
        self.model_.fit(features, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return _log_variance(self._xdawn.transform(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model_.predict(X), dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(X), dtype=float)
