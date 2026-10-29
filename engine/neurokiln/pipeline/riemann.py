from __future__ import annotations

import numpy as np

from .nodes import FitContext, Node


def estimate_covariances(X: np.ndarray, estimator: str = "oas"):
    try:
        from pyriemann.estimation import Covariances
    except ImportError as exc:  # pragma: no cover - dependency contract
        raise RuntimeError(
            "Riemannian baselines require pyriemann: uv pip install 'neurokiln-engine'"
        ) from exc
    if X.ndim != 3:
        raise ValueError("covariance features expect (n_trials, n_channels, n_times)")
    return Covariances(estimator=estimator).transform(X)


class RiemannMDM(Node):
    """Riemannian minimum-distance-to-mean with OAS covariance estimates."""

    name = "riemann_mdm"
    kind = "decoder"

    def __init__(self, metric: str = "riemann", covariance: str = "oas"):
        self.metric = metric
        self.covariance = covariance

    def fit(self, X, y, ctx: FitContext):
        try:
            from pyriemann.classification import MDM
        except ImportError as exc:  # pragma: no cover - dependency contract
            raise RuntimeError("Riemannian baselines require pyriemann") from exc
        if y is None:
            raise ValueError("RiemannMDM requires training labels")
        covs = estimate_covariances(X, self.covariance)
        self.classes_ = np.unique(y)
        self.model_ = MDM(
            metric=dict(mean=self.metric, distance=self.metric),
            n_jobs=1,
        )
        self.model_.fit(covs, y)
        return self

    def transform(self, X):
        return estimate_covariances(X, self.covariance)

    def predict(self, covs):
        return np.asarray(self.model_.predict(covs), dtype=int)

    def predict_proba(self, covs):
        return np.asarray(self.model_.predict_proba(covs), dtype=float)


class TangentSpaceLR(Node):
    """Riemannian tangent-space projection followed by standardized LR."""

    name = "tangent_space_lr"
    kind = "decoder"

    def __init__(
        self,
        metric: str = "riemann",
        covariance: str = "oas",
        C: float = 1.0,
        max_iter: int = 1000,
    ):
        self.metric = metric
        self.covariance = covariance
        self.C = float(C)
        self.max_iter = int(max_iter)

    def fit(self, X, y, ctx: FitContext):
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from pyriemann.tangentspace import TangentSpace
        except ImportError as exc:  # pragma: no cover - dependency contract
            raise RuntimeError("TangentSpaceLR requires pyriemann and scikit-learn") from exc
        if y is None:
            raise ValueError("TangentSpaceLR requires training labels")
        covs = estimate_covariances(X, self.covariance)
        self.tangent_ = TangentSpace(metric=self.metric)
        tangent = self.tangent_.fit_transform(covs)
        self.scaler_ = StandardScaler().fit(tangent)
        self.classes_ = np.unique(y)
        self.model_ = LogisticRegression(C=self.C, max_iter=self.max_iter)
        self.model_.fit(self.scaler_.transform(tangent), y)
        return self

    def transform(self, X):
        covs = estimate_covariances(X, self.covariance)
        return self.scaler_.transform(self.tangent_.transform(covs))

    def predict(self, X):
        return np.asarray(self.model_.predict(X), dtype=int)

    def predict_proba(self, X):
        return np.asarray(self.model_.predict_proba(X), dtype=float)
