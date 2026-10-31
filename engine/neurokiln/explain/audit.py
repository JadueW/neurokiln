"""Knowledge audit: how much of the model is explained by known EEG features.

A transparent logistic regression is trained on handcrafted bandpower/Hjorth
features from the training fold only. Comparing it to the model gives a
"closure" number: the share of the model's performance that established,
publishable features can already account for. A ridge surrogate additionally
measures how linearly predictable the model's own confidence is from those
features.
"""

from __future__ import annotations

import numpy as np

from ..evaluate import balanced_accuracy
from ..pipeline.nodes import FittedSplit
from .trace import decision_probs, spearman

AUDIT_BANDS = ((4.0, 8.0), (8.0, 13.0), (13.0, 20.0), (20.0, 30.0), (30.0, 45.0))


def _bandpower(X: np.ndarray, sfreq: float, bands) -> tuple[list[np.ndarray], list[str]]:
    window = np.hanning(X.shape[-1])
    spectrum = np.abs(np.fft.rfft(X * window, axis=-1)) ** 2
    frequencies = np.fft.rfftfreq(X.shape[-1], d=1.0 / sfreq)
    blocks, names = [], []
    for low, high in bands:
        mask = (frequencies >= low) & (frequencies < high)
        if not mask.any():
            continue
        power = spectrum[..., mask].mean(axis=-1)
        blocks.append(np.log(np.maximum(power, 1e-20)))
        names.append(f"{low:g}-{high:g}Hz")
    return blocks, names


def _hjorth(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    activity = np.var(X, axis=-1)
    first = np.diff(X, axis=-1)
    var_d1 = np.var(first, axis=-1)
    mobility = np.sqrt(var_d1 / np.maximum(activity, 1e-18))
    second = np.diff(first, axis=-1)
    var_d2 = np.var(second, axis=-1)
    mobility_d1 = np.sqrt(var_d2 / np.maximum(var_d1, 1e-18))
    complexity = mobility_d1 / np.maximum(mobility, 1e-18)
    return activity, mobility, complexity


def handcrafted_features(
    X: np.ndarray, sfreq: float, channel_names, bands=AUDIT_BANDS
) -> tuple[np.ndarray, list[str], list[str]]:
    """Per-trial bandpower + Hjorth + mean/std, with feature provenance."""
    X = np.asarray(X, dtype=float)
    block_values, band_labels = _bandpower(X, sfreq, bands)
    names: list[str] = []
    groups: list[str] = []
    columns: list[np.ndarray] = []
    for values, label in zip(block_values, band_labels):
        columns.append(values)
        names.extend([f"bp:{label}:{name}" for name in channel_names])
        groups.append(f"bandpower {label}")
    activity, mobility, complexity = _hjorth(X)
    for values, prefix in (
        (activity, "hjorth_activity"), (mobility, "hjorth_mobility"),
        (complexity, "hjorth_complexity"), (X.mean(axis=-1), "mean"),
        (X.std(axis=-1), "std"),
    ):
        columns.append(values)
        names.extend([f"{prefix}:{name}" for name in channel_names])
        groups.append(prefix)
    features = np.concatenate(columns, axis=1)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0), names, groups


def knowledge_audit(
    fitted: FittedSplit,
    *,
    occlusion_channel_importance=None,
    bands=AUDIT_BANDS,
) -> dict:
    try:
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.metrics import r2_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:  # pragma: no cover - scikit-learn is a hard dependency
        return {"available": False, "reason": "需要 scikit-learn"}

    X_train = np.asarray(fitted.train_epochs.data, dtype=float)
    X_test = np.asarray(fitted.test_epochs.data, dtype=float)
    y_train = np.asarray(fitted.train_epochs.labels)
    y_test = np.asarray(fitted.test_epochs.labels)
    sfreq = float(fitted.test_epochs.sfreq)
    channels = [str(name) for name in fitted.test_epochs.channel_names]

    train_features, names, groups = handcrafted_features(X_train, sfreq, channels, bands)
    test_features, _, _ = handcrafted_features(X_test, sfreq, channels, bands)

    scaler = StandardScaler().fit(train_features)
    train_scaled = scaler.transform(train_features)
    test_scaled = scaler.transform(test_features)

    student = LogisticRegression(
        max_iter=3000, class_weight="balanced", C=1.0
    )
    student.fit(train_scaled, y_train)
    student_pred = student.predict(test_scaled)
    student_ba = balanced_accuracy(y_test, student_pred)
    model_ba = balanced_accuracy(y_test, fitted.result["y_pred"])

    # Surrogate: can the known features linearly predict the model's own
    # confidence? Fit on training-fold predictions, evaluate on held-out ones.
    train_confidence = decision_probs(fitted, X_train).confidence
    test_confidence = decision_probs(fitted, X_test).confidence
    ridge = Ridge(alpha=1.0).fit(train_scaled, train_confidence)
    surrogate_pred = ridge.predict(test_scaled)
    r2 = float(r2_score(test_confidence, surrogate_pred)) if test_confidence.size > 2 else None

    coef = np.abs(np.asarray(student.coef_, dtype=float)).mean(axis=0)
    order = np.argsort(-coef)
    top_features = [
        {"name": names[index], "weight": round(float(coef[index]), 5)}
        for index in order[:10]
    ]

    bandpower_channels = np.zeros(len(channels))
    for index, name in enumerate(names):
        if name.startswith("bp:"):
            channel = name.rsplit(":", 1)[-1]
            if channel in channels:
                bandpower_channels[channels.index(channel)] += coef[index]

    agreement = None
    if occlusion_channel_importance is not None:
        rho = spearman(occlusion_channel_importance, bandpower_channels)
        agreement = None if rho is None else round(rho, 4)

    closure = None
    if model_ba > 1e-9:
        closure = float(student_ba / model_ba)

    return {
        "available": True,
        "n_features": len(names),
        "feature_groups": groups,
        "student": {
            "model": "LogisticRegression(class_weight=balanced)",
            "balanced_accuracy": round(float(student_ba), 5),
            "fit": "train_fold_only",
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
        },
        "model_balanced_accuracy": round(float(model_ba), 5),
        "closure_ratio": None if closure is None else round(closure, 4),
        "surrogate": {
            "model": "Ridge(alpha=1.0)",
            "r2": None if r2 is None else round(r2, 4),
            "target": "predicted_class_confidence",
            "fit": "train_fold_predictions",
        },
        "channel_importance": [round(float(value), 6) for value in bandpower_channels],
        "channel_agreement": agreement,
        "top_features": top_features,
        "interpretation": (
            "closure = 透明特征学生模型 BA / 被测模型 BA。接近 1 说明性能可由"
            "已知频带能量与 Hjorth 参数解释；明显小于 1 说明模型利用了这些特征"
            "之外的信息（也可能只是容量更大）。surrogate R² 衡量模型置信度能否"
            "由这些特征线性重建。"
        ),
    }


__all__ = ["AUDIT_BANDS", "handcrafted_features", "knowledge_audit"]
