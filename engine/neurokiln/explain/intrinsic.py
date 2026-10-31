"""Intrinsic, exact explanations read straight from fitted parameters.

These are not approximations: CSP patterns are the inverse of the fitted
filters, MDM class means are the covariance centroids, and the spatial
convolutions of EEGNet/ShallowConvNet are literal scalp maps. Every value is
therefore a property of the model that was evaluated, with no extra forward
passes and no assumption about the input distribution.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm, inv, sqrtm

from ..pipeline.baselines import NearestCentroid
from ..pipeline.csp import CSP, FilterBankCSP, LDA, Standardize
from ..pipeline.deep import TorchDecoder
from ..pipeline.erp import XdawnLDA
from ..pipeline.nodes import FittedSplit
from ..pipeline.riemann import RiemannMDM, TangentSpaceLR
from ..pipeline.ssvep import SSVEPCCA


def _class_label(fitted: FittedSplit, value: int) -> str:
    names = fitted.test_epochs.label_names
    index = int(value)
    if 0 <= index < len(names):
        return str(names[index])
    return f"类别 {index}"


def _entry(
    identifier: str,
    label: str,
    method: str,
    values,
    *,
    signed: bool,
    note: str = "",
    per_class: list[dict] | None = None,
    spectral: dict | None = None,
    extra: dict | None = None,
) -> dict:
    payload = {
        "id": identifier,
        "label": label,
        "method": method,
        "scope": "channel",
        "signed": bool(signed),
        "values": [round(float(value), 6) for value in np.asarray(values, dtype=float).ravel()],
        "note": note,
    }
    if per_class:
        payload["per_class"] = per_class
    if spectral:
        payload["spectral"] = spectral
    if extra:
        payload.update(extra)
    return payload


def _find(fitted: FittedSplit, cls) -> object | None:
    for trace in reversed(fitted.traces[:-1]):
        if isinstance(trace.node, cls):
            return trace.node
    return None


def _inverse_scale(fitted: FittedSplit) -> np.ndarray | float:
    """Undo ``Standardize`` so coefficients live in the CSP feature space."""
    scale = 1.0
    for trace in reversed(fitted.traces[:-1]):
        if isinstance(trace.node, Standardize):
            scale = scale / np.asarray(trace.node.scale_, dtype=float)
    return scale


def _lda_entries(fitted: FittedSplit, decoder: LDA) -> list[dict]:
    inv_scale = _inverse_scale(fitted)
    coef = np.atleast_2d(np.asarray(decoder.model_.coef_, dtype=float) * inv_scale)
    classes = [int(item) for item in np.asarray(decoder.classes_)]
    csp = _find(fitted, CSP)
    fbcsp = _find(fitted, FilterBankCSP)
    entries: list[dict] = []

    if csp is not None:
        patterns = np.asarray(csp.patterns_, dtype=float)
        if patterns.shape[1] != coef.shape[1]:
            return entries
        maps = patterns @ coef.T  # (n_channels, n_classes)
        binary = maps.shape[1] == 1
        entries.append(
            _entry(
                "csp_patterns",
                "CSP 空间模式 × LDA 权重",
                "patterns = pinv(filters); contribution = patterns @ coef",
                maps[:, 0] if binary else np.abs(maps).mean(axis=1),
                signed=binary,
                note=(
                    "每个通道对判别方向的投影强度；正值推向类别 "
                    f"{_class_label(fitted, classes[-1])}，负值推向 "
                    f"{_class_label(fitted, classes[0])}。"
                    if binary else
                    "多分类时取各类判别方向的绝对值平均；每类方向见 per_class。"
                ),
                per_class=None if binary else [
                    {
                        "class": classes[index],
                        "label": _class_label(fitted, classes[index]),
                        "values": [round(float(value), 6) for value in maps[:, index]],
                    }
                    for index in range(maps.shape[1])
                ],
            )
        )
    if fbcsp is not None:
        widths = [np.asarray(patterns).shape[1] for patterns in fbcsp.patterns_]
        if sum(widths) == coef.shape[1]:
            cursor = 0
            band_maps = []
            for index, (patterns, width) in enumerate(zip(fbcsp.patterns_, widths)):
                block = coef[:, cursor:cursor + width]
                cursor += width
                band_maps.append(np.asarray(patterns, dtype=float) @ block.T)
            total = np.stack(band_maps, axis=0).sum(axis=0)
            bands = fbcsp.bands[: len(band_maps)]
            entries.append(
                _entry(
                    "fbcsp_total",
                    "FBCSP 各频带贡献之和",
                    "sum_b patterns_b @ coef_b",
                    total[:, 0] if total.shape[1] == 1 else np.abs(total).mean(axis=1),
                    signed=total.shape[1] == 1,
                    note="每个子带的 CSP 贡献相加；用于观察整体空间来源。",
                )
            )
            for index, (map_values, band) in enumerate(zip(band_maps, bands)):
                if index >= 4:
                    break
                entries.append(
                    _entry(
                        f"fbcsp_band_{index}",
                        f"FBCSP 频带 {band[0]:g}–{band[1]:g} Hz",
                        "patterns_b @ coef_b",
                        map_values[:, 0] if map_values.shape[1] == 1
                        else np.abs(map_values).mean(axis=1),
                        signed=map_values.shape[1] == 1,
                        note="该子带单独的空间贡献。",
                    )
                )
    return entries


def _nearest_centroid_entries(fitted: FittedSplit, decoder: NearestCentroid) -> list[dict]:
    centroids = np.asarray(decoder.centroids_, dtype=float)
    classes = [int(item) for item in np.asarray(decoder.classes_)]
    if centroids.shape[0] == 2:
        values = centroids[1] - centroids[0]
        return [
            _entry(
                "centroid_direction",
                f"最近质心方向（推向{_class_label(fitted, classes[1])}）",
                "c1 - c0",
                values,
                signed=True,
                note="特征空间中的质心差；正值把试次推向第二个类别。",
            )
        ]
    maps = centroids - centroids.mean(axis=0, keepdims=True)
    return [
        _entry(
            "centroid_direction",
            "最近质心类别方向（绝对值平均）",
            "mean_c |centroid_c - mean(centroids)|",
            np.abs(maps).mean(axis=0),
            signed=False,
            note="多分类时取各类质心偏移的绝对值平均。",
            per_class=[
                {
                    "class": classes[index],
                    "label": _class_label(fitted, classes[index]),
                    "values": [round(float(value), 6) for value in maps[index]],
                }
                for index in range(maps.shape[0])
            ],
        )
    ]


def _mdm_entries(fitted: FittedSplit, decoder: RiemannMDM) -> list[dict]:
    means = [np.diag(np.asarray(cov, dtype=float)) for cov in decoder.model_.covmeans_]
    classes = [int(item) for item in np.asarray(decoder.classes_)]
    logs = np.log(np.stack([np.maximum(item, 1e-18) for item in means]))
    if logs.shape[0] == 2:
        values = logs[1] - logs[0]
        return [
            _entry(
                "mdm_logvar_ratio",
                f"类别 log 方差比（{_class_label(fitted, classes[1])} vs {_class_label(fitted, classes[0])}）",
                "log diag(cov_1) - log diag(cov_0)",
                values,
                signed=True,
                note="黎曼类均值协方差的对角线比；正值表示该通道在第二类中方差更大。",
            )
        ]
    deviation = logs - logs.mean(axis=0, keepdims=True)
    return [
        _entry(
            "mdm_logvar_ratio",
            "类别 log 方差偏移",
            "mean_c |log diag(cov_c) - mean_c|",
            np.abs(deviation).mean(axis=0),
            signed=False,
            per_class=[
                {
                    "class": classes[index],
                    "label": _class_label(fitted, classes[index]),
                    "values": [round(float(value), 6) for value in deviation[index]],
                }
                for index in range(deviation.shape[0])
            ],
        )
    ]


def _tangent_entries(fitted: FittedSplit, decoder: TangentSpaceLR) -> list[dict]:
    reference = getattr(decoder.tangent_, "reference_", None)
    if reference is None:
        return []
    reference = np.asarray(reference, dtype=float)
    n_channels = reference.shape[0]
    upper = np.triu_indices(n_channels)
    coef = np.atleast_2d(np.asarray(decoder.model_.coef_, dtype=float))
    if coef.shape[1] != upper[0].size:
        return []
    # ``coef`` lives in standardized tangent space; undo the decoder's own
    # scaler so the direction is expressed in the tangent basis itself.
    scaler = getattr(decoder, "scaler_", None)
    if scaler is not None and np.asarray(scaler.scale_).shape == (coef.shape[1],):
        coef = coef / np.maximum(np.asarray(scaler.scale_, dtype=float), 1e-12)
    root = sqrtm(reference)
    inv_root = inv(root)

    def to_covariance(row):
        tangent = np.zeros((n_channels, n_channels))
        values = row.copy()
        off = upper[0] != upper[1]
        values[off] = values[off] / np.sqrt(2.0)
        tangent[upper] = values
        tangent = tangent + tangent.T - np.diag(np.diag(tangent))
        return root @ expm(inv_root @ tangent @ inv_root) @ root

    classes = [int(item) for item in np.asarray(decoder.classes_)]
    covariances = [to_covariance(row) for row in coef]
    baselines = np.log(np.maximum(np.diag(reference), 1e-18))
    log_variances = [np.log(np.maximum(np.diag(cov), 1e-18)) - baselines for cov in covariances]
    if coef.shape[0] == 1 and len(classes) == 2:
        values = log_variances[0]
        signed = True
        label = f"切空间 LR 判别方向（推向{_class_label(fitted, classes[-1])}）"
        per_class = None
    elif coef.shape[0] == len(classes):
        values = np.abs(np.stack(log_variances)).mean(axis=0)
        signed = False
        label = "切空间 LR 类别方差偏移（绝对值平均）"
        per_class = [
            {
                "class": classes[index],
                "label": _class_label(fitted, classes[index]),
                "values": [round(float(value), 6) for value in log_variances[index]],
            }
            for index in range(len(classes))
        ]
    else:
        return []
    return [
        _entry(
            "tangent_logvar",
            label,
            "coef -> symmetric tangent -> expm -> log diag",
            values,
            signed=signed,
            note="把 LR 系数映回协方差空间，取对角线的对数（通道方差）。",
            per_class=per_class,
        )
    ]


def _xdawn_entries(fitted: FittedSplit, decoder: XdawnLDA) -> list[dict]:
    model = getattr(decoder, "xdawn_", None)
    if model is None:
        return []
    filters = np.asarray(model.filters_, dtype=float)   # (n_components, n_channels)
    patterns = np.asarray(model.patterns_, dtype=float)  # (n_components, n_channels)
    coef = np.atleast_2d(np.asarray(decoder.model_.coef_, dtype=float))
    entries = []
    if coef.shape[1] == patterns.shape[0]:
        values = patterns.T @ coef[0]
        entries.append(
            _entry(
                "xdawn_patterns",
                "xDAWN 空间模式 × LDA 权重",
                "patterns.T @ coef",
                values,
                signed=True,
                note="ERP 成分在头皮上的投影，按 LDA 权重加权。",
            )
        )
    if filters.shape[0] >= 2:
        for index in range(min(4, filters.shape[0])):
            entries.append(
                _entry(
                    f"xdawn_filter_{index}",
                    f"xDAWN 空间滤波器 {index + 1}",
                    "filters_[k]",
                    filters[index],
                    signed=True,
                    note="训练折 ERP 的广义特征向量；符号只有相对意义。",
                )
            )
    return entries


def _ssvep_entries(fitted: FittedSplit, decoder: SSVEPCCA) -> list[dict]:
    X = np.asarray(fitted.test_epochs.data, dtype=float)
    labels = fitted.test_epochs.labels
    classes = np.asarray(decoder.classes_)
    true_index = np.clip(np.searchsorted(classes, labels), 0, classes.size - 1)
    scores = decoder._scores(X)
    base = scores[np.arange(len(labels)), true_index]
    drops = np.zeros(X.shape[1])
    for channel in range(X.shape[1]):
        probe = X.copy()
        probe[:, channel, :] = 0.0
        probe_scores = decoder._scores(probe)
        drops[channel] = float(np.mean(base - probe_scores[np.arange(len(labels)), true_index]))
    return [
        _entry(
            "cca_channel_ablation",
            "通道遮盖后的 CCA 得分下降",
            "mean_trials(score_true - score_true_without_channel)",
            drops,
            signed=True,
            note="无监督 CCA 没有权重，用留一通道遮盖衡量每个电极对频率识别的贡献。",
        )
    ]


def _torch_entries(fitted: FittedSplit, decoder: TorchDecoder) -> list[dict]:
    torch = __import__("torch")
    n_channels = fitted.test_epochs.data.shape[1]
    spatial = temporal = None
    for module in decoder.model_.modules():
        if not isinstance(module, torch.nn.Conv2d):
            continue
        height, width = module.kernel_size
        if height == n_channels and width == 1 and spatial is None:
            spatial = module
        elif height == 1 and width > 1 and temporal is None:
            temporal = module
    entries: list[dict] = []
    if spatial is not None:
        weights = spatial.weight.detach().cpu().numpy()[:, 0, :, 0]  # (out, C)
        mean_weights = weights.mean(axis=0)
        entries.append(
            _entry(
                "conv_spatial",
                "空间卷积核（平均权重）",
                "mean_out conv_spatial_weight[out, :]",
                mean_weights,
                signed=True,
                note="网络第一层空间卷积的可视化；每个输出滤波器的头皮权重取平均。",
                extra={"n_filters": int(weights.shape[0]),
                       "magnitude": [round(float(value), 6)
                                     for value in np.abs(weights).mean(axis=0)]},
            )
        )
        entries.append(
            _entry(
                "conv_spatial_magnitude",
                "空间卷积核（绝对强度）",
                "mean_out |conv_spatial_weight[out, :]|",
                np.abs(weights).mean(axis=0),
                signed=False,
                note="不依赖符号的通道依赖强度，适合跨被试比较。",
            )
        )
    if temporal is not None:
        kernel = temporal.weight.detach().cpu().numpy()
        sfreq = float(decoder.target_sfreq)
        spectrum = np.abs(np.fft.rfft(kernel, n=512, axis=-1)) ** 2
        power = spectrum.reshape(-1, spectrum.shape[-1]).mean(axis=0)
        frequencies = np.fft.rfftfreq(512, d=1.0 / sfreq)
        keep = frequencies <= min(45.0, sfreq / 2.0)
        frequencies = frequencies[keep]
        power = power[keep]
        if power.max() > 0:
            power = power / power.max()
        stride = max(1, int(np.ceil(len(frequencies) / 96)))
        entries.append(
            {
                "id": "spectral",
                "label": "时域卷积的频率响应",
                "method": "|rfft(temporal_kernel)|^2",
                "scope": "frequency",
                "signed": False,
                "values": [],
                "note": "网络在输入层关注的频段；峰值频率是滤波器组的中心。",
                "spectral": {
                    "frequency": [round(float(value), 3) for value in frequencies[::stride]],
                    "power": [round(float(value), 5) for value in power[::stride]],
                    "peak_hz": round(float(frequencies[int(np.argmax(power))]), 3),
                    "sfreq": sfreq,
                },
            }
        )
    return entries


def extract_intrinsic(fitted: FittedSplit) -> list[dict]:
    """Return every exact explanation available for this fitted pipeline."""
    decoder = fitted.traces[-1].node
    if isinstance(decoder, TorchDecoder):
        return _torch_entries(fitted, decoder)
    if isinstance(decoder, TangentSpaceLR):
        return _tangent_entries(fitted, decoder)
    if isinstance(decoder, RiemannMDM):
        return _mdm_entries(fitted, decoder)
    if isinstance(decoder, XdawnLDA):
        return _xdawn_entries(fitted, decoder)
    if isinstance(decoder, SSVEPCCA):
        return _ssvep_entries(fitted, decoder)
    if isinstance(decoder, NearestCentroid):
        return _nearest_centroid_entries(fitted, decoder)
    if isinstance(decoder, LDA):
        return _lda_entries(fitted, decoder)
    return []


def channel_reference(fitted: FittedSplit) -> np.ndarray | None:
    """A signed per-channel importance to compare other methods against."""
    entries = extract_intrinsic(fitted)
    for identifier in ("csp_patterns", "fbcsp_total", "centroid_direction",
                       "mdm_logvar_ratio", "tangent_logvar", "xdawn_patterns",
                       "conv_spatial", "cca_channel_ablation"):
        for entry in entries:
            if entry["id"] == identifier and len(entry["values"]) == len(
                fitted.test_epochs.channel_names
            ):
                return np.asarray(entry["values"], dtype=float)
    return None


__all__ = ["channel_reference", "extract_intrinsic"]
