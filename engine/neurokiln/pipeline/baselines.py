from __future__ import annotations

import numpy as np

from .csp import CSP, FilterBankCSP, LDA, Standardize
from .deep import EEGNet, ShallowConvNet
from .erp import XdawnLDA
from .filters import Bandpass
from .nodes import FitContext, Node, Pipeline
from .riemann import RiemannMDM, TangentSpaceLR
from .ssvep import SSVEPCCA


class MeanCenter(Node):
    """Per-channel centering estimated on training trials only."""

    name = "mean_center"
    kind = "preprocess"
    cacheable = True

    def fit(self, X, y, ctx: FitContext):
        self.mean_ = X.mean(axis=(0, 2), keepdims=True)
        return self

    def transform(self, X):
        return X - self.mean_


class ChannelMean(Node):
    """Flatten trials into per-channel temporal means."""

    name = "channel_mean"
    kind = "feature"

    def transform(self, X):
        return X.mean(axis=2)


class ChannelStd(Node):
    """Flatten trials into per-channel temporal standard deviations."""

    name = "channel_std"
    kind = "feature"

    def transform(self, X):
        return X.std(axis=2)


class NearestCentroid(Node):
    name = "nearest_centroid"
    kind = "decoder"

    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    def fit(self, X, y, ctx: FitContext):
        if X.ndim != 2:
            raise ValueError("NearestCentroid expects 2D features (n_trials, n_features)")
        self.classes_ = np.unique(y)
        self.centroids_ = np.stack([X[y == cls].mean(axis=0) for cls in self.classes_])
        return self

    def _distances(self, X):
        return ((X[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)

    def predict(self, X):
        return self.classes_[self._distances(X).argmin(axis=1)]

    def predict_proba(self, X):
        logits = -self._distances(X) / max(self.temperature, 1e-6)
        logits -= logits.max(axis=1, keepdims=True)
        exponent = np.exp(logits)
        return exponent / exponent.sum(axis=1, keepdims=True)


PIPELINE_DEFINITIONS = (
    {
        "id": "raw_centroid",
        "name": "原始均值特征 + 最近质心",
        "family": "baseline",
        "paradigm": "MI/P300/SSVEP",
        "note": "最小可运行基线，用于验证协议与报告链路；跨范式通用",
        "nodes": [
            {"type": "feature.channel_mean", "label": "通道时域均值"},
            {"type": "decoder.nearest_centroid", "label": "最近质心分类器"},
        ],
    },
    {
        "id": "centered_centroid",
        "name": "训练集中心化 + 最近质心",
        "family": "baseline",
        "paradigm": "MI/P300/SSVEP",
        "note": "预处理统计量仅来自训练折的防泄漏基线",
        "nodes": [
            {"type": "preprocess.mean_center", "label": "逐通道中心化（仅训练折拟合）"},
            {"type": "feature.channel_mean", "label": "通道时域均值"},
            {"type": "decoder.nearest_centroid", "label": "最近质心分类器"},
        ],
    },
    {
        "id": "var_centroid",
        "name": "通道方差特征 + 最近质心",
        "family": "baseline",
        "paradigm": "MI/P300/SSVEP",
        "note": "类别间噪声强度不同时与均值特征形成对照",
        "nodes": [
            {"type": "feature.channel_std", "label": "通道时域标准差"},
            {"type": "decoder.nearest_centroid", "label": "最近质心分类器"},
        ],
    },
    {
        "id": "csp_lda",
        "name": "CSP + LDA",
        "family": "spatial",
        "paradigm": "MI",
        "note": "8–30 Hz 带通；CSP 空域滤波在训练折内估计；收缩 LDA 分类",
        "nodes": [
            {"type": "preprocess.bandpass", "label": "8–30 Hz 带通"},
            {"type": "feature.csp", "label": "CSP 对数方差（训练折拟合）"},
            {"type": "feature.standardize", "label": "特征标准化（训练折拟合）"},
            {"type": "decoder.lda", "label": "收缩 LDA"},
        ],
    },
    {
        "id": "fbcsp_lda",
        "name": "FBCSP + LDA",
        "family": "spatial",
        "paradigm": "MI",
        "note": "4–32 Hz 滤波器组，每带 CSP 2 分量；不基于测试折做特征选择",
        "nodes": [
            {"type": "feature.fbcsp", "label": "滤波器组 CSP（训练折拟合）"},
            {"type": "feature.standardize", "label": "特征标准化（训练折拟合）"},
            {"type": "decoder.lda", "label": "收缩 LDA"},
        ],
    },
    {
        "id": "riemann_mdm",
        "name": "Riemann MDM",
        "family": "riemann",
        "paradigm": "MI",
        "note": "OAS 协方差 + 黎曼均值最近距离分类；协方差中心仅来自训练折",
        "nodes": [
            {"type": "preprocess.bandpass", "label": "8–30 Hz 带通"},
            {"type": "decoder.riemann_mdm", "label": "Riemann MDM（训练折拟合）"},
        ],
    },
    {
        "id": "ts_lr",
        "name": "黎曼切空间 + LR",
        "family": "riemann",
        "paradigm": "MI",
        "note": "切空间参考矩阵仅由训练折估计；标准化后逻辑回归",
        "nodes": [
            {"type": "preprocess.bandpass", "label": "8–30 Hz 带通"},
            {"type": "decoder.tangent_space_lr", "label": "切空间 + LR（训练折拟合）"},
        ],
    },
    {
        "id": "eegnet",
        "name": "EEGNet",
        "family": "deep",
        "paradigm": "MI",
        "note": "128 Hz 重采样；训练折内部早停；权重与标准化均不接触测试折",
        "nodes": [
            {"type": "preprocess.bandpass", "label": "8–30 Hz 带通"},
            {"type": "decoder.eegnet", "label": "EEGNet（训练折早停）"},
        ],
    },
    {
        "id": "shallowconvnet",
        "name": "ShallowConvNet",
        "family": "deep",
        "paradigm": "MI",
        "note": "轻量时域-空域卷积；仅训练折拟合，内部验证折早停",
        "nodes": [
            {"type": "preprocess.bandpass", "label": "8–30 Hz 带通"},
            {"type": "decoder.shallowconvnet", "label": "ShallowConvNet（训练折早停）"},
        ],
    },
    {
        "id": "xdawn_lda",
        "name": "xDAWN + LDA（P300）",
        "family": "erp",
        "paradigm": "P300",
        "note": "1–16 Hz 带通；xDAWN 空间滤波与收缩 LDA 仅在训练折拟合",
        "nodes": [
            {"type": "preprocess.bandpass", "label": "1–16 Hz 带通"},
            {"type": "decoder.xdawn_lda", "label": "xDAWN + 收缩 LDA（训练折拟合）"},
        ],
    },
    {
        "id": "ssvep_cca",
        "name": "SSVEP CCA",
        "family": "ssvep",
        "paradigm": "SSVEP",
        "note": "刺激频率来自数据集元数据；CCA 无监督，参考信号按训练折类别对齐",
        "nodes": [
            {"type": "preprocess.bandpass", "label": "6–45 Hz 带通"},
            {"type": "decoder.ssvep_cca", "label": "CCA 频率识别"},
        ],
    },
    {
        "id": "ssvep_fbcca",
        "name": "SSVEP FBCCA",
        "family": "ssvep",
        "paradigm": "SSVEP",
        "note": "3 个子带的滤波器组 CCA，子带权重 1/(k+1)^1.25+0.25",
        "nodes": [
            {"type": "preprocess.bandpass", "label": "6–45 Hz 带通"},
            {"type": "decoder.ssvep_fbcca", "label": "FBCCA 频率识别"},
        ],
    },
)


def build_baseline(name: str) -> Pipeline:
    if name == "raw_centroid":
        return Pipeline(name, [ChannelMean(), NearestCentroid()])
    if name == "centered_centroid":
        return Pipeline(name, [MeanCenter(), ChannelMean(), NearestCentroid()])
    if name == "var_centroid":
        return Pipeline(name, [ChannelStd(), NearestCentroid()])
    if name == "csp_lda":
        return Pipeline(name, [Bandpass(8.0, 30.0), CSP(6), Standardize(), LDA()])
    if name == "fbcsp_lda":
        return Pipeline(name, [FilterBankCSP(2), Standardize(), LDA()])
    if name == "riemann_mdm":
        return Pipeline(name, [Bandpass(8.0, 30.0), RiemannMDM()])
    if name == "ts_lr":
        return Pipeline(name, [Bandpass(8.0, 30.0), TangentSpaceLR()])
    if name == "eegnet":
        return Pipeline(name, [Bandpass(8.0, 30.0), EEGNet()])
    if name == "shallowconvnet":
        return Pipeline(name, [Bandpass(8.0, 30.0), ShallowConvNet()])
    if name == "xdawn_lda":
        return Pipeline(name, [Bandpass(1.0, 16.0), XdawnLDA()])
    if name == "ssvep_cca":
        return Pipeline(name, [Bandpass(6.0, 45.0), SSVEPCCA(n_bands=0)])
    if name == "ssvep_fbcca":
        return Pipeline(name, [Bandpass(6.0, 45.0), SSVEPCCA(n_bands=3)])
    raise ValueError(f"unknown baseline: {name}")
