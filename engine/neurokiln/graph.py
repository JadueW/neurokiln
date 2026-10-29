from __future__ import annotations

import re
import threading
from dataclasses import replace
from typing import Any, Callable, Sequence

import numpy as np

from .cache import ArtifactCache
from .data.model import Epochs
from .pipeline import (
    EEGNet,
    LDA,
    AverageFusion,
    Bandpass,
    Bandpower,
    Baseline,
    CSP,
    ChannelMean,
    ChannelSelect,
    ChannelStd,
    CommonAverageReference,
    CovarianceVector,
    Crop,
    ErpWindowMean,
    FilterBankCSP,
    FilterBankSplit,
    FitContext,
    Hjorth,
    MeanCenter,
    NearestCentroid,
    Notch,
    Resample,
    RiemannMDM,
    SSVEPCCA,
    SelectK,
    ShallowConvNet,
    Standardize,
    TangentSpaceFeature,
    TangentSpaceLR,
    VoteFusion,
    Xdawn,
    XdawnLDA,
)
from .pipeline.fusion import DecoderPrediction
from .pipeline.nodes import Node, node_descriptor, probability_output, run_node_with_cache
from .protocol.types import Split

_NODE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MAX_NODES = 32


class GraphValidationError(ValueError):
    pass


class Concat(Node):
    """Concatenate two or more 2D feature branches along the feature axis."""

    name = "concat"
    kind = "merge"

    def transform(self, inputs):
        parts = list(inputs) if isinstance(inputs, (list, tuple)) else [inputs]
        if len(parts) < 2:
            raise ValueError("merge.concat requires at least two inputs")
        if any(np.asarray(part).ndim != 2 for part in parts):
            raise ValueError("merge.concat only accepts 2D feature matrices")
        return np.concatenate([np.asarray(part) for part in parts], axis=1)


def _param(
    name: str, label: str, kind: str, default: Any,
    **extra: Any,
) -> dict[str, Any]:
    return {"name": name, "label": label, "type": kind, "default": default, **extra}


NODE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "type": "preprocess.bandpass",
        "label": "带通滤波",
        "kind": "preprocess",
        "description": "Butterworth 零相位带通，仅使用采样率，不接触标签。",
        "params": [
            _param("low", "低频截止 (Hz)", "float", 8.0, min=0.1, max=120, step=0.5),
            _param("high", "高频截止 (Hz)", "float", 30.0, min=0.2, max=160, step=0.5),
            _param("order", "阶数", "int", 4, min=1, max=8, step=1),
        ],
        "factory": Bandpass,
    },
    {
        "type": "preprocess.filter_bank_split",
        "label": "滤波器组并行分支",
        "kind": "preprocess",
        "description": "把 epoch 切分为若干相邻频带，每个子带作为独立命名端口，供不同特征分支并行处理。",
        "params": [
            _param("low", "最低频率 (Hz)", "float", 4.0, min=0.5, max=100, step=0.5),
            _param("high", "最高频率 (Hz)", "float", 32.0, min=1.0, max=160, step=0.5),
            _param("n_bands", "子带数", "int", 4, min=2, max=8, step=1),
            _param("order", "阶数", "int", 4, min=1, max=8, step=1),
        ],
        "factory": FilterBankSplit,
    },
    {
        "type": "preprocess.notch",
        "label": "工频陷波",
        "kind": "preprocess",
        "description": "IIR 零相位陷波，按采样率设计；用于抑制 50/60 Hz 工频干扰。",
        "params": [
            _param("freq", "陷波频率 (Hz)", "float", 50.0, min=1.0, max=120.0, step=0.5),
            _param("q", "品质因数 Q", "float", 30.0, min=1.0, max=100.0, step=1.0),
        ],
        "factory": Notch,
    },
    {
        "type": "preprocess.car",
        "label": "共同平均参考",
        "kind": "preprocess",
        "description": "逐采样点减去全部通道均值（CAR），抑制共同参考噪声。",
        "params": [],
        "factory": CommonAverageReference,
    },
    {
        "type": "preprocess.crop",
        "label": "时间裁剪",
        "kind": "preprocess",
        "description": "按绝对 epoch 时间裁剪（0 为刺激起点）；时长为 0 表示保留到末尾。",
        "params": [
            _param("start", "起点 (s)", "float", 0.0, min=-5.0, max=10.0, step=0.05),
            _param("duration", "时长 (s，0=到末尾)", "float", 0.0, min=0.0, max=30.0, step=0.05),
        ],
        "factory": Crop,
    },
    {
        "type": "preprocess.resample",
        "label": "重采样",
        "kind": "preprocess",
        "description": "多相重采样到目标采样率；下游节点使用更新后的采样率与时间轴。",
        "params": [
            _param("target_sfreq", "目标采样率 (Hz)", "float", 128.0, min=1.0, max=2048.0, step=1.0),
        ],
        "factory": Resample,
    },
    {
        "type": "preprocess.baseline",
        "label": "基线校正",
        "kind": "preprocess",
        "description": "减去绝对时间窗口的均值；需要 epoch 包含刺激前采样点。",
        "params": [
            _param("start", "窗口起点 (s)", "float", -0.2, min=-10.0, max=10.0, step=0.05),
            _param("end", "窗口终点 (s)", "float", 0.0, min=-10.0, max=10.0, step=0.05),
        ],
        "factory": Baseline,
    },
    {
        "type": "preprocess.xdawn",
        "label": "xDAWN 空域滤波",
        "kind": "preprocess",
        "description": "按训练折标签估计 xDAWN 空域滤波器，输出每类 n_filters 个分量。",
        "params": [
            _param("n_filters", "每类滤波器数", "int", 4, min=1, max=32, step=1),
        ],
        "factory": Xdawn,
    },
    {
        "type": "preprocess.channel_select",
        "label": "通道选择",
        "kind": "preprocess",
        "description": "按训练折方差保留前 N 个通道；方差不使用标签，测试折不参与排序。",
        "params": [
            _param("n_channels", "保留通道数", "int", 8, min=1, max=128, step=1),
        ],
        "factory": ChannelSelect,
    },
    {
        "type": "preprocess.mean_center",
        "label": "逐通道中心化",
        "kind": "preprocess",
        "description": "通道均值仅在训练折估计。",
        "params": [],
        "factory": MeanCenter,
    },
    {
        "type": "feature.channel_mean",
        "label": "通道时域均值",
        "kind": "feature",
        "description": "将每个试次压成通道均值特征。",
        "params": [],
        "factory": ChannelMean,
    },
    {
        "type": "feature.channel_std",
        "label": "通道时域标准差",
        "kind": "feature",
        "description": "将每个试次压成通道标准差特征。",
        "params": [],
        "factory": ChannelStd,
    },
    {
        "type": "feature.csp",
        "label": "CSP 对数方差",
        "kind": "feature",
        "description": "在训练折内估计空域滤波器；多类使用一对多 CSP。",
        "params": [
            _param("n_components", "每类分量数", "int", 6, min=2, max=32, step=1),
            _param("reg", "正则化", "float", 1e-6, min=1e-6, max=0.1, step=1e-6),
        ],
        "factory": CSP,
    },
    {
        "type": "feature.fbcsp",
        "label": "滤波器组 CSP",
        "kind": "feature",
        "description": "固定 4–32 Hz 滤波器组，每带 CSP，不做测试折特征选择。",
        "params": [
            _param("n_components", "每带分量数", "int", 2, min=2, max=16, step=1),
            _param("reg", "正则化", "float", 1e-6, min=1e-6, max=0.1, step=1e-6),
        ],
        "factory": FilterBankCSP,
    },
    {
        "type": "feature.standardize",
        "label": "特征标准化",
        "kind": "feature",
        "description": "均值/方差仅在训练折估计。",
        "params": [],
        "factory": Standardize,
    },
    {
        "type": "feature.bandpower",
        "label": "频带功率",
        "kind": "feature",
        "description": "Welch 功率谱在指定频带内积分，输出逐通道带功率。",
        "params": [
            _param("low", "频带下界 (Hz)", "float", 8.0, min=0.5, max=120.0, step=0.5),
            _param("high", "频带上界 (Hz)", "float", 30.0, min=1.0, max=160.0, step=0.5),
            _param("relative", "相对功率", "bool", False),
            _param("log_power", "对数功率", "bool", True),
        ],
        "factory": Bandpower,
    },
    {
        "type": "feature.hjorth",
        "label": "Hjorth 参数",
        "kind": "feature",
        "description": "逐通道输出活动度、移动性与复杂性三个时域特征。",
        "params": [],
        "factory": Hjorth,
    },
    {
        "type": "feature.erp_window",
        "label": "ERP 窗口均值",
        "kind": "feature",
        "description": "对绝对时间窗口内的幅值取平均，输出逐通道特征。",
        "params": [
            _param("start", "窗口起点 (s)", "float", 0.2, min=-5.0, max=10.0, step=0.05),
            _param("end", "窗口终点 (s)", "float", 0.5, min=-4.0, max=15.0, step=0.05),
        ],
        "factory": ErpWindowMean,
    },
    {
        "type": "feature.covariance",
        "label": "协方差向量化",
        "kind": "feature",
        "description": "逐试次估计协方差并向量化上三角，非对角项乘 sqrt(2)。",
        "params": [
            _param("estimator", "协方差估计", "select", "oas",
                   options=["oas", "lwf", "scm"]),
        ],
        "factory": CovarianceVector,
    },
    {
        "type": "feature.tangent_space",
        "label": "黎曼切空间特征",
        "kind": "feature",
        "description": "协方差投影到切空间；参考矩阵仅在训练折估计。",
        "params": [
            _param("metric", "黎曼度量", "select", "riemann",
                   options=["riemann", "logeuclid"]),
            _param("covariance", "协方差估计", "select", "oas",
                   options=["oas", "lwf", "scm"]),
        ],
        "factory": TangentSpaceFeature,
    },
    {
        "type": "feature.select_k",
        "label": "特征选择 Top-K",
        "kind": "feature",
        "description": "ANOVA F 或互信息选择前 K 个特征；仅使用训练折标签。",
        "params": [
            _param("k", "保留特征数", "int", 10, min=1, max=512, step=1),
            _param("method", "评分方法", "select", "anova",
                   options=["anova", "mutual_info"]),
        ],
        "factory": SelectK,
    },
    {
        "type": "merge.concat",
        "label": "特征拼接",
        "kind": "merge",
        "description": "把多条 2D 特征分支按列拼接，再送入分类器。",
        "params": [],
        "factory": Concat,
    },
    {
        "type": "merge.average",
        "label": "概率平均融合",
        "kind": "merge",
        "description": "对齐多个解码器的类别概率后取平均，以平均概率最大的类别为最终预测；要求所有上游解码器提供概率输出。",
        "params": [],
        "inputs": ["labels"],
        "outputs": ["labels"],
        "factory": AverageFusion,
    },
    {
        "type": "merge.vote",
        "label": "多数投票融合",
        "kind": "merge",
        "description": "多个解码器硬投票决定最终类别；票数相同时用可用概率的均值裁决，仍相同则取较小类别。",
        "params": [],
        "inputs": ["labels"],
        "outputs": ["labels"],
        "factory": VoteFusion,
    },
    {
        "type": "decoder.nearest_centroid",
        "label": "最近质心",
        "kind": "decoder",
        "description": "欧氏距离最近质心分类器。",
        "params": [],
        "factory": NearestCentroid,
    },
    {
        "type": "decoder.lda",
        "label": "收缩 LDA",
        "kind": "decoder",
        "description": "线性判别分析，默认 Ledoit-Wolf 收缩。",
        "params": [
            _param("shrinkage", "收缩系数 (auto 或 0–1)", "text", "auto"),
        ],
        "factory": LDA,
    },
    {
        "type": "decoder.riemann_mdm",
        "label": "Riemann MDM",
        "kind": "decoder",
        "description": "OAS 协方差 + 黎曼均值最近距离。",
        "params": [
            _param("metric", "黎曼度量", "select", "riemann",
                   options=["riemann", "logeuclid", "euclid"]),
            _param("covariance", "协方差估计", "select", "oas",
                   options=["oas", "lwf", "scm"]),
        ],
        "factory": RiemannMDM,
    },
    {
        "type": "decoder.tangent_space_lr",
        "label": "切空间 + LR",
        "kind": "decoder",
        "description": "切空间参考矩阵来自训练折，随后逻辑回归。",
        "params": [
            _param("metric", "黎曼度量", "select", "riemann",
                   options=["riemann", "logeuclid"]),
            _param("covariance", "协方差估计", "select", "oas",
                   options=["oas", "lwf", "scm"]),
            _param("C", "正则强度", "float", 1.0, min=0.001, max=100, step=0.1),
        ],
        "factory": TangentSpaceLR,
    },
    {
        "type": "decoder.eegnet",
        "label": "EEGNet",
        "kind": "decoder",
        "description": "轻量卷积网络；128 Hz 重采样，训练折内部验证早停。",
        "params": [
            _param("max_epochs", "最大训练轮数", "int", 60, min=1, max=300, step=1),
            _param("batch_size", "批大小", "int", 32, min=4, max=256, step=4),
            _param("learning_rate", "学习率", "float", 0.001, min=0.00001, max=0.1, step=0.0001),
            _param("dropout", "Dropout", "float", 0.5, min=0.0, max=0.9, step=0.05),
        ],
        "factory": EEGNet,
    },
    {
        "type": "decoder.shallowconvnet",
        "label": "ShallowConvNet",
        "kind": "decoder",
        "description": "时域-空域卷积网络；仅训练折拟合。",
        "params": [
            _param("max_epochs", "最大训练轮数", "int", 60, min=1, max=300, step=1),
            _param("batch_size", "批大小", "int", 32, min=4, max=256, step=4),
            _param("learning_rate", "学习率", "float", 0.001, min=0.00001, max=0.1, step=0.0001),
            _param("dropout", "Dropout", "float", 0.5, min=0.0, max=0.9, step=0.05),
        ],
        "factory": ShallowConvNet,
    },
    {
        "type": "decoder.xdawn_lda",
        "label": "xDAWN + LDA",
        "kind": "decoder",
        "description": "P300/ERP 空间滤波与收缩 LDA；xDAWN 仅在训练折估计。",
        "params": [
            _param("n_filters", "空间滤波器数", "int", 4, min=1, max=16, step=1),
            _param("shrinkage", "LDA 收缩 (auto 或 0–1)", "text", "auto"),
        ],
        "factory": XdawnLDA,
    },
    {
        "type": "decoder.ssvep_cca",
        "label": "SSVEP CCA",
        "kind": "decoder",
        "description": "使用刺激频率元数据的典型相关分析；n_bands=0 为 CCA。",
        "params": [
            _param("n_harmonics", "谐波数", "int", 5, min=1, max=8, step=1),
            _param("n_bands", "滤波器组子带数 (0=CCA)", "int", 0, min=0, max=5, step=1),
        ],
        "factory": SSVEPCCA,
    },
    {
        "type": "decoder.ssvep_fbcca",
        "label": "SSVEP FBCCA",
        "kind": "decoder",
        "description": "3 个子带的滤波器组 CCA；频率表来自数据集元数据。",
        "params": [
            _param("n_harmonics", "谐波数", "int", 5, min=1, max=8, step=1),
            _param("n_bands", "滤波器组子带数", "int", 3, min=1, max=5, step=1),
        ],
        "factory": SSVEPCCA,
    },
)

_SPEC_LOCK = threading.RLock()
_PLUGIN_SPECS: dict[str, dict[str, Any]] = {}
_SPEC_BY_TYPE = {item["type"]: item for item in NODE_SPECS}

_NODE_KINDS = {"preprocess", "feature", "merge", "decoder"}
_PARAM_KINDS = {"int", "float", "select", "text", "bool"}
_PLUGIN_NODE_TYPE = re.compile(r"^plugin\.[a-z0-9][a-z0-9_-]{1,40}\.[a-z0-9_]{2,40}$")
_NODE_PARADIGMS = {"MI", "P300", "SSVEP", "ANY"}
_IO_TYPES = {"epochs", "features", "covariance", "labels", "probabilities", "any"}
_DEFAULT_IO = {
    "preprocess": (["epochs"], ["epochs"]),
    "feature": (["epochs", "features"], ["features"]),
    "merge": (["features"], ["features"]),
    "decoder": (["features"], ["labels"]),
}
_JSON_TYPES = {
    "int": "integer",
    "float": "number",
    "select": "string",
    "text": "string",
    "bool": "boolean",
}
_FUSION_TYPES = {"merge.average", "merge.vote"}
_MULTI_INPUT_TYPES = {"merge.concat", *_FUSION_TYPES}
_FEATURE_ONLY_TYPES = {
    "feature.standardize",
    "feature.select_k",
    "decoder.lda",
    "decoder.nearest_centroid",
}
_EPOCH_FEATURE_TYPES = {
    "feature.channel_mean",
    "feature.channel_std",
    "feature.csp",
    "feature.fbcsp",
    "feature.bandpower",
    "feature.hjorth",
    "feature.erp_window",
    "feature.covariance",
    "feature.tangent_space",
}
_EPOCH_ONLY_TYPES = {
    "decoder.riemann_mdm",
    "decoder.tangent_space_lr",
    "decoder.eegnet",
    "decoder.shallowconvnet",
    "decoder.xdawn_lda",
    "decoder.ssvep_cca",
    "decoder.ssvep_fbcca",
}
_IO_LABELS = {
    "epochs": "3D 试次",
    "features": "2D 特征",
    "labels": "类别预测",
    "covariance": "协方差矩阵",
    "probabilities": "类别概率",
    "any": "任意数据",
}


def _node_io(node_type: str, spec: dict[str, Any] | None = None) -> tuple[list[str], list[str]]:
    item = spec or _SPEC_BY_TYPE[node_type]
    default_inputs, default_outputs = _DEFAULT_IO[item["kind"]]
    if item.get("inputs") is not None:
        inputs = list(item["inputs"])
    elif node_type in _FUSION_TYPES:
        inputs = ["labels"]
    elif node_type in _FEATURE_ONLY_TYPES:
        inputs = ["features"]
    elif node_type in _EPOCH_ONLY_TYPES or node_type in _EPOCH_FEATURE_TYPES:
        inputs = ["epochs"]
    else:
        inputs = list(default_inputs)
    outputs = list(item.get("outputs") or default_outputs)
    return inputs, outputs


def _io_label(values: Sequence[str]) -> str:
    return " / ".join(_IO_LABELS.get(value, value) for value in values)


def _check_edge_io(
    source: dict[str, Any], target: dict[str, Any], source_type: str, target_inputs: Sequence[str]
) -> None:
    if source_type in target_inputs or "any" in target_inputs:
        return
    source_label = source["label"]
    target_label = target["label"]
    message = (
        f"节点 {target['id']}（{target_label}）只接受 {_io_label(target_inputs)} 输入，"
        f"但上游 {source['id']}（{source_label}）输出 {_io_label([source_type])}。"
    )
    if target["type"] in _EPOCH_ONLY_TYPES:
        message += "EEGNet/ShallowConvNet/Riemann/xDAWN/SSVEP 解码器需要直接接收预处理后的 3D 试次；若已提取手工特征，请改用 LDA 或最近质心。"
    elif target["type"] in _FEATURE_ONLY_TYPES:
        message += "请先接特征提取节点，再进入标准化、特征选择或特征分类器。"
    else:
        message += "请检查上游节点类型。"
    raise GraphValidationError(message)


def _default_params(node_type: str) -> dict[str, Any]:
    return {
        item["name"]: item["default"]
        for item in _SPEC_BY_TYPE[node_type]["params"]
    }


def _output_ports_for(
    node_type: str, params: dict[str, Any] | None = None
) -> list[dict[str, str]]:
    if node_type == "preprocess.filter_bank_split":
        values = {**_default_params(node_type), **(params or {})}
        return FilterBankSplit(
            low=values["low"],
            high=values["high"],
            n_bands=values["n_bands"],
            order=values["order"],
        ).port_specs()
    return [dict(item) for item in _SPEC_BY_TYPE[node_type].get("output_ports") or []]


def params_schema(params: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Generate a JSON Schema for a node's parameter table."""
    properties: dict[str, Any] = {}
    for param in params:
        name = str(param["name"])
        schema: dict[str, Any] = {
            "type": _JSON_TYPES[str(param["type"])],
            "title": str(param.get("label") or name),
        }
        if "default" in param:
            schema["default"] = param["default"]
        if param["type"] == "select":
            schema["enum"] = list(param.get("options") or [])
        for key in ("min", "max", "step"):
            if key in param:
                schema[key] = param[key]
        properties[name] = schema
    return {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }


def _plugin_io(raw: Any, node_type: str, kind: str, field: str) -> list[str]:
    default = _DEFAULT_IO[kind][0 if field == "inputs" else 1]
    if raw is None:
        return list(default)
    values = [raw] if isinstance(raw, str) else raw
    if not isinstance(values, (list, tuple)) or not values:
        raise GraphValidationError(f"插件节点 {node_type} 的 {field} 必须是非空数组")
    names = [str(value).strip() for value in values]
    invalid = [name for name in names if name not in _IO_TYPES]
    if invalid:
        raise GraphValidationError(
            f"插件节点 {node_type} 的 {field} 类型非法: {invalid}"
        )
    if len(set(names)) != len(names):
        raise GraphValidationError(f"插件节点 {node_type} 的 {field} 不能重复")
    return names


def _plugin_paradigm(raw: Any, node_type: str) -> str:
    if raw is None:
        return "ANY"
    values = [raw] if isinstance(raw, str) else raw
    if not isinstance(values, (list, tuple)) or not values:
        raise GraphValidationError(f"插件节点 {node_type} 的 paradigm 非法")
    supported: set[str] = set()
    for value in values:
        for part in str(value).split("/"):
            part = part.strip().upper()
            if not part:
                continue
            if part not in _NODE_PARADIGMS:
                raise GraphValidationError(
                    f"插件节点 {node_type} 的 paradigm 非法: {part}"
                )
            supported.add(part)
    if not supported:
        raise GraphValidationError(f"插件节点 {node_type} 的 paradigm 为空")
    if "ANY" in supported and len(supported) > 1:
        raise GraphValidationError(f"插件节点 {node_type} 的 ANY 不能与其他范式并存")
    return "/".join(sorted(supported))


def _plugin_schema(
    raw: Any, node_type: str, params: list[dict[str, Any]]
) -> dict[str, Any]:
    schema = params_schema(params) if raw is None else raw
    if not isinstance(schema, dict):
        raise GraphValidationError(f"插件节点 {node_type} 的 params_schema 必须是对象")
    if schema.get("type", "object") != "object":
        raise GraphValidationError(f"插件节点 {node_type} 的 params_schema.type 必须为 object")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise GraphValidationError(f"插件节点 {node_type} 的 params_schema.properties 必须是对象")
    declared = {param["name"] for param in params}
    extra = sorted(set(properties) - declared)
    missing = sorted(declared - set(properties))
    if extra or missing:
        raise GraphValidationError(
            f"插件节点 {node_type} 的 params_schema 与 params 不一致"
            + (f"；多出 {extra}" if extra else "")
            + (f"；缺少 {missing}" if missing else "")
        )
    required = schema.get("required", [])
    if not isinstance(required, list) or not set(required) <= declared:
        raise GraphValidationError(f"插件节点 {node_type} 的 params_schema.required 非法")
    for name, prop in properties.items():
        if not isinstance(prop, dict):
            raise GraphValidationError(
                f"插件节点 {node_type} 的 params_schema.properties.{name} 必须是对象"
            )
        expected = _JSON_TYPES[next(
            param["type"] for param in params if param["name"] == name
        )]
        actual = prop.get("type")
        if actual is not None and actual != expected:
            raise GraphValidationError(
                f"插件节点 {node_type} 的 params_schema 中 {name} 类型应为 {expected}"
            )
    return schema


def _factory_requires_fit(factory: Callable[..., Node]) -> bool:
    fit = getattr(factory, "fit", None)
    return fit is not Node.fit


def register_plugin_node_specs(specs: Sequence[dict[str, Any]]) -> list[str]:
    """Register node specs supplied by a verified plugin package."""
    validated: dict[str, dict[str, Any]] = {}
    for raw in specs:
        node_type = str(raw.get("type") or "").strip()
        if not _PLUGIN_NODE_TYPE.match(node_type):
            raise GraphValidationError(f"插件节点类型必须形如 plugin.<插件ID>.<名称>: {node_type!r}")
        kind = str(raw.get("kind") or "")
        if kind not in _NODE_KINDS:
            raise GraphValidationError(f"插件节点 {node_type} 的 kind 非法: {kind}")
        factory = raw.get("factory")
        if not callable(factory):
            raise GraphValidationError(f"插件节点 {node_type} 缺少可调用 factory")
        if not str(raw.get("label") or "").strip():
            raise GraphValidationError(f"插件节点 {node_type} 缺少 label")
        params = []
        seen = set()
        for param in raw.get("params") or []:
            if not isinstance(param, dict):
                raise GraphValidationError(f"插件节点 {node_type} 的参数定义必须是对象")
            name = str(param.get("name") or "").strip()
            if not re.match(r"^[a-z][a-z0-9_]{0,39}$", name):
                raise GraphValidationError(f"插件节点 {node_type} 参数名非法: {name!r}")
            if name in seen:
                raise GraphValidationError(f"插件节点 {node_type} 参数名重复: {name}")
            seen.add(name)
            if str(param.get("type") or "") not in _PARAM_KINDS:
                raise GraphValidationError(f"插件节点 {node_type}.{name} 参数类型非法")
            params.append({
                "name": name,
                "label": str(param.get("label") or name),
                "type": str(param["type"]),
                "default": param.get("default"),
                **{
                    key: param[key]
                    for key in ("min", "max", "step", "options")
                    if key in param
                },
            })
        validated[node_type] = {
            "type": node_type,
            "label": str(raw["label"])[:80],
            "kind": kind,
            "description": str(raw.get("description") or "插件提供的节点"),
            "params": params,
            "params_schema": _plugin_schema(
                raw.get("params_schema"), node_type, params
            ),
            "inputs": _plugin_io(raw.get("inputs"), node_type, kind, "inputs"),
            "outputs": _plugin_io(raw.get("outputs"), node_type, kind, "outputs"),
            "paradigm": _plugin_paradigm(raw.get("paradigm"), node_type),
            "requires_fit": bool(raw["requires_fit"])
            if "requires_fit" in raw
            else _factory_requires_fit(factory),
            "deterministic": bool(raw.get("deterministic", True)),
            "cacheable": bool(raw.get("cacheable", False)),
            "factory": factory,
            "plugin": True,
        }
    with _SPEC_LOCK:
        for node_type, spec in validated.items():
            _PLUGIN_SPECS[node_type] = spec
            _SPEC_BY_TYPE[node_type] = spec
    return sorted(validated)


def unregister_plugin_node_specs(prefix: str) -> list[str]:
    with _SPEC_LOCK:
        removed = [
            node_type for node_type in _PLUGIN_SPECS
            if node_type == prefix or node_type.startswith(prefix)
        ]
        for node_type in removed:
            _PLUGIN_SPECS.pop(node_type, None)
            _SPEC_BY_TYPE.pop(node_type, None)
    return sorted(removed)


def node_catalog() -> list[dict[str, Any]]:
    with _SPEC_LOCK:
        specs = [*NODE_SPECS, *_PLUGIN_SPECS.values()]
    catalog = []
    for item in specs:
        inputs, outputs = _node_io(item["type"], item)
        catalog.append(
            {
                "type": item["type"],
                "label": item["label"],
                "kind": item["kind"],
                "description": item["description"],
                "params": item["params"],
                "params_schema": item.get("params_schema")
                or params_schema(item["params"]),
                "inputs": inputs,
                "outputs": outputs,
                "paradigm": item.get("paradigm") or "ANY",
                "requires_fit": item.get(
                    "requires_fit", _factory_requires_fit(item["factory"])
                ),
                "deterministic": bool(item.get("deterministic", True)),
                "cacheable": bool(
                    item.get("cacheable")
                    if "cacheable" in item
                    else getattr(item["factory"], "cacheable", False)
                ),
                "plugin": bool(item.get("plugin")),
                "output_ports": _output_ports_for(item["type"]),
                "dynamic_ports": item["type"] == "preprocess.filter_bank_split",
            }
        )
    return catalog


def _build_node(node_type: str, params: dict[str, Any]) -> Node:
    spec = _SPEC_BY_TYPE.get(node_type)
    if spec is None:
        raise GraphValidationError(f"未知节点类型: {node_type}")
    factory: Callable[..., Node] = spec["factory"]
    try:
        return factory(**params)
    except TypeError as exc:
        raise GraphValidationError(f"{node_type} 参数不匹配: {exc}") from exc
    except (ValueError, RuntimeError) as exc:
        raise GraphValidationError(f"{node_type} 参数无效: {exc}") from exc


def _coerce_params(node_type: str, params: dict[str, Any]) -> dict[str, Any]:
    definitions = {item["name"]: item for item in _SPEC_BY_TYPE[node_type]["params"]}
    unknown = sorted(set(params) - set(definitions))
    if unknown:
        raise GraphValidationError(
            f"{node_type} 包含未知参数: {', '.join(unknown)}"
        )
    coerced: dict[str, Any] = {}
    for name, definition in definitions.items():
        if name not in params:
            coerced[name] = definition["default"]
            continue
        value = params[name]
        kind = definition["type"]
        try:
            if kind == "int":
                value = int(value)
            elif kind == "float":
                value = float(value)
            elif kind == "bool":
                if isinstance(value, str):
                    value = value.strip().lower() in {"1", "true", "yes", "on"}
                else:
                    value = bool(value)
            elif kind == "select":
                value = str(value)
                options = definition.get("options") or []
                if value not in options:
                    raise GraphValidationError(
                        f"{node_type}.{name} 必须是 {options} 之一"
                    )
            else:
                value = str(value)
        except (TypeError, ValueError) as exc:
            raise GraphValidationError(f"{node_type}.{name} 类型无效: {value!r}") from exc
        if "min" in definition and isinstance(value, (int, float)) and value < definition["min"]:
            raise GraphValidationError(f"{node_type}.{name} 不能小于 {definition['min']}")
        if "max" in definition and isinstance(value, (int, float)) and value > definition["max"]:
            raise GraphValidationError(f"{node_type}.{name} 不能大于 {definition['max']}")
        coerced[name] = value
    return coerced


def _check_coordinate(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return round(number, 1)


def normalize_graph(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate a graph spec and return a sanitized copy."""
    if not isinstance(spec, dict):
        raise GraphValidationError("图定义必须是对象")
    raw_nodes = spec.get("nodes")
    raw_edges = spec.get("edges", [])
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise GraphValidationError("图中至少需要一个节点")
    if len(raw_nodes) > _MAX_NODES:
        raise GraphValidationError(f"节点数不能超过 {_MAX_NODES}")
    if not isinstance(raw_edges, list):
        raise GraphValidationError("edges 必须是数组")

    nodes: list[dict[str, Any]] = []
    ids: set[str] = set()
    for raw in raw_nodes:
        if not isinstance(raw, dict):
            raise GraphValidationError("节点必须是对象")
        node_id = str(raw.get("id", "")).strip()
        if not _NODE_ID.match(node_id):
            raise GraphValidationError(f"节点 ID 非法: {node_id!r}")
        if node_id in ids:
            raise GraphValidationError(f"节点 ID 重复: {node_id}")
        ids.add(node_id)
        node_type = str(raw.get("type", "")).strip()
        if node_type not in _SPEC_BY_TYPE:
            raise GraphValidationError(f"未知节点类型: {node_type}")
        params = raw.get("params") or {}
        if not isinstance(params, dict):
            raise GraphValidationError(f"节点 {node_id} 的 params 必须是对象")
        params = _coerce_params(node_type, params)
        node = {
            "id": node_id,
            "type": node_type,
            "params": params,
            "label": str(raw.get("label") or _SPEC_BY_TYPE[node_type]["label"])[:80],
        }
        for key in ("x", "y"):
            coordinate = _check_coordinate(raw.get(key))
            if coordinate is not None:
                node[key] = coordinate
        _build_node(node_type, params)
        nodes.append(node)

    edges: list[dict[str, str]] = []
    seen_edges: set[tuple[str, str, str]] = set()
    incoming: dict[str, list[str]] = {node_id: [] for node_id in ids}
    outgoing: dict[str, list[str]] = {node_id: [] for node_id in ids}
    nodes_by_id = {node["id"]: node for node in nodes}
    for raw in raw_edges:
        if not isinstance(raw, dict):
            raise GraphValidationError("边必须是对象")
        source = str(raw.get("source", "")).strip()
        target = str(raw.get("target", "")).strip()
        if source not in ids or target not in ids:
            raise GraphValidationError(f"边引用了不存在的节点: {source} -> {target}")
        if source == target:
            raise GraphValidationError(f"不允许自环: {source}")
        source_port = str(raw.get("source_port") or "").strip()
        ports = {
            item["name"]
            for item in _output_ports_for(
                nodes_by_id[source]["type"], nodes_by_id[source]["params"]
            )
        }
        if ports:
            if not source_port:
                raise GraphValidationError(
                    f"节点 {source} 有多个输出端口，连线必须指定 source_port"
                    f"（可选 {', '.join(sorted(ports))}）"
                )
            if source_port not in ports:
                raise GraphValidationError(
                    f"节点 {source} 没有输出端口: {source_port}"
                )
        elif source_port:
            raise GraphValidationError(
                f"节点 {source} 只有一个默认输出，不接受 source_port"
            )
        key = (source, source_port, target)
        if key in seen_edges:
            label = f" {source_port}" if source_port else ""
            raise GraphValidationError(f"重复边: {source}{label} -> {target}")
        seen_edges.add(key)
        incoming[target].append(source)
        outgoing[source].append(target)
        edge = {"source": source, "target": target}
        if source_port:
            edge["source_port"] = source_port
        edges.append(edge)

    kinds = {
        node_id: _SPEC_BY_TYPE[nodes_by_id[node_id]["type"]]["kind"]
        for node_id in ids
    }
    decoders = [node["id"] for node in nodes if kinds[node["id"]] == "decoder"]
    fusion_nodes = [node["id"] for node in nodes if node["type"] in _FUSION_TYPES]
    if not decoders:
        raise GraphValidationError("图必须包含至少一个分类器节点")
    if len(fusion_nodes) > 1:
        raise GraphValidationError("图最多只能包含一个平均/投票融合节点")

    for node in nodes:
        node_id = node["id"]
        sources = incoming[node_id]
        if node["type"] == "merge.concat" and len(sources) < 2:
            raise GraphValidationError("特征拼接节点需要至少两个输入")
        if node["type"] in _FUSION_TYPES and len(sources) < 2:
            raise GraphValidationError("平均/投票融合节点需要至少两个解码器输入")
        if len(sources) > 1:
            if node["type"] == "merge.concat":
                if any(kinds[source] == "decoder" for source in sources):
                    raise GraphValidationError("特征拼接节点不能接收分类器输出")
            elif node["type"] in _FUSION_TYPES:
                for source in sources:
                    if kinds[source] != "decoder":
                        raise GraphValidationError(
                            "平均/投票融合节点只能接收分类器输出"
                        )
            else:
                raise GraphValidationError(
                    f"节点 {node_id} 有多个输入，只有特征拼接与融合节点支持多输入"
                )
        elif len(sources) == 1:
            if kinds[sources[0]] == "decoder" and node["type"] not in _FUSION_TYPES:
                raise GraphValidationError("分类器之后只能接平均或投票融合节点")

    if fusion_nodes:
        prediction_id = fusion_nodes[0]
        if outgoing[prediction_id]:
            raise GraphValidationError("融合节点必须是最终输出，不能再接其他节点")
        for decoder_id in decoders:
            if outgoing[decoder_id] != [prediction_id]:
                raise GraphValidationError(
                    f"分类器 {decoder_id} 必须且只能连接到融合节点 {prediction_id}"
                )
    else:
        if len(decoders) != 1:
            raise GraphValidationError(
                "图必须且只能包含一个分类器作为最终输出；"
                "多解码器请接入平均或投票融合"
            )
        prediction_id = decoders[0]
        if outgoing[prediction_id]:
            raise GraphValidationError("分类器节点必须是终点，不能有后继节点")

    indegree = {node_id: len(incoming[node_id]) for node_id in ids}
    ready = [node["id"] for node in nodes if indegree[node["id"]] == 0]
    if not ready:
        raise GraphValidationError("图中存在环：没有任何入度为 0 的输入节点")
    order: list[str] = []
    while ready:
        node_id = ready.pop(0)
        order.append(node_id)
        for target in outgoing[node_id]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if len(order) != len(ids):
        raise GraphValidationError("图中存在环，无法拓扑排序")

    reachable = {prediction_id}
    stack = [prediction_id]
    while stack:
        current = stack.pop()
        for source in incoming[current]:
            if source not in reachable:
                reachable.add(source)
                stack.append(source)
    orphans = sorted(ids - reachable)
    if orphans:
        raise GraphValidationError(f"以下节点未连接到最终输出: {', '.join(orphans)}")

    for edge in edges:
        source_node = nodes_by_id[edge["source"]]
        target_node = nodes_by_id[edge["target"]]
        _check_edge_io(
            source_node,
            target_node,
            _node_io(source_node["type"])[1][0],
            _node_io(target_node["type"])[0],
        )

    return {
        "nodes": nodes,
        "edges": edges,
        "topology": order,
        "decoder": decoders[0] if len(decoders) == 1 else prediction_id,
        "decoders": decoders,
        "prediction": prediction_id,
    }


def _select_port(values: Any, source: str, port: str) -> np.ndarray:
    if not port:
        return values
    if not isinstance(values, dict):
        raise GraphValidationError(f"节点 {source} 没有输出端口 {port}")
    if port not in values:
        raise GraphValidationError(f"节点 {source} 缺少输出端口 {port}")
    return values[port]


def _check_port_outputs(raw: Any, ports: Sequence[str], node_id: str) -> dict[str, np.ndarray]:
    if not isinstance(raw, dict):
        raise ValueError(f"节点 {node_id} 声明了命名端口，transform 必须返回 dict")
    missing = [port for port in ports if port not in raw]
    if missing:
        raise ValueError(f"节点 {node_id} 缺少输出端口: {', '.join(missing)}")
    return {port: np.asarray(raw[port]) for port in ports}


def _run_multi_output_node(
    node: Node,
    node_id: str,
    train_input,
    test_input,
    y_train: np.ndarray,
    node_ctx: FitContext,
    ports: Sequence[str],
    *,
    cache: ArtifactCache | None,
    data_signature: str,
    split_signature: str,
    seed: int,
    train_parents: Sequence[str],
    test_parents: Sequence[str],
    context: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], tuple[str, str]]:
    """Fit a named-port node without caching; keys still feed downstream hashes."""
    train_key = test_key = ""
    if cache is not None:
        descriptor = node_descriptor(node)
        train_key = ArtifactCache.node_key(
            data_signature=data_signature,
            split_signature=split_signature,
            seed=seed,
            parents=train_parents,
            descriptor=descriptor,
            role="train",
            context=context,
        )
        test_key = ArtifactCache.node_key(
            data_signature=data_signature,
            split_signature=split_signature,
            seed=seed,
            parents=test_parents,
            descriptor=descriptor,
            role="test",
            context=context,
        )
    node.fit(train_input, y_train, node_ctx)
    train_output = _check_port_outputs(node.transform(train_input), ports, node_id)
    test_output = _check_port_outputs(node.transform(test_input), ports, node_id)
    return train_output, test_output, (train_key, test_key)


class GraphPipeline:
    """Execute a validated DAG with branch/merge nodes.

    The fit order follows the topology. Every node only ever receives training
    arrays while fitting; test arrays are kept for transform/predict.
    """

    def __init__(self, name: str, spec: dict[str, Any]):
        self.name = name
        self.spec = normalize_graph(spec)
        self.order: list[str] = list(self.spec["topology"])
        self.decoder_ids: list[str] = list(self.spec["decoders"])
        self.prediction_id: str = self.spec["prediction"]
        self.decoder_id: str = self.spec["decoder"]
        self.nodes: dict[str, Node] = {
            node["id"]: _build_node(node["type"], node["params"])
            for node in self.spec["nodes"]
        }
        self.node_specs: dict[str, dict[str, Any]] = {
            node["id"]: node for node in self.spec["nodes"]
        }
        self.kinds: dict[str, str] = {
            node["id"]: _SPEC_BY_TYPE[node["type"]]["kind"]
            for node in self.spec["nodes"]
        }
        self.incoming: dict[str, list[dict[str, str]]] = {
            node_id: [] for node_id in self.order
        }
        for edge in self.spec["edges"]:
            self.incoming[edge["target"]].append(edge)

    def node_types(self) -> dict[str, str]:
        return {node["id"]: node["type"] for node in self.spec["nodes"]}

    def run_split(
        self,
        epochs: Epochs,
        split: Split,
        should_cancel=None,
        cache: ArtifactCache | None = None,
        data_signature: str | None = None,
    ) -> dict:
        train_epochs = epochs.select(split.train)
        test_epochs = epochs.select(split.test)
        y_train = train_epochs.labels
        ctx = FitContext(
            trial_ids=tuple(train_epochs.trial_ids.tolist()),
            indices=np.asarray(split.train, dtype=int),
            seed=split.seed,
            paradigm=epochs.paradigm,
            channel_names=epochs.channel_names,
            sfreq=float(epochs.sfreq),
            paradigm_meta=dict(epochs.paradigm_meta),
            should_cancel=should_cancel,
            tmin=float(epochs.tmin),
        )
        signature = data_signature or epochs.signature()
        cache_context = {
            "channel_names": list(epochs.channel_names),
            "paradigm": epochs.paradigm,
        }
        train_values: dict[str, Any] = {}
        test_values: dict[str, Any] = {}
        predictions: dict[str, DecoderPrediction] = {}
        fusion_result: DecoderPrediction | None = None
        keys: dict[str, tuple[str, str]] = {}
        timebases: dict[str, tuple[float, float]] = {}
        for node_id in self.order:
            node = self.nodes[node_id]
            node_spec = self.node_specs[node_id]
            node_type = node_spec["type"]
            edges_in = self.incoming[node_id]
            sources = [edge["source"] for edge in edges_in]
            ports = [str(edge.get("source_port") or "") for edge in edges_in]
            if node_type in _FUSION_TYPES:
                fusion_result = node.combine(
                    [predictions[source] for source in sources]
                )
                keys[node_id] = ("", "")
                timebases[node_id] = timebases[sources[0]]
                continue
            if not sources:
                train_input = train_epochs.data
                test_input = test_epochs.data
                node_sfreq = float(epochs.sfreq)
                node_tmin = float(epochs.tmin)
                train_parents: tuple[str, ...] = (signature,)
                test_parents: tuple[str, ...] = (signature,)
            else:
                node_sfreq, node_tmin = timebases[sources[0]]
                if len(sources) == 1:
                    train_input = _select_port(train_values[sources[0]], sources[0], ports[0])
                    test_input = _select_port(test_values[sources[0]], sources[0], ports[0])
                else:
                    train_input = [
                        _select_port(train_values[source], source, port)
                        for source, port in zip(sources, ports)
                    ]
                    test_input = [
                        _select_port(test_values[source], source, port)
                        for source, port in zip(sources, ports)
                    ]
                train_parents = tuple(keys[source][0] for source in sources)
                test_parents = tuple(keys[source][1] for source in sources)
            node_ctx = replace(ctx, sfreq=node_sfreq, tmin=node_tmin)
            context = {**cache_context, "sfreq": node_sfreq, "tmin": node_tmin}
            selected_ports = [port for port in ports if port]
            if selected_ports:
                context["source_ports"] = selected_ports

            output_ports = _output_ports_for(node_type, node_spec["params"])
            if output_ports:
                port_names = [item["name"] for item in output_ports]
                train_output, test_output, keys[node_id] = _run_multi_output_node(
                    node,
                    node_id,
                    train_input,
                    test_input,
                    y_train,
                    node_ctx,
                    port_names,
                    cache=cache,
                    data_signature=signature,
                    split_signature=split.signature(),
                    seed=split.seed,
                    train_parents=train_parents,
                    test_parents=test_parents,
                    context=context,
                )
                train_values[node_id] = train_output
                test_values[node_id] = test_output
            else:
                train_output, test_output, keys[node_id] = run_node_with_cache(
                    node,
                    train_input,
                    test_input,
                    y_train,
                    node_ctx,
                    cache=cache,
                    data_signature=signature,
                    split_signature=split.signature(),
                    seed=split.seed,
                    train_parents=train_parents,
                    test_parents=test_parents,
                    context=context,
                )
                if self.kinds[node_id] == "decoder":
                    y_pred = np.asarray(node.predict(test_output), dtype=int)
                    if y_pred.shape != test_epochs.labels.shape:
                        raise ValueError("decoder returned predictions with wrong shape")
                    y_proba, proba_classes = probability_output(node, test_output)
                    predictions[node_id] = DecoderPrediction(
                        y_pred=y_pred, proba=y_proba, classes=proba_classes
                    )
                else:
                    train_values[node_id] = train_output
                    test_values[node_id] = test_output
            timebases[node_id] = node.output_timebase(node_sfreq, node_tmin)

        final = fusion_result if fusion_result is not None else predictions.get(self.prediction_id)
        if final is None:
            raise GraphValidationError("图没有产生分类器输出")
        y_pred, y_proba, proba_classes = final.y_pred, final.proba, final.classes
        if y_pred.shape != test_epochs.labels.shape:
            raise ValueError("decoder returned predictions with wrong shape")
        return {
            "y_true": test_epochs.labels,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "proba_classes": proba_classes,
            "train_trial_ids": train_epochs.trial_ids,
            "test_trial_ids": test_epochs.trial_ids,
            "test_subject_ids": test_epochs.subject_ids,
            "test_session_ids": test_epochs.session_ids,
            "split": split,
        }

    def describe(self) -> dict[str, Any]:
        labels = {item["type"]: item["label"] for item in node_catalog()}
        return {
            "id": self.name,
            "name": self.name,
            "family": "custom",
            "graph": self.spec,
            "nodes": [
                {
                    "id": node["id"],
                    "type": node["type"],
                    "label": labels.get(node["type"], node["type"]),
                }
                for node in self.spec["nodes"]
            ],
        }


def build_graph_pipeline(name: str, spec: dict[str, Any]) -> GraphPipeline:
    return GraphPipeline(name, spec)


__all__ = [
    "GraphPipeline",
    "GraphValidationError",
    "NODE_SPECS",
    "build_graph_pipeline",
    "node_catalog",
    "normalize_graph",
    "register_plugin_node_specs",
    "unregister_plugin_node_specs",
]
