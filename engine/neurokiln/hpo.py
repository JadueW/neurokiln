"""Multi-objective hyper-parameter search for BCI pipelines.

The search runs Grey Wolf Optimizer (GWO) in a normalized parameter cube.
Every wolf is decoded into a valid DAG, evaluated on a proxy subset of the
protocol splits, and kept in an audit trail. Once the proxy budget is spent,
non-dominated candidates are re-evaluated on the full protocol before the
final Pareto front is reported.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from scipy.stats import rankdata

from .data.adapters import DatasetRegistry
from .data.model import Epochs
from .errors import RunCancelled
from .evaluate import run_arena
from .graph import (
    GraphValidationError,
    build_graph_pipeline,
    node_catalog,
    normalize_graph,
)
from .protocol import build_protocol
from .protocol.types import Split

OBJECTIVES: dict[str, dict[str, str]] = {
    "balanced_accuracy": {"label": "Balanced Accuracy", "direction": "max"},
    "cohen_kappa": {"label": "Cohen Kappa", "direction": "max"},
    "duration_ms": {"label": "单折耗时", "direction": "min", "unit": "ms"},
    "channel_count": {"label": "使用通道数", "direction": "min", "unit": "ch"},
}

_DEFAULT_OBJECTIVES = ("balanced_accuracy", "cohen_kappa")


@dataclass(frozen=True)
class ParamSpec:
    name: str
    label: str
    kind: str
    default: Any
    low: float | None = None
    high: float | None = None
    step: float | None = None
    options: tuple[Any, ...] = ()
    log: bool = False

    def describe(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "default": self.default,
        }
        if self.kind == "select":
            payload["options"] = list(self.options)
        else:
            payload.update({"low": self.low, "high": self.high, "step": self.step, "log": self.log})
        return payload

    def encode_default(self) -> float:
        if self.kind == "select":
            return self.options.index(self.default) / max(1, len(self.options) - 1)
        if self.kind == "int":
            return (float(self.default) - self.low) / max(1e-12, self.high - self.low)
        if self.log:
            return (math.log(float(self.default)) - math.log(self.low)) / max(
                1e-12, math.log(self.high) - math.log(self.low)
            )
        return (float(self.default) - self.low) / max(1e-12, self.high - self.low)


@dataclass(frozen=True)
class HPOTemplate:
    id: str
    name: str
    description: str
    family: str
    space: tuple[ParamSpec, ...]
    builder: Callable[[dict[str, Any], int], tuple[dict[str, Any], dict[str, Any]]]
    deep: bool = False
    source: dict[str, Any] | None = None

    def defaults(self) -> dict[str, Any]:
        return {item.name: item.default for item in self.space}

    def default_vector(self) -> np.ndarray:
        return np.asarray([item.encode_default() for item in self.space], dtype=float)

    def build(self, params: dict[str, Any], n_channels: int) -> tuple[dict[str, Any], dict[str, Any]]:
        spec, effective = self.builder(params, n_channels)
        return normalize_graph(spec), effective

    def describe(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "family": self.family,
            "deep": self.deep,
            "space": [item.describe() for item in self.space],
            "defaults": self.defaults(),
        }
        if self.source:
            payload["source"] = dict(self.source)
        return payload


@dataclass
class HPORequest:
    dataset_id: str
    subject_ids: list[str] = field(default_factory=list)
    classes: int | None = None
    options: dict[str, Any] = field(default_factory=dict)
    protocol: str = "within_subject"
    folds: int = 5
    repeats: int = 1
    seeds: list[int] = field(default_factory=lambda: [0])
    template_id: str = "csp_lda"
    objectives: list[str] = field(default_factory=lambda: list(_DEFAULT_OBJECTIVES))
    budget: int = 16
    population: int = 6
    proxy_folds: int = 2
    full_candidates: int = 3
    seed: int = 2026
    name: str = ""
    notes: str = ""


def _channel_plan(params: dict[str, Any], n_channels: int) -> tuple[int, bool]:
    ratio = float(params.get("channel_ratio", 1.0))
    used = int(round(n_channels * min(1.0, max(0.1, ratio))))
    used = max(2, min(n_channels, used)) if n_channels > 2 else n_channels
    return used, used < n_channels


def _bandpass(low: float, high: float, order: int) -> dict[str, Any]:
    return {
        "id": "bp",
        "type": "preprocess.bandpass",
        "params": {"low": float(low), "high": float(high), "order": int(order)},
    }


def _channel_node(used: int) -> dict[str, Any]:
    return {
        "id": "chan",
        "type": "preprocess.channel_select",
        "params": {"n_channels": int(used)},
    }


def _build_csp_lda(params: dict[str, Any], n_channels: int) -> tuple[dict[str, Any], dict[str, Any]]:
    used, select_channels = _channel_plan(params, n_channels)
    components = min(max(2, int(params["n_components"])), used)
    nodes = [_bandpass(params["low"], params["high"], params["order"])]
    edges = []
    source = "bp"
    if select_channels:
        nodes.append(_channel_node(used))
        edges.append({"source": "bp", "target": "chan"})
        source = "chan"
    nodes.append(
        {"id": "csp", "type": "feature.csp",
         "params": {"n_components": components, "reg": float(params["reg"])}}
    )
    nodes.append({"id": "lda", "type": "decoder.lda", "params": {"shrinkage": str(params["shrinkage"])}})
    edges.extend([{"source": source, "target": "csp"}, {"source": "csp", "target": "lda"}])
    return {"nodes": nodes, "edges": edges}, {**params, "n_components": components, "channels_used": used}


def _build_fbcsp_lda(params: dict[str, Any], n_channels: int) -> tuple[dict[str, Any], dict[str, Any]]:
    used, select_channels = _channel_plan(params, n_channels)
    components = min(max(1, int(params["n_components"])), used)
    nodes = []
    edges = []
    source = None
    if select_channels:
        nodes.append(_channel_node(used))
        source = "chan"
    nodes.append(
        {"id": "fbcsp", "type": "feature.fbcsp",
         "params": {"n_components": components, "reg": float(params["reg"])}}
    )
    nodes.append({"id": "lda", "type": "decoder.lda", "params": {"shrinkage": str(params["shrinkage"])}})
    if source:
        edges.append({"source": source, "target": "fbcsp"})
    edges.append({"source": "fbcsp", "target": "lda"})
    return {"nodes": nodes, "edges": edges}, {**params, "n_components": components, "channels_used": used}


def _build_riemann(params: dict[str, Any], n_channels: int) -> tuple[dict[str, Any], dict[str, Any]]:
    used, select_channels = _channel_plan(params, n_channels)
    nodes = [_bandpass(params["low"], params["high"], 4)]
    edges = []
    source = "bp"
    if select_channels:
        nodes.append(_channel_node(used))
        edges.append({"source": "bp", "target": "chan"})
        source = "chan"
    nodes.append(
        {"id": "mdm", "type": "decoder.riemann_mdm",
         "params": {"metric": str(params["metric"]), "covariance": str(params["covariance"])}}
    )
    edges.append({"source": source, "target": "mdm"})
    return {"nodes": nodes, "edges": edges}, {**params, "channels_used": used}


def _build_ts_lr(params: dict[str, Any], n_channels: int) -> tuple[dict[str, Any], dict[str, Any]]:
    used, select_channels = _channel_plan(params, n_channels)
    nodes = [_bandpass(params["low"], params["high"], 4)]
    edges = []
    source = "bp"
    if select_channels:
        nodes.append(_channel_node(used))
        edges.append({"source": "bp", "target": "chan"})
        source = "chan"
    nodes.append(
        {"id": "ts", "type": "decoder.tangent_space_lr",
         "params": {"metric": str(params["metric"]), "covariance": str(params["covariance"]),
                    "C": float(params["C"])}}
    )
    edges.append({"source": source, "target": "ts"})
    return {"nodes": nodes, "edges": edges}, {**params, "channels_used": used}


def _build_deep(decoder: str):
    def builder(params: dict[str, Any], n_channels: int) -> tuple[dict[str, Any], dict[str, Any]]:
        used, select_channels = _channel_plan(params, n_channels)
        nodes = []
        edges = []
        if select_channels:
            nodes.append(_channel_node(used))
        nodes.append(
            {
                "id": "net",
                "type": decoder,
                "params": {
                    "max_epochs": int(params["max_epochs"]),
                    "batch_size": int(params["batch_size"]),
                    "learning_rate": float(params["learning_rate"]),
                    "dropout": float(params["dropout"]),
                },
            }
        )
        return {"nodes": nodes, "edges": edges}, {**params, "channels_used": used}

    return builder


_SPACE_CSP = (
    ParamSpec("low", "低频截止 (Hz)", "float", 8.0, low=4.0, high=12.0, step=0.5),
    ParamSpec("high", "高频截止 (Hz)", "float", 30.0, low=20.0, high=40.0, step=0.5),
    ParamSpec("order", "滤波阶数", "int", 4, low=2, high=6, step=1),
    ParamSpec("n_components", "CSP 分量数", "int", 6, low=2, high=12, step=1),
    ParamSpec("reg", "CSP 正则", "select", "1e-6", options=("1e-6", "1e-4", "1e-3")),
    ParamSpec("shrinkage", "LDA 收缩", "select", "auto", options=("auto", "0.1", "0.3", "0.5")),
    ParamSpec("channel_ratio", "通道保留比例", "float", 1.0, low=0.5, high=1.0, step=0.05),
)

_SPACE_FBCSP = (
    ParamSpec("n_components", "每带 CSP 分量数", "int", 2, low=1, high=6, step=1),
    ParamSpec("reg", "CSP 正则", "select", "1e-6", options=("1e-6", "1e-4", "1e-3")),
    ParamSpec("shrinkage", "LDA 收缩", "select", "auto", options=("auto", "0.1", "0.3", "0.5")),
    ParamSpec("channel_ratio", "通道保留比例", "float", 1.0, low=0.5, high=1.0, step=0.05),
)

_SPACE_RIEMANN = (
    ParamSpec("low", "低频截止 (Hz)", "float", 8.0, low=4.0, high=12.0, step=0.5),
    ParamSpec("high", "高频截止 (Hz)", "float", 30.0, low=20.0, high=40.0, step=0.5),
    ParamSpec("metric", "黎曼度量", "select", "riemann", options=("riemann", "logeuclid")),
    ParamSpec("covariance", "协方差估计", "select", "oas", options=("oas", "lwf", "scm")),
    ParamSpec("channel_ratio", "通道保留比例", "float", 1.0, low=0.5, high=1.0, step=0.05),
)

_SPACE_TS = (
    ParamSpec("low", "低频截止 (Hz)", "float", 8.0, low=4.0, high=12.0, step=0.5),
    ParamSpec("high", "高频截止 (Hz)", "float", 30.0, low=20.0, high=40.0, step=0.5),
    ParamSpec("metric", "黎曼度量", "select", "riemann", options=("riemann", "logeuclid")),
    ParamSpec("covariance", "协方差估计", "select", "oas", options=("oas", "lwf", "scm")),
    ParamSpec("C", "LR 正则强度", "float", 1.0, low=0.01, high=10.0, log=True),
    ParamSpec("channel_ratio", "通道保留比例", "float", 1.0, low=0.5, high=1.0, step=0.05),
)

_SPACE_EEGNET = (
    ParamSpec("max_epochs", "最大训练轮数", "int", 40, low=20, high=80, step=10),
    ParamSpec("batch_size", "批大小", "select", 32, options=(16, 32, 64)),
    ParamSpec("learning_rate", "学习率", "float", 0.001, low=0.0001, high=0.01, log=True),
    ParamSpec("dropout", "Dropout", "float", 0.5, low=0.1, high=0.7, step=0.05),
    ParamSpec("channel_ratio", "通道保留比例", "float", 1.0, low=0.5, high=1.0, step=0.05),
)

_TEMPLATES: dict[str, HPOTemplate] = {
    template.id: template
    for template in (
        HPOTemplate(
            "csp_lda", "CSP + LDA", "搜索频带、CSP 分量、正则、LDA 收缩与通道保留比例。",
            "classical", _SPACE_CSP, _build_csp_lda,
        ),
        HPOTemplate(
            "fbcsp_lda", "FBCSP + LDA", "滤波器组 CSP 的分量数、正则、LDA 收缩与通道保留比例。",
            "classical", _SPACE_FBCSP, _build_fbcsp_lda,
        ),
        HPOTemplate(
            "riemann_mdm", "Riemann MDM", "搜索频带、黎曼度量、协方差估计与通道保留比例。",
            "riemann", _SPACE_RIEMANN, _build_riemann,
        ),
        HPOTemplate(
            "ts_lr", "Tangent Space + LR", "搜索频带、度量、协方差、LR 正则与通道保留比例。",
            "riemann", _SPACE_TS, _build_ts_lr,
        ),
        HPOTemplate(
            "eegnet", "EEGNet", "搜索学习率、dropout、批大小、训练轮数与通道保留比例；CPU 上较慢。",
            "deep", _SPACE_EEGNET, _build_deep("decoder.eegnet"), deep=True,
        ),
        HPOTemplate(
            "shallowconvnet", "ShallowConvNet", "搜索学习率、dropout、批大小、训练轮数与通道保留比例；CPU 上较慢。",
            "deep", _SPACE_EEGNET, _build_deep("decoder.shallowconvnet"), deep=True,
        ),
    )
}


def get_template(template_id: str) -> HPOTemplate:
    template = _TEMPLATES.get(str(template_id))
    if template is None:
        raise ValueError(f"unknown HPO template: {template_id}")
    return template


def template_catalog() -> list[dict[str, Any]]:
    return [template.describe() for template in _TEMPLATES.values()]


_SEARCH_KINDS = {"int", "float", "select"}
_STRUCTURAL_PARAMS = {("preprocess.filter_bank_split", "n_bands")}
_MAX_SEARCH_DIMS = 16
_DEEP_NODES = {"decoder.eegnet", "decoder.shallowconvnet"}


def _catalog_index() -> dict[str, dict[str, Any]]:
    return {item["type"]: item for item in node_catalog()}


def _number(
    value: Any, node_id: str, param: str, field: str
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{node_id}.{param} 的 {field} 必须是数字") from exc
    if not math.isfinite(number):
        raise ValueError(f"{node_id}.{param} 的 {field} 必须是有限数字")
    return number


def normalize_search_space(
    spec: dict[str, Any], dims: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Validate saved search dims against the node catalog and graph params."""
    base = normalize_graph(spec)
    if not isinstance(dims, (list, tuple)):
        raise ValueError("搜索空间必须是数组")
    if not dims:
        raise ValueError("搜索空间至少需要一个可调参数")
    if len(dims) > _MAX_SEARCH_DIMS:
        raise ValueError(f"搜索空间最多支持 {_MAX_SEARCH_DIMS} 个可调参数")
    nodes = {node["id"]: node for node in base["nodes"]}
    catalog = _catalog_index()
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(dims):
        if not isinstance(raw, dict):
            raise ValueError(f"第 {index + 1} 个搜索维度必须是对象")
        node_id = str(raw.get("node_id") or "").strip()
        param = str(raw.get("param") or "").strip()
        if not node_id or not param:
            raise ValueError(f"第 {index + 1} 个搜索维度缺少 node_id 或 param")
        node = nodes.get(node_id)
        if node is None:
            raise ValueError(f"搜索维度引用了不存在的节点: {node_id}")
        item = catalog.get(node["type"])
        if item is None:
            raise ValueError(f"节点 {node_id} 的类型 {node['type']} 不在节点库中")
        if item.get("plugin"):
            raise ValueError(
                f"节点 {node_id}（{item['label']}）是插件节点；搜索空间暂只支持内置节点"
            )
        definitions = {entry["name"]: entry for entry in item["params"]}
        definition = definitions.get(param)
        if definition is None:
            raise ValueError(f"节点 {node_id}（{item['label']}）没有参数 {param}")
        if (node["type"], param) in _STRUCTURAL_PARAMS or (
            item.get("dynamic_ports") and param == "n_bands"
        ):
            raise ValueError(
                f"{item['label']}.{param} 会改变输出端口，不能作为搜索维度"
            )
        kind = str(definition["type"])
        if kind not in _SEARCH_KINDS:
            raise ValueError(
                f"{item['label']}.{param} 的类型是 {kind}；只支持 int/float/select"
            )
        if (node_id, param) in seen:
            raise ValueError(f"搜索维度重复: {node_id}.{param}")
        seen.add((node_id, param))
        current = node["params"].get(param, definition.get("default"))
        label = f"{node.get('label') or item['label']} · {definition.get('label') or param}"
        entry: dict[str, Any] = {
            "node_id": node_id,
            "node_type": node["type"],
            "param": param,
            "kind": kind,
            "label": label,
        }
        if kind == "select":
            schema_options = [str(value) for value in definition.get("options") or []]
            raw_options = raw.get("options")
            options = (
                [str(value) for value in raw_options]
                if raw_options is not None
                else list(schema_options)
            )
            if len(set(options)) != len(options):
                raise ValueError(f"{node_id}.{param} 的选项不能重复")
            if len(options) < 2:
                raise ValueError(f"{node_id}.{param} 至少需要两个候选选项")
            invalid = [value for value in options if value not in schema_options]
            if invalid:
                raise ValueError(
                    f"{node_id}.{param} 包含非法选项: {', '.join(invalid)}"
                )
            default = str(current)
            if default not in options:
                raise ValueError(
                    f"{node_id}.{param} 当前值 {default} 不在搜索选项中；请保留当前值"
                )
            entry.update({"options": options, "default": default})
        else:
            if "min" not in definition and raw.get("low") is None:
                raise ValueError(f"{node_id}.{param} 缺少下界 low")
            if "max" not in definition and raw.get("high") is None:
                raise ValueError(f"{node_id}.{param} 缺少上界 high")
            low = _number(
                raw.get("low", definition.get("min")), node_id, param, "下界"
            )
            high = _number(
                raw.get("high", definition.get("max")), node_id, param, "上界"
            )
            if "min" in definition and low < float(definition["min"]):
                raise ValueError(
                    f"{node_id}.{param} 的下界不能小于节点允许的最小值 {definition['min']}"
                )
            if "max" in definition and high > float(definition["max"]):
                raise ValueError(
                    f"{node_id}.{param} 的上界不能大于节点允许的最大值 {definition['max']}"
                )
            if low >= high:
                raise ValueError(f"{node_id}.{param} 的下界必须小于上界")
            if node["type"] in {"feature.csp", "feature.fbcsp"}:
                if param == "n_components" and low < 2:
                    raise ValueError(f"{node_id}.n_components 的下界不能小于 2")
                if param == "reg" and low <= 0:
                    raise ValueError(
                        f"{node_id}.reg 的下界必须大于 0；CAR 之后协方差可能秩亏"
                    )
            default = float(current)
            if default < low or default > high:
                raise ValueError(
                    f"{node_id}.{param} 当前值 {current} 不在 [{low:g}, {high:g}] 内"
                )
            if kind == "int":
                if not float(low).is_integer() or not float(high).is_integer():
                    raise ValueError(f"{node_id}.{param} 是整数参数，上下界必须为整数")
                step = int(raw.get("step", definition.get("step", 1)))
                if step <= 0:
                    raise ValueError(f"{node_id}.{param} 的步长必须为正整数")
                if int(high) - int(low) < step:
                    raise ValueError(f"{node_id}.{param} 的区间小于一个步长")
                entry.update(
                    {
                        "low": int(low),
                        "high": int(high),
                        "step": step,
                        "default": int(current),
                    }
                )
            else:
                step_value = raw.get("step", definition.get("step"))
                step = (
                    None
                    if step_value in (None, "")
                    else _number(step_value, node_id, param, "步长")
                )
                if step is not None and step <= 0:
                    raise ValueError(f"{node_id}.{param} 的步长必须为正数")
                if step is not None and high - low < step:
                    raise ValueError(f"{node_id}.{param} 的区间小于一个步长")
                entry.update(
                    {
                        "low": float(low),
                        "high": float(high),
                        **({"step": float(step)} if step is not None else {}),
                        "default": float(current),
                    }
                )
        normalized.append(entry)
    return normalized


def custom_template(
    pipeline_id: int | str,
    name: str,
    description: str,
    spec: dict[str, Any],
    dims: Sequence[dict[str, Any]],
) -> HPOTemplate:
    """Build an HPO template from a saved graph plus explicit search dims."""
    base = normalize_graph(spec)
    normalized = normalize_search_space(base, dims)
    space = tuple(
        ParamSpec(
            name=f"{dim['node_id']}::{dim['param']}",
            label=str(dim["label"]),
            kind=str(dim["kind"]),
            default=dim["default"],
            low=dim.get("low"),
            high=dim.get("high"),
            step=dim.get("step"),
            options=tuple(dim.get("options") or ()),
        )
        for dim in normalized
    )

    def builder(
        params: dict[str, Any], n_channels: int
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        candidate = deepcopy(base)
        for dim in normalized:
            node = next(item for item in candidate["nodes"] if item["id"] == dim["node_id"])
            node["params"][dim["param"]] = params[f"{dim['node_id']}::{dim['param']}"]
        used = int(n_channels)
        for node in candidate["nodes"]:
            if node["type"] == "preprocess.channel_select":
                used = min(used, int(node["params"]["n_channels"]))
        effective = {
            f"{dim['node_id']}::{dim['param']}": params[
                f"{dim['node_id']}::{dim['param']}"
            ]
            for dim in normalized
        }
        effective["channels_used"] = used
        return candidate, effective

    template_id = f"custom:{pipeline_id}"
    deep = any(node["type"] in _DEEP_NODES for node in base["nodes"])
    canonical_spec = _canonical(base)
    canonical_space = _canonical(normalized)
    return HPOTemplate(
        template_id,
        name,
        description or f"自定义丹方 · {len(normalized)} 维搜索空间",
        "custom",
        space,
        builder,
        deep=deep,
        source={
            "kind": "custom_pipeline",
            "pipeline_id": int(pipeline_id),
            "spec_hash": hashlib.sha256(canonical_spec.encode("utf-8")).hexdigest()[:16],
            "space_hash": hashlib.sha256(
                canonical_space.encode("utf-8")
            ).hexdigest()[:16],
            "dims": len(normalized),
        },
    )


def objective_catalog() -> list[dict[str, str]]:
    return [{"name": name, **meta} for name, meta in OBJECTIVES.items()]


def _decode(space: Sequence[ParamSpec], vector: np.ndarray) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for spec, raw in zip(space, vector):
        value = min(1.0, max(0.0, float(raw)))
        if spec.kind == "select":
            index = min(len(spec.options) - 1, int(value * len(spec.options)))
            params[spec.name] = spec.options[index]
        elif spec.kind == "int":
            low, high = int(spec.low), int(spec.high)
            step = max(1, int(spec.step or 1))
            steps = int(round(value * (high - low) / step))
            params[spec.name] = int(min(high, max(low, low + steps * step)))
        else:
            if spec.log:
                low, high = math.log(float(spec.low)), math.log(float(spec.high))
                number = math.exp(low + value * (high - low))
            else:
                number = float(spec.low) + value * (float(spec.high) - float(spec.low))
            if spec.step:
                number = round(round(number / float(spec.step)) * float(spec.step), 6)
            params[spec.name] = float(min(float(spec.high), max(float(spec.low), number)))
    return params


def _canonical(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _metrics(row: dict[str, Any], channels_used: int) -> dict[str, float]:
    splits = row.get("splits") or []
    durations = [float(split.get("duration_ms", 0.0)) for split in splits]
    return {
        "balanced_accuracy": float(row["balanced_accuracy_mean"]),
        "balanced_accuracy_std": float(row.get("balanced_accuracy_std", 0.0)),
        "cohen_kappa": float(row.get("cohen_kappa_mean", 0.0)),
        "duration_ms": float(sum(durations) / len(durations)) if durations else 0.0,
        "channel_count": float(channels_used),
        "n_splits": float(len(splits)),
    }


def _dominates(left: dict[str, Any], right: dict[str, Any], objectives: Sequence[str]) -> bool:
    better_or_equal = True
    strictly_better = False
    for name in objectives:
        direction = OBJECTIVES[name]["direction"]
        a = float(left["metrics"][name])
        b = float(right["metrics"][name])
        if direction == "max":
            better_or_equal = better_or_equal and a >= b
            strictly_better = strictly_better or a > b
        else:
            better_or_equal = better_or_equal and a <= b
            strictly_better = strictly_better or a < b
    return better_or_equal and strictly_better


def _pareto(candidates: Sequence[dict[str, Any]], objectives: Sequence[str]) -> list[dict[str, Any]]:
    usable = [item for item in candidates if item.get("metrics")]
    front: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in usable:
        key = _canonical(item["params"])
        if key in seen:
            continue
        if any(
            other is not item
            and _canonical(other["params"]) != key
            and _dominates(other, item, objectives)
            for other in usable
        ):
            continue
        seen.add(key)
        front.append(item)
    return front


def _guide_ranks(
    candidates: Sequence[dict[str, Any]], objectives: Sequence[str]
) -> dict[int, float]:
    usable = [item for item in candidates if item.get("metrics")]
    if not usable:
        return {}
    ranks = np.zeros((len(usable), len(objectives)), dtype=float)
    for column, name in enumerate(objectives):
        values = np.asarray([float(item["metrics"][name]) for item in usable])
        ranks[:, column] = (
            rankdata(-values, method="average")
            if OBJECTIVES[name]["direction"] == "max"
            else rankdata(values, method="average")
        )
    means = ranks.mean(axis=1)
    return {item["index"]: float(means[position]) for position, item in enumerate(usable)}


def _select_full_candidates(
    proxy: Sequence[dict[str, Any]], objectives: Sequence[str], limit: int
) -> list[dict[str, Any]]:
    front = _pareto(proxy, objectives) or [item for item in proxy if item.get("metrics")]
    ranks = _guide_ranks(proxy, objectives)
    ordered = sorted(
        front,
        key=lambda item: (
            ranks.get(item["index"], 1e9),
            -float(item["metrics"]["balanced_accuracy"]),
        ),
    )
    return ordered[: max(0, limit)]


def _best(candidates: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    usable = [item for item in candidates if item.get("metrics")]
    if not usable:
        return None
    return max(
        usable,
        key=lambda item: (
            float(item["metrics"]["balanced_accuracy"]),
            float(item["metrics"]["cohen_kappa"]),
            -float(item["metrics"]["channel_count"]),
            -float(item["metrics"]["duration_ms"]),
        ),
    )


def _proxy_indices(n_splits: int, wanted: int) -> list[int]:
    if n_splits <= 0:
        return []
    count = max(1, min(n_splits, int(wanted)))
    return sorted({int(value) for value in np.linspace(0, n_splits - 1, count).round()})


def _candidate_record(
    index: int,
    stage: str,
    generation: int,
    role: str,
    vector: np.ndarray,
    params: dict[str, Any],
    metrics: dict[str, float] | None,
    error: str | None,
    duration_ms: float,
    splits: Sequence[Split],
) -> dict[str, Any]:
    return {
        "index": index,
        "stage": stage,
        "generation": generation,
        "role": role,
        # Full re-evaluation re-decodes this vector; rounding it here can move
        # fine-grained dims (e.g. reg with step 1e-6) away from the evaluated params.
        "vector": [float(value) for value in vector],
        "params": params,
        "metrics": metrics,
        "error": error,
        "duration_ms": round(float(duration_ms), 1),
        "n_splits": len(splits),
        "split_signatures": [split.signature()[:16] for split in splits],
    }


def _problem_signature(epochs: Epochs) -> str:
    return epochs.signature()


_BAND_NODES = {
    "preprocess.bandpass",
    "preprocess.filter_bank_split",
    "feature.bandpower",
}
_CSP_NODES = {"feature.csp", "feature.fbcsp"}


def _validate_candidate(spec: dict[str, Any], n_channels: int, sfreq: float) -> None:
    nodes = {str(node.get("id")): node for node in spec.get("nodes") or []}
    order = [str(item) for item in spec.get("topology") or nodes]
    sample_rate = float(sfreq)
    for node_id in order:
        node = nodes.get(node_id)
        if node is None:
            continue
        node_type = str(node.get("type"))
        params = node.get("params") or {}
        if node_type == "preprocess.resample":
            target = float(params.get("target_sfreq", sample_rate))
            if target > 0:
                sample_rate = target
        if node_type in _BAND_NODES:
            low = float(params.get("low", 0.0))
            high = float(params.get("high", 0.0))
            nyquist = sample_rate / 2.0
            if low >= high:
                raise ValueError(
                    f"{node_id} 的频带下界 {low:g} Hz 必须小于上界 {high:g} Hz"
                )
            if high > nyquist + 1e-9:
                raise ValueError(
                    f"{node_id} 的上界 {high:g} Hz 超过 Nyquist 频率 {nyquist:g} Hz"
                )
        if node_type in _CSP_NODES:
            n_components = int(params.get("n_components", 0))
            if n_components < 2:
                raise ValueError(f"{node_id}.n_components 必须不小于 2")
            if n_components > n_channels:
                raise ValueError(
                    f"{node_id}.n_components ({n_components}) 不能超过通道数 {n_channels}"
                )
            if float(params.get("reg", 0.0)) <= 0:
                raise ValueError(f"{node_id}.reg 必须大于 0")


_NUMBER_TOKEN = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _failure_summary(history: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    failed = 0
    for item in history:
        error = item.get("error")
        if not error:
            continue
        failed += 1
        text = str(error)
        message = text.split(":", 1)[1].strip() if ":" in text else text
        key = _NUMBER_TOKEN.sub("#", message)
        group = groups.setdefault(key, {"reason": message, "count": 0, "example": text})
        group["count"] += 1
    ordered = sorted(groups.values(), key=lambda item: (-item["count"], item["reason"]))
    return {
        "evaluations": len(history),
        "failed": failed,
        "groups": ordered[:8],
    }


def run_hpo(
    registry: DatasetRegistry,
    request: HPORequest,
    on_progress: Callable[[int, int, dict[str, Any]], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    epochs: Epochs | None = None,
    template: HPOTemplate | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    template = template or get_template(request.template_id)
    objectives = []
    for name in request.objectives or []:
        if name not in OBJECTIVES:
            raise ValueError(f"unknown objective: {name}")
        if name not in objectives:
            objectives.append(name)
    if "balanced_accuracy" not in objectives:
        objectives.insert(0, "balanced_accuracy")
    if request.budget < 4:
        raise ValueError("HPO budget must be at least 4 evaluations")
    if request.population < 3:
        raise ValueError("HPO population must be at least 3")
    if request.population >= request.budget:
        raise ValueError("HPO population must be smaller than the budget")
    if request.full_candidates < 1:
        raise ValueError("HPO full_candidates must be at least 1")
    if request.proxy_folds < 1:
        raise ValueError("HPO proxy_folds must be at least 1")

    epochs = epochs if epochs is not None else registry.load(
        request.dataset_id,
        subject_ids=request.subject_ids or None,
        classes=request.classes,
        options=request.options,
    )
    seeds = list(request.seeds) or [0]
    splits = build_protocol(
        request.protocol,
        folds=request.folds,
        repeats=request.repeats,
        seeds=tuple(seeds),
    ).split(epochs)
    proxy_indices = _proxy_indices(len(splits), request.proxy_folds)
    proxy_splits = [splits[index] for index in proxy_indices]
    n_channels = len(epochs.channel_names)
    rng = np.random.default_rng(int(request.seed))

    full_reserve = min(int(request.full_candidates), int(request.budget) - 1)
    proxy_budget = max(1, int(request.budget) - full_reserve)
    population = min(int(request.population), proxy_budget)
    history: list[dict[str, Any]] = []
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    counts = {"proxy": 0, "full": 0}

    def report(stage: str, params: dict[str, Any] | None, extra: dict[str, Any] | None = None) -> None:
        if on_progress is None:
            return
        done = counts["proxy"] + counts["full"]
        on_progress(
            done,
            int(request.budget),
            {
                "stage": stage,
                "params": params or {},
                "proxy": counts["proxy"],
                "full": counts["full"],
                **(extra or {}),
            },
        )

    def evaluate(
        stage: str,
        vector: np.ndarray,
        generation: int,
        role: str,
        stage_splits: Sequence[Split],
    ) -> dict[str, Any] | None:
        params = _decode(template.space, vector)
        try:
            spec, effective = template.build(params, n_channels)
            _validate_candidate(spec, n_channels, float(epochs.sfreq))
        except (GraphValidationError, ValueError, TypeError) as exc:
            record = _candidate_record(
                len(history), stage, generation, role, vector, params, None,
                f"{type(exc).__name__}: {exc}", 0.0, stage_splits,
            )
            history.append(record)
            counts[stage] += 1
            report(stage, record["params"], {"error": record["error"]})
            return record
        key = (stage, _canonical(effective))
        if key in cache:
            report(stage, cache[key]["params"], {"cached": True})
            return cache[key]
        if should_cancel and should_cancel():
            raise RunCancelled("HPO 已取消")
        started_candidate = time.perf_counter()
        try:
            pipeline = build_graph_pipeline(f"hpo-{template.id}", spec)
            rows = run_arena(
                epochs,
                [pipeline],
                list(stage_splits),
                should_cancel=should_cancel,
                cache=registry.artifact_cache,
                data_signature=epochs.signature(),
            )
            metrics = _metrics(rows[0], int(effective["channels_used"]))
            error = None
        except RunCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 - candidate failures are part of the audit
            metrics = None
            error = f"{type(exc).__name__}: {exc}"
        record = _candidate_record(
            len(history), stage, generation, role, vector, effective, metrics, error,
            (time.perf_counter() - started_candidate) * 1000.0, stage_splits,
        )
        history.append(record)
        cache[key] = record
        counts[stage] += 1
        report(stage, effective, {"error": error, "metrics": metrics})
        return record

    vectors = [template.default_vector()]
    for _ in range(max(0, population - 1)):
        vectors.append(rng.random(len(template.space)))
    proxy_records: list[dict[str, Any]] = []
    for vector in vectors:
        before = len(history)
        record = evaluate("proxy", vector, 0, "init", proxy_splits)
        if record and len(history) > before:
            proxy_records.append(record)

    iteration = 0
    max_generations = max(4, math.ceil(proxy_budget / max(1, population)) * 5 + 5)
    while counts["proxy"] < proxy_budget and iteration < max_generations:
        leaders = sorted(
            (item for item in proxy_records if item.get("metrics")),
            key=lambda item: (
                _guide_ranks(proxy_records, objectives).get(item["index"], 1e9),
                -float(item["metrics"]["balanced_accuracy"]),
            ),
        )[:3]
        if not leaders:
            break
        while len(leaders) < 3:
            leaders.append(leaders[-1])
        alpha, beta, delta = (np.asarray(item["vector"], dtype=float) for item in leaders)
        a = 2.0 - 2.0 * min(iteration, 10) / 10.0
        new_vectors: list[np.ndarray] = []
        progressed = False
        for position in range(population):
            if counts["proxy"] >= proxy_budget:
                break
            current = np.asarray(vectors[position % len(vectors)], dtype=float)
            movement = []
            for leader in (alpha, beta, delta):
                r1, r2 = rng.random(len(template.space)), rng.random(len(template.space))
                A = 2.0 * a * r1 - a
                C = 2.0 * r2
                movement.append(leader - A * np.abs(C * leader - current))
            candidate = np.clip(np.mean(movement, axis=0), 0.0, 1.0)
            role = "gwo"
            if rng.random() < 0.3:
                sigma = 0.02 + 0.12 * (1.0 - iteration / max_generations)
                candidate = np.clip(alpha + rng.normal(0.0, sigma, len(template.space)), 0.0, 1.0)
                role = "local"
            new_vectors.append(candidate)
            before_history = len(history)
            record = evaluate("proxy", candidate, iteration + 1, role, proxy_splits)
            if record and len(history) > before_history:
                proxy_records.append(record)
            progressed = progressed or len(history) > before_history
        if new_vectors:
            vectors = new_vectors
        iteration += 1
        if not progressed:
            while counts["proxy"] < proxy_budget:
                before = len(history)
                record = evaluate(
                    "proxy", rng.random(len(template.space)), iteration, "random_fill", proxy_splits
                )
                if record and len(history) > before:
                    proxy_records.append(record)
            break

    full_selection = _select_full_candidates(proxy_records, objectives, full_reserve)
    full_records: list[dict[str, Any]] = []
    for record in full_selection:
        if counts["proxy"] + counts["full"] >= int(request.budget):
            break
        full = evaluate(
            "full",
            np.asarray(record["vector"], dtype=float),
            record["generation"],
            "pareto",
            splits,
        )
        if full:
            full_records.append(full)

    final_candidates = [item for item in full_records if item.get("metrics")]
    if not final_candidates:
        final_candidates = [item for item in proxy_records if item.get("metrics")]
    front = _pareto(final_candidates, objectives)
    best = _best(front or final_candidates)
    best_definition = None
    best_graph = None
    if best:
        spec, _ = template.build(best["params"], n_channels)
        pipeline = build_graph_pipeline(f"hpo-{template.id}", spec)
        best_graph = spec
        best_definition = {
            **pipeline.describe(),
            "name": f"HPO · {template.name}",
            "family": "hpo",
            "note": f"{request.name or template.name} · 由 {template.name} 元启发式搜索得到",
        }

    manifest = registry.manifest(request.dataset_id, epochs)
    failures = _failure_summary(history)
    duration_ms = (time.perf_counter() - started) * 1000.0
    result = {
        "format": "neurokiln-hpo/v1",
        "name": request.name or f"HPO · {template.name}",
        "template": template.describe(),
        "objectives": [{"name": name, **OBJECTIVES[name]} for name in objectives],
        "search": {
            "algorithm": "GWO + elite local search + Pareto archive",
            "seed": int(request.seed),
            "budget": int(request.budget),
            "evaluations": counts["proxy"] + counts["full"],
            "proxy_evaluations": counts["proxy"],
            "full_evaluations": counts["full"],
            "failed_evaluations": failures["failed"],
            "population": population,
            "generations": iteration,
            "proxy_folds": len(proxy_splits),
            "full_splits": len(splits),
            "full_candidates": full_reserve,
            "proxy_indices": proxy_indices,
        },
        "dataset": {
            "id": request.dataset_id,
            "name": manifest.get("name", request.dataset_id),
            "signature": manifest.get("data_signature"),
            "n_trials": int(epochs.n_trials),
            "n_channels": n_channels,
            "n_classes": int(epochs.n_classes),
            "subjects": sorted(set(epochs.subjects)),
            "dataset_version": epochs.dataset_version,
            "selection": {
                "subject_ids": request.subject_ids,
                "classes": request.classes,
                "options": request.options,
            },
        },
        "protocol": {
            "name": request.protocol,
            "folds": int(request.folds),
            "repeats": int(request.repeats),
            "seeds": seeds,
        },
        "history": history,
        "failures": failures,
        "pareto": [
            {
                "index": item["index"],
                "stage": item["stage"],
                "params": item["params"],
                "metrics": item["metrics"],
            }
            for item in front
        ],
        "best": None if best is None else {
            "index": best["index"],
            "stage": best["stage"],
            "params": best["params"],
            "metrics": best["metrics"],
            "graph": best_graph,
            "definition": best_definition,
        },
        "duration_ms": duration_ms,
        "note": (
            "候选先在代理折上筛选，Pareto 非支配候选再用完整协议复评；"
            "所有参数与失败原因都记录在 history 中。"
        ),
        "cancelled": False,
    }
    if should_cancel and should_cancel():
        result["cancelled"] = True
    return result


__all__ = [
    "HPORequest",
    "HPOTemplate",
    "OBJECTIVES",
    "ParamSpec",
    "custom_template",
    "get_template",
    "normalize_search_space",
    "objective_catalog",
    "run_hpo",
    "template_catalog",
]
