from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from .data.adapters import DatasetRegistry
from .data.model import Epochs
from .evaluate import run_arena
from .graph import GraphPipeline, GraphValidationError, build_graph_pipeline, normalize_graph
from .pipeline import PIPELINE_DEFINITIONS, build_baseline
from .protocol import build_protocol
from .stats import compare_pipelines
from .store import ExperimentStore


@dataclass
class ExperimentRequest:
    dataset_id: str
    subject_ids: list[str] = field(default_factory=list)
    classes: int | None = None
    options: dict[str, Any] = field(default_factory=dict)
    protocol: str = "within_subject"
    folds: int = 5
    repeats: int = 1
    seeds: list[int] = field(default_factory=lambda: [0])
    pipelines: list[str] = field(default_factory=lambda: ["csp_lda"])
    name: str = ""
    notes: str = ""
    save: bool = True
    use_cache: bool = True
    job_id: str | None = None


def _dataset_public(manifest: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "id", "name", "paradigm", "paradigm_meta", "license", "citation", "dataset_version",
        "data_signature", "loaded_n_trials", "loaded_channels", "loaded_sfreq",
        "loaded_classes", "label_names", "subjects", "sessions", "tmin", "tmax",
    )
    return {key: manifest.get(key) for key in keys}


def infer_spec_paradigm(spec: dict[str, Any]) -> str:
    types = {str(node.get("type", "")) for node in spec.get("nodes", [])}
    if types & {"decoder.ssvep_cca", "decoder.ssvep_fbcca"}:
        return "SSVEP"
    if "decoder.xdawn_lda" in types:
        return "P300"
    return "MI"


def pipeline_paradigms(definition: dict[str, Any]) -> set[str]:
    label = str(definition.get("paradigm") or "")
    if not label:
        graph = definition.get("graph")
        if isinstance(graph, dict) and graph.get("nodes"):
            label = infer_spec_paradigm(graph)
        else:
            label = "MI"
    return {part.strip().upper() for part in label.split("/") if part.strip()}


def ensure_paradigm_compatible(
    definition: dict[str, Any], paradigm: str, identifier: str | None = None
) -> None:
    supported = pipeline_paradigms(definition)
    current = str(paradigm or "MI").upper()
    if current not in supported:
        name = identifier or definition.get("id") or "pipeline"
        raise ValueError(
            f"算法 {name} 仅支持 {'/'.join(sorted(supported))} 范式，"
            f"当前数据集范式为 {current}；请选择匹配的算法。"
        )


def resolve_pipelines(
    pipeline_ids: list[str],
    store: ExperimentStore | None = None,
    plugins: Any | None = None,
) -> tuple[list[dict[str, Any]], list[Any]]:
    definitions = {item["id"]: item for item in PIPELINE_DEFINITIONS}
    resolved: list[dict[str, Any]] = []
    instances = []
    if not pipeline_ids:
        raise ValueError("至少选择一个算法")
    for identifier in pipeline_ids:
        if identifier.startswith("plugin:"):
            if plugins is None:
                raise ValueError(f"插件算法需要插件服务: {identifier}")
            try:
                definition = plugins.definition(identifier)
                pipeline = build_graph_pipeline(identifier, definition["graph"])
            except GraphValidationError as exc:
                raise ValueError(f"插件算法 {identifier} 校验失败: {exc}") from exc
            resolved.append(definition)
            instances.append(pipeline)
            continue
        if identifier.startswith("custom:"):
            if store is None:
                raise ValueError(f"自定义算法需要实验库支持: {identifier}")
            try:
                custom_id = int(identifier.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(f"无效的自定义算法 ID: {identifier}") from exc
            custom = store.get_custom_pipeline(custom_id)
            if custom is None:
                raise ValueError(f"自定义算法不存在: {identifier}")
            try:
                spec = normalize_graph(custom["spec"])
                pipeline = build_graph_pipeline(identifier, spec)
            except GraphValidationError as exc:
                raise ValueError(f"自定义算法 {custom['name']} 校验失败: {exc}") from exc
            definition = {
                **pipeline.describe(),
                "name": custom["name"],
                "note": custom["description"] or "用户自定义 DAG",
                "paradigm": infer_spec_paradigm(spec),
                "graph": spec,
            }
        else:
            if identifier not in definitions:
                raise ValueError(f"unknown pipeline(s): {identifier}")
            definition = {**definitions[identifier]}
            pipeline = build_baseline(identifier)
        resolved.append(definition)
        instances.append(pipeline)
    return resolved, instances


def resolve_pipeline_definitions(
    definitions: list[dict[str, Any]],
    plugins: Any | None = None,
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Rebuild pipelines from an exported experiment bundle."""
    resolved: list[dict[str, Any]] = []
    instances: list[Any] = []
    for definition in definitions:
        identifier = str(definition.get("id") or "")
        if not identifier:
            raise ValueError("pipeline definition is missing an id")
        if definition.get("plugin"):
            if plugins is None:
                raise ValueError(
                    f"该实验使用插件算法 {identifier}，请在支持插件的环境中复现"
                )
            plugins.require_definition(definition)
        graph = definition.get("graph")
        if isinstance(graph, dict) and graph.get("nodes"):
            spec = normalize_graph(graph)
            pipeline = build_graph_pipeline(identifier, spec)
            resolved.append(
                {
                    **definition,
                    "graph": spec,
                    "paradigm": definition.get("paradigm") or infer_spec_paradigm(spec),
                }
            )
        else:
            if identifier.startswith("custom:"):
                raise ValueError(
                    f"自定义算法 {identifier} 缺少 graph 定义，无法脱离实验库复现"
                )
            resolved.append({**definition})
            pipeline = build_baseline(identifier)
        instances.append(pipeline)
    return resolved, instances


def experiment_name(manifest: dict[str, Any], protocol: str) -> str:
    return (
        f"{manifest.get('name', manifest.get('id', 'experiment'))} · {protocol} · "
        f"{datetime.now().strftime('%m-%d %H:%M')}"
    )


def run_experiment(
    registry: DatasetRegistry,
    store: ExperimentStore | None,
    request: ExperimentRequest,
    on_progress: Callable[[int, int, dict], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    epochs: Epochs | None = None,
    plugins: Any | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    epochs = epochs if epochs is not None else registry.load(
        request.dataset_id,
        subject_ids=request.subject_ids or None,
        classes=request.classes,
        options=request.options,
    )
    manifest = registry.manifest(request.dataset_id, epochs)
    manifest["dataset_version"] = epochs.dataset_version
    manifest["selection"] = {
        "subject_ids": request.subject_ids,
        "classes": request.classes,
        "options": request.options,
    }
    splits = build_protocol(
        request.protocol,
        folds=request.folds,
        repeats=request.repeats,
        seeds=tuple(request.seeds),
    ).split(epochs)
    definitions, pipelines = resolve_pipelines(request.pipelines, store, plugins)
    for definition in definitions:
        ensure_paradigm_compatible(definition, epochs.paradigm)
    cache = registry.artifact_cache if request.use_cache else None
    cache_before = cache.stats() if cache is not None else None
    rows = run_arena(
        epochs,
        pipelines,
        splits,
        on_progress=on_progress,
        should_cancel=should_cancel,
        cache=cache,
        data_signature=str(manifest["data_signature"]),
    )
    cache_activity = None
    if cache is not None and cache_before is not None:
        cache_after = cache.stats()
        cache_activity = {
            "hits": cache_after["hits"] - cache_before["hits"],
            "misses": cache_after["misses"] - cache_before["misses"],
            "writes": cache_after["writes"] - cache_before["writes"],
            "evictions": cache_after["evictions"] - cache_before["evictions"],
            "entries": cache_after["entries"],
            "bytes": cache_after["bytes"],
            "enabled": cache_after["enabled"],
        }
    duration_ms = (time.perf_counter() - started) * 1000.0
    statistics = compare_pipelines(rows)
    intervals = (statistics.get("ci") or {}).get("pipelines") or {}
    for row in rows:
        interval = intervals.get(str(row.get("pipeline")))
        if interval and row.get("balanced_accuracy_ci95") is None:
            row["balanced_accuracy_ci95"] = interval.get("ci95")
    protocol_params = {
        "folds": request.folds,
        "repeats": request.repeats,
        "seeds": request.seeds,
    }
    experiment_id = None
    name = request.name.strip() or experiment_name(manifest, request.protocol)
    if request.save:
        if store is None:
            raise ValueError("save=True 需要实验库")
        store.save_dataset(
            dataset_id=request.dataset_id,
            name=str(manifest.get("name", request.dataset_id)),
            adapter=str(manifest.get("adapter", "unknown")),
            signature=str(manifest["data_signature"]),
            manifest=manifest,
        )
        experiment_id = store.record_experiment(
            name=name,
            dataset_id=request.dataset_id,
            dataset_name=str(manifest.get("name", request.dataset_id)),
            dataset_version=str(manifest.get("dataset_version", "")),
            dataset_signature=str(manifest["data_signature"]),
            dataset_manifest=manifest,
            protocol=request.protocol,
            protocol_params=protocol_params,
            pipelines=definitions,
            seeds=request.seeds,
            arena_rows=rows,
            duration_ms=duration_ms,
            notes=request.notes,
            job_id=request.job_id,
        )
    return {
        "experiment_id": experiment_id,
        "saved": experiment_id is not None,
        "name": name,
        "dataset": _dataset_public(manifest),
        "manifest": manifest,
        "protocol": request.protocol,
        "protocol_params": protocol_params,
        "n_splits": len(splits),
        "rows": rows,
        "statistics": statistics,
        "duration_ms": duration_ms,
        "cache": cache_activity,
        "note": "所有预处理与分类器统计量均只在训练折拟合；预测来自留出折。",
    }


__all__ = [
    "ExperimentRequest",
    "experiment_name",
    "infer_spec_paradigm",
    "resolve_pipeline_definitions",
    "resolve_pipelines",
    "run_experiment",
]
