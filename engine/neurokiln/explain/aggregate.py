"""Aggregate explanation documents across folds, repeats and seeds.

The input is a list of ``global.json`` payloads produced by
:func:`neurokiln.explain.run_explanation`. Aggregation is descriptive: every
channel, band and metric is averaged over the documents that actually contain
it, and ``n`` is reported next to each statistic so a fold with a skipped
method never silently changes the denominator.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

FORMAT = "neurokiln.explain.aggregate"
FORMAT_VERSION = 1


class AggregateError(ValueError):
    """Aggregation rejected; ``code`` is stable for API clients."""

    def __init__(self, message: str, code: str):
        super().__init__(message)
        self.code = code


def _number(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _summary(values: Iterable[Any]) -> dict[str, Any]:
    numbers = np.asarray(
        [item for item in (_number(value) for value in values) if item is not None],
        dtype=float,
    )
    if not numbers.size:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": round(float(numbers.mean()), 6),
        "std": round(float(numbers.std(ddof=0)), 6),
        "min": round(float(numbers.min()), 6),
        "max": round(float(numbers.max()), 6),
        "n": int(numbers.size),
    }


def _round_vector(values: np.ndarray) -> list[float | None]:
    return [
        None if not np.isfinite(value) else round(float(value), 6)
        for value in values
    ]


def _vector_summary(vectors: Sequence[np.ndarray], width: int) -> dict[str, Any]:
    usable = [np.asarray(vector, dtype=float) for vector in vectors if vector.size == width]
    if not usable:
        return {
            "mean": [None] * width,
            "std": [None] * width,
            "n": [0] * width,
        }
    stack = np.vstack(usable)
    finite = np.isfinite(stack)
    counts = finite.sum(axis=0)
    safe = np.where(finite, stack, 0.0)
    mean = np.divide(
        safe.sum(axis=0),
        np.maximum(counts, 1),
        out=np.full(width, np.nan),
        where=counts > 0,
    )
    squared = np.where(finite, (stack - mean) ** 2, 0.0)
    variance = np.divide(
        squared.sum(axis=0),
        np.maximum(counts, 1),
        out=np.full(width, np.nan),
        where=counts > 0,
    )
    return {
        "mean": _round_vector(mean),
        "std": _round_vector(np.sqrt(variance)),
        "n": [int(value) for value in counts],
    }


def _aligned(
    document: dict[str, Any], values: Any, channels: Sequence[str]
) -> np.ndarray | None:
    source_channels = [str(item) for item in (document.get("dataset") or {}).get("channel_names") or []]
    if not isinstance(values, (list, tuple)) or len(values) != len(source_channels):
        return None
    by_name = {name: _number(value) for name, value in zip(source_channels, values)}
    if set(source_channels) != set(channels):
        return None
    return np.asarray([by_name.get(name, np.nan) for name in channels], dtype=float)


def _prepare(documents: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared = [
        document
        for document in documents
        if isinstance(document, dict) and document.get("format") == "neurokiln.explain"
    ]
    if not prepared:
        raise AggregateError("没有可聚合的已完成解释", "explain.aggregate_empty")
    pipeline_ids = {str((document.get("pipeline") or {}).get("id")) for document in prepared}
    signatures = {
        str((document.get("dataset") or {}).get("signature")) for document in prepared
    }
    if len(pipeline_ids) != 1:
        raise AggregateError(
            "只能聚合同一算法的解释", "explain.aggregate_pipeline_mismatch"
        )
    if len(signatures) != 1:
        raise AggregateError(
            "只能聚合同一数据签名的解释；跨数据集结果不可平均",
            "explain.aggregate_dataset_mismatch",
        )
    return prepared


def aggregate_explanations(documents: Sequence[dict[str, Any]]) -> dict[str, Any]:
    prepared = _prepare(documents)
    first = prepared[0]
    dataset = first.get("dataset") or {}
    pipeline = first.get("pipeline") or {}
    channels = [str(item) for item in dataset.get("channel_names") or []]

    occlusion_vectors = [
        vector
        for vector in (
            _aligned(document, (document.get("occlusion") or {}).get("channel_importance"), channels)
            for document in prepared
        )
        if vector is not None
    ]
    occlusion = _vector_summary(occlusion_vectors, len(channels))

    intrinsic_entries: dict[str, dict[str, Any]] = {}
    for document in prepared:
        for entry in document.get("intrinsic") or []:
            identifier = str(entry.get("id") or "")
            if not identifier:
                continue
            bucket = intrinsic_entries.setdefault(
                identifier,
                {
                    "id": identifier,
                    "label": entry.get("label") or identifier,
                    "method": entry.get("method") or "",
                    "signed": bool(entry.get("signed")),
                    "note": entry.get("note") or "",
                    "vectors": [],
                },
            )
            vector = _aligned(document, entry.get("values"), channels)
            if vector is not None:
                bucket["vectors"].append(vector)
    intrinsic = []
    for bucket in intrinsic_entries.values():
        summary = _vector_summary(bucket.pop("vectors"), len(channels))
        intrinsic.append({**bucket, **summary})

    band_keys: list[str] = []
    for document in prepared:
        bands = document.get("bands") or {}
        for low, high in bands.get("bands") or []:
            key = f"{float(low):g}-{float(high):g}"
            if key not in band_keys:
                band_keys.append(key)
        for key in (bands.get("channel_importance") or {}):
            if key not in band_keys:
                band_keys.append(str(key))
    bands_output = []
    for key in band_keys:
        vectors = [
            vector
            for vector in (
                _aligned(
                    document,
                    ((document.get("bands") or {}).get("channel_importance") or {}).get(key),
                    channels,
                )
                for document in prepared
            )
            if vector is not None
        ]
        lows, highs = key.split("-", 1) if "-" in key else (key, "")
        drops = []
        for document in prepared:
            bands = document.get("bands") or {}
            for index, (low, high) in enumerate(bands.get("bands") or []):
                if f"{float(low):g}-{float(high):g}" == key:
                    values = bands.get("drop") or []
                    if index < len(values):
                        drops.append(values[index])
                    break
        bands_output.append(
            {
                "key": key,
                "band": [
                    float(lows) if lows else None,
                    float(highs) if highs else None,
                ],
                "drop": _summary(drops),
                "channel_importance": _vector_summary(vectors, len(channels)),
            }
        )

    gradient_signed = [
        vector
        for vector in (
            _aligned(document, (document.get("gradient") or {}).get("channel_importance"), channels)
            for document in prepared
        )
        if vector is not None
    ]
    gradient_magnitude = [
        vector
        for vector in (
            _aligned(document, (document.get("gradient") or {}).get("channel_magnitude"), channels)
            for document in prepared
        )
        if vector is not None
    ]
    audit_vectors = [
        vector
        for vector in (
            _aligned(document, (document.get("audit") or {}).get("channel_importance"), channels)
            for document in prepared
        )
        if vector is not None
    ]

    performance = first.get("performance") or {}
    faithfulness_rows = [document.get("faithfulness") or {} for document in prepared]
    audit_rows = [document.get("audit") or {} for document in prepared]
    sanity_rows = [document.get("sanity") or {} for document in prepared]
    agreement_intrinsic = [
        ((row.get("agreement") or {}).get("intrinsic") or {}).get("spearman")
        for row in faithfulness_rows
    ]
    agreement_gradient = [
        ((row.get("agreement") or {}).get("gradient") or {}).get("spearman")
        for row in faithfulness_rows
    ]
    gradient_summary = _vector_summary(gradient_signed, len(channels))
    audit_summary = _vector_summary(audit_vectors, len(channels))

    channel_table = []
    for index, name in enumerate(channels):
        intrinsic_mean = intrinsic[0]["mean"][index] if intrinsic else None
        channel_table.append(
            {
                "name": name,
                "occlusion_mean": occlusion["mean"][index],
                "occlusion_std": occlusion["std"][index],
                "support": occlusion["n"][index],
                "intrinsic_mean": intrinsic_mean,
                "gradient_mean": (
                    gradient_summary["mean"][index] if gradient_signed else None
                ),
                "band_power_mean": (
                    audit_summary["mean"][index] if audit_vectors else None
                ),
            }
        )
    channel_table.sort(
        key=lambda item: abs(item["occlusion_mean"] or 0.0), reverse=True
    )

    warnings: dict[str, int] = {}
    for document in prepared:
        for warning in document.get("warnings") or []:
            if warning:
                warnings[str(warning)] = warnings.get(str(warning), 0) + 1

    splits = []
    for index, document in enumerate(prepared):
        split = document.get("split") or {}
        splits.append(
            {
                "explanation_id": document.get("id"),
                "fold": split.get("fold"),
                "repeat": split.get("repeat"),
                "seed": split.get("seed"),
                "balanced_accuracy": (document.get("performance") or {}).get(
                    "balanced_accuracy"
                ),
                "deletion_auc": (document.get("faithfulness") or {}).get("deletion_auc"),
                "warnings": len(document.get("warnings") or []),
            }
        )

    return {
        "format": FORMAT,
        "version": FORMAT_VERSION,
        "dataset": {
            "name": dataset.get("name"),
            "version": dataset.get("version"),
            "signature": dataset.get("signature"),
            "n_channels": len(channels),
            "channel_names": channels,
            "n_classes": dataset.get("n_classes"),
        },
        "pipeline": {
            "id": pipeline.get("id"),
            "name": pipeline.get("name"),
        },
        "provenance": {
            "n_explanations": len(prepared),
            "n_unique_splits": len(
                {
                    (
                        (document.get("split") or {}).get("fold"),
                        (document.get("split") or {}).get("repeat"),
                        (document.get("split") or {}).get("seed"),
                    )
                    for document in prepared
                }
            ),
            "backgrounds": sorted(
                {
                    str((document.get("provenance") or {}).get("background"))
                    for document in prepared
                }
            ),
            "engine_versions": sorted(
                {
                    str((document.get("provenance") or {}).get("engine_version"))
                    for document in prepared
                }
            ),
        },
        "performance": {
            "balanced_accuracy": _summary(
                (document.get("performance") or {}).get("balanced_accuracy")
                for document in prepared
            ),
            "cohen_kappa": _summary(
                (document.get("performance") or {}).get("cohen_kappa")
                for document in prepared
            ),
            "n_test_total": int(
                sum(
                    int((document.get("performance") or {}).get("n_test") or 0)
                    for document in prepared
                )
            ),
        },
        "faithfulness": {
            "deletion_auc": _summary(row.get("deletion_auc") for row in faithfulness_rows),
            "insertion_auc": _summary(row.get("insertion_auc") for row in faithfulness_rows),
            "random_deletion_auc": _summary(
                row.get("random_deletion_auc") for row in faithfulness_rows
            ),
            "aopc": _summary(row.get("aopc") for row in faithfulness_rows),
            "half_split_spearman": _summary(
                ((row.get("stability") or {}).get("half_split_spearman"))
                for row in faithfulness_rows
            ),
            "subject_consistency_spearman": _summary(
                ((row.get("subject_consistency") or {}).get("spearman"))
                for row in faithfulness_rows
            ),
            "agreement_intrinsic_spearman": _summary(agreement_intrinsic),
            "agreement_gradient_spearman": _summary(agreement_gradient),
            "available_count": sum(
                1 for row in faithfulness_rows if row.get("deletion_auc") is not None
            ),
        },
        "audit": {
            "closure_ratio": _summary(row.get("closure_ratio") for row in audit_rows),
            "available_count": sum(1 for row in audit_rows if row.get("available")),
            "channel_importance": audit_summary,
        },
        "sanity": {
            "spearman": _summary(row.get("spearman") for row in sanity_rows),
            "available_count": sum(1 for row in sanity_rows if row.get("available")),
        },
        "occlusion": {
            "channel_importance": occlusion,
            "available_count": len(occlusion_vectors),
        },
        "intrinsic": intrinsic,
        "bands": bands_output,
        "gradient": {
            "methods": sorted(
                {
                    str((document.get("gradient") or {}).get("method"))
                    for document in prepared
                    if (document.get("gradient") or {}).get("available")
                }
            ),
            "channel_importance": gradient_summary,
            "channel_magnitude": _vector_summary(gradient_magnitude, len(channels)),
            "available_count": len(gradient_magnitude),
        },
        "channels": channel_table,
        "top_channels": channel_table[:10],
        "splits": splits,
        "warnings": [
            {"text": text, "count": count}
            for text, count in sorted(warnings.items(), key=lambda item: (-item[1], item[0]))
        ],
    }


__all__ = ["FORMAT", "FORMAT_VERSION", "AggregateError", "aggregate_explanations"]
