"""Explainability for fitted pipelines (P0-P3).

Public surface:

* :func:`run_explanation` - one call produces intrinsic maps, occlusion and
  band attribution, optional Captum gradients, faithfulness checks, a
  knowledge audit and the on-disk artifact bundle.
* :func:`extract_intrinsic` - exact explanations read from fitted parameters.
* :func:`occlusion_blocks` / :func:`band_ablation` - model-agnostic attribution.
* :func:`topomap_png` - dependency-free scalp rendering.

Everything is evaluated against the *same* fitted split that produced the
reported metrics, and every reference signal comes from the training fold.
"""

from .aggregate import AggregateError, aggregate_explanations
from .attribution import (
    DEFAULT_BANDS,
    band_ablation,
    gradient_attribution,
    occlusion_blocks,
    probe_confidence,
    weight_randomization_check,
)
from .audit import AUDIT_BANDS, handcrafted_features, knowledge_audit
from .faithfulness import faithfulness
from .intrinsic import channel_reference, extract_intrinsic
from .montage import channel_positions, normalize_name, topomap_png
from .runner import FORMAT, FORMAT_VERSION, ExplanationConfig, run_explanation

__all__ = [
    "AUDIT_BANDS",
    "AggregateError",
    "DEFAULT_BANDS",
    "ExplanationConfig",
    "FORMAT",
    "FORMAT_VERSION",
    "aggregate_explanations",
    "band_ablation",
    "channel_positions",
    "channel_reference",
    "extract_intrinsic",
    "faithfulness",
    "gradient_attribution",
    "handcrafted_features",
    "knowledge_audit",
    "normalize_name",
    "occlusion_blocks",
    "probe_confidence",
    "run_explanation",
    "topomap_png",
    "weight_randomization_check",
]
