"""Model-agnostic attribution with one shared perturbation protocol.

All methods replace part of the *raw* epoch with the training-fold mean and
re-run the transform-only suffix, so every decoder - linear, Riemannian or
convolutional - is probed through exactly the same interface. The target is
always the model's own decision, and the reference signal never contains test
data.
"""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np

from ..errors import RunCancelled
from ..pipeline.deep import TorchDecoder
from ..pipeline.nodes import FittedSplit
from .trace import (
    SCORE_UNIT,
    channel_groups,
    decision_probs,
    decision_scores,
    log_odds,
    score_source,
    spearman,
    target_log_odds,
    train_baseline,
    window_slices,
)

DEFAULT_BANDS = ((4.0, 8.0), (8.0, 13.0), (13.0, 20.0), (20.0, 30.0), (30.0, 45.0))


def _check_cancel(should_cancel) -> None:
    if should_cancel is not None and should_cancel():
        raise RunCancelled("explanation cancelled")


def probe_confidence(fitted: FittedSplit, probe: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Probability the model still assigns to its original decision."""
    proba = decision_probs(fitted, probe).proba
    return proba[np.arange(proba.shape[0]), np.clip(target, 0, proba.shape[1] - 1)]


def probe_scores(fitted: FittedSplit, probe: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Log-odds with which the model still prefers its original decision.

    Unbounded, unlike probability: a saturated model still shows how much a
    perturbation moved the margin instead of returning a flat 1.0. The
    log-odds form is also shift-invariant, so decoders whose raw scores carry
    an arbitrary offset (convolutional logits, distance negatives) remain
    comparable across probes.
    """
    return target_log_odds(fitted, probe, target)


_probe_confidence = probe_confidence


def occlusion_blocks(
    fitted: FittedSplit,
    *,
    window_ms: int = 250,
    max_blocks: int = 120,
    max_windows: int = 8,
    should_cancel=None,
    on_step=None,
) -> dict:
    X = np.asarray(fitted.test_epochs.data, dtype=float)
    n_trials, n_channels, n_times = X.shape
    sfreq = float(fitted.test_epochs.sfreq)
    decision = decision_probs(fitted)
    base = decision.confidence
    target = decision.predicted_index
    base_score = log_odds(decision_scores(fitted), target)
    baseline = train_baseline(fitted)

    raw_windows = max(1, int(round(n_times / max(1, window_ms / 1000.0 * sfreq))))
    n_windows = max(1, min(max_windows, raw_windows))
    windows = window_slices(n_times, n_windows)
    max_groups = max(1, min(n_channels, max_blocks // max(1, len(windows))))
    groups = channel_groups(n_channels, max_groups)

    drops = np.zeros((n_trials, len(groups), len(windows)), dtype=float)
    total = len(groups) * len(windows)
    step = 0
    for group_index, channels in enumerate(groups):
        for window_index, window in enumerate(windows):
            _check_cancel(should_cancel)
            probe = X.copy()
            probe[:, channels, window] = baseline[channels][None, :, None]
            drops[:, group_index, window_index] = base_score - probe_scores(fitted, probe, target)
            step += 1
            if on_step is not None:
                on_step(step, total)

    per_trial_channel = np.zeros((n_trials, n_channels), dtype=float)
    for group_index, channels in enumerate(groups):
        per_trial_channel[:, channels] = drops[:, group_index, :].mean(axis=1)[:, None]

    tmin = float(fitted.test_epochs.tmin)
    window_times = [
        [round(tmin + window.start / sfreq, 4), round(tmin + window.stop / sfreq, 4)]
        for window in windows
    ]
    blocks = [
        {"group": group_index, "window": window_index,
         "channels": [int(item) for item in channels]}
        for group_index, channels in enumerate(groups)
        for window_index in range(len(windows))
    ]
    block_values = drops.mean(axis=0).ravel().tolist()
    return {
        "window_ms": int(window_ms),
        "n_windows": len(windows),
        "window_times": window_times,
        "groups": [[int(item) for item in channels] for channels in groups],
        "blocks": blocks,
        "block_values": [round(float(value), 6) for value in block_values],
        "channel_importance": [
            round(float(value), 6) for value in per_trial_channel.mean(axis=0)
        ],
        "time_importance": [
            round(float(value), 6) for value in drops.mean(axis=(0, 1))
        ],
        "matrix": [
            [round(float(value), 6) for value in row] for row in drops.mean(axis=0)
        ],
        "per_trial_channel": per_trial_channel,
        "base_confidence": base,
        "base_score": base_score,
        "score_unit": SCORE_UNIT,
        "score_source": score_source(fitted),
        "target": target,
        "predicted_classes": decision.classes,
        "n_trials": int(n_trials),
        "windows": [(int(window.start), int(window.stop)) for window in windows],
    }


def _band_weights(frequencies: np.ndarray, low: float, high: float, ramp: float = 1.0) -> np.ndarray:
    weights = np.zeros_like(frequencies)
    weights[(frequencies >= low) & (frequencies <= high)] = 1.0
    left = (frequencies >= low - ramp) & (frequencies < low)
    right = (frequencies > high) & (frequencies <= high + ramp)
    weights[left] = 0.5 * (1 + np.cos(np.pi * (low - frequencies[left]) / ramp))
    weights[right] = 0.5 * (1 + np.cos(np.pi * (frequencies[right] - high) / ramp))
    return weights


def band_ablation(
    fitted: FittedSplit,
    *,
    bands=DEFAULT_BANDS,
    max_groups: int = 12,
    should_cancel=None,
    on_step=None,
) -> dict:
    X = np.asarray(fitted.test_epochs.data, dtype=float)
    _, n_channels, n_times = X.shape
    sfreq = float(fitted.test_epochs.sfreq)
    nyquist = sfreq / 2.0
    frequencies = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    decision = decision_probs(fitted)
    base = decision.confidence
    target = decision.predicted_index
    base_score = log_odds(decision_scores(fitted), target)
    spectrum = np.fft.rfft(X, axis=-1)

    usable = [
        (float(low), float(high)) for low, high in bands if float(low) < nyquist * 0.95
    ]
    if not usable:
        return {"available": False, "reason": "采样率过低，没有可用频带"}

    groups = channel_groups(n_channels, max_groups)
    total = len(usable) * (1 + len(groups))
    step = 0
    band_drops = np.zeros(len(usable))
    channel_band = np.zeros((len(groups), len(usable)))
    for band_index, (low, high) in enumerate(usable):
        _check_cancel(should_cancel)
        weights = _band_weights(frequencies, low, min(high, nyquist * 0.99))
        removed = 1.0 - weights
        probe = np.fft.irfft(spectrum * removed[None, None, :], n=n_times, axis=-1)
        band_drops[band_index] = float(np.mean(base_score - probe_scores(fitted, probe, target)))
        step += 1
        if on_step is not None:
            on_step(step, total)
        for group_index, channels in enumerate(groups):
            _check_cancel(should_cancel)
            shaped = spectrum.copy()
            shaped[:, channels, :] *= removed[None, None, :]
            probe = np.fft.irfft(shaped, n=n_times, axis=-1)
            channel_band[group_index, band_index] = float(
                np.mean(base_score - probe_scores(fitted, probe, target))
            )
            step += 1
            if on_step is not None:
                on_step(step, total)

    per_channel = np.zeros((n_channels, len(usable)))
    for group_index, channels in enumerate(groups):
        per_channel[channels] = channel_band[group_index][None, :]
    return {
        "available": True,
        "bands": [[round(low, 3), round(high, 3)] for low, high in usable],
        "drop": [round(float(value), 6) for value in band_drops],
        "relative": [
            round(float(value / max(abs(np.mean(base_score)), 1e-9)), 4)
            for value in band_drops
        ],
        "groups": [[int(item) for item in channels] for channels in groups],
        "channel_band": per_channel.round(6).tolist(),
        "channel_importance": {
            f"{low:g}-{high:g}": [
                round(float(value), 6) for value in per_channel[:, index]
            ]
            for index, (low, high) in enumerate(usable)
        },
        "base_confidence": float(np.mean(base)),
        "base_score": float(np.mean(base_score)),
        "score_unit": SCORE_UNIT,
        "score_source": score_source(fitted),
    }


def gradient_attribution(
    fitted: FittedSplit,
    *,
    method: str = "integrated_gradients",
    n_steps: int = 32,
    max_trials: int = 64,
) -> dict:
    decoder = fitted.traces[-1].node
    if not isinstance(decoder, TorchDecoder):
        return {"available": False, "reason": "梯度归因仅适用于 EEGNet / ShallowConvNet 等 torch 解码器"}
    try:
        import torch
        from captum.attr import DeepLift, IntegratedGradients
    except ImportError:
        return {
            "available": False,
            "reason": "未安装 captum；执行 uv pip install 'neurokiln-engine[xai]' 后可用",
        }
    X = np.asarray(fitted.test_epochs.data[:max_trials], dtype=float)
    if X.shape[0] == 0:
        return {"available": False, "reason": "没有测试试次"}
    prepared = np.asarray(decoder.prepare(X), dtype=np.float32)
    inputs = torch.from_numpy(prepared)
    baselines = torch.zeros_like(inputs)
    decision = decision_probs(fitted)
    target = torch.from_numpy(decision.predicted_index[: X.shape[0]].astype(np.int64))

    def forward(tensor):
        return decoder.model_(tensor.unsqueeze(1))

    try:
        if method == "deeplift":
            engine = DeepLift(forward)
            attributes = engine.attribute(inputs, baselines=baselines, target=target)
        else:
            engine = IntegratedGradients(forward)
            attributes = engine.attribute(
                inputs, baselines=baselines, target=target, n_steps=int(n_steps)
            )
    except Exception as exc:  # captum raises many backend-specific errors
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    attr = attributes.detach().cpu().numpy()
    sfreq = float(fitted.test_epochs.sfreq)
    valid = int(round(fitted.test_epochs.data.shape[-1] * decoder.target_sfreq / sfreq))
    valid = max(1, min(valid, attr.shape[-1]))
    attr = attr[:, :, :valid]
    signed = attr.mean(axis=(0, 2))
    magnitude = np.abs(attr).mean(axis=(0, 2))
    return {
        "available": True,
        "method": method,
        "n_steps": int(n_steps) if method == "integrated_gradients" else None,
        "n_trials": int(attr.shape[0]),
        "channel_importance": [round(float(value), 6) for value in signed],
        "channel_magnitude": [round(float(value), 6) for value in magnitude],
        "time_importance": [
            round(float(value), 6) for value in attr.mean(axis=(0, 1))
        ],
        "target_frequency": float(decoder.target_sfreq),
        "valid_samples": valid,
    }


def channel_occlusion_scores(
    fitted: FittedSplit, *, max_trials: int = 48
) -> np.ndarray | None:
    """Cheap channel-only occlusion used by the weight-randomization check."""
    X = np.asarray(fitted.test_epochs.data[:max_trials], dtype=float)
    if X.shape[0] == 0:
        return None
    decision = decision_probs(fitted, X)
    target = decision.predicted_index
    base = log_odds(decision_scores(fitted, X), target)
    baseline = train_baseline(fitted)
    scores = np.zeros(X.shape[1])
    for channel in range(X.shape[1]):
        probe = X.copy()
        probe[:, channel, :] = baseline[channel]
        scores[channel] = float(np.mean(base - probe_scores(fitted, probe, target)))
    return scores


def weight_randomization_check(
    fitted: FittedSplit, reference, *, max_trials: int = 48, seed: int = 2027
) -> dict:
    """Sanity check: random weights must destroy the attribution structure."""
    decoder = fitted.traces[-1].node
    if not isinstance(decoder, TorchDecoder):
        return {"available": False, "reason": "权重随机化检验仅适用于 torch 解码器"}
    reference = np.asarray(reference, dtype=float) if reference is not None else None
    if reference is None or reference.size != len(fitted.test_epochs.channel_names):
        return {"available": False, "reason": "缺少可比对的通道归因"}
    try:
        import torch
    except ImportError:  # pragma: no cover
        return {"available": False, "reason": "PyTorch 不可用"}
    probe_decoder = copy.deepcopy(decoder)
    with torch.no_grad():
        torch.manual_seed(int(seed))
        for module in probe_decoder.model_.modules():
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()
    probe_trace = replace(fitted.traces[-1], node=probe_decoder)
    probe_fitted = replace(fitted, traces=[*fitted.traces[:-1], probe_trace])
    randomized = channel_occlusion_scores(probe_fitted, max_trials=max_trials)
    if randomized is None:
        return {"available": False, "reason": "随机化模型没有产生有效输出"}
    rho = spearman(reference, randomized)
    return {
        "available": True,
        "spearman": None if rho is None else round(rho, 4),
        "max_trials": int(max_trials),
        "note": (
            "权重随机化后重新计算通道遮挡；与原模型的相关性应接近 0，"
            "若仍高度相关说明归因主要反映了输入统计而非模型。"
        ),
    }


__all__ = [
    "DEFAULT_BANDS",
    "band_ablation",
    "channel_occlusion_scores",
    "gradient_attribution",
    "occlusion_blocks",
    "weight_randomization_check",
]
