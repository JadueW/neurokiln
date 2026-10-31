"""Electrode positions and dependency-free topographic map rendering.

The engine deliberately avoids matplotlib (it stays a light, optional-free
core), so topomaps are produced by inverse-distance interpolation onto a
unit-disk grid and encoded as PNG with zlib. Coordinates follow the extended
10-10 system projected to a top-down view (x = right, y = anterior).
"""

from __future__ import annotations

import re
import struct
import zlib
from functools import lru_cache

import numpy as np

# Head radius is 1.0; y grows towards the nose.
_BASE_10_20 = {
    "Fp1": (-0.30, 0.92), "Fp2": (0.30, 0.92),
    "F7": (-0.81, 0.55), "F3": (-0.42, 0.68), "Fz": (0.0, 0.72),
    "F4": (0.42, 0.68), "F8": (0.81, 0.55),
    "T7": (-0.94, 0.0), "C3": (-0.46, 0.0), "Cz": (0.0, 0.0),
    "C4": (0.46, 0.0), "T8": (0.94, 0.0),
    "P7": (-0.81, -0.55), "P3": (-0.42, -0.68), "Pz": (0.0, -0.72),
    "P4": (0.42, -0.68), "P8": (0.81, -0.55),
    "O1": (-0.30, -0.92), "Oz": (0.0, -0.96), "O2": (0.30, -0.92),
}

_EXTENDED_10_10 = {
    "Fpz": (0.0, 0.95), "Iz": (0.0, -1.0),
    "AF7": (-0.60, 0.76), "AF3": (-0.36, 0.80), "AFz": (0.0, 0.84),
    "AF4": (0.36, 0.80), "AF8": (0.60, 0.76),
    "F5": (-0.63, 0.62), "F1": (-0.21, 0.70), "F2": (0.21, 0.70), "F6": (0.63, 0.62),
    "FC5": (-0.70, 0.35), "FC3": (-0.42, 0.33), "FC1": (-0.20, 0.35),
    "FCz": (0.0, 0.36), "FC2": (0.20, 0.35), "FC4": (0.42, 0.33), "FC6": (0.70, 0.35),
    "FT7": (-0.88, 0.30), "FT8": (0.88, 0.30),
    "FT9": (-0.95, 0.21), "FT10": (0.95, 0.21),
    "T9": (-0.99, 0.0), "T10": (0.99, 0.0),
    "C5": (-0.70, 0.0), "C1": (-0.23, 0.0), "C2": (0.23, 0.0), "C6": (0.70, 0.0),
    "CP5": (-0.70, -0.35), "CP3": (-0.42, -0.33), "CP1": (-0.20, -0.35),
    "CPz": (0.0, -0.36), "CP2": (0.20, -0.35), "CP4": (0.42, -0.33), "CP6": (0.70, -0.35),
    "TP7": (-0.88, -0.30), "TP8": (0.88, -0.30),
    "TP9": (-0.95, -0.21), "TP10": (0.95, -0.21),
    "P5": (-0.63, -0.62), "P1": (-0.21, -0.70), "P2": (0.21, -0.70), "P6": (0.63, -0.62),
    "PO7": (-0.60, -0.78), "PO3": (-0.36, -0.82), "POz": (0.0, -0.86),
    "PO4": (0.36, -0.82), "PO8": (0.60, -0.78),
    "I1": (-0.20, -0.99), "I2": (0.20, -0.99),
    "M1": (-0.95, -0.10), "M2": (0.95, -0.10),
}

COORDINATES: dict[str, tuple[float, float]] = {**_BASE_10_20, **_EXTENDED_10_10}

_ALIASES = {
    "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8",
    "FP1": "Fp1", "FP2": "Fp2", "FPZ": "Fpz",
    "A1": "M1", "A2": "M2",
    "T1": "FT9", "T2": "FT10",
    "M1": "M1", "M2": "M2",
    "IZ": "Iz", "OZ": "Oz", "CZ": "Cz", "PZ": "Pz", "FZ": "Fz",
    "LA": "M1", "RA": "M2",
}

_LOOKUP = {name.upper(): name for name in COORDINATES}
_LOOKUP.update(_ALIASES)

_TOKEN = re.compile(r"(?:F|FC|FT|C|CP|T|TP|P|PO|O|I|AF|M|A)[0-9]{0,2}[zZ]?")
_NOISE = re.compile(r"(?:EEG|EOG|ECG|EMG|REF|AVG|LE|RE)[-_ ]*")
_SUFFIX = re.compile(r"[-_. ]*(?:REF|AVG|LE|RE)$", re.IGNORECASE)

HEAD_MASK_RADIUS = 1.0


def normalize_name(raw: object) -> str | None:
    """Map a free-form channel label to a canonical 10-10 name (or None)."""
    text = str(raw or "").strip()
    if not text:
        return None
    upper = text.upper()
    if upper in _LOOKUP:
        return _LOOKUP[upper]

    cleaned = _NOISE.sub("", upper)
    cleaned = _SUFFIX.sub("", cleaned)
    for separator in ("-", "_", " ", "/", "|", ":"):
        cleaned = cleaned.replace(separator, "")
    if cleaned in _LOOKUP:
        return _LOOKUP[cleaned]

    match = _TOKEN.search(cleaned)
    if match:
        token = match.group(0).upper()
        if token in _LOOKUP:
            return _LOOKUP[token]
    return None


def channel_positions(channel_names) -> tuple[np.ndarray, dict]:
    """Return per-channel (x, y) positions plus layout metadata.

    Unknown montages degrade to an evenly spaced ring. That is explicitly
    reported so downstream consumers never mistake it for real geometry.
    """
    names = [str(name) for name in channel_names]
    canonical = [normalize_name(name) for name in names]
    known = sum(1 for item in canonical if item is not None)
    if known >= max(4, int(0.6 * len(names))) and known >= 4:
        coords = np.array(
            [COORDINATES[item] if item else (-1.2, -1.2) for item in canonical],
            dtype=float,
        )
        # Unknown leftovers are parked at the centre; they still take part in
        # interpolation but are drawn without a marker.
        unknown = [name for name, item in zip(names, canonical) if item is None]
        for index, item in enumerate(canonical):
            if item is None:
                coords[index] = _nearest_free_spot(coords[:index])
        return coords, {
            "layout": "10-10",
            "mapped": known,
            "unknown": unknown,
            "canonical": [item or names[i] for i, item in enumerate(canonical)],
        }
    coords = _ring(len(names))
    return coords, {
        "layout": "ring",
        "mapped": 0,
        "unknown": names,
        "canonical": names,
        "note": "未识别到 10-10 电极命名，按等距环形排布，仅用于相对比较。",
    }


def _nearest_free_spot(previous: np.ndarray) -> tuple[float, float]:
    if previous.size == 0:
        return (0.0, 0.0)
    candidates = _ring(previous.shape[0] + 1) * 0.85
    for candidate in candidates:
        if np.min(np.linalg.norm(previous - candidate, axis=1)) > 0.12:
            return float(candidate[0]), float(candidate[1])
    return (0.0, 0.0)


def _ring(count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0, 2))
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, count, endpoint=False)
    return np.stack([np.cos(angles) * 0.78, np.sin(angles) * 0.78], axis=1)


def _grid(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(-1.15, 1.15, size)
    gx, gy = np.meshgrid(axis, axis)
    mask = (gx ** 2 + gy ** 2) <= HEAD_MASK_RADIUS ** 2
    return gx, gy, mask


def interpolate(
    values: np.ndarray, coords: np.ndarray, size: int = 64, power: float = 2.0
) -> np.ndarray:
    """Inverse-distance interpolation of channel values onto a head grid."""
    values = np.asarray(values, dtype=float).ravel()
    coords = np.asarray(coords, dtype=float)
    if values.size != coords.shape[0]:
        raise ValueError("values and coordinates must have the same length")
    gx, gy, mask = _grid(size)
    inside = np.flatnonzero(mask.ravel())
    points = np.stack([gx.ravel()[inside], gy.ravel()[inside]], axis=1)
    distance = np.linalg.norm(points[:, None, :] - coords[None, :, :], axis=2)
    weight = 1.0 / np.maximum(distance, 1e-3) ** power
    total = weight.sum(axis=1, keepdims=True)
    smooth = (weight * np.nan_to_num(values, nan=0.0)[None, :]).sum(axis=1, keepdims=True) / total
    exact = distance.min(axis=1) < 1e-6
    if np.any(exact):
        nearest = distance.argmin(axis=1)
        smooth[exact, 0] = np.nan_to_num(values, nan=0.0)[nearest[exact]]
    grid = np.full(mask.shape, np.nan)
    grid.ravel()[inside] = smooth[:, 0]
    return _blur(grid, mask)


def _blur(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    padded = np.pad(np.nan_to_num(grid), 1, mode="edge")
    smoothed = (
        padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
        + 4.0 * padded[1:-1, 1:-1]
    ) / 8.0
    result = np.where(mask, np.where(np.isnan(grid), smoothed, 0.55 * grid + 0.45 * smoothed), np.nan)
    return result


def _diverging(values: np.ndarray, limit: float) -> np.ndarray:
    low = np.array([43, 87, 151], dtype=float)
    mid = np.array([247, 248, 250], dtype=float)
    high = np.array([178, 52, 45], dtype=float)
    ratio = np.clip(values / max(limit, 1e-12), -1.0, 1.0)
    positive = ratio >= 0
    t = np.abs(ratio)[..., None]
    return np.where(positive[..., None], mid + (high - mid) * t, mid + (low - mid) * t)


def _sequential(values: np.ndarray, limit: float) -> np.ndarray:
    low = np.array([247, 251, 255], dtype=float)
    high = np.array([8, 48, 107], dtype=float)
    ratio = np.clip(np.abs(values) / max(limit, 1e-12), 0.0, 1.0)
    return low + (high - low) * ratio[..., None]


def _draw_head(rgb: np.ndarray) -> None:
    height, width, _ = rgb.shape
    axis = np.linspace(-1.15, 1.15, width)
    gx, gy = np.meshgrid(axis, axis)
    radius = np.sqrt(gx ** 2 + gy ** 2)
    ring = np.abs(radius - HEAD_MASK_RADIUS) < (2.4 / width)
    rgb[ring] = np.array([43, 58, 74], dtype=np.float32)

    # Nose: a small triangle above the circle.
    for y in np.linspace(0.97, 1.12, 26):
        half = (1.12 - y) * 0.55
        columns = np.abs(gx[0] + 0.0) <= half
        row = int(round((1.15 - y) / 2.30 * (height - 1)))
        if 0 <= row < height:
            rgb[row, columns] = np.array([43, 58, 74], dtype=np.float32)
    # Ears.
    for side in (-1.0, 1.0):
        for y in np.linspace(-0.18, 0.18, 30):
            x = side * (1.0 + 0.07 * np.cos(y / 0.18 * np.pi / 2))
            column = int(round((x + 1.15) / 2.30 * (width - 1)))
            row = int(round((1.15 - y) / 2.30 * (height - 1)))
            if 0 <= row < height:
                start, end = max(0, column - 1), min(width, column + 2)
                rgb[row, start:end] = np.array([43, 58, 74], dtype=np.float32)


def _draw_dots(rgb: np.ndarray, coords: np.ndarray) -> None:
    height, width, _ = rgb.shape
    for x, y in np.asarray(coords):
        if not np.isfinite(x) or abs(x) > 1.12 or abs(y) > 1.12:
            continue
        column = int(round((x + 1.15) / 2.30 * (width - 1)))
        row = int(round((1.15 - y) / 2.30 * (height - 1)))
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dx * dx + dy * dy > 4:
                    continue
                rr, cc = row + dy, column + dx
                if 0 <= rr < height and 0 <= cc < width:
                    rgb[rr, cc] = np.array([28, 38, 50], dtype=np.float32)


def _png_bytes(rgb: np.ndarray) -> bytes:
    rgb = np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))
    height, width, _ = rgb.shape
    raw = bytearray()
    for row in rgb:
        raw.append(0)
        raw.extend(row.tobytes())

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )


def topomap_png(
    values,
    channel_names,
    *,
    signed: bool = True,
    size: int = 140,
    limit: float | None = None,
    dot_channels: bool = True,
) -> tuple[bytes, dict]:
    """Render one topomap. Returns (png_bytes, meta)."""
    values = np.asarray(values, dtype=float).ravel()
    coords, layout = channel_positions(channel_names)
    finite = np.isfinite(values)
    if not finite.any():
        values = np.zeros_like(values)
        finite = np.ones_like(values, dtype=bool)
    fill = float(np.nanmean(values[finite]))
    smooth_values = np.where(finite, values, fill)
    scale = float(limit) if limit is not None else float(np.nanmax(np.abs(smooth_values)) or 1.0)
    scale = max(scale, 1e-9)
    grid = interpolate(smooth_values, coords, size=size)
    colors = _diverging(grid, scale) if signed else _sequential(grid, scale)
    colors = np.where(np.isnan(grid)[..., None], np.array([246, 248, 251], dtype=float), colors)
    _draw_head(colors)
    if dot_channels:
        _draw_dots(colors, coords)
    meta = {
        **{key: value for key, value in layout.items() if key != "canonical"},
        "signed": bool(signed),
        "limit": round(float(scale), 6),
        "size": int(size),
        "channels": [str(name) for name in channel_names],
    }
    return _png_bytes(colors), meta


def topomap_data_uri(values, channel_names, **kwargs) -> tuple[str, dict]:
    import base64

    payload, meta = topomap_png(values, channel_names, **kwargs)
    return "data:image/png;base64," + base64.b64encode(payload).decode("ascii"), meta


@lru_cache(maxsize=1)
def montage_table() -> dict[str, tuple[float, float]]:
    return dict(COORDINATES)


__all__ = [
    "COORDINATES",
    "channel_positions",
    "interpolate",
    "montage_table",
    "normalize_name",
    "topomap_data_uri",
    "topomap_png",
]
