"""Split-safe Parquet cache for preprocessing artifacts.

Only nodes that explicitly opt in with ``cacheable = True`` are stored. The
cache key binds the dataset content signature, the exact split signature, the
fit seed and the node's public parameters, so a training-fold artifact can
never be served to another fold or another dataset version.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_DEFAULT_MAX_BYTES = 512 * 1024 * 1024
_DEFAULT_MAX_ENTRIES = 256


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, set)):
        return list(value)
    return str(value)


class ArtifactCache:
    """Bounded on-disk cache for fitted preprocessing outputs."""

    def __init__(
        self,
        root: str | Path,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        enabled: bool = True,
    ):
        self.root = Path(root).expanduser().resolve()
        self.max_bytes = max(1, int(max_bytes))
        self.max_entries = max(1, int(max_entries))
        self.enabled = bool(enabled)
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.evictions = 0
        self._entries: int | None = None
        self._bytes: int | None = None
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_data_root(cls, data_root: str | Path, enabled: bool | None = None) -> "ArtifactCache":
        if enabled is None:
            enabled = os.environ.get("NEUROKILN_CACHE", "1").strip().lower() not in {
                "0", "false", "off", "no",
            }
        return cls(
            Path(data_root) / "cache" / "preprocess",
            max_bytes=int(os.environ.get("NEUROKILN_CACHE_BYTES", str(_DEFAULT_MAX_BYTES))),
            max_entries=int(os.environ.get("NEUROKILN_CACHE_MAX_ENTRIES", str(_DEFAULT_MAX_ENTRIES))),
            enabled=enabled,
        )

    @staticmethod
    def node_key(
        *,
        data_signature: str,
        split_signature: str,
        seed: int,
        parents: Sequence[str],
        descriptor: dict[str, Any],
        role: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        payload = {
            "format": "neurokiln-preprocess-cache/v1",
            "data": data_signature,
            "split": split_signature,
            "seed": int(seed),
            "parents": list(parents),
            "node": descriptor,
            "role": role,
            "context": context or {},
        }
        encoded = json.dumps(
            payload, sort_keys=True, ensure_ascii=False, default=_json_default
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _paths(self, key: str) -> tuple[Path, Path]:
        directory = self.root / key[:2] / key
        return directory, directory / "meta.json"

    def _entry_directories(self) -> list[Path]:
        if not self.root.exists():
            return []
        return [
            path for path in self.root.glob("*/*")
            if path.is_dir() and not path.name.startswith(".tmp-")
        ]

    def _refresh_usage(self) -> None:
        entries = self._entry_directories()
        self._entries = len(entries)
        self._bytes = int(
            sum(file.stat().st_size for entry in entries for file in entry.glob("*") if file.is_file())
        )

    def _ensure_usage(self) -> None:
        if self._entries is None or self._bytes is None:
            self._refresh_usage()

    def get(self, key: str) -> np.ndarray | None:
        if not self.enabled:
            return None
        directory, meta_path = self._paths(key)
        if not meta_path.exists():
            with self._lock:
                self.misses += 1
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("key") != key:
                raise ValueError("cache key mismatch")
            shape = tuple(int(value) for value in meta["shape"])
            dtype = np.dtype(meta.get("dtype", "float64"))
            if meta.get("format") == "parquet":
                import pyarrow.parquet as pq

                table = pq.read_table(directory / "data.parquet")
                column = table.column("values").combine_chunks()
                flat = np.asarray(column.flatten(), dtype=dtype)
                array = flat.reshape(shape)
            else:
                array = np.load(directory / "data.npy", allow_pickle=False).astype(dtype, copy=False)
                if array.shape != shape:
                    raise ValueError("cached array shape mismatch")
        except (OSError, ValueError, KeyError, TypeError, ImportError):
            with self._lock:
                self.misses += 1
            return None
        try:
            os.utime(meta_path, None)
        except OSError:
            pass
        with self._lock:
            self.hits += 1
        return np.array(array, copy=True)

    def put(self, key: str, array: np.ndarray, meta: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        data = np.asarray(array)
        if data.ndim == 0:
            raise ValueError("cache only accepts arrays with at least one dimension")
        directory, meta_path = self._paths(key)
        payload = {
            "key": key,
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "created_at": time.time(),
            "nbytes": int(data.nbytes),
            **(meta or {}),
        }
        with self._lock:
            if meta_path.exists():
                return
            directory.parent.mkdir(parents=True, exist_ok=True)
            temp = directory.parent / f".tmp-{uuid.uuid4().hex}"
            temp.mkdir(parents=True, exist_ok=True)
            try:
                rows = data.reshape(data.shape[0], -1)
                try:
                    import pyarrow as pa
                    import pyarrow.parquet as pq

                    flat = pa.array(np.ascontiguousarray(rows).reshape(-1), type=pa.float64())
                    values = pa.FixedSizeListArray.from_arrays(flat, rows.shape[1])
                    pq.write_table(
                        pa.table({"values": values}),
                        temp / "data.parquet",
                        compression="zstd",
                    )
                    payload["format"] = "parquet"
                except ImportError:
                    np.save(temp / "data.npy", data, allow_pickle=False)
                    payload["format"] = "npy"
                (temp / "meta.json").write_text(
                    json.dumps(payload, ensure_ascii=False, sort_keys=True, default=_json_default),
                    encoding="utf-8",
                )
                if directory.exists():
                    if meta_path.exists():
                        shutil.rmtree(temp, ignore_errors=True)
                        return
                    shutil.rmtree(directory, ignore_errors=True)
                os.replace(temp, directory)
            except Exception:
                shutil.rmtree(temp, ignore_errors=True)
                return
            self.writes += 1
            if self._entries is not None:
                self._entries += 1
            if self._bytes is not None:
                self._bytes += int(data.nbytes)
            self._evict_locked()

    def _evict_locked(self) -> None:
        self._ensure_usage()
        entries = self._entry_directories()
        entries.sort(key=lambda path: (path / "meta.json").stat().st_mtime if (path / "meta.json").exists() else 0.0)
        while entries and (
            (self._entries or 0) > self.max_entries
            or (self._bytes or 0) > self.max_bytes
        ):
            victim = entries.pop(0)
            try:
                self._bytes = (self._bytes or 0) - int(
                    sum(file.stat().st_size for file in victim.glob("*") if file.is_file())
                )
                shutil.rmtree(victim, ignore_errors=True)
                self._entries = max(0, (self._entries or 1) - 1)
                self.evictions += 1
            except OSError:
                break

    def stats(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_usage()
            return {
                "enabled": self.enabled,
                "root": str(self.root),
                "entries": int(self._entries or 0),
                "bytes": int(self._bytes or 0),
                "max_bytes": self.max_bytes,
                "max_entries": self.max_entries,
                "hits": self.hits,
                "misses": self.misses,
                "writes": self.writes,
                "evictions": self.evictions,
                "hit_rate": round(
                    self.hits / max(1, self.hits + self.misses), 4
                ),
            }

    def clear(self) -> dict[str, Any]:
        with self._lock:
            if self.root.exists():
                for child in self.root.iterdir():
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        child.unlink(missing_ok=True)
            self._entries = 0
            self._bytes = 0
            self.hits = self.misses = self.writes = self.evictions = 0
            return self.stats()


__all__ = ["ArtifactCache"]
