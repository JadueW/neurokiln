from __future__ import annotations

import copy
import os
import threading
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

from ..errors import RunCancelled
from .nodes import FitContext, Node

_TRAIN_LOCK = threading.Lock()


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency contract
        raise RuntimeError(
            "Deep baselines require PyTorch: uv pip install torch"
        ) from exc
    return torch


def _resample(X: np.ndarray, source: float, target: float) -> np.ndarray:
    if abs(source - target) < 1e-9:
        return np.asarray(X, dtype=np.float32)
    ratio = Fraction(target / source).limit_denominator(1000)
    resampled = resample_poly(X, ratio.numerator, ratio.denominator, axis=-1)
    return resampled.astype(np.float32, copy=False)


def _pad_time(X: np.ndarray, multiple: int, minimum: int) -> np.ndarray:
    length = X.shape[-1]
    target = max(minimum, int(np.ceil(length / multiple) * multiple))
    if target <= length:
        return X
    return np.pad(X, ((0, 0), (0, 0), (0, target - length)), mode="edge")


def _build_eegnet(torch, n_channels: int, n_times: int, n_classes: int,
                  dropout: float, f1: int, depth_multiplier: int, f2: int):
    nn = torch.nn
    features = nn.Sequential(
        nn.Conv2d(1, f1, (1, 64), padding="same", bias=False),
        nn.BatchNorm2d(f1),
        nn.Conv2d(f1, f1 * depth_multiplier, (n_channels, 1), groups=f1, bias=False),
        nn.BatchNorm2d(f1 * depth_multiplier),
        nn.ELU(),
        nn.AvgPool2d((1, 4)),
        nn.Dropout(dropout),
        nn.Conv2d(
            f1 * depth_multiplier, f1 * depth_multiplier, (1, 16),
            padding="same", groups=f1 * depth_multiplier, bias=False,
        ),
        nn.Conv2d(f1 * depth_multiplier, f2, (1, 1), bias=False),
        nn.BatchNorm2d(f2),
        nn.ELU(),
        nn.AvgPool2d((1, 8)),
        nn.Dropout(dropout),
        nn.Flatten(),
    )
    with torch.no_grad():
        flat = features(torch.zeros(1, 1, n_channels, n_times)).shape[1]
    return nn.Sequential(features, nn.Linear(flat, n_classes))


def _build_shallow(torch, n_channels: int, n_times: int, n_classes: int,
                   dropout: float, filters: int, target_sfreq: float):
    nn = torch.nn

    class Square(nn.Module):
        def forward(self, tensor):
            return tensor * tensor

    class SafeLog(nn.Module):
        def forward(self, tensor):
            return torch.log(tensor.clamp_min(1e-6))

    temporal = max(3, int(round(0.1 * target_sfreq)))
    pool = max(3, int(round(0.3 * target_sfreq)))
    stride = max(1, int(round(0.06 * target_sfreq)))
    pool = max(pool, stride + 1)
    features = nn.Sequential(
        nn.Conv2d(1, filters, (1, temporal), bias=False),
        nn.Conv2d(filters, filters, (n_channels, 1), bias=False),
        nn.BatchNorm2d(filters),
        nn.AvgPool2d((1, pool), stride=(1, stride)),
        Square(),
        nn.AvgPool2d((1, 4)),
        SafeLog(),
        nn.Dropout(dropout),
        nn.Flatten(),
    )
    with torch.no_grad():
        flat = features(torch.zeros(1, 1, n_channels, n_times)).shape[1]
    return nn.Sequential(features, nn.Linear(flat, n_classes))


class TorchDecoder(Node):
    """Shared, deterministic training loop for small EEG convolutional nets.

    Everything (resampling statistics, per-channel normalization, validation
    split used for early stopping and the network weights) is estimated from
    the training fold passed to :meth:`fit`; test trials never touch fitting.
    """

    kind = "decoder"
    architecture = "torch"
    pad_multiple = 32
    minimum_length = 128
    seed_offset = 0

    def __init__(
        self,
        max_epochs: int = 60,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-3,
        val_fraction: float = 0.15,
        patience: int = 10,
        target_sfreq: float = 128.0,
        dropout: float = 0.5,
    ):
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.val_fraction = float(val_fraction)
        self.patience = int(patience)
        self.target_sfreq = float(target_sfreq)
        self.dropout = float(dropout)

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError("deep decoders expect (n_trials, n_channels, n_times)")
        X = _resample(X, self.sfreq_, self.target_sfreq)
        X = (X - self.mean_) / self.scale_
        return _pad_time(X, self.pad_multiple, self.minimum_length)

    def _build(self, torch, n_channels: int, n_times: int, n_classes: int):
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray | None, ctx: FitContext):
        if y is None:
            raise ValueError("deep decoders require training labels")
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError("deep decoders expect (n_trials, n_channels, n_times)")
        torch = _torch()
        self.sfreq_ = float(ctx.sfreq)
        self.classes_ = np.unique(y)
        if self.classes_.size < 2:
            raise ValueError("deep decoders require at least two classes")

        resampled = _resample(X, self.sfreq_, self.target_sfreq)
        self.mean_ = resampled.mean(axis=(0, 2), keepdims=True)
        scale = resampled.std(axis=(0, 2), keepdims=True)
        self.scale_ = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
        prepared = self._prepare(X)
        self.n_times_ = int(prepared.shape[-1])

        seed = int(ctx.seed) + self.seed_offset
        with _TRAIN_LOCK:
            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

            y_index = np.searchsorted(self.classes_, y)
            train_idx, val_idx = self._validation_split(y_index, seed)
            model = self._build(
                torch, prepared.shape[1], prepared.shape[-1], self.classes_.size
            )

            counts = np.bincount(y_index, minlength=self.classes_.size).astype(np.float64)
            weights = counts.sum() / (self.classes_.size * np.maximum(counts, 1.0))
            criterion = torch.nn.CrossEntropyLoss(
                weight=torch.tensor(weights, dtype=torch.float32)
            )
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )

            X_train = torch.from_numpy(prepared[train_idx])
            y_train = torch.from_numpy(y_index[train_idx]).long()
            if val_idx.size:
                X_val = torch.from_numpy(prepared[val_idx])
                y_val = torch.from_numpy(y_index[val_idx]).long()
            else:
                X_val = y_val = None

            rng = np.random.default_rng(seed)
            best_state = copy.deepcopy(model.state_dict())
            best_loss = float("inf")
            stale = 0
            model.train()
            for _epoch in range(self.max_epochs):
                if ctx.should_cancel is not None and ctx.should_cancel():
                    raise RunCancelled("arena run cancelled")
                order = rng.permutation(len(X_train))
                for start in range(0, len(order), self.batch_size):
                    batch = torch.from_numpy(order[start:start + self.batch_size])
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(X_train[batch].unsqueeze(1))
                    loss = criterion(logits, y_train[batch])
                    loss.backward()
                    optimizer.step()
                if X_val is None:
                    continue
                val_loss = self._evaluate_loss(torch, model, X_val, y_val, criterion)
                if val_loss < best_loss - 1e-5:
                    best_loss = val_loss
                    best_state = copy.deepcopy(model.state_dict())
                    stale = 0
                else:
                    stale += 1
                    if stale >= self.patience:
                        break
            if X_val is not None:
                model.load_state_dict(best_state)
            model.eval()
        self.model_ = model
        return self

    def _validation_split(self, y_index: np.ndarray, seed: int):
        if self.val_fraction <= 0 or y_index.size < 12:
            return np.arange(y_index.size), np.array([], dtype=int)
        rng = np.random.default_rng(seed + 991)
        train_parts: list[np.ndarray] = []
        val_parts: list[np.ndarray] = []
        for cls in np.unique(y_index):
            members = np.flatnonzero(y_index == cls)
            rng.shuffle(members)
            n_val = int(round(members.size * self.val_fraction))
            if members.size >= 4:
                n_val = min(max(1, n_val), members.size - 2)
            else:
                n_val = 0
            val_parts.append(members[:n_val])
            train_parts.append(members[n_val:])
        val_idx = np.sort(np.concatenate(val_parts)) if any(p.size for p in val_parts) else np.array([], dtype=int)
        train_idx = np.sort(np.concatenate(train_parts))
        return train_idx, val_idx

    @staticmethod
    def _evaluate_loss(torch, model, X, y, criterion) -> float:
        model.eval()
        total = 0.0
        with torch.no_grad():
            for start in range(0, len(X), 128):
                logits = model(X[start:start + 128].unsqueeze(1))
                total += float(criterion(logits, y[start:start + 128])) * len(logits)
        model.train()
        return total / max(len(X), 1)

    def prepare(self, X: np.ndarray) -> np.ndarray:
        """Resample, normalize and pad raw epochs for the network."""
        if not hasattr(self, "mean_"):
            raise RuntimeError("deep decoder must be fitted before transform")
        return self._prepare(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Identity by design: ``predict``/``predict_proba`` prepare internally.

        Resampling here as well double-applied the 250 -> 128 Hz conversion
        whenever a pipeline called ``transform`` before ``predict``, which
        silently changed the input length for epochs longer than the model's
        minimum and broke the final linear layer.
        """
        if not hasattr(self, "mean_"):
            raise RuntimeError("deep decoder must be fitted before transform")
        return np.asarray(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        torch = _torch()
        prepared = self._prepare(X)
        self.model_.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(prepared), 128):
                batch = torch.from_numpy(prepared[start:start + 128]).unsqueeze(1)
                predictions.append(self.model_(batch).argmax(dim=1).numpy())
        indices = np.concatenate(predictions) if predictions else np.array([], dtype=int)
        return self.classes_[indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        torch = _torch()
        prepared = self._prepare(X)
        self.model_.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(prepared), 128):
                batch = torch.from_numpy(prepared[start:start + 128]).unsqueeze(1)
                chunks.append(torch.softmax(self.model_(batch), dim=1).cpu().numpy())
        return np.concatenate(chunks, axis=0) if chunks else np.empty((0, self.classes_.size))


class EEGNet(TorchDecoder):
    name = "eegnet"
    architecture = "eegnet"
    seed_offset = 0

    def __init__(self, f1: int = 8, depth_multiplier: int = 2, f2: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.f1 = int(f1)
        self.depth_multiplier = int(depth_multiplier)
        self.f2 = int(f2)

    def _build(self, torch, n_channels, n_times, n_classes):
        return _build_eegnet(
            torch, n_channels, n_times, n_classes, self.dropout,
            self.f1, self.depth_multiplier, self.f2,
        )


class ShallowConvNet(TorchDecoder):
    name = "shallowconvnet"
    architecture = "shallowconvnet"
    pad_multiple = 16
    minimum_length = 160
    seed_offset = 1

    def __init__(self, filters: int = 40, **kwargs):
        super().__init__(**kwargs)
        self.filters = int(filters)

    def _build(self, torch, n_channels, n_times, n_classes):
        return _build_shallow(
            torch, n_channels, n_times, n_classes, self.dropout,
            self.filters, self.target_sfreq,
        )


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def configure_threads() -> int:
    """Keep CPU inference predictable when several jobs run in parallel."""
    torch = _torch()
    requested = int(os.environ.get("NEUROKILN_TORCH_THREADS", "0"))
    threads = requested if requested > 0 else max(1, min(4, os.cpu_count() or 1))
    torch.set_num_threads(threads)
    return threads
