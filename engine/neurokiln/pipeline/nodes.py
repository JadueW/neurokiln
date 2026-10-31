from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Sequence

import numpy as np

from ..cache import ArtifactCache
from ..data.model import Epochs
from ..protocol.types import Split


@dataclass(frozen=True)
class FitContext:
    """Everything a node may see while fitting. No test data by construction."""

    trial_ids: tuple[str, ...]
    indices: np.ndarray
    seed: int
    paradigm: str
    channel_names: tuple[str, ...]
    sfreq: float
    paradigm_meta: dict = field(default_factory=dict)
    should_cancel: Callable[[], bool] | None = None
    tmin: float = 0.0


class Node:
    name = "node"
    kind = "transform"

    def fit(self, X: np.ndarray, y: np.ndarray | None, ctx: FitContext) -> "Node":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def output_timebase(self, sfreq: float, tmin: float) -> tuple[float, float]:
        """Return the sampling rate and epoch start seen by downstream nodes."""
        return sfreq, tmin


def probability_output(
    decoder: Node, X: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (proba, classes) when the decoder exposes a valid predict_proba."""
    if not hasattr(decoder, "predict_proba"):
        return None, None
    classes = getattr(decoder, "classes_", None)
    if classes is None:
        return None, None
    try:
        proba = np.asarray(decoder.predict_proba(X), dtype=float)
    except (AttributeError, NotImplementedError, RuntimeError, ValueError):
        return None, None
    labels = np.asarray(classes)
    if (
        proba.ndim != 2
        or proba.shape[1] != labels.size
        or proba.shape[0] != np.asarray(X).shape[0]
        or not np.all(np.isfinite(proba))
    ):
        return None, None
    return proba, labels


def node_descriptor(node: Node) -> dict[str, Any]:
    params = {
        key: value
        for key, value in getattr(node, "__dict__", {}).items()
        if not key.startswith("_") and not key.endswith("_")
    }
    return {
        "class": f"{type(node).__module__}.{type(node).__name__}",
        "params": params,
    }


def run_node_with_cache(
    node: Node,
    train_input,
    test_input,
    y_train: np.ndarray | None,
    ctx: FitContext,
    *,
    cache: ArtifactCache | None = None,
    data_signature: str = "",
    split_signature: str = "",
    seed: int = 0,
    train_parents: Sequence[str] = (),
    test_parents: Sequence[str] = (),
    context: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, str]]:
    """Fit/transform one node, reusing split-bound artifacts when allowed."""
    cacheable = cache is not None and bool(getattr(node, "cacheable", False))
    descriptor = node_descriptor(node) if cache is not None else None
    train_key = test_key = ""
    if cache is not None:
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
        if cacheable:
            cached_train = cache.get(train_key)
            cached_test = cache.get(test_key)
            expected_train = (
                None if isinstance(train_input, (list, tuple))
                else int(np.asarray(train_input).shape[0])
            )
            expected_test = (
                None if isinstance(test_input, (list, tuple))
                else int(np.asarray(test_input).shape[0])
            )
            if (
                cached_train is not None
                and cached_test is not None
                and cached_train.ndim >= 1
                and cached_test.ndim >= 1
                and (expected_train is None or cached_train.shape[0] == expected_train)
                and (expected_test is None or cached_test.shape[0] == expected_test)
            ):
                return cached_train, cached_test, (train_key, test_key)
    node.fit(train_input, y_train, ctx)
    train_output = np.asarray(node.transform(train_input))
    test_output = np.asarray(node.transform(test_input))
    if cacheable:
        cache.put(
            train_key,
            train_output,
            meta={"node": descriptor["class"], "role": "train"},
        )
        cache.put(
            test_key,
            test_output,
            meta={"node": descriptor["class"], "role": "test"},
        )
    return train_output, test_output, (train_key, test_key)


@dataclass
class NodeTrace:
    """One fitted node plus the arrays it saw. Used by explainability only."""

    index: int
    name: str
    kind: str
    node: Node
    train_input: np.ndarray
    test_input: np.ndarray
    train_output: np.ndarray
    test_output: np.ndarray
    sfreq: float
    tmin: float


@dataclass
class FittedSplit:
    """A fitted pipeline for one split, with per-node activations retained."""

    pipeline: "Pipeline"
    train_epochs: Epochs
    test_epochs: Epochs
    traces: list[NodeTrace]
    result: dict

    def transform_from_raw(self, X: np.ndarray) -> np.ndarray:
        """Run the transform-only suffix (no refitting) on raw epochs."""
        values = np.asarray(X)
        for trace in self.traces:
            values = np.asarray(trace.node.transform(values))
        return values

    def predict_from_raw(self, X: np.ndarray):
        """Transform-only pass: no refitting, safe for perturbation oracles."""
        values = self.transform_from_raw(X)
        decoder = self.traces[-1].node
        y_pred = np.asarray(decoder.predict(values), dtype=int)
        y_proba, proba_classes = probability_output(decoder, values)
        return y_pred, y_proba, proba_classes


@dataclass
class Pipeline:
    name: str
    nodes: list[Node] = field(default_factory=list)

    def fit_split(
        self,
        epochs: Epochs,
        split: Split,
        should_cancel: Callable[[], bool] | None = None,
        cache: ArtifactCache | None = None,
        data_signature: str | None = None,
    ) -> tuple[dict, FittedSplit]:
        """Fit one split and keep every node's train/test activations.

        ``run_split`` delegates here; the returned arrays are identical to the
        ones used for prediction, so explanations always describe the model
        that actually produced the reported metrics.
        """
        train_epochs = epochs.select(split.train)
        test_epochs = epochs.select(split.test)
        X_train = train_epochs.data
        y_train = train_epochs.labels
        X_test = test_epochs.data
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
        node_sfreq = float(epochs.sfreq)
        node_tmin = float(epochs.tmin)
        train_parent = test_parent = signature
        traces: list[NodeTrace] = []
        for index, node in enumerate(self.nodes):
            input_sfreq, input_tmin = node_sfreq, node_tmin
            input_train = np.asarray(X_train)
            input_test = np.asarray(X_test)
            node_ctx = replace(ctx, sfreq=node_sfreq, tmin=node_tmin)
            X_train, X_test, (train_parent, test_parent) = run_node_with_cache(
                node,
                X_train,
                X_test,
                y_train,
                node_ctx,
                cache=cache,
                data_signature=signature,
                split_signature=split.signature(),
                seed=split.seed,
                train_parents=(train_parent,),
                test_parents=(test_parent,),
                context={**cache_context, "sfreq": node_sfreq, "tmin": node_tmin},
            )
            traces.append(
                NodeTrace(
                    index=index,
                    name=getattr(node, "name", type(node).__name__),
                    kind=getattr(node, "kind", "transform"),
                    node=node,
                    train_input=input_train,
                    test_input=input_test,
                    train_output=np.asarray(X_train),
                    test_output=np.asarray(X_test),
                    sfreq=input_sfreq,
                    tmin=input_tmin,
                )
            )
            node_sfreq, node_tmin = node.output_timebase(node_sfreq, node_tmin)
        decoder = self.nodes[-1]
        y_pred = np.asarray(decoder.predict(X_test), dtype=int)
        y_proba, proba_classes = probability_output(decoder, X_test)
        if y_pred.shape != test_epochs.labels.shape:
            raise ValueError("decoder returned predictions with wrong shape")
        result = {
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
        fitted = FittedSplit(
            pipeline=self,
            train_epochs=train_epochs,
            test_epochs=test_epochs,
            traces=traces,
            result=result,
        )
        return result, fitted

    def run_split(
        self,
        epochs: Epochs,
        split: Split,
        should_cancel: Callable[[], bool] | None = None,
        cache: ArtifactCache | None = None,
        data_signature: str | None = None,
    ) -> dict:
        result, _ = self.fit_split(
            epochs,
            split,
            should_cancel=should_cancel,
            cache=cache,
            data_signature=data_signature,
        )
        return result
