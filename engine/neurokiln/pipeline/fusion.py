from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .nodes import Node


@dataclass(frozen=True)
class DecoderPrediction:
    y_pred: np.ndarray
    proba: np.ndarray | None
    classes: np.ndarray | None


def _classes_for(results: list[DecoderPrediction]) -> np.ndarray:
    labels = set()
    for result in results:
        if result.classes is not None:
            labels.update(int(label) for label in result.classes)
        else:
            labels.update(int(label) for label in np.unique(result.y_pred))
    if not labels:
        raise ValueError("融合节点没有可用的类别")
    return np.asarray(sorted(labels), dtype=int)


def _one_hot(y_pred: np.ndarray, classes: np.ndarray) -> np.ndarray:
    positions = {int(label): index for index, label in enumerate(classes)}
    encoded = np.zeros((np.asarray(y_pred).size, classes.size), dtype=float)
    for row, label in enumerate(np.asarray(y_pred).tolist()):
        encoded[row, positions[int(label)]] = 1.0
    return encoded


def _aligned_proba(result: DecoderPrediction, classes: np.ndarray) -> np.ndarray | None:
    if result.proba is None or result.classes is None:
        return None
    positions = {int(label): index for index, label in enumerate(result.classes)}
    aligned = np.zeros((result.proba.shape[0], classes.size), dtype=float)
    for column, label in enumerate(classes.tolist()):
        source = positions.get(int(label))
        if source is not None:
            aligned[:, column] = result.proba[:, source]
    return aligned


def _check_results(results: list[DecoderPrediction]) -> int:
    if len(results) < 2:
        raise ValueError("融合节点需要至少两个解码器输出")
    sizes = {np.asarray(result.y_pred).size for result in results}
    if len(sizes) != 1:
        raise ValueError("融合节点的解码器输出试次数不一致")
    return sizes.pop()


def average_predictions(results: list[DecoderPrediction]) -> DecoderPrediction:
    _check_results(results)
    classes = _classes_for(results)
    aligned = [_aligned_proba(result, classes) for result in results]
    if any(item is None for item in aligned):
        raise ValueError("概率平均融合需要所有解码器提供概率输出")
    combined = np.mean(np.stack(aligned), axis=0)
    combined /= np.maximum(combined.sum(axis=1, keepdims=True), 1e-12)
    y_pred = classes[np.argmax(combined, axis=1)]
    return DecoderPrediction(y_pred=y_pred, proba=combined, classes=classes)


def vote_predictions(results: list[DecoderPrediction]) -> DecoderPrediction:
    n_trials = _check_results(results)
    classes = _classes_for(results)
    votes = np.zeros((n_trials, classes.size), dtype=float)
    for result in results:
        votes += _one_hot(result.y_pred, classes)
    proba = votes / votes.sum(axis=1, keepdims=True)
    backed = [_aligned_proba(result, classes) for result in results]
    if all(item is not None for item in backed):
        ranking = np.mean(np.stack(backed), axis=0)
    else:
        ranking = None
    y_pred = np.empty(n_trials, dtype=int)
    for row in range(n_trials):
        tied = np.flatnonzero(votes[row] == votes[row].max())
        if tied.size > 1 and ranking is not None:
            chosen = int(tied[int(np.argmax(ranking[row, tied]))])
        else:
            chosen = int(tied[0])
        y_pred[row] = classes[chosen]
    return DecoderPrediction(y_pred=y_pred, proba=proba, classes=classes)


class AverageFusion(Node):
    name = "average_fusion"
    kind = "merge"

    def combine(self, results: list[DecoderPrediction]) -> DecoderPrediction:
        return average_predictions(results)


class VoteFusion(Node):
    name = "vote_fusion"
    kind = "merge"

    def combine(self, results: list[DecoderPrediction]) -> DecoderPrediction:
        return vote_predictions(results)


__all__ = [
    "AverageFusion",
    "DecoderPrediction",
    "VoteFusion",
    "average_predictions",
    "vote_predictions",
]
