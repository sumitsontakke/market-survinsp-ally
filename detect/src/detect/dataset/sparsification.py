"""Adaptive sparsification - replaces Phase 1's static tau = 0.70.

Two methods, numpy-only (no torch dep so the data layer is exercisable
end-to-end without the trainer image installed):

  * ``knn_sparsify``        - per-source-node top-k by edge weight
  * ``percentile_sparsify`` - global threshold = ``quantile(weights, q)``

Both return masks over the input edge arrays. The PyG builder applies
the mask to ``edge_index`` and ``edge_attr`` in lock-step.

Reference
---------
Cont, R. (2001). Empirical properties of asset returns: stylized facts
and statistical implications. Quantitative Finance, 1(2), 223-236.

Hamilton, W. L., Ying, R., Leskovec, J. (2017). Inductive Representation
Learning on Large Graphs. NeurIPS.
"""
from __future__ import annotations

import numpy as np


def knn_sparsify(
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    k: int,
) -> np.ndarray:
    """Boolean mask keeping the top-k outgoing edges per source node.

    Parameters
    ----------
    edge_index : np.ndarray of shape (2, E)
        Rows: [source_idx, target_idx] for each directed edge.
    edge_weight : np.ndarray of shape (E,)
        Scalar weight (e.g. traded_volume) used for top-k ranking.
    k : int
        Edges to keep per source.

    Returns
    -------
    mask : np.ndarray of bool of shape (E,)
        True for edges that survive sparsification.
    """
    if edge_index.size == 0:
        return np.zeros(0, dtype=bool)
    if edge_weight.shape[0] != edge_index.shape[1]:
        raise ValueError(
            f"edge_weight length {edge_weight.shape[0]} != edge count {edge_index.shape[1]}"
        )
    sources = edge_index[0]
    keep = np.zeros(sources.shape[0], dtype=bool)
    # group by source node
    unique_sources = np.unique(sources)
    for src in unique_sources:
        idx = np.where(sources == src)[0]
        if idx.size <= k:
            keep[idx] = True
            continue
        # top-k by weight
        weights = edge_weight[idx]
        # argpartition is faster than argsort for the top-k case
        top_local = np.argpartition(-weights, k)[:k]
        keep[idx[top_local]] = True
    return keep


def percentile_sparsify(
    edge_weight: np.ndarray,
    q: float = 0.7,
) -> np.ndarray:
    """Boolean mask keeping edges with weight at or above the q-th percentile."""
    if edge_weight.size == 0:
        return np.zeros(0, dtype=bool)
    threshold = float(np.quantile(edge_weight, q))
    return edge_weight >= threshold


def static_threshold(
    edge_weight: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Phase 1's static cut. Provided for ablation / comparison only."""
    if edge_weight.size == 0:
        return np.zeros(0, dtype=bool)
    return edge_weight >= float(threshold)
