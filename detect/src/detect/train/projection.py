"""Edge-prob -> trader-score projection (SHARED across rungs).

Phase 1's edge projection used a 0.7 * max_edge_prob + 0.3 * top3_edge_prob
combination (see ``scripts/run_edge_level_experiment.py:_project_trader_scores``)
to derive a per-trader risk score from the model's per-edge predictions.
Rung 3 and Rung 4 must use the SAME projection so the trader-level
numbers being compared are derived identically from each rung's edge
output.

Reference
---------
Cont, R. (2001). Empirical properties of asset returns: stylized facts
and statistical implications. Quantitative Finance, 1(2), 223-236.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

# Aggregator weights - matches Phase 1 verbatim.
W_MAX = 0.7
W_TOP3 = 0.3


def _top_k_mean(series: pd.Series, k: int = 3) -> float:
    values = np.sort(series.to_numpy(dtype=float))
    if values.size == 0:
        return 0.0
    return float(np.mean(values[-k:]))


def project_edge_probs_to_traders(
    edges: pd.DataFrame,
    *,
    edge_threshold: float,
    group_cols: Iterable[str] = ("run_id", "run_name", "run_family"),
) -> pd.DataFrame:
    """Project per-edge probabilities to per-trader scores.

    Required columns on ``edges``:
      - ``sell_trader_id``, ``buy_trader_id``: edge endpoints
      - ``seller_core_label``, ``buyer_core_label``: per-trader truth
      - ``edge_prob``: model's positive-class probability for the edge
    Plus all of ``group_cols`` (e.g. run_id, run_family).

    Returns one row per (group_cols..., trader_id, label_core) with:
      - ``max_edge_prob``, ``mean_edge_prob``, ``top3_edge_prob``
      - ``trader_score = 0.7 * max + 0.3 * top3``
      - ``trader_pred = (trader_score >= edge_threshold).astype(int)``
      - ``positive_incident_edges``, ``incident_edges``
    """
    group_cols = list(group_cols)
    edges = edges.copy()
    edges["edge_pred"] = (edges["edge_prob"] >= edge_threshold).astype(int)

    seller = edges[
        group_cols + ["sell_trader_id", "seller_core_label", "edge_prob", "edge_pred"]
    ].rename(columns={
        "sell_trader_id": "trader_id",
        "seller_core_label": "label_core",
    })
    buyer = edges[
        group_cols + ["buy_trader_id", "buyer_core_label", "edge_prob", "edge_pred"]
    ].rename(columns={
        "buy_trader_id": "trader_id",
        "buyer_core_label": "label_core",
    })
    combined = pd.concat([seller, buyer], axis=0, ignore_index=True)

    traders = (
        combined.groupby(group_cols + ["trader_id", "label_core"], as_index=False)
        .agg(
            max_edge_prob=("edge_prob", "max"),
            mean_edge_prob=("edge_prob", "mean"),
            top3_edge_prob=("edge_prob", _top_k_mean),
            positive_incident_edges=("edge_pred", "sum"),
            incident_edges=("edge_prob", "count"),
        )
    )
    traders["trader_score"] = (
        W_MAX * traders["max_edge_prob"] + W_TOP3 * traders["top3_edge_prob"]
    )
    traders["trader_pred"] = (traders["trader_score"] >= edge_threshold).astype(int)
    return traders
