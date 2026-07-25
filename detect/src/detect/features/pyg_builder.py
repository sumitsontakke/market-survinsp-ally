"""PyG ``Data`` object builder for Rung 4 GraphSAGE.

Two-layer design:

  ``build_graph_arrays(run, ...)``    returns numpy arrays
      (edge_index, edge_attr, x, y, node_id_index, metadata).
      No torch dependency - exercisable without the trainer image.

  ``build_pyg_data(run, ...)``         calls build_graph_arrays then
      wraps the result in ``torch_geometric.data.Data``. Imports torch
      lazily so the rest of the package works without it.

Reference
---------
Cont, R. (2001). Empirical properties of asset returns: stylized facts
and statistical implications. Quantitative Finance, 1(2), 223-236.

Hamilton, W. L., Ying, R., Leskovec, J. (2017). Inductive Representation
Learning on Large Graphs. NeurIPS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from detect.dataset.loader import Run
from detect.dataset.sparsification import knn_sparsify, percentile_sparsify
from detect.features.edge_directed import (
    EXPECTED_COLUMNS,
    compute_directed_edge_features,
    compute_edge_labels,
)
from detect.features.node_engineered import (
    ENGINEERED_NODE_FEATURE_NAMES,
    engineered_node_feature_frame,
)


# Default edge attribute order - the 8 features the Rung 4 SAGE YAML uses.
# (Lead-lag is summarized via best_lag_value + best_lag rather than the
# 7 per-lag scalars; if the user wants the full 7 in edge_attr they can
# pass ``edge_feature_names`` explicitly.)
DEFAULT_EDGE_FEATURE_NAMES: tuple[str, ...] = (
    "best_lag_value",
    "best_lag",
    "signed_imbalance",
    "traded_volume",
    "interaction_count",
    "pre_window_drift",
    "post_window_drift",
    "vwap_imbalance",
)

DEFAULT_NODE_FEATURE_NAMES: tuple[str, ...] = (
    "trader_total_volume",
    "trader_unique_counterparties",
)

# Path A2 (Phase H) feature-augmented set: the two topology features plus
# the six engineered manipulation-signature features. Opt-in only, via
# build_graph_arrays(..., node_feature_names=AUGMENTED_NODE_FEATURE_NAMES).
# The default path stays 2-dim so existing v1 checkpoints still evaluate.
AUGMENTED_NODE_FEATURE_NAMES: tuple[str, ...] = (
    DEFAULT_NODE_FEATURE_NAMES + ENGINEERED_NODE_FEATURE_NAMES
)


@dataclass
class GraphArrays:
    """torch-free container of a single run's directed graph."""

    edge_index: np.ndarray          # shape (2, E)
    edge_attr: np.ndarray           # shape (E, F_edge)
    x: np.ndarray                   # shape (N, F_node)
    y: np.ndarray                   # shape (E,) {0, 1}
    node_ids: list[str]             # length N, position -> trader_id
    edge_feature_names: tuple[str, ...]
    node_feature_names: tuple[str, ...]
    metadata: dict

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def num_positives(self) -> int:
        return int(self.y.sum())


# ---------------------------------------------------------------------------
# Node feature computation
# ---------------------------------------------------------------------------

def _build_node_features(trades: pd.DataFrame,
                         orders: pd.DataFrame,
                         node_ids: Sequence[str],
                         feature_names: Sequence[str]) -> pd.DataFrame:
    """Per-trader features used as ``x`` in the PyG graph.

    Always provides the two topology features (total volume, unique
    counterparties). The six engineered manipulation-signature features
    are computed only when ``feature_names`` requests at least one of
    them (Path A2 feature-augmented retrain) - so the default 2-feature
    v1 path pays no extra computation.
    """
    ids = [str(n) for n in node_ids]

    # --- two topology features (cheap, always computed) ---
    if trades is None or trades.empty:
        base = pd.DataFrame(
            index=ids,
            data={"trader_total_volume": 0.0,
                  "trader_unique_counterparties": 0.0},
        )
    else:
        t = trades.copy()
        t["quantity"] = pd.to_numeric(t["quantity"], errors="coerce").fillna(0.0)
        seller_vol = t.groupby("sell_trader_id")["quantity"].sum()
        buyer_vol = t.groupby("buy_trader_id")["quantity"].sum()
        total_vol = seller_vol.add(buyer_vol, fill_value=0.0)

        # Counterparties = unique partners regardless of direction.
        cps_seller = (
            t.groupby("sell_trader_id")["buy_trader_id"].nunique().rename("from_buy")
        )
        cps_buyer = (
            t.groupby("buy_trader_id")["sell_trader_id"].nunique().rename("from_sell")
        )
        # Combine: a trader's unique counterparty count is the union, not sum.
        # We approximate with max(from_buy, from_sell) which is a safe lower
        # bound; the true union would require per-trader counterparty sets.
        # For the SAGE input this magnitude is what matters, not the exact count.
        cps = pd.concat([cps_seller, cps_buyer], axis=1).fillna(0.0).max(axis=1)

        base = pd.DataFrame(
            index=ids,
            data={
                "trader_total_volume":
                    [float(total_vol.get(n, 0.0)) for n in ids],
                "trader_unique_counterparties":
                    [float(cps.get(n, 0.0)) for n in ids],
            },
        )

    # --- six engineered features (only when requested) ---
    wants_engineered = any(n in ENGINEERED_NODE_FEATURE_NAMES
                           for n in feature_names)
    if wants_engineered:
        eng = engineered_node_feature_frame(orders, trades, ids)
        return base.join(eng, how="left").fillna(0.0)
    return base


# ---------------------------------------------------------------------------
# Public entry point - torch-free
# ---------------------------------------------------------------------------

def build_graph_arrays(
    run: Run,
    *,
    edge_feature_names: Sequence[str] = DEFAULT_EDGE_FEATURE_NAMES,
    node_feature_names: Sequence[str] = DEFAULT_NODE_FEATURE_NAMES,
    sparsification: Optional[str] = "knn",
    k: int = 12,
    percentile_q: float = 0.7,
    target_label: str = "label_any_edge",
) -> GraphArrays:
    """Build a single run's directed graph as numpy arrays.

    Edge weight used by k-NN sparsification is ``traded_volume`` so that
    Phase 1's "most-traded pair survives" intuition holds. Per-source
    top-k is the default; ``percentile_sparsify`` is available for
    ablation.
    """
    edge_feats = compute_directed_edge_features(run)
    edge_labels = compute_edge_labels(run)
    if edge_feats.empty or edge_labels.empty:
        return GraphArrays(
            edge_index=np.zeros((2, 0), dtype=np.int64),
            edge_attr=np.zeros((0, len(edge_feature_names)), dtype=np.float32),
            x=np.zeros((0, len(node_feature_names)), dtype=np.float32),
            y=np.zeros(0, dtype=np.int64),
            node_ids=[],
            edge_feature_names=tuple(edge_feature_names),
            node_feature_names=tuple(node_feature_names),
            metadata={"run_id": run.run_id, "family": run.family,
                      "note": "empty"},
        )

    merged = edge_feats.merge(
        edge_labels, on=["sell_trader_id", "buy_trader_id"], how="left"
    ).fillna({"label_any_edge": 0, "label_core_edge": 0})

    # Node-id catalogue: the union of seller and buyer ids in this graph.
    all_ids = pd.unique(
        pd.concat(
            [merged["sell_trader_id"], merged["buy_trader_id"]],
            ignore_index=True,
        ).astype(str)
    ).tolist()
    all_ids.sort()
    id_to_idx = {n: i for i, n in enumerate(all_ids)}

    src = merged["sell_trader_id"].map(id_to_idx).to_numpy(dtype=np.int64)
    dst = merged["buy_trader_id"].map(id_to_idx).to_numpy(dtype=np.int64)
    edge_index = np.stack([src, dst], axis=0)

    # Validate feature names
    missing = [n for n in edge_feature_names if n not in merged.columns]
    if missing:
        raise ValueError(
            f"build_graph_arrays: missing edge feature columns {missing}. "
            f"Available: {list(merged.columns)}"
        )
    edge_attr = merged[list(edge_feature_names)].to_numpy(dtype=np.float32)
    y = merged[target_label].astype(int).to_numpy(dtype=np.int64)

    # Sparsification - applied AFTER feature compute so we never lose
    # signal we already extracted.
    if sparsification == "knn":
        mask = knn_sparsify(edge_index, merged["traded_volume"].to_numpy(), k=k)
    elif sparsification == "percentile":
        mask = percentile_sparsify(merged["traded_volume"].to_numpy(), q=percentile_q)
    elif sparsification in (None, "none", "static"):
        mask = np.ones(edge_index.shape[1], dtype=bool)
    else:
        raise ValueError(f"unknown sparsification: {sparsification!r}")
    edge_index = edge_index[:, mask]
    edge_attr = edge_attr[mask]
    y = y[mask]

    # Node features over the same catalogue
    nf_df = _build_node_features(run.trades, run.orders, all_ids,
                                 node_feature_names)
    missing_nf = [n for n in node_feature_names if n not in nf_df.columns]
    if missing_nf:
        raise ValueError(
            f"build_graph_arrays: missing node feature columns {missing_nf}"
        )
    x = nf_df[list(node_feature_names)].to_numpy(dtype=np.float32)

    # Defensive NaN scrubbing (the trainer rejects NaNs downstream).
    edge_attr = np.nan_to_num(edge_attr, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    return GraphArrays(
        edge_index=edge_index,
        edge_attr=edge_attr,
        x=x,
        y=y,
        node_ids=all_ids,
        edge_feature_names=tuple(edge_feature_names),
        node_feature_names=tuple(node_feature_names),
        metadata={
            "run_id": run.run_id,
            "family": run.family,
            "scenario_ids": sorted(set(run.scenarios.get("scenario_id", pd.Series(dtype=str)).astype(str)))
                if not run.scenarios.empty else [],
            "sparsification": sparsification,
            "k": k if sparsification == "knn" else None,
        },
    )


# ---------------------------------------------------------------------------
# torch-aware wrapper - imports torch lazily
# ---------------------------------------------------------------------------

def build_pyg_data(run: Run, **kwargs):
    """Wrap :func:`build_graph_arrays` output in ``torch_geometric.data.Data``.

    Imports torch + torch_geometric lazily; raises a clear error if they
    are unavailable. Use ``build_graph_arrays`` directly when running
    outside the trainer Docker image.
    """
    arrays = build_graph_arrays(run, **kwargs)
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:  # noqa: BLE001
        raise ImportError(
            "torch + torch_geometric required for build_pyg_data. "
            "Run inside the trainer container (calibration_service/trainer/) "
            "or use build_graph_arrays() instead."
        ) from exc

    return Data(
        x=torch.from_numpy(arrays.x).float(),
        edge_index=torch.from_numpy(arrays.edge_index).long(),
        edge_attr=torch.from_numpy(arrays.edge_attr).float(),
        y=torch.from_numpy(arrays.y).long(),
        node_ids=arrays.node_ids,
        edge_feature_names=arrays.edge_feature_names,
        node_feature_names=arrays.node_feature_names,
        metadata=arrays.metadata,
    )
