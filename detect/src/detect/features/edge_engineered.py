"""Engineered edge features (Rung 3 v1).

Faithful port of ``scripts/feature_engineering_edges_v1.py`` into the
new harness. Behavior is unchanged - the same inputs produce the same
feature DataFrame - so the future Rung 3 retraining (deferred per
Option A) can plug straight in.

What this computes per directed (seller, buyer) pair across one run:

  - aggregate stats (trade_count, total_quantity, mean/std/max quantity,
    mean/std price, span, density, instrument concentration)
  - 60-second and 300-second window features (peaks, shares, burstiness)
  - graph features (degrees, weighted flow, PageRank, SCC/WCC sizes,
    reverse-edge reciprocity, cycle closure, common neighbors)
  - reciprocity balances and edge-share ratios
  - labels (label_any_edge, label_core_edge) and per-trader core labels

Reference
---------
Cont, R. (2001). Empirical properties of asset returns: stylized facts
and statistical implications. Quantitative Finance, 1(2), 223-236.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from detect.dataset.loader import Run


@dataclass(frozen=True)
class _ScenarioRecord:
    scenario_id: str
    scenario_type: str
    participants: set[str]
    ring_edges: set[tuple[str, str]]


def _build_scenario_index(scenarios: pd.DataFrame) -> dict[str, _ScenarioRecord]:
    if scenarios is None or scenarios.empty:
        return {}
    out: dict[str, _ScenarioRecord] = {}
    for _, row in scenarios.iterrows():
        sid = str(row.get("scenario_id", "")).strip()
        if not sid:
            continue
        participants: set[str] = set()
        raw = row.get("participant_ids")
        if raw is not None and not (isinstance(raw, float) and pd.isna(raw)):
            if isinstance(raw, (list, tuple, set, np.ndarray)):
                participants = {str(x) for x in raw}
            else:
                try:
                    participants = {str(x) for x in json.loads(raw)}
                except Exception:  # noqa: BLE001
                    participants = set()
        ring_edges: set[tuple[str, str]] = set()
        raw_ring = row.get("ring_order")
        if raw_ring is not None and not (isinstance(raw_ring, float) and pd.isna(raw_ring)):
            if isinstance(raw_ring, (list, tuple, set, np.ndarray)):
                order = [str(x) for x in raw_ring]
            else:
                try:
                    order = [str(x) for x in json.loads(raw_ring)]
                except Exception:  # noqa: BLE001
                    order = []
            if len(order) >= 2:
                for i, src in enumerate(order):
                    dst = order[(i + 1) % len(order)]
                    ring_edges.add((src, dst))
        out[sid] = _ScenarioRecord(
            scenario_id=sid,
            scenario_type=str(row.get("scenario_type", "")),
            participants=participants,
            ring_edges=ring_edges,
        )
    return out


def _window_features(trades: pd.DataFrame, window_seconds: int, prefix: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    base_t = trades["timestamp"].min()
    f = trades.copy()
    f["window_idx"] = ((f["timestamp"] - base_t).dt.total_seconds() // window_seconds).astype(int)
    if "trade_id" not in f.columns:
        f["trade_id"] = [f"trade_{i:06d}" for i in range(len(f))]
    grouped = (
        f.groupby(["sell_trader_id", "buy_trader_id", "window_idx"])
        .agg(window_trade_count=("trade_id", "count"), window_quantity=("quantity", "sum"))
        .reset_index()
    )
    if grouped.empty:
        return pd.DataFrame()

    def _burst(s: pd.Series) -> float:
        m = float(s.mean())
        sd = float(s.std(ddof=0))
        return sd / (m + 1e-6) if m > 0 else 0.0

    summary = (
        grouped.groupby(["sell_trader_id", "buy_trader_id"])
        .agg(
            active_windows=("window_idx", "nunique"),
            peak_trade_count=("window_trade_count", "max"),
            mean_trade_count=("window_trade_count", "mean"),
            std_trade_count=("window_trade_count", "std"),
            peak_quantity=("window_quantity", "max"),
            mean_quantity=("window_quantity", "mean"),
            std_quantity=("window_quantity", "std"),
        )
        .reset_index()
    )
    totals = (
        grouped.groupby(["sell_trader_id", "buy_trader_id"])
        .agg(
            total_window_trade_count=("window_trade_count", "sum"),
            total_window_quantity=("window_quantity", "sum"),
        )
        .reset_index()
    )
    summary = summary.merge(totals, on=["sell_trader_id", "buy_trader_id"], how="left")
    summary[f"{prefix}_peak_trade_share"] = (
        summary["peak_trade_count"] / (summary["total_window_trade_count"] + 1e-6)
    )
    summary[f"{prefix}_peak_quantity_share"] = (
        summary["peak_quantity"] / (summary["total_window_quantity"] + 1e-6)
    )
    burst = (
        grouped.groupby(["sell_trader_id", "buy_trader_id"])
        .agg(
            trade_count_burstiness=("window_trade_count", _burst),
            quantity_burstiness=("window_quantity", _burst),
        )
        .reset_index()
    )
    summary = summary.merge(burst, on=["sell_trader_id", "buy_trader_id"], how="left")

    rename_map = {
        "active_windows": f"{prefix}_active_windows",
        "peak_trade_count": f"{prefix}_peak_trade_count",
        "mean_trade_count": f"{prefix}_mean_trade_count",
        "std_trade_count": f"{prefix}_std_trade_count",
        "peak_quantity": f"{prefix}_peak_quantity",
        "mean_quantity": f"{prefix}_mean_quantity",
        "std_quantity": f"{prefix}_std_quantity",
        "trade_count_burstiness": f"{prefix}_trade_count_burstiness",
        "quantity_burstiness": f"{prefix}_quantity_burstiness",
    }
    summary = summary.rename(columns=rename_map)
    keep_cols = [
        "sell_trader_id", "buy_trader_id",
        *rename_map.values(),
        f"{prefix}_peak_trade_share", f"{prefix}_peak_quantity_share",
    ]
    return summary[keep_cols]


def _graph_features(edge_base: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    g = nx.DiGraph()
    for _, row in edge_base.iterrows():
        g.add_edge(
            str(row["sell_trader_id"]),
            str(row["buy_trader_id"]),
            weight=float(row["total_quantity"]),
            trade_count=float(row["trade_count"]),
        )
    if g.number_of_nodes() == 0:
        return pd.DataFrame(), {}

    pagerank = nx.pagerank(g, weight="weight") if g.number_of_edges() else {}
    undirected = g.to_undirected()

    scc_id: dict[str, str] = {}
    scc_size: dict[str, int] = {}
    for i, comp in enumerate(nx.strongly_connected_components(g)):
        cid = f"scc_{i}"
        size = len(comp)
        for node in comp:
            scc_id[str(node)] = cid
            scc_size[str(node)] = size

    wcc_size: dict[str, int] = {}
    for comp in nx.connected_components(undirected):
        size = len(comp)
        for node in comp:
            wcc_size[str(node)] = size

    metrics: dict[str, dict[str, Any]] = {}
    for node in g.nodes():
        succ = set(g.successors(node))
        pred = set(g.predecessors(node))
        out_w = float(sum(g[node][n].get("weight", 0.0) for n in succ))
        in_w = float(sum(g[n][node].get("weight", 0.0) for n in pred))
        metrics[str(node)] = {
            "out_degree": g.out_degree(node),
            "in_degree": g.in_degree(node),
            "out_weight": out_w,
            "in_weight": in_w,
            "pagerank": float(pagerank.get(node, 0.0)),
            "scc_id": scc_id.get(str(node), f"scc_{node}"),
            "scc_size": scc_size.get(str(node), 1),
            "wcc_size": wcc_size.get(str(node), 1),
            "successors": succ,
            "predecessors": pred,
        }

    rows: list[dict[str, Any]] = []
    for _, row in edge_base.iterrows():
        s = str(row["sell_trader_id"])
        d = str(row["buy_trader_id"])
        sm = metrics.get(s, {})
        dm = metrics.get(d, {})
        s_succ = sm.get("successors", set())
        s_pred = sm.get("predecessors", set())
        d_succ = dm.get("successors", set())
        d_pred = dm.get("predecessors", set())
        rev_exists = 1.0 if g.has_edge(d, s) else 0.0
        rev_w = float(g[d][s].get("weight", 0.0)) if g.has_edge(d, s) else 0.0
        rev_tc = float(g[d][s].get("trade_count", 0.0)) if g.has_edge(d, s) else 0.0
        rows.append({
            "sell_trader_id": s,
            "buy_trader_id": d,
            "seller_out_degree": float(sm.get("out_degree", 0.0)),
            "seller_in_degree": float(sm.get("in_degree", 0.0)),
            "buyer_out_degree": float(dm.get("out_degree", 0.0)),
            "buyer_in_degree": float(dm.get("in_degree", 0.0)),
            "seller_out_weight": float(sm.get("out_weight", 0.0)),
            "seller_in_weight": float(sm.get("in_weight", 0.0)),
            "buyer_out_weight": float(dm.get("out_weight", 0.0)),
            "buyer_in_weight": float(dm.get("in_weight", 0.0)),
            "seller_pagerank": float(sm.get("pagerank", 0.0)),
            "buyer_pagerank": float(dm.get("pagerank", 0.0)),
            "seller_scc_size": float(sm.get("scc_size", 1.0)),
            "buyer_scc_size": float(dm.get("scc_size", 1.0)),
            "seller_wcc_size": float(sm.get("wcc_size", 1.0)),
            "buyer_wcc_size": float(dm.get("wcc_size", 1.0)),
            "reverse_edge_exists": rev_exists,
            "reverse_total_quantity": rev_w,
            "reverse_trade_count": rev_tc,
            "cycle_closure_count": float(len(d_succ & s_pred)),
            "common_successor_count": float(len(s_succ & d_succ)),
            "common_predecessor_count": float(len(s_pred & d_pred)),
            "shared_neighbor_count": float(
                len((s_succ | s_pred) & (d_succ | d_pred))
            ),
            "same_scc": 1.0 if sm.get("scc_id") == dm.get("scc_id") else 0.0,
        })
    return pd.DataFrame(rows), metrics


def _edge_labels(
    edge_trades: pd.DataFrame,
    scenarios: dict[str, _ScenarioRecord],
) -> tuple[int, int, set[str]]:
    any_pos = 0
    core_pos = 0
    types: set[str] = set()
    if edge_trades.empty:
        return any_pos, core_pos, types
    for _, t in edge_trades.iterrows():
        if not bool(t.get("is_manipulative", False)):
            continue
        any_pos = 1
        seller = str(t.get("sell_trader_id"))
        buyer = str(t.get("buy_trader_id"))
        sid = str(t.get("scenario_id", ""))
        scen = scenarios.get(sid)
        stype = str(t.get("scenario_type", scen.scenario_type if scen else ""))
        if stype:
            types.add(stype)
        if scen is None:
            core_pos = 1
            continue
        if scen.scenario_type == "circular_trading_ring":
            if (seller, buyer) in scen.ring_edges:
                core_pos = 1
        elif seller in scen.participants and buyer in scen.participants:
            core_pos = 1
    return any_pos, core_pos, types


def compute_edge_features_v1(run: Run) -> pd.DataFrame:
    """Compute the v1 engineered edge feature set for a single run.

    Returns a DataFrame with columns:
      - sell_trader_id, buy_trader_id  (composite key)
      - ~30 numeric feature columns (aggregates + windows + graph)
      - label_any_edge, label_core_edge
      - seller_core_label, buyer_core_label
      - edge_has_ring_type, edge_has_clique_type, scenario_type_count
    """
    trades = run.trades
    if trades.empty:
        return pd.DataFrame()

    trades = trades.copy()
    trades["sell_trader_id"] = trades["sell_trader_id"].astype(str)
    trades["buy_trader_id"] = trades["buy_trader_id"].astype(str)
    if "trade_id" not in trades.columns:
        trades["trade_id"] = [f"trade_{i:06d}" for i in range(len(trades))]
    if "price" not in trades.columns:
        trades["price"] = 0.0
    if "instrument_id" not in trades.columns:
        trades["instrument_id"] = "INST_0"
    trades["price"] = pd.to_numeric(trades["price"], errors="coerce").fillna(0.0)
    trades["quantity"] = pd.to_numeric(trades["quantity"], errors="coerce").fillna(0.0)

    scenarios = _build_scenario_index(run.scenarios)

    # Aggregate edge base.
    edge_base = (
        trades.groupby(["sell_trader_id", "buy_trader_id"])
        .agg(
            trade_count=("trade_id", "count"),
            total_quantity=("quantity", "sum"),
            mean_quantity=("quantity", "mean"),
            std_quantity=("quantity", "std"),
            max_quantity=("quantity", "max"),
            mean_price=("price", "mean"),
            std_price=("price", "std"),
            first_timestamp=("timestamp", "min"),
            last_timestamp=("timestamp", "max"),
            unique_instruments=("instrument_id", "nunique"),
        )
        .reset_index()
    )
    if edge_base.empty:
        return pd.DataFrame()

    edge_base["active_seconds_span"] = (
        edge_base["last_timestamp"] - edge_base["first_timestamp"]
    ).dt.total_seconds().fillna(0.0)
    edge_base["edge_direction_density"] = (
        edge_base["trade_count"] / (edge_base["active_seconds_span"] + 1.0)
    )

    instrument_counts = (
        trades.groupby(["sell_trader_id", "buy_trader_id", "instrument_id"])
        .agg(instrument_quantity=("quantity", "sum"))
        .reset_index()
    )
    instrument_max = (
        instrument_counts.groupby(["sell_trader_id", "buy_trader_id"])["instrument_quantity"]
        .max()
        .rename("max_instrument_quantity")
        .reset_index()
    )
    edge_base = edge_base.merge(instrument_max, on=["sell_trader_id", "buy_trader_id"], how="left")
    edge_base["instrument_concentration"] = (
        edge_base["max_instrument_quantity"] / (edge_base["total_quantity"] + 1e-6)
    )
    edge_base = edge_base.drop(columns=["max_instrument_quantity"])

    temporal_60 = _window_features(trades, 60, "w60")
    temporal_300 = _window_features(trades, 300, "w300")
    graph_df, _ = _graph_features(edge_base)

    merged = edge_base.merge(temporal_60, on=["sell_trader_id", "buy_trader_id"], how="left")
    merged = merged.merge(temporal_300, on=["sell_trader_id", "buy_trader_id"], how="left")
    merged = merged.merge(graph_df, on=["sell_trader_id", "buy_trader_id"], how="left")

    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        seller = str(row["sell_trader_id"])
        buyer = str(row["buy_trader_id"])
        edge_trades = trades[
            (trades["sell_trader_id"] == seller) & (trades["buy_trader_id"] == buyer)
        ]
        any_pos, core_pos, types = _edge_labels(edge_trades, scenarios)
        rev_tc = float(row.get("reverse_trade_count", 0.0))
        rev_q = float(row.get("reverse_total_quantity", 0.0))
        tot_q = float(row.get("total_quantity", 0.0))
        tc = float(row.get("trade_count", 0.0))
        s_out = float(row.get("seller_out_weight", 0.0))
        b_in = float(row.get("buyer_in_weight", 0.0))
        rev_balance = (
            min(tot_q, rev_q) / (max(tot_q, rev_q) + 1e-6)
            if (tot_q + rev_q) else 0.0
        )
        tc_recip = (
            min(tc, rev_tc) / (max(tc, rev_tc) + 1e-6)
            if (tc + rev_tc) else 0.0
        )
        rows.append({
            **row.to_dict(),
            "seller_edge_share_of_outflow": tot_q / (s_out + 1e-6),
            "buyer_edge_share_of_inflow": tot_q / (b_in + 1e-6),
            "quantity_reciprocity_balance": rev_balance,
            "trade_count_reciprocity_balance": tc_recip,
            "label_core_edge": core_pos,
            "label_any_edge": any_pos,
            "scenario_type_count": len(types),
            "edge_has_ring_type": int("circular_trading_ring" in types),
            "edge_has_clique_type": int("collusive_clique" in types),
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    manip_participants = {
        p
        for s in scenarios.values()
        for p in s.participants
        if s.scenario_type != "generic_background"
    }
    frame["seller_core_label"] = frame["sell_trader_id"].isin(manip_participants).astype(int)
    frame["buyer_core_label"] = frame["buy_trader_id"].isin(manip_participants).astype(int)
    frame = frame.fillna(0.0)
    drop_cols = ["first_timestamp", "last_timestamp"]
    frame = frame.drop(columns=[c for c in drop_cols if c in frame.columns])
    return frame
