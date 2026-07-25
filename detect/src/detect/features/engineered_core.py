"""Engineered manipulation-detection features - shared computation core.

Six per-trader scalars, each scoped to the trader's busiest 5-minute
window. This module is the single source of truth for the feature math;
two callers import it so the numbers can never drift apart:

  - scripts/compute_engineered_features.py  - analysis CLI + per-feature
    ROC AUC (the tier-2 study, vault note 46/48).
  - training/features/node_engineered.py    - GNN node-feature injection
    for the Path A2 feature-augmented GraphSAGE retrain.

Because both paths call ``compute_features_frame`` here, the features
analysed in the tier-2 study and the features fed to the GNN are
provably the same code.

Features (per trader, scoped to busiest 5-minute window):
  1. burst_concentration         - fraction of orders in peak 5-min window
  2. side_entropy_in_burst       - Shannon entropy of buy/sell in burst
  3. counterparty_hhi_burst      - Herfindahl-Hirschman of CP shares in burst
  4. order_qty_cov               - coefficient of variation of order qty
  5. top_partner_trade_share     - fraction of burst trades vs top-1 CP
  6. co_active_top_count         - count of co-active top-quantile traders

Pure CPU pandas/numpy - no torch, no sklearn - so it imports cleanly
inside the trainer container as well as the analysis environment.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import pandas as pd


FEATURE_NAMES: tuple[str, ...] = (
    "burst_concentration",
    "side_entropy_in_burst",
    "counterparty_hhi_burst",
    "order_qty_cov",
    "top_partner_trade_share",
    "co_active_top_count",
)

FEATURE_DESCRIPTIONS = {
    "burst_concentration": (
        "Fraction of the trader's orders inside their busiest 5-minute "
        "window. Clique manipulators concentrate bursts (high); "
        "diversified traders spread (low)."
    ),
    "side_entropy_in_burst": (
        "Normalised Shannon entropy of buy/sell in the burst window. "
        "0 = fully one-sided (clique signature); 1 = balanced (wash signature)."
    ),
    "counterparty_hhi_burst": (
        "Herfindahl-Hirschman index of counterparty shares during the burst. "
        "1.0 = single partner (ring signature); near 0 = diversified."
    ),
    "order_qty_cov": (
        "Coefficient of variation (std/mean) of order quantities. "
        "Synth manipulators use uniform burst quantities (low); "
        "legitimate trading varies more (high)."
    ),
    "top_partner_trade_share": (
        "Fraction of burst-window trades against the single most-frequent "
        "counterparty. Ring members concentrate (high); benign spreads (low)."
    ),
    "co_active_top_count": (
        "Number of other top-quantile-active traders co-active in this "
        "trader's burst window. High values flag traders whose peaks "
        "overlap with many other 'busy' traders - a structural group marker."
    ),
}

# Columns the per-trader computation emits in addition to FEATURE_NAMES.
_META_COLS: tuple[str, ...] = (
    "burst_start", "burst_end", "n_orders", "n_burst_orders",
)


def _shannon_entropy_binary(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    q = 1.0 - p
    return -(p * math.log2(p) + q * math.log2(q))


def _find_burst_window(timestamps: pd.Series, window_min: int = 5
                       ) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Returns (start, end) of the 5-min rolling window with most orders."""
    if timestamps.empty:
        return pd.NaT, pd.NaT
    ts = pd.to_datetime(timestamps).sort_values()
    by_min = ts.dt.floor("1min").value_counts().sort_index()
    if len(by_min) <= window_min:
        return ts.min(), ts.max()
    rolling = by_min.rolling(f"{window_min}min").sum()
    end_ts = rolling.idxmax()
    start_ts = end_ts - pd.Timedelta(minutes=window_min - 1)
    return start_ts, end_ts


def _default_feature_row(trader_id: str, n_orders: int = 0) -> dict:
    """Neutral row for a node with no usable order data."""
    return {
        "trader_id":               str(trader_id),
        "burst_concentration":     0.0,
        "side_entropy_in_burst":   0.0,
        "counterparty_hhi_burst":  0.0,
        "order_qty_cov":           0.0,
        "top_partner_trade_share": 0.0,
        "co_active_top_count":     0,
        "burst_start":             "",
        "burst_end":               "",
        "n_orders":                int(n_orders),
        "n_burst_orders":          0,
    }


def _features_for_one_trader(trader_id: str,
                             own_orders: pd.DataFrame,
                             my_trades: pd.DataFrame,
                             orders_all: pd.DataFrame,
                             top_active_set: set) -> dict:
    """The six engineered features for a single trader. No label column.

    This is the verbatim per-trader computation extracted from the
    original ``compute_features_for_run`` loop body, so analysis-mode
    output is unchanged.
    """
    # 1. burst_concentration
    start_ts, end_ts = _find_burst_window(own_orders["timestamp"], 5)
    in_burst = own_orders[(own_orders["timestamp"] >= start_ts)
                          & (own_orders["timestamp"] <= end_ts)]
    burst_concentration = (len(in_burst) / len(own_orders)
                           if len(own_orders) else 0.0)

    # 2. side_entropy_in_burst
    if len(in_burst) == 0:
        side_entropy = 1.0
    else:
        p_buy = float((in_burst["side"] == "buy").mean())
        side_entropy = _shannon_entropy_binary(p_buy)

    # Burst trades for this trader.
    if my_trades is not None and not my_trades.empty:
        burst_trades = my_trades[(my_trades["timestamp"] >= start_ts)
                                 & (my_trades["timestamp"] <= end_ts)]
    else:
        burst_trades = pd.DataFrame()

    # Counterparty IDs per burst trade (vectorised).
    if not burst_trades.empty:
        buy_mask = burst_trades["buy_trader_id"].values == trader_id
        cp_arr = np.where(
            buy_mask,
            burst_trades["sell_trader_id"].values,
            burst_trades["buy_trader_id"].values,
        )
        cp_counts = pd.Series(cp_arr).value_counts()
    else:
        cp_counts = pd.Series(dtype=int)

    # 3. counterparty_hhi_burst
    if len(cp_counts) > 0:
        shares = cp_counts.values / cp_counts.sum()
        cp_hhi = float(np.sum(shares ** 2))
    else:
        cp_hhi = 0.0

    # 4. order_qty_cov
    qty = own_orders["quantity"].astype(float)
    order_qty_cov = (float(qty.std() / qty.mean())
                     if len(qty) >= 2 and qty.mean() > 0 else 0.0)

    # 5. top_partner_trade_share
    if len(cp_counts) > 0:
        top_partner_share = float(cp_counts.iloc[0] / cp_counts.sum())
    else:
        top_partner_share = 0.0

    # 6. co_active_top_count
    if pd.notna(start_ts) and pd.notna(end_ts):
        in_window = orders_all[(orders_all["timestamp"] >= start_ts)
                               & (orders_all["timestamp"] <= end_ts)]
        co_active = set(in_window["trader_id"].unique()) & top_active_set
        co_active.discard(trader_id)
        co_active_count = len(co_active)
    else:
        co_active_count = 0

    return {
        "trader_id":               str(trader_id),
        "burst_concentration":     float(burst_concentration),
        "side_entropy_in_burst":   float(side_entropy),
        "counterparty_hhi_burst":  float(cp_hhi),
        "order_qty_cov":           float(order_qty_cov),
        "top_partner_trade_share": float(top_partner_share),
        "co_active_top_count":     int(co_active_count),
        "burst_start":             str(start_ts) if pd.notna(start_ts) else "",
        "burst_end":               str(end_ts) if pd.notna(end_ts) else "",
        "n_orders":                int(len(own_orders)),
        "n_burst_orders":          int(len(in_burst)),
    }


def _prepare(orders: pd.DataFrame,
             trades: Optional[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Copy + normalise timestamps / dtypes. Idempotent on parsed input."""
    orders = orders.copy()
    if "timestamp" in orders.columns:
        orders["timestamp"] = pd.to_datetime(orders["timestamp"],
                                             errors="coerce")
    if "trader_id" in orders.columns:
        orders["trader_id"] = orders["trader_id"].astype(str)
    # Some synth exports name the order size column "volume".
    if "quantity" not in orders.columns and "volume" in orders.columns:
        orders = orders.rename(columns={"volume": "quantity"})

    if trades is None or trades.empty:
        trades = pd.DataFrame()
    else:
        trades = trades.copy()
        if "timestamp" in trades.columns:
            trades["timestamp"] = pd.to_datetime(trades["timestamp"],
                                                 errors="coerce")
        for col in ("buy_trader_id", "sell_trader_id"):
            if col in trades.columns:
                trades[col] = trades[col].astype(str)
    return orders, trades


def compute_features_frame(orders: pd.DataFrame,
                           trades: Optional[pd.DataFrame],
                           *,
                           trader_ids: Optional[Sequence[str]] = None,
                           keep_top_active: int = 200,
                           min_orders: int = 3) -> pd.DataFrame:
    """Six engineered features per trader. No label column.

    ``trader_ids is None`` -> analysis mode: featurise the
        ``keep_top_active`` most-active traders plus all manipulators;
        skip traders with fewer than ``min_orders`` orders. Row count
        varies. This reproduces the original analysis behaviour exactly.

    ``trader_ids`` given   -> node mode (Path A2): emit exactly one row
        per id, in the given order; ids with no orders get a neutral
        default row. ``keep_top_active`` / ``min_orders`` are ignored.

    Returns columns: trader_id, the six FEATURE_NAMES, then burst_start,
    burst_end, n_orders, n_burst_orders.
    """
    orders, trades = _prepare(orders, trades)
    node_mode = trader_ids is not None

    if orders.empty or "trader_id" not in orders.columns:
        if node_mode:
            return pd.DataFrame(
                [_default_feature_row(str(t)) for t in trader_ids])
        return pd.DataFrame(columns=["trader_id", *FEATURE_NAMES, *_META_COLS])

    by_trader = orders.groupby("trader_id").size()
    top_quantile_cut = by_trader.quantile(0.75) if len(by_trader) else 0.0
    top_active_set = set(by_trader[by_trader >= top_quantile_cut].index)

    if node_mode:
        target_ids = [str(t) for t in trader_ids]
        keep_ids: set = set(target_ids)
    else:
        activity = by_trader.sort_values(ascending=False)
        keep_ids = set(activity.head(int(keep_top_active)).index)
        if "is_manipulative" in orders.columns:
            keep_ids |= set(
                orders.loc[orders["is_manipulative"] == True,   # noqa: E712
                           "trader_id"].astype(str).unique())
        target_ids = None  # analysis mode iterates the groupby

    orders_kept = orders[orders["trader_id"].isin(keep_ids)].copy()
    own_orders_by = {tid: g for tid, g in orders_kept.groupby("trader_id")}

    # Pre-index trades by trader -> DataFrame slice.
    trader_trade_idx: dict[str, pd.DataFrame] = {}
    if not trades.empty:
        for tid in keep_ids:
            mask = ((trades["buy_trader_id"] == tid)
                    | (trades["sell_trader_id"] == tid))
            sub = trades.loc[mask]
            if not sub.empty:
                trader_trade_idx[tid] = sub

    rows: list[dict] = []
    if node_mode:
        for tid in target_ids:
            own = own_orders_by.get(tid)
            if own is None or len(own) == 0:
                rows.append(_default_feature_row(tid, n_orders=0))
                continue
            rows.append(_features_for_one_trader(
                tid, own, trader_trade_idx.get(tid, pd.DataFrame()),
                orders, top_active_set))
    else:
        for tid, own in own_orders_by.items():
            if len(own) < min_orders:
                continue
            rows.append(_features_for_one_trader(
                tid, own, trader_trade_idx.get(tid, pd.DataFrame()),
                orders, top_active_set))

    if not rows:
        return pd.DataFrame(columns=["trader_id", *FEATURE_NAMES, *_META_COLS])
    return pd.DataFrame(rows)[["trader_id", *FEATURE_NAMES, *_META_COLS]]
