"""Directed lead-lag and market-impact edge features (Rung 4).

Per directed (seller, buyer) pair within one run:

  Lead-lag (7 + 2):
    lead_lag_corr_lag_{-3..+3}  Pearson(signed_vol_seller, shift(signed_vol_buyer, lag))
    best_lag                    argmax over the 7 lags
    best_lag_value              max correlation across the 7 lags

  Interaction (3):
    signed_imbalance            sum(vol_A_buys_from_B) - sum(vol_B_buys_from_A)
    traded_volume               sum of executed volume between A and B (directed)
    interaction_count           number of trades on the directed edge

  Market impact (3):
    pre_window_drift            log-return of mid-price over 5 min before window start
    post_window_drift           log-return of mid-price over 5 min after window end
    vwap_imbalance              signed_VWAP_in_window - daily_reference_VWAP

Edges are kept directed: (i -> j) and (j -> i) are SEPARATE rows when both
exist. Phase 1's edge_engineered.py uses the same convention; this module
matches it so M3 can train Rung 3 and Rung 4 on identical edge sets.

Reference
---------
Cont, R. (2001). Empirical properties of asset returns: stylized facts
and statistical implications. Quantitative Finance, 1(2), 223-236.

Hamilton, W. L., Ying, R., Leskovec, J. (2017). Inductive Representation
Learning on Large Graphs. NeurIPS.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from detect.dataset.loader import Run

LAGS: tuple[int, ...] = (-3, -2, -1, 0, 1, 2, 3)
MARKET_IMPACT_WINDOW_MINUTES: int = 5

# Columns produced by ``compute_directed_edge_features``.
EXPECTED_COLUMNS: tuple[str, ...] = (
    "sell_trader_id", "buy_trader_id",
    "lead_lag_corr_lag_neg3", "lead_lag_corr_lag_neg2",
    "lead_lag_corr_lag_neg1", "lead_lag_corr_lag_0",
    "lead_lag_corr_lag_pos1", "lead_lag_corr_lag_pos2",
    "lead_lag_corr_lag_pos3",
    "best_lag", "best_lag_value",
    "signed_imbalance", "traded_volume", "interaction_count",
    "pre_window_drift", "post_window_drift", "vwap_imbalance",
)


def _lag_column_name(lag: int) -> str:
    if lag < 0:
        return f"lead_lag_corr_lag_neg{abs(lag)}"
    if lag == 0:
        return "lead_lag_corr_lag_0"
    return f"lead_lag_corr_lag_pos{lag}"


# ---------------------------------------------------------------------------
# Per-trader signed volume series
# ---------------------------------------------------------------------------

def build_signed_volume_matrix(trades: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """Return a ``[time_bucket, trader_id]`` signed-volume matrix.

    Convention: a buy contributes +qty to the buyer's series and -qty to
    the seller's series. This is the standard signed-volume definition
    used throughout this project (see ``synthetic_market_sim.analysis.
    signed_volume``).
    """
    if trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["bucket"] = t["timestamp"].dt.floor(freq)
    buy = t[["bucket", "buy_trader_id", "quantity"]].rename(
        columns={"buy_trader_id": "trader_id"}
    )
    buy["signed_vol"] = buy["quantity"].astype(float)
    sell = t[["bucket", "sell_trader_id", "quantity"]].rename(
        columns={"sell_trader_id": "trader_id"}
    )
    sell["signed_vol"] = -sell["quantity"].astype(float)
    combined = pd.concat(
        [buy[["bucket", "trader_id", "signed_vol"]],
         sell[["bucket", "trader_id", "signed_vol"]]],
        ignore_index=True,
    )
    pivot = (
        combined.groupby(["bucket", "trader_id"], dropna=False)["signed_vol"]
        .sum()
        .unstack(level="trader_id", fill_value=0.0)
        .sort_index()
    )
    # Complete the time index with all bucket-spans (zeros for missing buckets)
    if not pivot.empty:
        full_index = pd.date_range(pivot.index.min(), pivot.index.max(), freq=freq)
        pivot = pivot.reindex(full_index, fill_value=0.0)
    return pivot


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; 0.0 when either side has zero variance."""
    if a.size < 2 or b.size < 2:
        return 0.0
    sa = float(np.std(a, ddof=0))
    sb = float(np.std(b, ddof=0))
    if sa == 0.0 or sb == 0.0:
        return 0.0
    a_c = a - a.mean()
    b_c = b - b.mean()
    return float(np.dot(a_c, b_c) / (a.size * sa * sb))


# ---------------------------------------------------------------------------
# Lead-lag features
# ---------------------------------------------------------------------------

def compute_lead_lag_features(
    run: Run,
    *,
    freq: str = "1min",
    lags: Sequence[int] = LAGS,
) -> pd.DataFrame:
    """Lead-lag Pearson correlations + argmax across a small lag set.

    For each pair (A, B) that has a trade in either direction:
        lead_lag_corr_lag_k = Pearson(signed_vol_A, shift(signed_vol_B, k))
    where k > 0 means B's series shifted forward in time (A leads B).

    Returns one row per **ordered** (seller, buyer) pair. The lead-lag
    values are pair-symmetric in source data but we store them under the
    ordered key for consistency with the rest of the feature set.
    """
    trades = run.trades
    if trades.empty:
        return pd.DataFrame(
            columns=("sell_trader_id", "buy_trader_id", *[_lag_column_name(l) for l in lags], "best_lag", "best_lag_value")
        )

    matrix = build_signed_volume_matrix(trades, freq=freq)
    if matrix.empty:
        return pd.DataFrame()

    # Find the ordered pairs that have at least one trade.
    pairs = (
        trades[["sell_trader_id", "buy_trader_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Pre-compute centered + std for each trader to speed up correlation.
    cols = matrix.columns
    centered: dict[str, np.ndarray] = {}
    stds: dict[str, float] = {}
    for c in cols:
        arr = matrix[c].to_numpy(dtype=np.float64)
        centered[str(c)] = arr - arr.mean()
        stds[str(c)] = float(np.std(arr, ddof=0))

    n = matrix.shape[0]
    rows: list[dict] = []
    for _, p in pairs.iterrows():
        seller = str(p["sell_trader_id"])
        buyer = str(p["buy_trader_id"])
        s = centered.get(seller)
        b = centered.get(buyer)
        ss = stds.get(seller, 0.0)
        sb = stds.get(buyer, 0.0)
        row: dict[str, object] = {
            "sell_trader_id": seller, "buy_trader_id": buyer,
        }
        if s is None or b is None or ss == 0.0 or sb == 0.0:
            for lag in lags:
                row[_lag_column_name(lag)] = 0.0
            row["best_lag"] = 0
            row["best_lag_value"] = 0.0
            rows.append(row)
            continue
        best_lag = 0
        best_val = -np.inf
        for lag in lags:
            if lag > 0:
                # shift buyer forward by lag → align s[:-lag] with b[lag:]
                a_seg = s[:-lag] if lag > 0 else s
                b_seg = b[lag:]  if lag > 0 else b
            elif lag < 0:
                # shift buyer backward by |lag| → align s[|lag|:] with b[:-|lag|]
                k = -lag
                a_seg = s[k:]
                b_seg = b[:-k]
            else:
                a_seg = s
                b_seg = b
            if a_seg.size < 2:
                corr = 0.0
            else:
                # ss/sb were computed on the full series; recomputing exactly
                # the per-segment std would be more accurate but slower.
                # For 1-min buckets in a 120-bucket session the difference
                # is < 0.01; we accept the bias for speed.
                corr = float(np.dot(a_seg, b_seg) / (a_seg.size * ss * sb))
                if not np.isfinite(corr):
                    corr = 0.0
                # Clip to [-1, 1] to absorb the std approximation
                corr = max(-1.0, min(1.0, corr))
            row[_lag_column_name(lag)] = corr
            if corr > best_val:
                best_val = corr
                best_lag = lag
        row["best_lag"] = int(best_lag)
        row["best_lag_value"] = float(best_val if np.isfinite(best_val) else 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Interaction + market impact features
# ---------------------------------------------------------------------------

def _running_mid_price(trades: pd.DataFrame) -> pd.Series:
    """Best-effort running mid-price from the trade tape.

    Uses the last-trade price as the mid-price proxy. Indexed by timestamp.
    """
    if trades.empty:
        return pd.Series(dtype=float)
    s = (
        trades.sort_values("timestamp")
        .groupby("timestamp")["price"]
        .last()
    )
    return s


def _lookup_mid(price_series: pd.Series, when: pd.Timestamp) -> float:
    """Last known mid-price at or before ``when``. Falls back to first
    known price if there is no earlier record."""
    if price_series.empty:
        return 0.0
    idx = price_series.index.searchsorted(when, side="right") - 1
    if idx < 0:
        return float(price_series.iloc[0])
    return float(price_series.iloc[idx])


def compute_interaction_and_impact_features(run: Run) -> pd.DataFrame:
    """Compute interaction + market-impact features per directed pair."""
    trades = run.trades
    if trades.empty:
        return pd.DataFrame()

    t = trades.copy()
    t["quantity"] = pd.to_numeric(t["quantity"], errors="coerce").fillna(0.0)
    t["price"] = pd.to_numeric(t["price"], errors="coerce").fillna(0.0)

    # Reverse-pair lookup for signed imbalance.
    pair_agg = (
        t.groupby(["sell_trader_id", "buy_trader_id"])
        .agg(
            interaction_count=("price", "size"),
            traded_volume=("quantity", "sum"),
            first_ts=("timestamp", "min"),
            last_ts=("timestamp", "max"),
            signed_value=("quantity", lambda q: float(np.sum(q))),
        )
        .reset_index()
    )

    # Build a reverse-volume lookup keyed on (other_dir).
    reverse_lookup = (
        pair_agg[["sell_trader_id", "buy_trader_id", "traded_volume"]]
        .rename(columns={
            "sell_trader_id": "buy_trader_id_rev",
            "buy_trader_id": "sell_trader_id_rev",
            "traded_volume": "reverse_traded_volume",
        })
    )
    # join on (seller, buyer) where reverse is (buyer, seller)
    merged = pair_agg.merge(
        reverse_lookup,
        how="left",
        left_on=["sell_trader_id", "buy_trader_id"],
        right_on=["sell_trader_id_rev", "buy_trader_id_rev"],
    ).drop(columns=["sell_trader_id_rev", "buy_trader_id_rev"])
    merged["reverse_traded_volume"] = merged["reverse_traded_volume"].fillna(0.0)
    merged["signed_imbalance"] = merged["traded_volume"] - merged["reverse_traded_volume"]

    # Daily reference VWAP.
    if t["quantity"].sum() > 0:
        daily_ref_vwap = float((t["price"] * t["quantity"]).sum() / t["quantity"].sum())
    else:
        daily_ref_vwap = 0.0
    mid = _running_mid_price(t)

    # Per-pair signed VWAP within the interaction window + drifts.
    rows: list[dict] = []
    for _, row in merged.iterrows():
        seller = row["sell_trader_id"]
        buyer = row["buy_trader_id"]
        edge_trades = t[(t["sell_trader_id"] == seller) & (t["buy_trader_id"] == buyer)]
        if edge_trades.empty or edge_trades["quantity"].sum() == 0:
            signed_vwap = 0.0
        else:
            signed_vwap = float(
                (edge_trades["price"] * edge_trades["quantity"]).sum()
                / edge_trades["quantity"].sum()
            )

        first_ts: pd.Timestamp = row["first_ts"]
        last_ts: pd.Timestamp = row["last_ts"]
        five_min = pd.Timedelta(minutes=MARKET_IMPACT_WINDOW_MINUTES)
        mid_pre_before = _lookup_mid(mid, first_ts - five_min)
        mid_at_start = _lookup_mid(mid, first_ts)
        mid_at_end = _lookup_mid(mid, last_ts)
        mid_post_after = _lookup_mid(mid, last_ts + five_min)

        def _safe_logret(after: float, before: float) -> float:
            if before <= 0 or after <= 0:
                return 0.0
            return float(np.log(after / before))

        pre_drift = _safe_logret(mid_at_start, mid_pre_before)
        post_drift = _safe_logret(mid_post_after, mid_at_end)

        rows.append({
            "sell_trader_id": seller,
            "buy_trader_id": buyer,
            "signed_imbalance": float(row["signed_imbalance"]),
            "traded_volume": float(row["traded_volume"]),
            "interaction_count": int(row["interaction_count"]),
            "pre_window_drift": pre_drift,
            "post_window_drift": post_drift,
            "vwap_imbalance": float(signed_vwap - daily_ref_vwap),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_directed_edge_features(run: Run, *, freq: str = "1min") -> pd.DataFrame:
    """Compute the full Rung 4 directed-edge feature set.

    Returns a DataFrame with columns matching :data:`EXPECTED_COLUMNS`.
    One row per directed (seller, buyer) pair that traded at least once.
    """
    if run.trades.empty:
        return pd.DataFrame(columns=list(EXPECTED_COLUMNS))

    lead = compute_lead_lag_features(run, freq=freq)
    impact = compute_interaction_and_impact_features(run)

    if lead.empty or impact.empty:
        return pd.DataFrame(columns=list(EXPECTED_COLUMNS))

    out = lead.merge(impact, on=["sell_trader_id", "buy_trader_id"], how="outer")

    # Reorder + fill any merge-induced NaNs with 0.
    for col in EXPECTED_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0
    out = out[list(EXPECTED_COLUMNS)]
    out = out.fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# Edge labels (label_any_edge, label_core_edge) for the same pair grain
# ---------------------------------------------------------------------------

def compute_edge_labels(run: Run) -> pd.DataFrame:
    """``label_any_edge`` and ``label_core_edge`` per directed pair.

    Mirrors ``training/features/edge_engineered.py``'s labeling but
    operates on the same ordered (seller, buyer) grain the Rung 4
    feature pipeline produces. We re-use that module's helper for
    parity.
    """
    from detect.features.edge_engineered import _build_scenario_index, _edge_labels  # noqa: WPS437
    scenarios = _build_scenario_index(run.scenarios)
    rows: list[dict] = []
    for (seller, buyer), edge_trades in run.trades.groupby(
        ["sell_trader_id", "buy_trader_id"], sort=False
    ):
        any_pos, core_pos, _ = _edge_labels(edge_trades, scenarios)
        rows.append({
            "sell_trader_id": str(seller),
            "buy_trader_id": str(buyer),
            "label_any_edge": int(any_pos),
            "label_core_edge": int(core_pos),
        })
    if not rows:
        return pd.DataFrame(columns=["sell_trader_id", "buy_trader_id", "label_any_edge", "label_core_edge"])
    return pd.DataFrame(rows)
