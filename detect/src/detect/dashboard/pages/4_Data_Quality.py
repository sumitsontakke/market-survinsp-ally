"""Data Quality page.

For a chosen date, runs the stylized-facts checks against the NSE
bhavcopy + computes the cross-sectional concentration distribution.
Renders a pass/fail grid that mirrors the Phase 2 Review 2 deliverable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats as sci_stats

sys.path.insert(0, "/app")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from synth.calibration.core.config import (  # noqa: E402
    EMPIRICAL_RETURN_DF_RANGE,
    EMPIRICAL_VOLUME_ALPHA_RANGE,
    NIFTY_LIQUID_20,
    SYNTHETIC_BASELINES,
)
from synth.calibration.core.database import MarketDataDB  # noqa: E402

st.set_page_config(page_title="Data Quality", page_icon="📊", layout="wide")
st.title("Data Quality")
st.caption(
    "Stylized-facts checks against the bhavcopy store. Three of the five facts "
    "from the R2 deck (heavy tails, no autocorr, intraday volume) require "
    "1-minute data. The cross-sectional Pareto check on activity is computable "
    "from EOD alone."
)


@st.cache_resource
def get_db() -> MarketDataDB:
    return MarketDataDB()


def _hill_alpha(x: np.ndarray, q: float = 0.9) -> "tuple[float, np.ndarray, float]":
    x = np.sort(x[x > 0])
    if x.size < 30:
        return float("nan"), x, float("nan")
    x_min = float(np.percentile(x, q * 100))
    tail = x[x > x_min]
    if tail.size < 5 or x_min <= 0:
        return float("nan"), x, x_min
    denom = float(np.sum(np.log(tail / x_min)))
    return (tail.size / denom) if denom > 0 else float("nan"), x, x_min


def render() -> None:
    db = get_db()
    available = db.get_available_daily_dates()
    if not available:
        st.warning("No bhavcopy dates yet. Fetch some first.")
        return

    target_date = st.selectbox(
        "Date for cross-sectional checks",
        options=list(reversed(available)),
        index=0,
    )

    window_days = st.slider(
        "Trailing window for return-distribution check (days)",
        min_value=5, max_value=120, value=30, step=5,
    )

    # ----------------------------------------------------------------
    # Cross-sectional volume distribution (Fact E in Phase 2 framing)
    # ----------------------------------------------------------------
    vols = db.get_daily_universe_volumes(target_date)
    cross_section_ok = not vols.empty

    if cross_section_ok:
        alpha, sorted_x, x_min = _hill_alpha(vols.to_numpy())
        median = float(vols.median())
        p99 = float(vols.quantile(0.99))
        p99_med_ratio = p99 / max(median, 1)
        a_lo, a_hi = EMPIRICAL_VOLUME_ALPHA_RANGE
        alpha_in_band = a_lo <= alpha <= a_hi
    else:
        alpha = float("nan"); median = 0.0; p99 = 0.0; p99_med_ratio = 0.0
        alpha_in_band = False; sorted_x = np.array([]); x_min = float("nan")

    # ----------------------------------------------------------------
    # Daily-return distribution (Fact A: heavy tails)
    # ----------------------------------------------------------------
    end = pd.Timestamp(target_date)
    start = (end - pd.Timedelta(days=int(window_days * 1.6 + 7))).date().isoformat()
    df = db.get_daily_market_data(NIFTY_LIQUID_20, start, target_date)
    return_dist_ok = False
    df_fit = float("nan"); kurt = float("nan"); n = 0; rets = np.array([])
    if not df.empty:
        df = df.sort_values(["ticker", "trade_date"])
        df["log_ret"] = np.log(df["close"].astype(float) / df["prev_close"].astype(float).replace(0, np.nan))
        df["log_ret"] = df["log_ret"].fillna(
            np.log(df["close"].astype(float)).groupby(df["ticker"]).diff()
        )
        rets = df["log_ret"].dropna().to_numpy()
        if rets.size >= 30:
            df_fit, _, _ = sci_stats.t.fit(rets, floc=0.0)
            kurt = float(sci_stats.kurtosis(rets, fisher=True, bias=False))
            n = int(rets.size)
            return_dist_ok = True

    df_lo, df_hi = EMPIRICAL_RETURN_DF_RANGE
    df_in_band = (df_lo <= df_fit <= df_hi) if return_dist_ok else False

    # ----------------------------------------------------------------
    # Pass/fail grid
    # ----------------------------------------------------------------
    st.markdown("### Stylized-facts pass/fail")
    grid = [
        {
            "Fact": "A. Heavy-tailed daily returns",
            "Available?": "✓" if return_dist_ok else "✗ (needs ≥30 returns)",
            "NSE value": (f"Student-t df = {df_fit:.2f}, kurt = {kurt:.2f}" if return_dist_ok else "—"),
            "Synthetic baseline": f"df = {SYNTHETIC_BASELINES['return_df']:.2f}",
            "Empirical band": f"df ∈ [{df_lo}, {df_hi}]",
            "Status": ("PASS" if df_in_band else "GAP" if return_dist_ok else "skip"),
        },
        {
            "Fact": "B. Volatility clustering",
            "Available?": "✗ needs intraday",
            "NSE value": "—",
            "Synthetic baseline": "ACF(|r|²) lag-1 ≈ 0.27",
            "Empirical band": "many lags > CI",
            "Status": "skip (EOD)",
        },
        {
            "Fact": "C. Absence of return autocorr",
            "Available?": "✗ needs intraday",
            "NSE value": "—",
            "Synthetic baseline": "mean |ACF| ≈ 0.013",
            "Empirical band": "≤ CI",
            "Status": "skip (EOD)",
        },
        {
            "Fact": "D. Intraday volume U-shape",
            "Available?": "✗ needs intraday",
            "NSE value": "—",
            "Synthetic baseline": "open > middle, close > middle",
            "Empirical band": "U-shape",
            "Status": "skip (EOD)",
        },
        {
            "Fact": "E. Cross-sectional volume Pareto",
            "Available?": "✓" if cross_section_ok else "✗ no rows",
            "NSE value": (f"α = {alpha:.2f},  p99/median = {p99_med_ratio:.0f}×"
                         if cross_section_ok else "—"),
            "Synthetic baseline": f"{SYNTHETIC_BASELINES['volume_alpha']}",
            "Empirical band": f"α ∈ [{a_lo}, {a_hi}]",
            "Status": ("PASS" if alpha_in_band else "GAP" if cross_section_ok else "skip"),
        },
    ]

    grid_df = pd.DataFrame(grid)
    def style_status(val: str) -> str:
        if val == "PASS":
            return "background-color: #DCFCE7; color: #14532D; font-weight: bold;"
        if val == "GAP":
            return "background-color: #FEE2E2; color: #991B1B; font-weight: bold;"
        return "background-color: #F4F6FA; color: #5C6480;"
    styled = grid_df.style.map(style_status, subset=["Status"])
    st.dataframe(styled, hide_index=True, use_container_width=True)

    # ----------------------------------------------------------------
    # Plots
    # ----------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Plots")

    plot_col1, plot_col2 = st.columns(2)

    with plot_col1:
        st.markdown("**Cross-sectional volume (Zipf log-log)**")
        if cross_section_ok and sorted_x.size > 0:
            v = sorted_x[sorted_x > 0][::-1]
            rank = np.arange(1, v.size + 1)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.loglog(rank, v, "o", color="#1E2761", markersize=2.5, alpha=0.5,
                      label="rank vs volume")
            if not np.isnan(alpha) and alpha > 0:
                y_fit = v[0] * (rank / rank[0]) ** (-1.0 / max(alpha, 1e-3))
                ax.loglog(rank, y_fit, color="#C8102E", linewidth=2,
                          label=f"Pareto fit (α = {alpha:.2f})")
            ax.set_xlabel("Rank (log)")
            ax.set_ylabel("Volume (log)")
            ax.set_title(f"Cross-sectional Volume — {target_date}")
            ax.legend(fontsize=9)
            ax.grid(True, which="both", linestyle=":", alpha=0.4)
            fig.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No data for this date.")

    with plot_col2:
        st.markdown("**Daily log-return distribution**")
        if return_dist_ok:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(rets, bins=60, density=True, alpha=0.55, color="#1E2761",
                    label="Pooled NIFTY 20 daily returns")
            grid_x = np.linspace(rets.min(), rets.max(), 400)
            scale = float(np.std(rets, ddof=1))
            ax.plot(grid_x, sci_stats.t.pdf(grid_x, df_fit, loc=0.0, scale=scale),
                    color="#C8102E", linewidth=2,
                    label=f"Student-t fit (df = {df_fit:.2f})")
            ax.plot(grid_x,
                    sci_stats.norm.pdf(grid_x, np.mean(rets), scale),
                    color="#0F7A4D", linewidth=2, linestyle="--",
                    label="Gaussian (for comparison)")
            ax.set_yscale("log")
            ax.set_xlabel("Daily log-return")
            ax.set_ylabel("Density (log)")
            ax.set_title(f"Daily Return Distribution — {window_days}d window")
            ax.legend(fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)
        else:
            st.info(
                "Insufficient pooled returns for a Student-t fit. Increase the trailing window or fetch more bhavcopy dates."
            )

    # ----------------------------------------------------------------
    # Numeric summary card
    # ----------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Summary")
    if cross_section_ok:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tickers", f"{len(vols):,}")
        c2.metric("Median volume", f"{median:,.0f}")
        c3.metric("p99 volume", f"{p99:,.0f}")
        c4.metric("p99 / median", f"{p99_med_ratio:.0f}×",
                  delta=f"vs synth 1.11×",
                  delta_color="inverse")


render()
