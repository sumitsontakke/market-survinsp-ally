"""Calibrate page.

Runs ``NSEDailyCalibrator`` (or the intraday path) for a chosen date,
renders the four parameters, the gap-vs-baseline panel, and any warnings.
Also previews a Python snippet showing how to wire CalibrationParams
into ``synthetic_market_sim``.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, "/app")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from synth.calibration.core.config import (  # noqa: E402
    EMPIRICAL_RETURN_DF_RANGE,
    EMPIRICAL_VOLUME_ALPHA_RANGE,
    NIFTY_LIQUID_20,
    SYNTHETIC_BASELINES,
)
from synth.calibration.core.database import MarketDataDB  # noqa: E402
from synth.calibration.core.nse_calibrator_daily import NSEDailyCalibrator  # noqa: E402

st.set_page_config(page_title="Calibrate", page_icon="🎯", layout="wide")
st.title("Calibrate")
st.caption(
    "Compute the four NSE-derived parameters that feed into "
    "`synthetic_market_sim`. Daily mode reads the bhavcopy store; intraday "
    "mode reads the 1-minute yfinance store."
)


@st.cache_resource
def get_db() -> MarketDataDB:
    return MarketDataDB()


def _badge(in_band: bool, text_in: str = "in band", text_out: str = "out of band") -> str:
    return (
        f"<span class='badge badge-ok'>{text_in}</span>"
        if in_band
        else f"<span class='badge badge-missing'>{text_out}</span>"
    )


def render() -> None:
    db = get_db()

    # ----------------------------------------------------------------
    # Inputs
    # ----------------------------------------------------------------
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        mode = st.radio(
            "Mode",
            ["daily (bhavcopy)", "intraday (yfinance)"],
            help="Daily mode is the recommended default; intraday requires the yfinance fetcher to have populated the 1-minute store.",
        )
    with c2:
        if mode.startswith("daily"):
            available = db.get_available_daily_dates()
        else:
            available = db.get_available_dates()
        if available:
            target_date = st.selectbox(
                "Calibration date",
                options=["latest", *list(reversed(available))],
                index=0,
            )
        else:
            target_date = st.text_input(
                "Calibration date",
                value="latest",
                help="No dates in the store yet. Fetch some first.",
            )
    with c3:
        if mode.startswith("daily"):
            window_days = st.number_input(
                "Trailing window (days)",
                min_value=2, max_value=120, value=30, step=1,
                help="Pool returns over this trailing window for the Student-t fit and realized-vol estimate.",
            )
        else:
            window_days = None

    if not available:
        st.warning(
            "No dates available in the selected store. Run **Fetch Bhavcopy** first "
            "(daily mode) or the yfinance fetcher (intraday mode)."
        )
        return

    run = st.button("Calibrate", type="primary", use_container_width=True)
    if not run:
        return

    # ----------------------------------------------------------------
    # Run
    # ----------------------------------------------------------------
    t0 = time.perf_counter()
    try:
        if mode.startswith("daily"):
            cal = NSEDailyCalibrator(
                db_path=db.db_path,
                focus_universe=NIFTY_LIQUID_20,
                pool_window_days=int(window_days),
            )
            params = cal.calibrate(target_date if target_date != "latest" else None)
        else:
            from synth.calibration.core.nse_calibrator import NSECalibrator
            cal = NSECalibrator(db.db_path)
            params = cal.calibrate(target_date if target_date != "latest" else None)
    except ValueError as exc:
        st.error(f"Calibration failed: {exc}")
        return
    elapsed = time.perf_counter() - t0
    st.success(f"Done in {elapsed:.2f}s — calibration date `{params.calibration_date}`")

    # ----------------------------------------------------------------
    # Headline parameter cards
    # ----------------------------------------------------------------
    df_lo, df_hi = EMPIRICAL_RETURN_DF_RANGE
    a_lo, a_hi = EMPIRICAL_VOLUME_ALPHA_RANGE
    df_in = df_lo <= params.return_df <= df_hi
    a_in = a_lo <= params.volume_alpha <= a_hi

    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("realized_volatility", f"{params.realized_volatility:.4f}")
    cc2.metric("return_df", f"{params.return_df:.3f}",
               delta="OK" if df_in else "GAP",
               delta_color="off" if df_in else "inverse")
    cc3.metric("volume_alpha", f"{params.volume_alpha:.3f}",
               delta="OK" if a_in else "GAP",
               delta_color="off" if a_in else "inverse")
    cc4.metric("Tickers used", f"{len(params.tickers_used)}")

    # ----------------------------------------------------------------
    # Gap vs synthetic baseline
    # ----------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Gap vs Phase 1 synthetic baseline")
    st.markdown(
        f"""
        | Parameter | NSE value | Phase 1 synthetic | Empirical band | Status |
        |---|---|---|---|---|
        | `realized_volatility` | `{params.realized_volatility:.4f}` | `{SYNTHETIC_BASELINES['realized_volatility']}` | n/a | <span class='badge badge-ok'>computed</span> |
        | `return_df` (Student-t) | `{params.return_df:.3f}` | `{SYNTHETIC_BASELINES['return_df']}` (df=2.00) | [{df_lo}–{df_hi}] | {_badge(df_in)} |
        | `return_scale` | `{params.return_scale:.5f}` | n/a | n/a | <span class='badge badge-ok'>computed</span> |
        | `volume_alpha` (Hill α) | `{params.volume_alpha:.3f}` | `{SYNTHETIC_BASELINES['volume_alpha']}` | [{a_lo}–{a_hi}] | {_badge(a_in)} |
        """,
        unsafe_allow_html=True,
    )

    # ----------------------------------------------------------------
    # Warnings + raw summary
    # ----------------------------------------------------------------
    if params.warnings:
        with st.expander(f"Warnings ({len(params.warnings)})", expanded=True):
            for w in params.warnings:
                st.write(f"- {w}")

    with st.expander("Full text summary", expanded=False):
        st.code(params.summary(), language="text")

    # ----------------------------------------------------------------
    # Inline plots if intraday
    # ----------------------------------------------------------------
    if params.intraday_volume_profile:
        st.markdown("---")
        st.markdown("### Intraday volume profile (375 minutes)")
        prof = pd.Series(params.intraday_volume_profile)
        st.line_chart(prof)

    # ----------------------------------------------------------------
    # R3 wiring snippet
    # ----------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Wire into `synthetic_market_sim`")
    snippet = f"""# Generated by NSE Calibration Workbench  ·  {params.calibration_date}
# Drop into a synthetic_market_sim run-config builder before generation.

from scipy.stats import pareto
import numpy as np

CALIBRATION = {{
    "volatility_scale":      {params.realized_volatility:.6f},
    "return_shock_df":       {params.return_df:.3f},
    "return_shock_scale":    {params.return_scale:.6f},
    "trader_activity_alpha": {params.volume_alpha:.3f},
}}

def make_pareto_activity_multipliers(n_traders: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = pareto.rvs(b=CALIBRATION["trader_activity_alpha"],
                     size=n_traders, random_state=rng)
    raw = np.clip(raw, 0, np.percentile(raw, 99))
    return raw / raw.mean()
"""
    st.code(snippet, language="python")
    st.caption(
        "This snippet expresses the daily-mode parameters; the intraday volume "
        "profile (375 slots) is empty in daily mode and consumers should fall "
        "back to the Phase 1 synthetic profile or run the intraday calibrator "
        "when 1-minute data is available."
    )


render()
