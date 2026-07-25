"""NSE Calibration Workbench — Streamlit dashboard.

Multipage app. The home page is a "what's in the system right now"
dashboard. Other pages live in ``pages/`` and Streamlit auto-routes them
into the sidebar.

Reference
---------
Cont, R. (2001). Empirical properties of asset returns: stylized facts
and statistical implications. Quantitative Finance, 1(2), 223-236.
"""
from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import streamlit as st
import pandas as pd

# Allow ``core`` import when the host bind-mounts /app/core into this image.
sys.path.insert(0, "/app")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synth.calibration.core.config import (  # noqa: E402
    DB_PATH,
    EMPIRICAL_RETURN_DF_RANGE,
    EMPIRICAL_VOLUME_ALPHA_RANGE,
    NIFTY_LIQUID_20,
    SYNTHETIC_BASELINES,
)
from synth.calibration.core.database import MarketDataDB  # noqa: E402

# ---------------------------------------------------------------------------
# Page config + theming
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NSE Calibration Workbench",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Match the existing project palette (Midnight Executive).
PALETTE = {
    "navy": "#1E2761",
    "ice": "#CADCFC",
    "accent": "#C8102E",
    "ink": "#1A1A2E",
    "muted": "#5C6480",
    "success": "#0F7A4D",
    "warn": "#A16207",
    "soft_bg": "#F4F6FA",
}

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: #FFFFFF; }}
    section[data-testid="stSidebar"] {{ background-color: {PALETTE['soft_bg']}; }}
    h1, h2, h3 {{ color: {PALETTE['navy']}; font-family: Georgia, serif; }}
    .stat-card {{
        background: {PALETTE['soft_bg']};
        border-left: 4px solid {PALETTE['navy']};
        padding: 14px 18px; border-radius: 4px; margin-bottom: 12px;
    }}
    .stat-card .label {{
        font-size: 11px; text-transform: uppercase;
        letter-spacing: 0.08em; color: {PALETTE['accent']}; font-weight: 600;
    }}
    .stat-card .value {{
        font-family: Georgia, serif; font-size: 32px;
        color: {PALETTE['navy']}; font-weight: bold; margin-top: 4px;
    }}
    .stat-card .sub {{ font-size: 12px; color: {PALETTE['muted']}; }}
    .badge {{
        display: inline-block; padding: 2px 8px; border-radius: 12px;
        font-size: 11px; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    .badge-ok {{ background: #DCFCE7; color: #14532D; }}
    .badge-holiday {{ background: #FEF9C3; color: #713F12; }}
    .badge-missing {{ background: #FEE2E2; color: #991B1B; }}
    .badge-cached {{ background: #E0E7FF; color: #312E81; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Shared resources
# ---------------------------------------------------------------------------
@st.cache_resource
def get_db() -> MarketDataDB:
    """One DB handle shared across reruns. The DB file lives on a Docker
    named volume so the same handle works regardless of which page mutated
    it last."""
    db_path = Path(os.environ.get("DB_PATH", str(DB_PATH)))
    return MarketDataDB(db_path)


def stat_card(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f"""
        <div class="stat-card">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          <div class="sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Home dashboard
# ---------------------------------------------------------------------------
def render_dashboard() -> None:
    st.markdown("# NSE Calibration Workbench")
    st.markdown(
        f"<div style='color:{PALETTE['muted']}; font-style:italic; "
        "margin-bottom:24px;'>"
        "Pull NSE bhavcopy, compute calibration parameters, inspect data quality, "
        "and feed the results into <code>synthetic_market_sim</code>."
        "</div>",
        unsafe_allow_html=True,
    )

    db = get_db()
    stats = db.stats()

    daily_lo, daily_hi = stats["daily_date_range"]
    intra_lo, intra_hi = stats["intraday_date_range"]
    latest_cal = db.get_latest_calibration()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        stat_card(
            "Bhavcopy rows",
            f"{stats['daily_market_data_rows']:,}",
            f"{stats['daily_fetch_log_rows']} dates" if stats["daily_fetch_log_rows"] else "no dates",
        )
    with col2:
        stat_card(
            "Bhavcopy date range",
            f"{daily_lo or '—'}",
            f"to {daily_hi or '—'}",
        )
    with col3:
        stat_card(
            "Intraday rows (yfinance)",
            f"{stats['market_data_rows']:,}",
            f"{intra_lo or '—'} → {intra_hi or '—'}" if intra_lo else "not populated",
        )
    with col4:
        stat_card(
            "Calibration runs",
            str(stats["calibration_runs_rows"]),
            f"latest: {latest_cal.calibration_date}" if latest_cal else "no runs yet",
        )

    st.markdown("---")
    st.markdown("### Get started")
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.markdown("**1. Fetch bhavcopy**")
        st.caption("Pull a window of NSE end-of-day data into the local SQLite store. Free, no auth, no rate limit.")
        if st.button("Open Fetch page", use_container_width=True, key="goto_fetch"):
            st.switch_page("pages/2_Fetch_Bhavcopy.py")
    with g2:
        st.markdown("**2. Calibrate a date**")
        st.caption("Compute realized vol, Student-t fit, cross-sectional Hill alpha. See gap vs Phase 1 baselines.")
        if st.button("Open Calibrate page", use_container_width=True, key="goto_cal"):
            st.switch_page("pages/3_Calibrate.py")
    with g3:
        st.markdown("**3. Run the full demo flow**")
        st.caption("Fetch 30 days → calibrate latest date → render all artifacts. ~60 seconds end-to-end.")
        if st.button("Open Demo Flow", use_container_width=True, key="goto_demo"):
            st.switch_page("pages/7_Demo_Flow.py")
    with g4:
        st.markdown("**4. Compare rungs 1-4+**")
        st.caption("Four-rung representation ladder — measured locked-stress recall across statistical, trader-ML, edge-ML, and GraphSAGE (CPU + GPU boosted).")
        if st.button("Open Compare page", use_container_width=True, key="goto_compare"):
            st.switch_page("pages/8_Compare.py")

    st.markdown("---")
    st.markdown("### Latest calibration snapshot")
    if latest_cal is None:
        st.info("No calibration runs yet. Fetch some bhavcopy first, then run the calibrator.")
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            df_lo, df_hi = EMPIRICAL_RETURN_DF_RANGE
            a_lo, a_hi = EMPIRICAL_VOLUME_ALPHA_RANGE
            df_in = df_lo <= latest_cal.return_df <= df_hi
            a_in = a_lo <= latest_cal.volume_alpha <= a_hi
            df_status = "badge-ok" if df_in else "badge-missing"
            a_status = "badge-ok" if a_in else "badge-missing"

            st.markdown(
                f"""
                **Calibration date:** `{latest_cal.calibration_date}`<br>
                **Tickers used:** {len(latest_cal.tickers_used)} of {len(NIFTY_LIQUID_20)}<br>
                **Observations:** {latest_cal.n_observations:,}
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                | Parameter | NSE value | Phase 1 synth | Empirical band | Status |
                |---|---|---|---|---|
                | realized_volatility | `{latest_cal.realized_volatility:.4f}` | `{SYNTHETIC_BASELINES['realized_volatility']}` | n/a | <span class='badge badge-ok'>computed</span> |
                | return_df | `{latest_cal.return_df:.3f}` | `{SYNTHETIC_BASELINES['return_df']}` | [{df_lo}–{df_hi}] | <span class='badge {df_status}'>{'in band' if df_in else 'out of band'}</span> |
                | volume_alpha | `{latest_cal.volume_alpha:.3f}` | `{SYNTHETIC_BASELINES['volume_alpha']}` | [{a_lo}–{a_hi}] | <span class='badge {a_status}'>{'in band' if a_in else 'out of band'}</span> |
                """,
                unsafe_allow_html=True,
            )
        with c2:
            if latest_cal.warnings:
                st.warning("Warnings:\n" + "\n".join(f"- {w}" for w in latest_cal.warnings))
            else:
                st.success("No warnings on this run.")

    st.markdown("---")
    st.caption(
        "Reference: Cont, R. (2001). *Empirical properties of asset returns: stylized facts "
        "and statistical implications.* Quantitative Finance, 1(2), 223-236."
    )


# ---------------------------------------------------------------------------
# Sidebar status block (visible from every page)
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    db = get_db()
    stats = db.stats()
    daily_lo, daily_hi = stats["daily_date_range"]
    st.sidebar.markdown("### System status")
    st.sidebar.markdown(
        f"""
        - **DB**: `{db.db_path}`
        - **Bhavcopy**: {stats['daily_market_data_rows']:,} rows, {stats['daily_fetch_log_rows']} dates
        - **Range**: {daily_lo or '—'} → {daily_hi or '—'}
        - **Calibrations**: {stats['calibration_runs_rows']}
        """
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Phase 2 calibration workbench<br>"
        "M.Tech dissertation, PES University",
        unsafe_allow_html=True,
    )


def main() -> None:
    render_sidebar()
    render_dashboard()


if __name__ == "__main__":
    main()
