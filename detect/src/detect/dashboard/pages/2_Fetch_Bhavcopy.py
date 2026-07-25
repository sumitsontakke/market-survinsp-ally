"""Fetch Bhavcopy page.

Lets the user pull NSE bhavcopy CSVs for a chosen date range. Calls
``BhavcopyFetcher`` directly in-process; streams progress to the UI.
"""
from __future__ import annotations

import io
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import streamlit as st

sys.path.insert(0, "/app")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from synth.calibration.core.config import NIFTY_LIQUID_20  # noqa: E402
from synth.calibration.core.database import MarketDataDB  # noqa: E402
from synth.calibration.core.nse_bhavcopy import BhavcopyFetcher  # noqa: E402

st.set_page_config(page_title="Fetch Bhavcopy", page_icon="📥", layout="wide")
st.title("Fetch Bhavcopy")
st.caption(
    "Pull NSE end-of-day OHLCV from `nsearchives.nseindia.com`. Free, no API "
    "auth, no rate limit. Idempotent — re-running on the same window is a no-op."
)


@st.cache_resource
def get_db() -> MarketDataDB:
    return MarketDataDB()


def render() -> None:
    db = get_db()
    today = date.today()

    # ----------------------------------------------------------------
    # Date range picker with sane defaults
    # ----------------------------------------------------------------
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("### Window")
        preset = st.radio(
            "Quick presets",
            ["Last 5 trading days", "Last 30 days", "Last 90 days", "Custom"],
            horizontal=True,
        )
        if preset == "Last 5 trading days":
            default_start = today - timedelta(days=8)
        elif preset == "Last 30 days":
            default_start = today - timedelta(days=30)
        elif preset == "Last 90 days":
            default_start = today - timedelta(days=90)
        else:
            default_start = today - timedelta(days=14)

        start_d = st.date_input("Start date (UTC)", value=default_start)
        end_d = st.date_input("End date (UTC)", value=today)

    with col_right:
        st.markdown("### Universe")
        universe_choice = st.radio(
            "Which symbols to keep",
            ["All NSE EQ (~2,450 tickers/day)", "NIFTY 20 only"],
            help="`All` is recommended — needed for the cross-sectional Pareto / Hill α."
            " `NIFTY 20` is faster and uses less disk if you only care about the focus universe.",
        )
        force_refetch = st.checkbox(
            "Force re-fetch even if cached",
            value=False,
            help="Use only when troubleshooting. Normal runs are idempotent and skip cached dates.",
        )

    if start_d > end_d:
        st.error("Start date must be on or before end date.")
        return

    n_calendar = (end_d - start_d).days + 1
    n_weekdays = sum(
        1 for i in range(n_calendar)
        if (start_d + timedelta(days=i)).weekday() < 5
    )
    st.info(
        f"Window: **{start_d.isoformat()} → {end_d.isoformat()}** "
        f"({n_calendar} calendar days, ~{n_weekdays} weekdays)"
    )

    # ----------------------------------------------------------------
    # Run button
    # ----------------------------------------------------------------
    run_btn = st.button(
        "Fetch", type="primary", use_container_width=True,
        help="Pulls each weekday's CSV from NSE archives. Saturdays/Sundays are skipped automatically.",
    )

    if not run_btn:
        return

    universe = NIFTY_LIQUID_20 if universe_choice.startswith("NIFTY") else None
    fetcher = BhavcopyFetcher(db_path=db.db_path, universe=universe)

    log_buf = io.StringIO()
    handler = logging.StreamHandler(log_buf)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger("core.nse_bhavcopy")
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    progress = st.progress(0.0, text="Starting...")
    log_box = st.empty()

    t0 = time.perf_counter()
    try:
        # Run synchronously - the Bhavcopy CSVs are small (~400KB each)
        # and the loop is bounded by the number of weekdays in the window
        # so we don't need a worker thread.
        summary = fetcher.fetch_window(
            start_d.isoformat(), end_d.isoformat(), force=force_refetch
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Fetch failed: {exc!r}")
        return
    finally:
        root.removeHandler(handler)

    elapsed = time.perf_counter() - t0
    progress.progress(1.0, text=f"Done in {elapsed:.1f}s")
    log_box.code(log_buf.getvalue() or "(no log lines)", language="text")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Result")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Dates processed", summary.dates_processed)
    c2.metric("With data", summary.dates_with_data)
    c3.metric("Holidays (404)", summary.dates_holiday)
    c4.metric("Cached / skipped", summary.dates_cached)
    c5.metric("Rows inserted", f"{summary.rows_inserted:,}")

    if summary.warnings:
        with st.expander(f"Warnings ({len(summary.warnings)})", expanded=False):
            for w in summary.warnings:
                st.write(f"- {w}")
    if summary.errors:
        st.error(
            "Errors:\n" + "\n".join(f"- {e}" for e in summary.errors)
        )

    if summary.rows_inserted > 0:
        st.success(
            f"Inserted {summary.rows_inserted:,} rows across "
            f"{summary.dates_with_data} trading days. "
            "Open **Calibrate** to compute parameters from this window."
        )
        if st.button("Go to Calibrate"):
            st.switch_page("pages/3_Calibrate.py")


render()
