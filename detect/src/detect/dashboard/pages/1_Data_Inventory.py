"""Data Inventory page.

Shows every date the bhavcopy fetcher has touched, the status (ok / holiday
/ missing), the row count, and lets the user drill into a specific date
to see what's stored.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, "/app")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from synth.calibration.core.config import NIFTY_LIQUID_20  # noqa: E402
from synth.calibration.core.database import MarketDataDB  # noqa: E402

st.set_page_config(page_title="Data Inventory", page_icon="📅", layout="wide")
st.title("Data Inventory")
st.caption(
    "Every date the bhavcopy fetcher has touched, with status. Drill into a "
    "row to see the per-ticker contents."
)


@st.cache_resource
def get_db() -> MarketDataDB:
    return MarketDataDB()


def _badge(status: str) -> str:
    klass = {
        "ok": "badge-ok",
        "holiday": "badge-holiday",
        "missing": "badge-missing",
    }.get(status, "badge-missing")
    return f"<span class='badge {klass}'>{status}</span>"


def render() -> None:
    db = get_db()

    with db._connect() as conn:
        log = pd.read_sql_query(
            """
            SELECT trade_date, status, row_count, fetched_at
            FROM daily_fetch_log
            ORDER BY trade_date DESC
            """,
            conn,
        )
        market = pd.read_sql_query(
            """
            SELECT trade_date, COUNT(*) AS row_count_actual,
                   COUNT(DISTINCT ticker) AS tickers
            FROM daily_market_data
            GROUP BY trade_date
            ORDER BY trade_date DESC
            """,
            conn,
        )

    if log.empty:
        st.info(
            "No bhavcopy dates fetched yet. Open the **Fetch Bhavcopy** page to pull "
            "a window of NSE end-of-day data."
        )
        if st.button("Go to Fetch Bhavcopy"):
            st.switch_page("pages/2_Fetch_Bhavcopy.py")
        return

    log["fetched_at"] = pd.to_datetime(log["fetched_at"], errors="coerce")
    merged = log.merge(market, how="left", on="trade_date")
    merged["row_count_actual"] = merged["row_count_actual"].fillna(0).astype(int)
    merged["tickers"] = merged["tickers"].fillna(0).astype(int)

    # Stat strip
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Dates fetched", len(merged))
    with c2:
        n_ok = int((merged["status"] == "ok").sum())
        st.metric("OK days", n_ok)
    with c3:
        n_hol = int((merged["status"] == "holiday").sum())
        st.metric("Holidays / 404", n_hol)
    with c4:
        st.metric("Total rows", int(merged["row_count_actual"].sum()))

    st.markdown("---")
    st.markdown("### Date table")

    # Filters
    col1, col2 = st.columns([2, 1])
    with col1:
        status_filter = st.multiselect(
            "Status filter",
            options=sorted(merged["status"].unique().tolist()),
            default=sorted(merged["status"].unique().tolist()),
        )
    with col2:
        ticker_focus = st.checkbox("Show NIFTY 20 ticker count only", value=False)

    visible = merged.loc[merged["status"].isin(status_filter)].copy()

    if ticker_focus and not visible.empty:
        nifty_bare = [t.split(".")[0] for t in NIFTY_LIQUID_20]
        with db._connect() as conn:
            placeholders = ",".join("?" for _ in nifty_bare)
            n_df = pd.read_sql_query(
                f"""
                SELECT trade_date, COUNT(*) AS nifty_rows
                FROM daily_market_data
                WHERE ticker IN ({placeholders})
                GROUP BY trade_date
                """,
                conn,
                params=nifty_bare,
            )
        visible = visible.merge(n_df, how="left", on="trade_date")
        visible["nifty_rows"] = visible["nifty_rows"].fillna(0).astype(int)

    # Render with rich badges
    table_html = ["<table style='width:100%; border-collapse:collapse;'>"]
    table_html.append(
        "<tr style='background:#1E2761; color:white;'>"
        "<th style='padding:8px;text-align:left'>Trade date</th>"
        "<th style='padding:8px;text-align:left'>Status</th>"
        "<th style='padding:8px;text-align:right'>Logged rows</th>"
        "<th style='padding:8px;text-align:right'>Stored rows</th>"
        "<th style='padding:8px;text-align:right'>Distinct tickers</th>"
        + ("<th style='padding:8px;text-align:right'>NIFTY 20 rows</th>" if ticker_focus else "")
        + "<th style='padding:8px;text-align:left'>Fetched</th>"
        "</tr>"
    )
    for i, row in visible.iterrows():
        bg = "#FFFFFF" if i % 2 == 0 else "#F4F6FA"
        nifty_cell = (
            f"<td style='padding:6px 8px;text-align:right'>{int(row['nifty_rows'])}</td>"
            if ticker_focus
            else ""
        )
        fetched = row["fetched_at"].strftime("%Y-%m-%d %H:%M") if pd.notna(row["fetched_at"]) else "—"
        table_html.append(
            f"<tr style='background:{bg};'>"
            f"<td style='padding:6px 8px'><b>{row['trade_date']}</b></td>"
            f"<td style='padding:6px 8px'>{_badge(row['status'])}</td>"
            f"<td style='padding:6px 8px;text-align:right'>{int(row['row_count'])}</td>"
            f"<td style='padding:6px 8px;text-align:right'>{int(row['row_count_actual'])}</td>"
            f"<td style='padding:6px 8px;text-align:right'>{int(row['tickers'])}</td>"
            f"{nifty_cell}"
            f"<td style='padding:6px 8px;color:#5C6480'>{fetched}</td>"
            f"</tr>"
        )
    table_html.append("</table>")
    st.markdown("".join(table_html), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Drill into a date")
    drill_date = st.selectbox(
        "Pick a date",
        options=visible.loc[visible["status"] == "ok", "trade_date"].tolist(),
        index=0 if not visible.empty else None,
    )
    if drill_date:
        with db._connect() as conn:
            day_df = pd.read_sql_query(
                "SELECT * FROM daily_market_data WHERE trade_date = ? ORDER BY volume DESC",
                conn,
                params=[drill_date],
            )
        st.markdown(f"#### {drill_date} — {len(day_df):,} rows")

        # Top by volume
        st.markdown("**Top 15 by volume**")
        st.dataframe(
            day_df.head(15)[
                ["ticker", "series", "open", "high", "low", "close", "volume", "deliv_pct"]
            ],
            hide_index=True,
            use_container_width=True,
        )

        # Bottom of distribution
        st.markdown("**Bottom 5 by volume (sanity check)**")
        st.dataframe(
            day_df.tail(5)[
                ["ticker", "series", "open", "high", "low", "close", "volume", "deliv_pct"]
            ],
            hide_index=True,
            use_container_width=True,
        )

        # Cross-sectional summary
        vols = day_df["volume"].astype(float)
        vols = vols[vols > 0]
        if len(vols) > 0:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Tickers w/ volume", f"{len(vols):,}")
            c2.metric("Median volume", f"{vols.median():,.0f}")
            c3.metric("p99 volume", f"{vols.quantile(0.99):,.0f}")
            c4.metric("p99 / median", f"{vols.quantile(0.99) / max(vols.median(), 1):.1f}×")


render()
