"""Calibration history & trends.

Lists every saved calibration run and plots the four parameters over
time. Useful for spotting regime shifts or stress events; doubles as
the "see we ran this every day" panel for review demos.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, "/app")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from synth.calibration.core.config import (  # noqa: E402
    EMPIRICAL_RETURN_DF_RANGE,
    EMPIRICAL_VOLUME_ALPHA_RANGE,
    SYNTHETIC_BASELINES,
)
from synth.calibration.core.database import MarketDataDB  # noqa: E402

st.set_page_config(page_title="Calibration History", page_icon="📜", layout="wide")
st.title("Calibration History")
st.caption(
    "Every saved calibration run, latest first. Compare two runs side by side, "
    "or watch the four parameters trend across calibration dates."
)


@st.cache_resource
def get_db() -> MarketDataDB:
    return MarketDataDB()


def _load_history() -> pd.DataFrame:
    db = get_db()
    with db._connect() as conn:
        return pd.read_sql_query(
            """
            SELECT calibration_date, n_observations,
                   realized_vol, return_df, return_loc, return_scale,
                   volume_alpha, created_at
            FROM calibration_runs
            ORDER BY calibration_date DESC
            """,
            conn,
        )


def render() -> None:
    df = _load_history()
    if df.empty:
        st.info("No calibration runs yet. Run a calibration first.")
        return

    df["calibration_date"] = pd.to_datetime(df["calibration_date"])

    # ----------------------------------------------------------------
    # Trend plots
    # ----------------------------------------------------------------
    st.markdown("### Trends")
    st.caption("Lines below are calibration-date series. The dashed bands mark the empirical reference ranges.")
    df_lo, df_hi = EMPIRICAL_RETURN_DF_RANGE
    a_lo, a_hi = EMPIRICAL_VOLUME_ALPHA_RANGE
    synth_df = float(SYNTHETIC_BASELINES["return_df"])

    plot_data = df.sort_values("calibration_date").set_index("calibration_date")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**realized_volatility**")
        st.line_chart(plot_data[["realized_vol"]], height=240)
    with c2:
        st.markdown(f"**return_df**  (band: {df_lo}-{df_hi}, synth baseline: {synth_df})")
        df_chart = plot_data[["return_df"]].copy()
        df_chart["band_lo"] = df_lo
        df_chart["band_hi"] = df_hi
        df_chart["synth"] = synth_df
        st.line_chart(df_chart, height=240)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown(f"**volume_alpha**  (band: {a_lo}-{a_hi})")
        v_chart = plot_data[["volume_alpha"]].copy()
        v_chart["band_lo"] = a_lo
        v_chart["band_hi"] = a_hi
        st.line_chart(v_chart, height=240)
    with c4:
        st.markdown("**return_scale**")
        st.line_chart(plot_data[["return_scale"]], height=240)

    # ----------------------------------------------------------------
    # Run table
    # ----------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Runs table")
    show = df.copy()
    show["calibration_date"] = show["calibration_date"].dt.strftime("%Y-%m-%d")
    st.dataframe(
        show.rename(
            columns={
                "calibration_date": "date",
                "n_observations": "n_obs",
                "realized_vol": "realized_vol",
                "return_df": "df",
                "return_loc": "loc",
                "return_scale": "scale",
                "volume_alpha": "α",
                "created_at": "computed",
            }
        ).round({"realized_vol": 4, "df": 3, "loc": 6, "scale": 6, "α": 3}),
        hide_index=True,
        use_container_width=True,
    )

    # ----------------------------------------------------------------
    # Compare two runs
    # ----------------------------------------------------------------
    st.markdown("---")
    st.markdown("### Side-by-side compare")
    if len(df) < 2:
        st.caption("Need at least two calibrations to compare.")
        return

    options = df["calibration_date"].dt.strftime("%Y-%m-%d").tolist()
    cc1, cc2 = st.columns(2)
    with cc1:
        date_a = st.selectbox("Run A", options=options, index=0)
    with cc2:
        date_b = st.selectbox("Run B", options=options, index=min(1, len(options) - 1))

    if date_a == date_b:
        st.info("Pick two different dates to see the diff.")
        return

    row_a = df.loc[df["calibration_date"].dt.strftime("%Y-%m-%d") == date_a].iloc[0]
    row_b = df.loc[df["calibration_date"].dt.strftime("%Y-%m-%d") == date_b].iloc[0]

    cmp_df = pd.DataFrame(
        {
            "Run A ({0})".format(date_a): {
                "realized_vol": float(row_a["realized_vol"]),
                "return_df": float(row_a["return_df"]),
                "return_scale": float(row_a["return_scale"]),
                "volume_alpha": float(row_a["volume_alpha"]),
                "n_obs": int(row_a["n_observations"]),
            },
            "Run B ({0})".format(date_b): {
                "realized_vol": float(row_b["realized_vol"]),
                "return_df": float(row_b["return_df"]),
                "return_scale": float(row_b["return_scale"]),
                "volume_alpha": float(row_b["volume_alpha"]),
                "n_obs": int(row_b["n_observations"]),
            },
        }
    )
    cmp_df["Δ (B - A)"] = cmp_df.iloc[:, 1] - cmp_df.iloc[:, 0]
    cmp_df["Δ %"] = (cmp_df["Δ (B - A)"] / cmp_df.iloc[:, 0].replace(0, pd.NA)) * 100
    st.dataframe(cmp_df.round(6), use_container_width=True)


render()
