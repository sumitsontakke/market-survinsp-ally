"""Metric Timeline — drift surveillance for the model registry.

Reads /data/model_registry (the shared Docker named volume) and renders a
rolling timeline of the four headline metrics across all registered model
versions:

  * cv_auc
  * locked_clique_recall
  * locked_ring_recall
  * locked_mixed_recall

Pieces on the page:

  1. **Headline panel**  current champion + challenger summary
  2. **Timeline chart**   one line per metric, x = registered_at
  3. **Drift indicator**  rolling-window band (mean ± 2σ) — entries
                          outside the band are flagged
  4. **Champion / challenger comparison**  side-by-side metric table
                                            with delta column
  5. **Registry table**   sortable inventory with download links

This is the MLOps continuous-evaluation surface promised in Chapter 6
of LIMITATIONS.md. Defaults to a "growing horizon" view — as new model
fits land in /data/model_registry they appear here automatically.

Reference
---------
Cont, R. (2001). Empirical properties of asset returns. QF 1(2), 223-236.
Hamilton, W. L., Ying, R., Leskovec, J. (2017). Inductive Representation
Learning on Large Graphs. NeurIPS.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Metric Timeline — drift surveillance",
                   page_icon="📈", layout="wide")

PALETTE = {
    "navy":   "#1E2761",
    "ice":    "#7C8FC9",
    "accent": "#C8102E",
    "ink":    "#1A1A2E",
    "muted":  "#5C6480",
    "success":"#0F7A4D",
    "warn":   "#A16207",
    "soft":   "#F4F6FA",
}

REGISTRY_ROOT = Path(os.environ.get("MODEL_REGISTRY_DIR", "/data/model_registry"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def section_header(title: str, eyebrow: str, kicker: str = "") -> None:
    st.markdown(
        f"<div style='font-size:11px;text-transform:uppercase;"
        f"letter-spacing:0.10em;color:{PALETTE['accent']};font-weight:700;'>"
        f"{escape(eyebrow)}</div>"
        f"<div style='font-family:Georgia,serif;color:{PALETTE['navy']};"
        f"font-size:24px;font-weight:bold;margin-top:2px;line-height:1.1;'>"
        f"{escape(title)}</div>"
        + (f"<div style='color:{PALETTE['muted']};font-style:italic;"
           f"font-size:13px;margin-top:4px;margin-bottom:12px;'>"
           f"{escape(kicker)}</div>" if kicker else ""),
        unsafe_allow_html=True,
    )


def stat_card(label: str, value: str, sub: str = "", accent: bool = False) -> str:
    border = PALETTE["accent"] if accent else PALETTE["navy"]
    return (
        f"<div style='background:{PALETTE['soft']};"
        f"border-left:4px solid {border};padding:14px 18px;"
        f"border-radius:4px;margin-bottom:10px;height:112px;'>"
        f"<div style='font-size:10px;text-transform:uppercase;"
        f"letter-spacing:0.08em;color:{PALETTE['accent']};font-weight:600;'>"
        f"{escape(label)}</div>"
        f"<div style='font-family:Georgia,serif;font-size:26px;"
        f"color:{PALETTE['navy']};font-weight:bold;margin-top:4px;"
        f"line-height:1.0;'>{escape(value)}</div>"
        f"<div style='font-size:11px;color:{PALETTE['muted']};"
        f"margin-top:6px;'>{escape(sub)}</div>"
        f"</div>"
    )


def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f < 0:
        return None
    return f


# ---------------------------------------------------------------------------
# Registry loader
# ---------------------------------------------------------------------------
@st.cache_data(ttl=30)
def load_registry(root: str) -> pd.DataFrame:
    """Walk every model-registry subdirectory and assemble a tidy DataFrame.

    Cache is short (30s) so dropping a new model entry into /data/model_registry
    is reflected on the next page interaction without a manual refresh.
    """
    root_p = Path(root)
    if not root_p.exists():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for entry in sorted(root_p.iterdir()):
        if not entry.is_dir():
            continue
        cfg = entry / "config.yaml"
        metrics = entry / "metrics.json"
        if not (cfg.exists() and metrics.exists()):
            continue
        # YAML/JSON config — try JSON first since seed wrote JSON
        try:
            cfg_payload = json.loads(cfg.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            try:
                import yaml
                cfg_payload = yaml.safe_load(cfg.read_text(encoding="utf-8"))
            except Exception:
                continue
        try:
            m_payload = json.loads(metrics.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        ts = cfg_payload.get("registered_at")
        if ts is None:
            # Fall back to the timestamp encoded in the directory name.
            name_parts = entry.name.split("__")
            ts = name_parts[1] if len(name_parts) >= 2 else "?"
        rows.append({
            "entry":                entry.name,
            "experiment_id":        cfg_payload.get("experiment_id"),
            "registered_at":        ts,
            "device":               cfg_payload.get("device", "?"),
            "cohort":               cfg_payload.get("data", {}).get("runs",
                                       cfg_payload.get("cohort", "?")),
            "parent":               cfg_payload.get("parent_model"),
            "cv_auc":               _safe_float(m_payload.get("cv_auc")),
            "cv_f1":                _safe_float(m_payload.get("cv_f1")),
            "locked_clique_recall": _safe_float(m_payload.get("locked_clique_recall")),
            "locked_ring_recall":   _safe_float(m_payload.get("locked_ring_recall")),
            "locked_mixed_recall":  _safe_float(m_payload.get("locked_mixed_recall")),
            "locked_benign_alarm":  _safe_float(m_payload.get("locked_benign_alarm")),
            "n_train_runs":         m_payload.get("n_train_runs"),
            "n_eval_runs":          m_payload.get("n_eval_runs"),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Normalize timestamps to pandas datetime for sorting + plotting.
    df["registered_at_dt"] = pd.to_datetime(df["registered_at"], errors="coerce",
                                             utc=True)
    return df.sort_values("registered_at_dt")


@st.cache_data(ttl=30)
def load_champion(root: str) -> str | None:
    path = Path(root) / "CHAMPION.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").strip() or None
    return None


# ---------------------------------------------------------------------------
# Page render
# ---------------------------------------------------------------------------
def render() -> None:
    st.markdown(
        f"<div style='font-family:Georgia,serif;font-size:32px;"
        f"font-weight:bold;color:{PALETTE['navy']};margin-bottom:2px;'>"
        f"Metric Timeline — model registry surveillance</div>"
        f"<div style='color:{PALETTE['muted']};font-style:italic;font-size:14px;"
        f"margin-bottom:16px;'>"
        f"Continuous evaluation across every registered model version. Defaults "
        f"to the &#x201C;growing horizon&#x201D; view — new fits in "
        f"<code>/data/model_registry</code> appear automatically on next refresh."
        f"</div>",
        unsafe_allow_html=True,
    )

    df = load_registry(str(REGISTRY_ROOT))
    if df.empty:
        st.warning(
            f"No model versions in `{REGISTRY_ROOT}`. "
            "Seed the registry first:\n\n"
            "```\ndocker exec nse-webapp python /app/seed_model_registry.py\n```"
        )
        st.stop()

    champion_name = load_champion(str(REGISTRY_ROOT))
    champion_row = (
        df[df["entry"] == champion_name].iloc[0]
        if champion_name and (df["entry"] == champion_name).any()
        else df.iloc[-1]
    )
    # Challenger = most recent non-champion, or the previous version.
    not_champion = df[df["entry"] != champion_row["entry"]]
    challenger_row = (not_champion.iloc[-1]
                      if not not_champion.empty else None)

    # -----------------------------------------------------------------------
    # 1. Headline — champion + challenger summary cards
    # -----------------------------------------------------------------------
    section_header(
        "Current champion",
        "Production state",
        "Promoted via the `CHAMPION.txt` pointer in the registry root. "
        "All inference traffic in a live deployment would route to this model.",
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(stat_card(
            "Champion",
            champion_row["experiment_id"].split("_")[0].upper()
            if isinstance(champion_row["experiment_id"], str) else "—",
            champion_row["entry"][:48] + "…"
            if len(str(champion_row["entry"])) > 48 else str(champion_row["entry"]),
            accent=True,
        ), unsafe_allow_html=True)
    with c2:
        cr = champion_row["locked_clique_recall"]
        st.markdown(stat_card(
            "Clique recall",
            f"{cr:.3f}" if cr is not None else "—",
            f"holdout n_eval={int(champion_row['n_eval_runs'])}"
            if not pd.isna(champion_row['n_eval_runs']) else "—",
        ), unsafe_allow_html=True)
    with c3:
        rr = champion_row["locked_ring_recall"]
        st.markdown(stat_card(
            "Ring recall",
            f"{rr:.3f}" if rr is not None else "—",
            "vs Rung-3 baseline 0.500",
        ), unsafe_allow_html=True)
    with c4:
        mr = champion_row["locked_mixed_recall"]
        st.markdown(stat_card(
            "Mixed recall",
            f"{mr:.3f}" if mr is not None else "—",
            "vs Rung-3 baseline 0.154",
        ), unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # 2. Timeline chart
    # -----------------------------------------------------------------------
    section_header(
        "Timeline — locked-stress recall by version",
        "Drift surveillance",
        "One point per registered model version. Hover any point to see the "
        "version it came from. As scaled-cohort or nightly fits land, new "
        "dots appear automatically.",
    )

    # Metric selector (default: all four)
    metric_options = {
        "Clique recall":   "locked_clique_recall",
        "Ring recall":     "locked_ring_recall",
        "Mixed recall":    "locked_mixed_recall",
        "CV AUC":          "cv_auc",
    }
    metric_pick = st.multiselect(
        "Metrics to plot",
        list(metric_options.keys()),
        default=list(metric_options.keys()),
    )

    if not metric_pick:
        st.info("Pick at least one metric to render the timeline.")
    else:
        fig, ax = plt.subplots(figsize=(11, 3.6), dpi=170)
        colors = {
            "Clique recall": PALETTE["navy"],
            "Ring recall":   PALETTE["ice"],
            "Mixed recall":  PALETTE["accent"],
            "CV AUC":        PALETTE["warn"],
        }
        x = df["registered_at_dt"]
        for label in metric_pick:
            col = metric_options[label]
            y = df[col]
            ax.plot(x, y, marker="o", markersize=8, linewidth=1.6,
                    color=colors[label], label=label, alpha=0.92)
            # Annotate the experiment_id near the last point.
            for xi, yi in zip(x, y):
                if pd.notna(yi):
                    ax.annotate(f"{yi:.3f}", xy=(xi, yi),
                                xytext=(5, 6), textcoords="offset points",
                                fontsize=8, color=colors[label])
        # Drift band (mean ± 2σ over the last 10 entries)
        recent = df.tail(10)
        for label in metric_pick:
            col = metric_options[label]
            vals = recent[col].dropna()
            if len(vals) >= 3:
                mu, sd = float(vals.mean()), float(vals.std())
                ax.fill_between(recent["registered_at_dt"],
                                mu - 2 * sd, mu + 2 * sd,
                                color=colors[label], alpha=0.06)
        ax.set_ylim(0.0, 1.08)
        ax.set_ylabel("Locked-stress recall (or CV AUC)",
                      color=PALETTE["navy"], fontsize=10)
        ax.set_xlabel("Registered at (UTC)", color=PALETTE["navy"], fontsize=10)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.legend(loc="lower right", frameon=False, fontsize=9, ncol=2)
        fig.autofmt_xdate(rotation=15)
        fig.tight_layout()
        st.pyplot(fig)

    st.caption(
        "Shaded bands are the rolling mean ± 2σ over the last 10 registered "
        "versions. A new model fit landing outside the band is a drift "
        "alert. With only 2 entries seeded the band is degenerate — once 3+ "
        "versions exist (e.g. after the scaled cohort runs), the band becomes "
        "meaningful."
    )

    # -----------------------------------------------------------------------
    # 3. Drift indicator
    # -----------------------------------------------------------------------
    if len(df) >= 3:
        section_header(
            "Drift alerts",
            "Out-of-band check",
            "Versions where any tracked metric fell outside the rolling "
            "mean ± 2σ band over the last 10 entries.",
        )
        recent = df.tail(10)
        alerts = []
        for _, row in df.iterrows():
            for label, col in metric_options.items():
                v = row[col]
                if pd.isna(v):
                    continue
                vals = recent[col].dropna()
                if len(vals) < 3:
                    continue
                mu, sd = float(vals.mean()), float(vals.std())
                if not (mu - 2 * sd <= v <= mu + 2 * sd):
                    alerts.append({
                        "entry": row["entry"],
                        "metric": label,
                        "value": float(v),
                        "expected": f"{mu:.3f} ± {2*sd:.3f}",
                    })
        if alerts:
            st.error(f"{len(alerts)} drift alert(s) on the timeline.")
            st.dataframe(pd.DataFrame(alerts), use_container_width=True)
        else:
            st.success("All registered versions are inside the ±2σ band — no drift detected.")
    else:
        st.info(
            "Drift alerting needs ≥3 versions to compute a reliable band. "
            "Run more fits to populate the registry. Each "
            "`run_m3_boosted.py` / `run_m3.py` invocation adds one entry."
        )

    # -----------------------------------------------------------------------
    # 4. Champion / challenger comparison
    # -----------------------------------------------------------------------
    if challenger_row is not None and len(df) >= 2:
        section_header(
            "Champion vs challenger",
            "Promotion gate",
            "Side-by-side metrics for the current production model and the "
            "most recent non-champion fit. A real promotion gate would "
            "require both a positive delta on every metric AND a passing "
            "paired bootstrap test on per-trader predictions.",
        )
        cmp_rows = []
        for label, col in metric_options.items():
            ch_val = champion_row[col]
            cl_val = challenger_row[col]
            delta = (None if (pd.isna(ch_val) or pd.isna(cl_val))
                     else float(cl_val) - float(ch_val))
            cmp_rows.append({
                "Metric": label,
                "Champion": f"{ch_val:.3f}" if pd.notna(ch_val) else "—",
                "Challenger": f"{cl_val:.3f}" if pd.notna(cl_val) else "—",
                "Δ (Challenger − Champion)": (f"{delta:+.3f}"
                                                if delta is not None else "—"),
                "Verdict": (
                    "🟢 challenger improves" if delta is not None and delta > 0
                    else "🟡 no change" if delta is not None and abs(delta) < 1e-9
                    else "🔴 challenger regresses" if delta is not None
                    else "—"
                ),
            })
        st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True,
                     hide_index=True)
        st.caption(
            f"Champion: `{champion_row['entry']}`  ·  "
            f"Challenger: `{challenger_row['entry']}`"
        )

    # -----------------------------------------------------------------------
    # 5. Registry inventory table
    # -----------------------------------------------------------------------
    section_header(
        "Registry inventory",
        "All versions",
        "Click into a row to see its config.yaml, metrics.json, and notes "
        "from the filesystem at /data/model_registry/&lt;entry&gt;.",
    )

    show_df = df.copy()
    show_df["registered_at"] = show_df["registered_at_dt"].dt.strftime("%Y-%m-%d %H:%M UTC")
    show_cols = ["experiment_id", "registered_at", "device", "cohort",
                 "cv_auc", "locked_clique_recall",
                 "locked_ring_recall", "locked_mixed_recall",
                 "n_train_runs", "n_eval_runs"]
    show_df = show_df[show_cols + ["entry"]].rename(columns={
        "experiment_id": "Experiment",
        "registered_at": "Registered",
        "device": "Device",
        "cohort": "Cohort",
        "cv_auc": "CV AUC",
        "locked_clique_recall": "Clique",
        "locked_ring_recall": "Ring",
        "locked_mixed_recall": "Mixed",
        "n_train_runs": "n_train",
        "n_eval_runs": "n_eval",
    })
    for c in ("CV AUC", "Clique", "Ring", "Mixed"):
        show_df[c] = show_df[c].apply(
            lambda v: "—" if pd.isna(v) else f"{float(v):.3f}"
        )
    st.dataframe(
        show_df.set_index("entry"),
        use_container_width=True,
        height=240,
    )

    st.markdown("---")
    st.caption(
        f"Registry root: `{REGISTRY_ROOT}`  ·  "
        f"Currently {len(df)} version(s) registered.  ·  "
        "Hit refresh after a new fit to update."
    )


render()
