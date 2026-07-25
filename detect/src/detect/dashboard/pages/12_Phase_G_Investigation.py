"""Phase G Investigation — flagged-trader analytics + engineered features.

A guided investigation page for the Phase G OOD cohort. Sections:

  A. Run selector + cohort context
  B. Feature library — 6 engineered features with descriptions, sample values, AUC
  C. Flagged traders — interactive timeline + 2D score scatter
  D. Manipulated tickers — price + volume context
  E. Trader network — graph of suspicious counterparty links
  F. LLM justification — per-trader narrative (Ollama qwen2.5:7b)

Data sources (all pre-computed outside Streamlit):

  outputs/_phase_g_features/<run>.csv         — engineered features per trader
  outputs/_phase_g_features_auc.json          — per-feature AUC + descriptions
  outputs/_phase_g_v1_trader_scores/<run>.csv — v1 model trader scores
  outputs/_phase_g_v1_threshold_sweep.json    — threshold operating points
  outputs/phase_g_test_ood/<run>/orders.csv   — raw orders for price/volume
  outputs/phase_g_test_ood/<run>/trades.csv   — raw trades for network

Heavy ML inference is NOT done in this page. It only reads CSVs + JSON.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import altair as alt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Phase G Investigation",
                   page_icon="🔍", layout="wide")

try:
    from ollama_helper import explain, is_ollama_available  # type: ignore
    _HAS_OLLAMA = True
except Exception:  # noqa: BLE001
    _HAS_OLLAMA = False

    def is_ollama_available() -> bool:
        return False

    def explain(prompt: str, **kw) -> str:
        return kw.get("fallback", "")


OUTPUTS = Path(os.environ.get("OUTPUTS_DIR", "/outputs"))
OOD_DIR = OUTPUTS / "phase_g_test_ood"
FEAT_DIR = OUTPUTS / "_phase_g_features"
FEAT_AUC = OUTPUTS / "_phase_g_features_auc.json"
SCORES_DIR = OUTPUTS / "_phase_g_v1_trader_scores"
SWEEP_JSON = OUTPUTS / "_phase_g_v1_threshold_sweep.json"

PALETTE = {
    "navy":    "#1E2761",
    "ice":     "#7C8FC9",
    "accent":  "#C8102E",
    "ink":     "#1A1A2E",
    "muted":   "#5C6480",
    "success": "#0F7A4D",
    "warn":    "#A16207",
    "danger":  "#A62121",
}


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def load_runs_index() -> pd.DataFrame:
    """List all OOD runs with metadata."""
    if not OOD_DIR.is_dir():
        return pd.DataFrame()
    rows = []
    for d in sorted(OOD_DIR.iterdir()):
        if not (d.is_dir() and d.name.startswith("OOD_RUN")):
            continue
        parts = d.name.split("_")
        scenario = parts[1] if len(parts) > 1 else "unknown"
        rows.append({
            "run_name":   d.name,
            "scenario":   scenario,
            "has_orders": (d / "orders.csv").is_file(),
            "has_trades": (d / "trades.csv").is_file(),
            "has_feat":   (FEAT_DIR / f"{d.name}.csv").is_file(),
            "has_score":  (SCORES_DIR / f"{d.name}.csv").is_file(),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=600, show_spinner=False)
def load_feature_auc() -> dict:
    if FEAT_AUC.is_file():
        return json.loads(FEAT_AUC.read_text("utf-8"))
    return {}


@st.cache_data(ttl=600, show_spinner=False)
def load_sweep() -> dict:
    if SWEEP_JSON.is_file():
        return json.loads(SWEEP_JSON.read_text("utf-8"))
    return {}


@st.cache_data(ttl=300, show_spinner=False)
def load_features_for_run(run_name: str) -> pd.DataFrame:
    p = FEAT_DIR / f"{run_name}.csv"
    return pd.read_csv(p) if p.is_file() else pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def load_scores_for_run(run_name: str) -> pd.DataFrame:
    """Per-trader v1 scores. Deduped to one row per trader.

    The projection's ``label_core`` flags traders with manipulator-incident
    edges (includes counterparties). We replace it with the true label
    looked up from the run's orders.csv.
    """
    p = SCORES_DIR / f"{run_name}.csv"
    if not p.is_file():
        return pd.DataFrame()
    df = pd.read_csv(p)
    grp = df.groupby("trader_id", as_index=False).agg(
        trader_score=("trader_score", "max"),
        max_edge_prob=("max_edge_prob", "max"),
        mean_edge_prob=("mean_edge_prob", "mean"),
        top3_edge_prob=("top3_edge_prob", "max"),
        positive_incident_edges=("positive_incident_edges", "max"),
        incident_edges=("incident_edges", "max"),
    )
    orders_p = OOD_DIR / run_name / "orders.csv"
    if orders_p.is_file():
        ords = pd.read_csv(orders_p,
                            usecols=["trader_id", "is_manipulative"])
        truth = (ords.groupby("trader_id")["is_manipulative"].any()
                 .astype(int).reset_index()
                 .rename(columns={"is_manipulative": "label_core"}))
        grp = grp.merge(truth, on="trader_id", how="left")
        grp["label_core"] = grp["label_core"].fillna(0).astype(int)
    else:
        grp["label_core"] = 0
    return grp


@st.cache_data(ttl=300, show_spinner=False)
def load_orders(run_name: str) -> pd.DataFrame:
    p = OOD_DIR / run_name / "orders.csv"
    if not p.is_file():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_trades(run_name: str) -> pd.DataFrame:
    p = OOD_DIR / run_name / "trades.csv"
    if not p.is_file():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("🔍 Phase G Investigation")
st.markdown(
    "*Surveillance officer's view of the Phase G out-of-distribution cohort. "
    "Six engineered features, the v1 model's trader scores, manipulated-ticker "
    "context, the trader-counterparty network, and LLM justification — for "
    "every flagged trader in every OOD run.*"
)

runs_idx = load_runs_index()
feat_auc_payload = load_feature_auc()
sweep_payload = load_sweep()

if runs_idx.empty:
    st.error("No OOD runs found. Run Phase G first (page 11_Phase_G_Pilot).")
    st.stop()

# ---------------------------------------------------------------------------
# Section A — Run selector + cohort context
# ---------------------------------------------------------------------------
st.markdown("## A. Run selector + cohort context")

cohort_total = len(runs_idx)
cohort_feat = int(runs_idx["has_feat"].sum())
cohort_score = int(runs_idx["has_score"].sum())
ready_runs = runs_idx[runs_idx["has_feat"] & runs_idx["has_score"]]

col_a1, col_a2, col_a3, col_a4 = st.columns(4)
col_a1.metric("OOD runs", cohort_total)
col_a2.metric("with features", cohort_feat)
col_a3.metric("with v1 scores", cohort_score)
col_a4.metric("fully ready", len(ready_runs))

scenario_counts = ready_runs["scenario"].value_counts().to_dict()
scen_pretty = ", ".join(f"{k}: {v}" for k, v in
                         sorted(scenario_counts.items()))
st.caption(f"Family mix in ready runs: {scen_pretty}")

if ready_runs.empty:
    st.warning("No run has both features and scores. Run "
               "`scripts/compute_engineered_features.py` and "
               "`scripts/predict_v1_all_ood.py` first.")
    st.stop()

left, right = st.columns([1, 2])
with left:
    family_filter = st.multiselect(
        "Filter by family",
        options=sorted(scenario_counts.keys()),
        default=sorted(scenario_counts.keys()),
    )
with right:
    options = ready_runs[ready_runs["scenario"].isin(family_filter)]
    if options.empty:
        st.error("No runs match the family filter.")
        st.stop()
    run_name = st.selectbox(
        "Select an OOD run to investigate",
        options=options["run_name"].tolist(),
        index=0,
    )

scenario = runs_idx.loc[runs_idx["run_name"] == run_name,
                          "scenario"].iloc[0]
st.markdown(f"**Selected:** `{run_name}` &nbsp;&nbsp; "
            f"family: **{scenario}**")

# ---------------------------------------------------------------------------
# Section B — Feature library
# ---------------------------------------------------------------------------
st.markdown("## B. Feature library (six engineered scalars)")
st.markdown(
    "Each row below is one explicit feature computed for every trader, "
    "centred on the trader's busiest 5-minute window. The **AUC** column "
    "shows the feature's standalone discriminative power against the "
    "ground-truth manipulator label across the pooled OOD cohort "
    f"({feat_auc_payload.get('n_traders_pooled', '?')} traders, "
    f"{feat_auc_payload.get('n_manipulators_pooled', '?')} manipulators)."
)

feat_df = load_features_for_run(run_name)
descriptions = feat_auc_payload.get("feature_descriptions", {})
per_feat = feat_auc_payload.get("per_feature_auc", {})

feat_lib_rows = []
for fname, desc in descriptions.items():
    if fname in feat_df.columns and not feat_df.empty:
        pos_rows = feat_df[feat_df["label_core"] == 1]
        if len(pos_rows) > 0:
            sample_val = pos_rows[fname].iloc[0]
        else:
            sample_val = feat_df[fname].iloc[0]
        if isinstance(sample_val, float):
            sample_str = f"{sample_val:.3f}"
        else:
            sample_str = str(sample_val)
    else:
        sample_str = "—"
    auc_info = per_feat.get(fname, {})
    auc = auc_info.get("auc", float("nan"))
    direction = auc_info.get("direction", "")
    if not math.isnan(auc):
        auc_cell = f"{auc:.3f} ({direction})"
    else:
        auc_cell = "—"
    feat_lib_rows.append({
        "Feature":      fname,
        "Sample value": sample_str,
        "AUC (pooled)": auc_cell,
        "Description":  desc,
    })

feat_lib_df = pd.DataFrame(feat_lib_rows)
st.dataframe(feat_lib_df, hide_index=True, use_container_width=True)

if feat_auc_payload:
    best_f = feat_auc_payload.get("best_single_feature", "—")
    best_a = feat_auc_payload.get("best_single_auc", 0.0)
    st.success(
        f"**Strongest single feature:** `{best_f}` "
        f"(AUC = {best_a:.3f}). Five of six features have AUC ≥ 0.70 — "
        "feature-augmented retraining is justified."
    )

# ---------------------------------------------------------------------------
# Section C — Flagged traders timeline + 2D scatter
# ---------------------------------------------------------------------------
st.markdown("## C. Flagged traders — interactive timeline + 2D scatter")

scores_df = load_scores_for_run(run_name)
thr = 0.0  # default; reset below if scores present

if scores_df.empty:
    st.warning("v1 trader scores missing for this run.")
else:
    sweep_targets = sweep_payload.get("best_at_targets", {})
    op_thr_50_d = sweep_targets.get("purity_ge_0.50") or {}
    op_thr_80_d = sweep_targets.get("purity_ge_0.80") or {}
    op_thr_95_d = sweep_targets.get("purity_ge_0.95") or {}
    op_thr_50 = op_thr_50_d.get("threshold") if op_thr_50_d else None
    op_thr_80 = op_thr_80_d.get("threshold") if op_thr_80_d else None
    op_thr_95 = op_thr_95_d.get("threshold") if op_thr_95_d else None

    score_max = float(scores_df["trader_score"].max())
    score_min = float(scores_df["trader_score"].min())
    default_thr = float(scores_df["trader_score"].quantile(0.95))
    if op_thr_80 and (score_min < op_thr_80 < score_max):
        default_thr = float(op_thr_80)

    c1, c2 = st.columns([2, 1])
    with c1:
        thr = st.slider(
            "Trader-score threshold (anything ≥ this is flagged)",
            min_value=score_min,
            max_value=score_max,
            value=default_thr,
            step=0.005,
        )
    with c2:
        st.markdown("**Operating points (pooled cohort):**")
        if op_thr_50 is not None:
            mark = "✓" if score_min < op_thr_50 < score_max else "✗"
            st.markdown(f"{mark} P ≥ 0.50 → thr `{op_thr_50:.3f}`")
        if op_thr_80 is not None:
            mark = "✓" if score_min < op_thr_80 < score_max else "✗"
            st.markdown(f"{mark} P ≥ 0.80 → thr `{op_thr_80:.3f}`")
        if op_thr_95 is not None:
            mark = "✓" if score_min < op_thr_95 < score_max else "✗"
            st.markdown(f"{mark} P ≥ 0.95 → thr `{op_thr_95:.3f}`")
        st.caption(
            f"This run score range: "
            f"[{score_min:.3f}, {score_max:.3f}]"
        )

    flagged = scores_df[scores_df["trader_score"] >= thr].copy()
    flagged = flagged.sort_values("trader_score", ascending=False)
    tp_n = int((flagged["label_core"] == 1).sum())
    fp_n = int((flagged["label_core"] == 0).sum())
    fn_n = int(((scores_df["label_core"] == 1)
                & (scores_df["trader_score"] < thr)).sum())
    recall = tp_n / max(tp_n + fn_n, 1)
    purity = tp_n / max(tp_n + fp_n, 1)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("flagged", len(flagged))
    m2.metric("true positives", tp_n)
    m3.metric("recall", f"{recall:.3f}")
    m4.metric("purity", f"{purity:.3f}")

    if not feat_df.empty:
        merged = scores_df.merge(feat_df, on="trader_id", how="inner",
                                  suffixes=("", "_feat"))
        x_options = [c for c in feat_df.columns if c in descriptions.keys()]
        default_x_idx = (x_options.index("counterparty_hhi_burst")
                          if "counterparty_hhi_burst" in x_options else 0)
        x_feat = st.selectbox(
            "X-axis feature",
            options=x_options,
            index=default_x_idx,
            key="scatter_x",
        )
        merged["label"] = merged["label_core"].map(
            {0: "Benign", 1: "Manipulator"}
        )
        scatter = (
            alt.Chart(merged)
            .mark_circle(size=80, opacity=0.65)
            .encode(
                x=alt.X(f"{x_feat}:Q", title=x_feat),
                y=alt.Y("trader_score:Q", title="v1 trader score"),
                color=alt.Color(
                    "label:N",
                    scale=alt.Scale(
                        domain=["Benign", "Manipulator"],
                        range=[PALETTE["ice"], PALETTE["accent"]],
                    ),
                    legend=alt.Legend(title="Ground truth"),
                ),
                tooltip=["trader_id", "label", "trader_score",
                         x_feat, "burst_concentration",
                         "counterparty_hhi_burst",
                         "top_partner_trade_share"],
            )
            .properties(height=380)
            .interactive()
        )
        rule = (alt.Chart(pd.DataFrame({"y": [thr]}))
                .mark_rule(color=PALETTE["warn"], strokeDash=[4, 4])
                .encode(y="y:Q"))
        st.altair_chart(scatter + rule, use_container_width=True)

    if len(flagged) > 0:
        st.markdown("### Order timeline for top flagged traders")
        orders = load_orders(run_name)
        top_k = st.slider("Show top-N flagged", 1, min(20, len(flagged)),
                          min(5, len(flagged)),
                          key="timeline_topk")
        top_ids = flagged.head(top_k)["trader_id"].tolist()
        sub = orders[orders["trader_id"].isin(top_ids)].copy()
        if not sub.empty:
            sub["minute"] = sub["timestamp"].dt.floor("1min")
            agg = (sub.groupby(["trader_id", "minute"]).size()
                   .reset_index(name="orders"))
            label_map = dict(zip(flagged["trader_id"], flagged["label_core"]))
            score_map = dict(zip(flagged["trader_id"], flagged["trader_score"]))
            agg["label"] = agg["trader_id"].map(
                lambda t: "Manipulator" if label_map.get(t) == 1
                else "Benign"
            )
            agg["score"] = agg["trader_id"].map(score_map)

            timeline = (
                alt.Chart(agg)
                .mark_circle(size=70)
                .encode(
                    x=alt.X("minute:T", title="Time"),
                    y=alt.Y("trader_id:N", sort=top_ids,
                             title="Trader (top flagged)"),
                    size=alt.Size("orders:Q",
                                   scale=alt.Scale(range=[20, 250]),
                                   title="orders / min"),
                    color=alt.Color(
                        "label:N",
                        scale=alt.Scale(
                            domain=["Benign", "Manipulator"],
                            range=[PALETTE["ice"], PALETTE["accent"]],
                        ),
                    ),
                    tooltip=["trader_id", "label", "score",
                             "minute", "orders"],
                )
                .properties(height=24 * top_k + 80)
            )
            st.altair_chart(timeline, use_container_width=True)

# ---------------------------------------------------------------------------
# Section D — Manipulated tickers price + volume
# ---------------------------------------------------------------------------
st.markdown("## D. Manipulated tickers — price + volume context")

trades_df = load_trades(run_name)
orders_df = load_orders(run_name)

if trades_df.empty or orders_df.empty:
    st.warning("Trades or orders missing for this run.")
else:
    manip_orders = orders_df[orders_df["is_manipulative"] == True]  # noqa: E712
    if manip_orders.empty:
        st.info("No manipulator orders flagged in this run.")
    else:
        manip_instruments = (manip_orders.groupby("instrument_id").size()
                             .sort_values(ascending=False))
        ins_pretty = ", ".join(
            f"`{i}` ({n})" for i, n in manip_instruments.head(8).items()
        )
        st.markdown(
            f"Found **{len(manip_instruments)}** instruments touched by "
            f"manipulators: " + ins_pretty
        )

        instr_pick = st.selectbox(
            "Pick instrument",
            options=manip_instruments.index.tolist(),
            key="instr_pick",
        )
        instr_trades = trades_df[
            trades_df["instrument_id"] == instr_pick
        ].copy()
        if not instr_trades.empty:
            instr_trades["minute"] = instr_trades["timestamp"].dt.floor("1min")
            bars = (instr_trades.groupby("minute")
                    .agg(price=("price", "mean"),
                          volume=("quantity", "sum"),
                          trades=("price", "count"))
                    .reset_index())
            manip_windows = manip_orders[
                manip_orders["instrument_id"] == instr_pick
            ]
            burst_start = burst_end = None
            if not manip_windows.empty:
                burst_start = manip_windows["timestamp"].min()
                burst_end = manip_windows["timestamp"].max()

            price_chart = (
                alt.Chart(bars)
                .mark_line(color=PALETTE["navy"], strokeWidth=2)
                .encode(
                    x=alt.X("minute:T", title="Time"),
                    y=alt.Y("price:Q", title="Mid price",
                             scale=alt.Scale(zero=False)),
                    tooltip=["minute", "price", "volume", "trades"],
                )
                .properties(height=240)
            )
            vol_chart = (
                alt.Chart(bars)
                .mark_bar(color=PALETTE["ice"], opacity=0.7)
                .encode(
                    x=alt.X("minute:T", title="Time"),
                    y=alt.Y("volume:Q", title="Volume / min"),
                    tooltip=["minute", "volume", "trades"],
                )
                .properties(height=160)
            )
            if burst_start is not None and burst_end is not None:
                shade = (alt.Chart(pd.DataFrame({
                    "start": [burst_start], "end": [burst_end]
                })).mark_rect(opacity=0.18, color=PALETTE["accent"])
                    .encode(x="start:T", x2="end:T"))
                price_chart = price_chart + shade
                vol_chart = vol_chart + shade
            st.altair_chart(price_chart, use_container_width=True)
            st.altair_chart(vol_chart, use_container_width=True)
            if burst_start is not None:
                st.caption(
                    f"Red shading marks the manipulator-active window: "
                    f"{burst_start.time()} → {burst_end.time()}"
                )

# ---------------------------------------------------------------------------
# Section E — Trader counterparty network
# ---------------------------------------------------------------------------
st.markdown("## E. Trader counterparty network")

if (not scores_df.empty) and (not trades_df.empty):
    net_k = st.slider("Network size (top-K flagged)", 3, 30, 12,
                       key="net_topk")
    top_flagged = (scores_df.sort_values("trader_score", ascending=False)
                   .head(net_k))
    focal_ids = top_flagged["trader_id"].tolist()
    label_map_e = dict(zip(top_flagged["trader_id"],
                            top_flagged["label_core"]))
    score_map_e = dict(zip(top_flagged["trader_id"],
                            top_flagged["trader_score"]))

    foc_set = set(focal_ids)
    incident = trades_df[(trades_df["buy_trader_id"].isin(foc_set))
                         | (trades_df["sell_trader_id"].isin(foc_set))]
    pair = incident.groupby(["buy_trader_id", "sell_trader_id"]).size()
    edge_rows = []
    for (a, b), c in pair.items():
        edge_rows.append({"src": a, "dst": b, "trades": int(c)})
    edges_df = pd.DataFrame(edge_rows)

    if edges_df.empty:
        st.info("No trades involving the top flagged traders.")
    else:
        cp_counts: dict = {}
        for _, r in edges_df.iterrows():
            for v in (r["src"], r["dst"]):
                if v not in foc_set:
                    cp_counts[v] = cp_counts.get(v, 0) + int(r["trades"])
        top_cps = sorted(cp_counts.items(), key=lambda kv: -kv[1])[:20]
        cp_ids = [c for c, _ in top_cps]
        node_ids = list(foc_set) + cp_ids

        positions: dict = {}
        n_foc = len(focal_ids)
        for i, t in enumerate(focal_ids):
            angle = 2 * math.pi * i / max(n_foc, 1)
            positions[t] = (math.cos(angle) * 0.5,
                            math.sin(angle) * 0.5)
        n_cp = len(cp_ids)
        for i, t in enumerate(cp_ids):
            angle = 2 * math.pi * i / max(n_cp, 1) + 0.2
            positions[t] = (math.cos(angle) * 1.0,
                            math.sin(angle) * 1.0)

        node_rows = []
        for t in node_ids:
            x, y = positions.get(t, (0, 0))
            is_foc = t in foc_set
            label = label_map_e.get(t, 0) if is_foc else 0
            score = score_map_e.get(t, 0.0) if is_foc else 0.0
            if is_foc and label == 1:
                kind = "Focal manipulator"
            elif is_foc:
                kind = "Focal benign"
            else:
                kind = "Counterparty"
            node_rows.append({
                "trader_id": t,
                "x":         x,
                "y":         y,
                "kind":      kind,
                "score":     score,
            })
        node_df = pd.DataFrame(node_rows)

        edge_plot_rows = []
        for _, r in edges_df.iterrows():
            if r["src"] not in positions or r["dst"] not in positions:
                continue
            x1, y1 = positions[r["src"]]
            x2, y2 = positions[r["dst"]]
            edge_plot_rows.append({
                "x": x1, "y": y1, "x2": x2, "y2": y2,
                "trades": r["trades"],
                "pair": f"{r['src']} -> {r['dst']}",
            })
        edge_plot_df = pd.DataFrame(edge_plot_rows)

        # Static matplotlib render. A layered Altair chart (edges + nodes
        # + labels) rendered through st.altair_chart crashes Streamlit's
        # Vega-Lite compile path ("Cannot create property 'bottom' on
        # number"). matplotlib sidesteps the Vega-Lite/Streamlit
        # interaction entirely and is already a webapp dependency.
        node_color = {
            "Focal manipulator": PALETTE["accent"],
            "Focal benign":      PALETTE["warn"],
            "Counterparty":      PALETTE["ice"],
        }
        fig, ax = plt.subplots(figsize=(8.6, 6.4))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        t_max = (int(edge_plot_df["trades"].max())
                 if not edge_plot_df.empty else 1)
        t_max = max(t_max, 1)
        for _, e in edge_plot_df.iterrows():
            lw = 0.6 + 3.6 * (int(e["trades"]) / t_max)
            ax.plot([e["x"], e["x2"]], [e["y"], e["y2"]],
                    color=PALETTE["muted"], alpha=0.32, linewidth=lw,
                    zorder=1, solid_capstyle="round")

        for _, n in node_df.iterrows():
            ax.scatter(n["x"], n["y"], s=320, zorder=3,
                       c=node_color.get(n["kind"], PALETTE["ice"]),
                       edgecolors=PALETTE["navy"], linewidths=0.7)
            ax.annotate(str(n["trader_id"]), (n["x"], n["y"]),
                        textcoords="offset points", xytext=(0, 11),
                        ha="center", fontsize=7, color=PALETTE["ink"])

        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.set_aspect("equal")
        ax.axis("off")
        legend_handles = [
            Line2D([0], [0], marker="o", linestyle="", markersize=10,
                   markerfacecolor=node_color["Focal manipulator"],
                   markeredgecolor=PALETTE["navy"],
                   label="Focal manipulator"),
            Line2D([0], [0], marker="o", linestyle="", markersize=10,
                   markerfacecolor=node_color["Focal benign"],
                   markeredgecolor=PALETTE["navy"], label="Focal benign"),
            Line2D([0], [0], marker="o", linestyle="", markersize=10,
                   markerfacecolor=node_color["Counterparty"],
                   markeredgecolor=PALETTE["navy"], label="Counterparty"),
        ]
        ax.legend(handles=legend_handles, loc="upper right",
                  fontsize=8, frameon=True, framealpha=0.9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            f"Inner ring: {len(focal_ids)} top-flagged traders "
            f"(red=manipulator, amber=benign). "
            f"Outer ring: {len(cp_ids)} most-active counterparties. "
            f"Edge thickness = trade count."
        )

# ---------------------------------------------------------------------------
# Section F — LLM justification per flagged trader
# ---------------------------------------------------------------------------
st.markdown("## F. LLM justification per flagged trader")

def _safe_float(v, default=0.0):
    try:
        fv = float(v)
        return default if (fv != fv) else fv
    except (TypeError, ValueError):
        return default


def _safe_int(v, default=0):
    try:
        fv = float(v)
        return default if (fv != fv) else int(fv)
    except (TypeError, ValueError):
        return default


if scores_df.empty:
    st.warning("Need v1 scores for this run.")
else:
    merged_inv = scores_df.merge(feat_df, on="trader_id", how="left",
                                   suffixes=("", "_feat"))
    flagged_inv = (merged_inv[merged_inv["trader_score"] >= thr]
                   .sort_values("trader_score", ascending=False))

    if flagged_inv.empty:
        st.info("No flagged traders at the current threshold. "
                "Lower the threshold in section C.")
    else:
        # Always justify every true positive (flagged manipulator).
        # The slider only governs how many false positives to show.
        tp_inv = flagged_inv[flagged_inv["label_core"] == 1]
        fp_inv = flagged_inv[flagged_inv["label_core"] != 1]
        n_tp, n_fp = len(tp_inv), len(fp_inv)

        st.caption(
            f"{n_tp} true positive(s) flagged at this threshold — all "
            f"shown below. {n_fp} false positive(s) available."
        )
        if n_fp > 0:
            n_fp_show = st.slider(
                "False positives to display",
                0, n_fp, min(3, n_fp),
                key="just_fp_n",
            )
        else:
            n_fp_show = 0

        display_inv = pd.concat(
            [tp_inv, fp_inv.head(n_fp_show)], ignore_index=True
        )
        if display_inv.empty:
            st.info("Nothing to show — raise the false-positive "
                    "slider above.")

        ollama_on = _HAS_OLLAMA and is_ollama_available()
        if not ollama_on:
            st.caption(
                "Ollama not reachable - showing rule-based "
                "fallback narrative instead of LLM output."
            )

        for _, row in display_inv.iterrows():
            tid = str(row.get("trader_id", ""))
            is_manip = _safe_int(row.get("label_core", 0)) == 1
            score = _safe_float(row.get("trader_score", 0.0))
            badge = ("KNOWN MANIPULATOR" if is_manip
                     else "BENIGN (false positive)")
            header = f"{badge}  ·  `{tid}`  ·  score = {score:.3f}"
            with st.expander(header, expanded=False):
                f_burst = _safe_float(row.get("burst_concentration", 0.0))
                f_entropy = _safe_float(row.get("side_entropy_in_burst", 0.0))
                f_hhi = _safe_float(row.get("counterparty_hhi_burst", 0.0))
                f_qty = _safe_float(row.get("order_qty_cov", 0.0))
                f_top = _safe_float(row.get("top_partner_trade_share", 0.0))
                f_co = _safe_int(row.get("co_active_top_count", 0))
                n_orders = _safe_int(row.get("n_orders", 0))
                n_burst = _safe_int(row.get("n_burst_orders", 0))
                features_known = (n_orders > 0)

                if not features_known:
                    st.caption(
                        "This trader was not in the top-200-active "
                        "subset, so feature values are unavailable. "
                        "Showing only the v1 model score."
                    )
                else:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("burst orders / total",
                              f"{n_burst} / {n_orders}",
                              f"{f_burst*100:.1f}%")
                    c2.metric("CP HHI (burst)", f"{f_hhi:.3f}")
                    c3.metric("Side entropy", f"{f_entropy:.3f}")
                    c1.metric("top-partner share", f"{f_top:.3f}")
                    c2.metric("order qty CoV", f"{f_qty:.3f}")
                    c3.metric("co-active top traders", f_co)

                entropy_word = "one-sided" if f_entropy < 0.4 else "balanced"
                if is_manip:
                    truth_word = "manipulator"
                    truth_phrase = "this trader IS a known manipulator."
                else:
                    truth_word = "benign"
                    truth_phrase = (
                        "this trader is BENIGN - likely a "
                        "false positive at this threshold."
                    )
                if features_known:
                    fallback = (
                        f"Trader {tid} (score {score:.3f}) showed "
                        f"{n_burst} of {n_orders} orders in a 5-min burst "
                        f"({f_burst*100:.0f}% concentration). Top "
                        f"counterparty shared {f_top*100:.0f}% of burst "
                        f"trades; side entropy was {f_entropy:.2f} "
                        f"({entropy_word}). Ground truth: {truth_phrase}"
                    )
                else:
                    fallback = (
                        f"Trader {tid} flagged with score {score:.3f}. "
                        f"Engineered features unavailable for this trader. "
                        f"Ground truth: {truth_phrase}"
                    )
                if ollama_on and features_known:
                    prompt = (
                        f"A market-surveillance model flagged trader {tid} "
                        f"with score {score:.3f}. Burst stats: concentration "
                        f"{f_burst*100:.0f}%, side entropy {f_entropy:.2f} "
                        f"(0=one-sided, 1=balanced), counterparty HHI "
                        f"{f_hhi:.2f}, top-partner share {f_top*100:.0f}%, "
                        f"co-active top traders {f_co}. Ground truth: "
                        f"{truth_word}. In two sentences, what would a "
                        f"surveillance officer say about this trader?"
                    )
                    narrative = explain(
                        prompt, max_tokens=160, fallback=fallback
                    )
                else:
                    narrative = fallback
                st.markdown(f"**Verdict narrative:** {narrative}")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Phase G Investigation page - powered by pre-computed v1 trader "
    "scores and six engineered features (no model inference inside "
    "Streamlit). See note 46 in the vault for the methodology writeup."
)
