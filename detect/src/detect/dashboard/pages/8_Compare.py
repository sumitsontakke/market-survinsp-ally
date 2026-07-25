"""Compare — Four-rung representation ladder, measured numbers.

This is the M4 deliverable of the R3 plan: a Streamlit view that reads
the locked-stress comparison CSV produced by training (rungs 1-4+) and
renders it as the dissertation's headline contrast table, plus per-family
charts, per-run drill-down, and training loss curves.

Inputs (all read-only):
  - outputs/r3_locked_stress.csv         four-rung roll-up
  - outputs/_m3_full_metrics.json        M3 baseline (CPU) per-run recall
  - outputs/_m3_boosted_metrics.json     M3+ boosted (GPU) per-run recall
  - outputs/_m3_full_loss_curve.json     per-epoch loss (M3 baseline)
  - outputs/_m3_boosted_loss_curve.json  per-epoch loss (M3+ boosted)

The page intentionally does no training — it only renders artifacts that
already exist on disk. If a file is missing the section degrades quietly.

Reference
---------
Cont, R. (2001). Empirical properties of asset returns. QF 1(2), 223-236.
Hamilton, W. L., Ying, R., Leskovec, J. (2017). Inductive Representation
Learning on Large Graphs. NeurIPS.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Compare — Four Rungs", page_icon="📐", layout="wide")

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

# Inside Docker the host outputs/ is bind-mounted at /outputs. Outside
# Docker (rare; mostly for local dev) we fall back to the repo path.
DEFAULT_PATHS = [
    Path("/outputs"),
    Path(os.environ.get("OUTPUTS_DIR", "")),
    Path(__file__).resolve().parent.parent.parent.parent / "outputs",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _outputs_root() -> Path | None:
    for p in DEFAULT_PATHS:
        if p and p.exists():
            return p
    return None


def _read_json(path: Path) -> Any | None:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _fmt_recall(x: float | str) -> str:
    """Render a recall cell: '1.000', '0.956', or '—' for NOT_COMPUTED (-1)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v < 0:
        return "—"
    return f"{v:.3f}"


def _badge(label: str, kind: str = "ok") -> str:
    colors = {
        "ok":   ("#DCFCE7", "#14532D"),
        "warn": ("#FEF9C3", "#713F12"),
        "bad":  ("#FEE2E2", "#991B1B"),
        "info": ("#E0E7FF", "#312E81"),
    }
    bg, fg = colors.get(kind, colors["info"])
    return (
        f"<span style='display:inline-block;padding:2px 8px;border-radius:12px;"
        f"font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.04em;"
        f"background:{bg};color:{fg};'>{label}</span>"
    )


# ---------------------------------------------------------------------------
# data loaders
# ---------------------------------------------------------------------------
@st.cache_data(ttl=30)
def load_four_rung(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # The CSV mixes numeric and 'n/a'/'recovered' tokens. Coerce the recall
    # columns to nullable floats so charts work; keep the originals for the
    # display column.
    for col in ("locked_clique_recall", "locked_ring_recall",
                "locked_mixed_recall", "locked_benign_alarm",
                "cv_f1", "cv_auc"):
        if col in df.columns:
            df[col + "_raw"] = df[col]
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_metrics(out_root: Path, fname: str) -> dict | None:
    return _read_json(out_root / fname)


def load_loss_curve(out_root: Path, fname: str) -> pd.DataFrame | None:
    raw = _read_json(out_root / fname)
    if raw is None:
        return None
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return pd.DataFrame(raw)
    # Tolerate flat list of train_loss values (older format)
    if isinstance(raw, list):
        return pd.DataFrame({"epoch": range(1, len(raw) + 1), "train_loss": raw})
    return None


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------
def render() -> None:
    st.markdown(
        f"<h1 style='color:{PALETTE['navy']};font-family:Georgia,serif;'>"
        "Four-Rung Representation Ladder</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='color:{PALETTE['muted']};font-style:italic;margin-bottom:18px;'>"
        "Measured locked-stress trader-level recall on the R01-R24 cohort, "
        "comparing statistical, trader-ML, edge-engineered ML, and learned-interaction "
        "(GraphSAGE) representations."
        "</div>",
        unsafe_allow_html=True,
    )

    out_root = _outputs_root()
    if out_root is None:
        st.error("Could not locate the outputs/ folder.")
        st.stop()

    # Data-source toggle — synth vs ABIDES. ABIDES cohort comes from Phase D
    # (services/abides-synth/src/run_cohort.py); both CSVs have identical
    # column schemas so the rest of the page renders unchanged.
    abides_csv = out_root / "r3_abides_locked_stress.csv"
    sources = {
        "synthetic_market_sim  (our synthesizer)": out_root / "r3_locked_stress.csv",
    }
    if abides_csv.exists():
        sources["ABIDES  (peer-reviewed, Byrd et al. 2019)"] = abides_csv

    if len(sources) > 1:
        st.markdown(
            f"<div style='background:#E0E7FF;border-left:4px solid {PALETTE['navy']};"
            f"padding:10px 14px;border-radius:4px;margin:6px 0 12px 0;font-size:13px;"
            f"color:{PALETTE['ink']};'>"
            "🎛️ <strong>Dual-track:</strong> the four-rung table renders against "
            "either data source. <em>synthetic_market_sim</em> is our own "
            "calibrated generator (Phase 2 R3 numbers, M3+ headline). "
            "<em>ABIDES</em> is the JPMC fork of the peer-reviewed agent-based "
            "simulator (Byrd, Hybinette, Balch 2019) — Phase D-1 cohort, "
            "Rungs 1-4+ evaluation pending on this cohort."
            "</div>",
            unsafe_allow_html=True,
        )
        source_label = st.radio(
            "Data source",
            list(sources.keys()),
            index=0,
            horizontal=True,
            help=("Switch between our synthesizer and the ABIDES-generated "
                  "cohort. Both share the same per-run MSA schema."),
        )
        csv_path = sources[source_label]
    else:
        csv_path = sources["synthetic_market_sim  (our synthesizer)"]
        source_label = "synthetic_market_sim  (our synthesizer)"

    if not csv_path.exists():
        st.error(f"`{csv_path}` not found. Run M3 + M3+ training first.")
        st.stop()

    df = load_four_rung(csv_path)

    # Banner with the per-source caveat where applicable
    if "abides" in source_label.lower():
        st.markdown(
            f"<div style='background:#FEF9C3;border-left:4px solid #A16207;"
            f"padding:10px 14px;border-radius:4px;margin:0 0 12px 0;"
            f"font-size:12px;color:{PALETTE['ink']};'>"
            "ℹ️ <strong>ABIDES cohort is generated and on disk, but rungs are "
            "not yet evaluated against it.</strong> "
            "Cells show <code>—</code> while we wait on the per-rung detectors "
            "to re-run on this cohort (Rung-1 is fast; Rung-4 needs GPU retrain). "
            "Until then the dual-track view confirms the infrastructure is in "
            "place; numbers fill in as evaluators land."
            "</div>",
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------------
    # 1. Headline table
    # -----------------------------------------------------------------------
    st.markdown("### Headline: trader-level recall on the locked-stress holdout")

    display_cols = {
        "rung":            "Rung",
        "representation":  "Representation",
        "model":           "Model",
        "locked_clique_recall": "Clique recall",
        "locked_ring_recall":   "Ring recall",
        "locked_mixed_recall":  "Mixed recall",
        "locked_benign_alarm":  "Benign alarm",
        "cv_auc":          "CV AUC",
        "source":          "Source",
    }
    avail = [c for c in display_cols if c in df.columns]
    show = df[avail].rename(columns=display_cols).copy()
    for col in ("Clique recall", "Ring recall", "Mixed recall", "Benign alarm"):
        if col in show.columns:
            show[col] = show[col].apply(_fmt_recall)
    if "CV AUC" in show.columns:
        show["CV AUC"] = show["CV AUC"].apply(
            lambda v: "—" if pd.isna(v) else f"{float(v):.3f}"
        )

    st.dataframe(
        show.set_index("Rung"),
        use_container_width=True,
        height=240,
    )

    # The thesis sentence, restated under the table for the reviewer.
    st.markdown(
        f"<div style='background:{PALETTE['soft_bg']};border-left:4px solid {PALETTE['navy']};"
        "padding:14px 18px;border-radius:4px;margin:8px 0 20px 0;'>"
        "<strong>The thesis sentence:</strong> trader-to-trader interaction modeling "
        "(<em>Rung 4</em>) beats statistical baselines (Rung 1), trader-level ML (Rung 2), "
        "and engineered edge ML (Rung 3) on the ring and mixed manipulation families. "
        "The boosted GPU run (<em>Rung 4+</em>) extends coverage to the clique family "
        "without losing the ladder ordering."
        "</div>",
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # 2. Per-family bar chart
    # -----------------------------------------------------------------------
    st.markdown("### Per-family recall, side by side")
    st.caption(
        "Grouped bars across rungs. `—` cells (NOT_COMPUTED / no holdout coverage) "
        "are omitted from the chart but shown in the table above."
    )

    melted_rows = []
    for _, r in df.iterrows():
        rung_label = f"R{r['rung']}"  # e.g. R1, R2, R3, R4, R4+
        for col, family in [
            ("locked_clique_recall", "clique"),
            ("locked_ring_recall",   "ring"),
            ("locked_mixed_recall",  "mixed"),
        ]:
            v = r.get(col)
            if pd.isna(v) or v < 0:
                continue
            melted_rows.append({"rung": rung_label, "family": family, "recall": float(v)})

    if melted_rows:
        chart_df = pd.DataFrame(melted_rows)
        pivot = chart_df.pivot(index="rung", columns="family", values="recall")
        # Preserve rung order: 1 < 2 < 3 < 4 < 4+
        pivot = pivot.reindex(["R1", "R2", "R3", "R4", "R4+"]).dropna(how="all")
        st.bar_chart(pivot, height=320)
    else:
        st.info("No measured recall numbers yet. Run M3 / M3+ training first.")

    # -----------------------------------------------------------------------
    # 3. Training loss curves (M3 vs M3+)
    # -----------------------------------------------------------------------
    st.markdown("### Training convergence — M3 baseline vs M3+ boosted")

    m3_loss = load_loss_curve(out_root, "_m3_full_loss_curve.json")
    mp_loss = load_loss_curve(out_root, "_m3_boosted_loss_curve.json")

    cols = st.columns(2)
    with cols[0]:
        st.markdown(
            f"<div style='font-weight:600;color:{PALETTE['navy']};'>"
            "Rung 4 — M3 baseline (CPU, 2-layer, 50 epochs cfg)</div>",
            unsafe_allow_html=True,
        )
        if m3_loss is None:
            st.info("`_m3_full_loss_curve.json` not present.")
        else:
            keep = [c for c in ("train_loss", "val_loss") if c in m3_loss.columns]
            st.line_chart(m3_loss.set_index("epoch")[keep] if keep else m3_loss, height=300)
            stopped = int(m3_loss["epoch"].max()) if "epoch" in m3_loss.columns else len(m3_loss)
            st.caption(f"Stopped at epoch **{stopped}** (early stopping).")
    with cols[1]:
        st.markdown(
            f"<div style='font-weight:600;color:{PALETTE['navy']};'>"
            "Rung 4+ — M3+ boosted (GPU sm_120, 3-layer, 200 epochs cfg)</div>",
            unsafe_allow_html=True,
        )
        if mp_loss is None:
            st.info("`_m3_boosted_loss_curve.json` not present.")
        else:
            keep = [c for c in ("train_loss", "val_loss") if c in mp_loss.columns]
            st.line_chart(mp_loss.set_index("epoch")[keep] if keep else mp_loss, height=300)
            stopped = int(mp_loss["epoch"].max()) if "epoch" in mp_loss.columns else len(mp_loss)
            st.caption(f"Stopped at epoch **{stopped}** (early stopping with patience 20).")

    # -----------------------------------------------------------------------
    # 4. Per-run drill-down
    # -----------------------------------------------------------------------
    st.markdown("### Per-run holdout breakdown")
    st.caption(
        "Trader-level recall on each individual run in the holdout. Pick a "
        "Rung-4 variant and an individual run to see its training configuration "
        "and direct links to the metrics + observation artifacts."
    )

    variants: dict[str, dict] = {}
    m3_metrics = load_metrics(out_root, "_m3_full_metrics.json")
    if m3_metrics and m3_metrics.get("locked_per_run"):
        variants["M3 baseline (CPU, 2-layer)"] = {
            "metrics": m3_metrics,
            "metrics_path": "_m3_full_metrics.json",
            "loss_path":    "_m3_full_loss_curve.json",
            "obs_path":     "../training/M3_FULL_OBSERVATIONS.md",
            "config": {
                "device":     "cpu (forced; sm_120 unsupported in stable torch)",
                "hidden":     "[128, 64]",
                "epochs_cfg": 50,
                "patience":   8,
                "seed":       42,
                "holdout":    "R09 R10 R11 (ring) + R17 R18 R19 (mixed)",
            },
        }
    mp_metrics = load_metrics(out_root, "_m3_boosted_metrics.json")
    if mp_metrics and mp_metrics.get("locked_per_run"):
        variants["M3+ boosted (GPU sm_120, 3-layer)"] = {
            "metrics": mp_metrics,
            "metrics_path": "_m3_boosted_metrics.json",
            "loss_path":    "_m3_boosted_loss_curve.json",
            "obs_path":     "../training/M3_BOOSTED_OBSERVATIONS.md",
            "config": {
                "device":     "cuda (RTX 5060 Ti, sm_120) — PyTorch nightly cu128",
                "hidden":     "[256, 128, 64]",
                "epochs_cfg": 200,
                "patience":   20,
                "seed":       42,
                "holdout":    "R03 R07 (clique) + R09 R11 (ring) + R17 R19 (mixed)",
            },
        }

    if not variants:
        st.info("No per-run metrics on disk. Run M3 / M3+ training first.")
        return

    picked = st.selectbox("Pick a Rung-4 variant", list(variants.keys()))
    chosen = variants[picked]
    per_run = chosen["metrics"]["locked_per_run"]

    # Per-run table with sortable columns + hyperlink-style row selection
    rows = []
    for run_id, recall in per_run.items():
        fam = "clique" if "clique" in run_id else "ring" if "ring" in run_id else "mixed" if "mixed" in run_id else "?"
        rows.append({"run": run_id, "family": fam, "recall": float(recall)})
    per_run_df = pd.DataFrame(rows).sort_values(["family", "run"])

    a, b = st.columns([3, 2])
    with a:
        st.markdown("#### Per-run recall")
        st.dataframe(
            per_run_df.set_index("run"),
            use_container_width=True,
            height=260,
            column_config={
                "family": st.column_config.TextColumn("Family"),
                "recall": st.column_config.NumberColumn(
                    "Trader-level recall",
                    format="%.3f",
                    min_value=0.0,
                    max_value=1.0,
                ),
            },
        )

        run_pick = st.selectbox(
            "Drill into a run",
            per_run_df["run"].tolist(),
            help="Selecting here surfaces the run's family + recall + a direct "
                 "link to the run directory under outputs/runs/.",
        )
        if run_pick:
            recall = float(per_run.get(run_pick, 0.0))
            fam = per_run_df.loc[per_run_df["run"] == run_pick, "family"].iloc[0]
            color = "ok" if recall >= 0.85 else "warn" if recall >= 0.50 else "bad"
            st.markdown(
                f"""
                <div style='background:{PALETTE['soft_bg']};padding:12px 16px;
                            border-radius:4px;border-left:4px solid {PALETTE['navy']};
                            margin-top:8px;'>
                  <div style='font-family:Georgia,serif;font-size:18px;color:{PALETTE['navy']};'>
                    <strong>{run_pick}</strong>
                  </div>
                  <div style='margin-top:6px;'>
                    Family: <code>{fam}</code> &nbsp;·&nbsp;
                    Trader-level recall: <code>{recall:.3f}</code> &nbsp;·&nbsp;
                    {_badge(color.upper(), color)}
                  </div>
                  <div style='margin-top:6px;font-size:12px;color:{PALETTE['muted']};'>
                    Artifacts: <code>/outputs/runs/{run_pick}/</code>
                    (or <code>/outputs/calibrated_runs/{run_pick}/</code>).
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with b:
        st.markdown("#### Training config")
        cfg = chosen["config"]
        cfg_md = "\n".join(f"- **{k}**: `{v}`" for k, v in cfg.items())
        st.markdown(cfg_md)

        st.markdown("#### Headline metrics")
        m = chosen["metrics"]
        n_edges = m.get("n_edges_train")
        n_edges_disp = f"{int(n_edges):,}" if isinstance(n_edges, (int, float)) else "—"
        st.markdown(
            f"""
            - **locked_clique_recall** = `{_fmt_recall(m.get('locked_clique_recall', -1))}`
            - **locked_ring_recall**   = `{_fmt_recall(m.get('locked_ring_recall', -1))}`
            - **locked_mixed_recall**  = `{_fmt_recall(m.get('locked_mixed_recall', -1))}`
            - **locked_benign_alarm**  = `{_fmt_recall(m.get('locked_benign_alarm', -1))}`
            - **cv_auc**               = `{float(m.get('cv_auc', 0)):.4f}`
            - **cv_f1**                = `{float(m.get('cv_f1', 0)):.4f}`
            - **edges (train)**        = `{n_edges_disp}`
            - **positives (train)**    = `{m.get('n_pos_edges_train', '—')}`
            """
        )

        st.markdown("#### Artifacts on disk")
        st.markdown(
            f"""
            - metrics: `outputs/{chosen['metrics_path']}`
            - loss curve: `outputs/{chosen['loss_path']}`
            - observations: `training/{Path(chosen['obs_path']).name}`
            - comparison CSV: `outputs/r3_locked_stress.csv`
            """
        )

    # -----------------------------------------------------------------------
    # 5. Footer
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.caption(
        "Reference: Cont, R. (2001). *Empirical properties of asset returns.* "
        "Quantitative Finance, 1(2), 223-236. · "
        "Hamilton, W. L., Ying, R., Leskovec, J. (2017). *Inductive Representation "
        "Learning on Large Graphs.* NeurIPS."
    )


render()
