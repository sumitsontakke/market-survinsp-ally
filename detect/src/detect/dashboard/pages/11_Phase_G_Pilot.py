"""Phase G Pilot — browser-driven 4-stage wizard.

The user runs the Phase G pipeline as 4 stages: (1) generate cohort plans,
(2) synthesize all runs, (3) continual training, (4) OOD evaluation. This
page dashboards each stage:

  * Shows the EXACT command to copy-paste
  * Watches the filesystem for expected output files
  * When outputs appear, renders summary cards + sample data
  * Calls Ollama (qwen2.5:7b) for a plain-English explanation per stage

Designed for the pilot config (5 days x 50 runs x 1500 traders, 50 OOD test
runs). Same UI works at full scale (240 days x 50 x 5000) — the file-watcher
just dashboards more.

Lives at calibration_service/webapp_v2/pages/11_Phase_G_Pilot.py. Read-only
view of /outputs (mounted into webapp_v2 container at /outputs:ro). State
files for the page itself land in /data/phase_g_ui_state/ (the shared
market_data volume).
"""
from __future__ import annotations

import json
import os
import time
from html import escape
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Phase G Pilot — control panel",
                   page_icon="🧪", layout="wide")

try:
    from ollama_helper import explain  # type: ignore
    _HAS_OLLAMA = True
except Exception:  # noqa: BLE001
    _HAS_OLLAMA = False

# Optional: docker-socket integration. If the SDK is installed and the
# socket is reachable, the page renders Run/Cancel/Logs buttons. Otherwise
# it falls back to the copy-paste mode and shows a banner explaining why.
try:
    import docker_runner  # type: ignore
    _CAN_LAUNCH, _LAUNCH_REASON = docker_runner.available()
except Exception as _exc:  # noqa: BLE001
    docker_runner = None       # type: ignore[assignment]
    _CAN_LAUNCH = False
    _LAUNCH_REASON = f"docker_runner import failed: {_exc!r}"

OUTPUTS = Path(os.environ.get("OUTPUTS_DIR", "/outputs"))
COHORT_DIR = OUTPUTS / "phase_g_cohort"
OOD_DIR    = OUTPUTS / "phase_g_test_ood"
STATE_DIR  = OUTPUTS / "phase_g_state"
EVAL_JSON  = OUTPUTS / "_phase_g_eval_results.json"

PALETTE = {
    "navy":    "#1E2761",
    "ice":     "#7C8FC9",
    "accent":  "#C8102E",
    "ink":     "#1A1A2E",
    "muted":   "#5C6480",
    "success": "#0F7A4D",
    "warn":    "#A16207",
    "soft":    "#F4F6FA",
    "danger":  "#A62121",
}


# ---------------------------------------------------------------------------
# Status detection — each stage is "done" when its expected outputs exist.
# ---------------------------------------------------------------------------
def cohort_status() -> dict[str, Any]:
    """Stage 1: did regen_phase_g_cohort.py run?"""
    plan = COHORT_DIR / "PLAN.json"
    test_plan = OOD_DIR / "TEST_PLAN.json"
    has_train = plan.is_file()
    has_test = test_plan.is_file()
    train_meta = json.loads(plan.read_text("utf-8")) if has_train else {}
    test_meta = json.loads(test_plan.read_text("utf-8")) if has_test else {}
    return {
        "done":          has_train and has_test,
        "train_present": has_train,
        "test_present":  has_test,
        "train_meta":    train_meta,
        "test_meta":     test_meta,
    }


def synth_status() -> dict[str, Any]:
    """Stage 2: how many runs have orders.csv?"""
    train_total = train_done = 0
    test_total = test_done = 0
    if COHORT_DIR.is_dir():
        for day in sorted(COHORT_DIR.iterdir()):
            if not (day.is_dir() and day.name.startswith("DAY_")):
                continue
            for r in sorted(day.iterdir()):
                if r.is_dir() and r.name.startswith("DAY"):
                    train_total += 1
                    if (r / "orders.csv").is_file():
                        train_done += 1
    if OOD_DIR.is_dir():
        for r in sorted(OOD_DIR.iterdir()):
            if r.is_dir() and r.name.startswith("OOD_RUN"):
                test_total += 1
                if (r / "orders.csv").is_file():
                    test_done += 1
    return {
        "train_done":  train_done,
        "train_total": train_total,
        "test_done":   test_done,
        "test_total":  test_total,
        "done":        (train_total > 0 and train_done == train_total
                        and test_total > 0 and test_done == test_total),
    }


def train_status() -> dict[str, Any]:
    """Stage 3: how many day_NNN_checkpoint.pt files?

    Always returns the same key set so callers don't have to defend against
    early-return paths. (The original early-return dict was missing
    n_days_total and crashed the Stage 3 metric cards.)
    """
    cohort = cohort_status()
    n_days = int(cohort.get("train_meta", {}).get("days", 0)) or 0
    if not STATE_DIR.is_dir():
        return {
            "done":         False,
            "checkpoints":  0,
            "n_days_total": n_days,
            "history":      [],
        }
    ckpts = sorted(STATE_DIR.glob("day_*_checkpoint.pt"))
    history = []
    for ck in ckpts:
        met = ck.with_name(ck.stem.replace("_checkpoint", "_metrics") + ".json")
        if met.is_file():
            try:
                history.append(json.loads(met.read_text("utf-8")))
            except Exception:  # noqa: BLE001
                pass
    return {
        "done":         (n_days > 0 and len(ckpts) >= n_days),
        "checkpoints":  len(ckpts),
        "n_days_total": n_days,
        "history":      history,
    }


def eval_status() -> dict[str, Any]:
    """Stage 4: did _phase_g_eval_results.json materialize?"""
    if not EVAL_JSON.is_file():
        return {"done": False}
    try:
        payload = json.loads(EVAL_JSON.read_text("utf-8"))
    except Exception:  # noqa: BLE001
        return {"done": False, "error": "could not parse eval JSON"}
    return {"done": True, "payload": payload}


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def chip(label: str, status: str) -> str:
    """status is 'done', 'pending', 'running', 'blocked'."""
    color = {
        "done":    PALETTE["success"],
        "pending": PALETTE["muted"],
        "running": PALETTE["warn"],
        "blocked": PALETTE["danger"],
    }.get(status, PALETTE["muted"])
    icon = {"done": "OK", "pending": "...", "running": ">>",
            "blocked": "X"}.get(status, "?")
    return (
        f"<span style='display:inline-block;padding:4px 10px;"
        f"border-radius:12px;background:{color};color:white;"
        f"font-size:12px;font-weight:600;margin-right:6px;'>"
        f"{escape(icon)} {escape(label)}</span>"
    )


def stat_card(label: str, value: str, sub: str = "") -> str:
    return (
        f"<div style='background:{PALETTE['soft']};border-left:4px solid "
        f"{PALETTE['navy']};padding:12px 16px;border-radius:4px;"
        f"margin-bottom:10px;'>"
        f"<div style='font-size:10px;text-transform:uppercase;"
        f"letter-spacing:0.08em;color:{PALETTE['accent']};font-weight:600;'>"
        f"{escape(label)}</div>"
        f"<div style='font-family:Georgia,serif;font-size:22px;"
        f"color:{PALETTE['navy']};font-weight:bold;margin-top:2px;'>"
        f"{escape(value)}</div>"
        f"<div style='font-size:11px;color:{PALETTE['muted']};"
        f"margin-top:4px;'>{escape(sub)}</div></div>"
    )


def command_block(label: str, cmd: str) -> None:
    st.markdown(f"**{escape(label)}**")
    st.code(cmd, language="bash")


def ollama_explain(prompt: str, fallback: str) -> str:
    if not _HAS_OLLAMA:
        return fallback
    try:
        text = explain(prompt)
        return text or fallback
    except Exception:  # noqa: BLE001
        return fallback


# ---------------------------------------------------------------------------
# Header strip — at-a-glance status of all 4 stages
# ---------------------------------------------------------------------------
st.title("Phase G Pilot — control panel")
st.markdown(
    f"<div style='color:{PALETTE['muted']};font-style:italic;font-size:13px;"
    f"margin-bottom:16px;'>"
    f"Run the four pilot stages from your terminal — this page dashboards each "
    f"stage's progress, renders a summary when it completes, and shows a sample "
    f"of the output data.</div>",
    unsafe_allow_html=True,
)

cs = cohort_status()
ss = synth_status()
ts = train_status()
es = eval_status()

s1 = "done" if cs["done"] else "pending"
s2 = ("done" if ss["done"] else
      ("running" if ss["train_done"] > 0 else
       ("blocked" if not cs["done"] else "pending")))
s3 = ("done" if ts["done"] else
      ("running" if ts["checkpoints"] > 0 else
       ("blocked" if not ss["done"] else "pending")))
s4 = ("done" if es["done"] else
      ("blocked" if not ts["done"] else "pending"))

st.markdown(
    chip("Stage 1: Cohort", s1) +
    chip("Stage 2: Synth",  s2) +
    chip("Stage 3: Train",  s3) +
    chip("Stage 4: Eval",   s4),
    unsafe_allow_html=True,
)

# Docker-socket integration banner.
if _CAN_LAUNCH:
    st.success(f"Docker socket reachable — Run/Cancel/Logs buttons are live below.")
else:
    st.warning(
        "Run-from-browser is disabled. " + _LAUNCH_REASON + "  "
        "Copy the commands shown below into your terminal instead."
    )

if st.button("Refresh status", help="Re-scan the outputs directory"):
    st.rerun()

st.divider()


def stage_controls(stage: str, label: str) -> None:
    """Render Run/Cancel/Logs buttons for one stage. No-op if launching is
    not available — the user falls back to the copy-paste command."""
    if not _CAN_LAUNCH or docker_runner is None:
        return
    job = docker_runner.status(stage)
    is_running = job is not None and job.state == "running"
    cols = st.columns([1, 1, 4])
    with cols[0]:
        run_btn = st.button(
            f"Run {label}",
            key=f"run_{stage}",
            disabled=is_running,
            type="primary",
            help="Launch this stage in a detached trainer container",
        )
    with cols[1]:
        cancel_btn = st.button(
            "Cancel",
            key=f"cancel_{stage}",
            disabled=not is_running,
            help="docker stop the running container",
        )
    with cols[2]:
        if job is None:
            st.caption("No job started yet.")
        else:
            elapsed = (
                (job.finished_at or __import__("time").time())
                - job.started_at
            )
            badge = {
                "running":   "RUNNING",
                "done":      "DONE",
                "failed":    f"FAILED (exit {job.exit_code})",
                "cancelled": "CANCELLED",
            }.get(job.state, job.state.upper())
            st.caption(
                f"{badge}  |  container {job.container_id[:12]}  |  "
                f"elapsed {elapsed:.0f}s"
            )
    # IMPORTANT: st.rerun() raises RerunException by design. If we wrap it
    # in `try/except Exception`, the exception gets swallowed and the page
    # never refreshes (Streamlit surfaces it back as StreamlitAPIException).
    # So: do the launch inside try/except, and only call st.rerun() OUTSIDE
    # if launch succeeded. Same pattern for cancel.
    launched_ok = False
    if run_btn:
        try:
            docker_runner.launch(stage)
            launched_ok = True
        except Exception as exc:  # noqa: BLE001
            st.error(f"Launch failed: {exc!r}")
    if launched_ok:
        st.toast(f"Launched {label}")   # no icon — Streamlit expects a
                                         # single emoji char, not a string.
        st.rerun()

    cancelled_ok = False
    if cancel_btn:
        try:
            cancelled_ok = bool(docker_runner.cancel(stage))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Cancel failed: {exc!r}")
    if cancelled_ok:
        st.toast(f"Cancelled {label}")
        st.rerun()
    if job is not None:
        with st.expander("Live container logs (tail 200 lines)",
                         expanded=is_running):
            st.code(docker_runner.logs_tail(stage, n=200) or "(no output)",
                    language="text")


# ---------------------------------------------------------------------------
# Stage 1 — generate cohort plans
# ---------------------------------------------------------------------------
st.header("Stage 1 — generate cohort plans")
command_block(
    "Run on host (≈30 seconds):",
    "cd E:\\PesFinal\\market-survinsp-ally\\calibration_service\n"
    "docker compose run --rm trainer python /app/scripts/run_phase_g.py --stage cohort",
)
st.caption("Writes ~300 small `scenario_config.json` files. No GPU needed. "
           "Fast, idempotent — safe to re-run.")
stage_controls("cohort", "Stage 1")


if cs["done"]:
    c1, c2, c3, c4 = st.columns(4)
    tm = cs["train_meta"]
    em = cs["test_meta"]
    c1.markdown(stat_card("Train days", str(tm.get("days", "?")),
                          f"runs/day = {tm.get('runs_per_day', '?')}"),
                unsafe_allow_html=True)
    c2.markdown(stat_card("Train runs total", str(tm.get("total_runs", "?")),
                          f"traders = {tm.get('n_traders', '?')}"),
                unsafe_allow_html=True)
    c3.markdown(stat_card("OOD test runs", str(em.get("n_runs", "?")),
                          "held-out seeds + parameter ranges"),
                unsafe_allow_html=True)
    fc = tm.get("family_counts", {})
    c4.markdown(stat_card("Family balance",
                          f"{fc.get('clique', 0)} / {fc.get('ring', 0)} / "
                          f"{fc.get('mixed', 0)}",
                          "clique / ring / mixed"),
                unsafe_allow_html=True)

    with st.expander("Sample: a generated scenario_config.json", expanded=False):
        # Show the first run config in the first day, prettified.
        try:
            day0 = next(d for d in sorted(COHORT_DIR.iterdir())
                        if d.is_dir() and d.name.startswith("DAY_"))
            run0 = next(r for r in sorted(day0.iterdir())
                        if r.is_dir() and r.name.startswith("DAY"))
            cfg = json.loads((run0 / "scenario_config.json").read_text("utf-8"))
            st.json(cfg, expanded=False)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not load sample: {exc!r}")

    with st.expander("Sample: a day's parameter envelope (params.json)",
                     expanded=False):
        try:
            day0 = next(d for d in sorted(COHORT_DIR.iterdir())
                        if d.is_dir() and d.name.startswith("DAY_"))
            params = json.loads((day0 / "params.json").read_text("utf-8"))
            st.json(params, expanded=True)
            st.caption("Each day independently samples a parameter envelope "
                       "from one of five regimes (balanced / low_intensity / "
                       "high_intensity / short_burst / long_horizon). The "
                       "model is then sequentially exposed to all 5 days.")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not load params: {exc!r}")

    st.success(
        ollama_explain(
            f"Cohort generation produced {tm.get('total_runs', '?')} training "
            f"runs across {tm.get('days', '?')} simulated days plus "
            f"{em.get('n_runs', '?')} out-of-distribution test runs. The "
            f"training cohort family balance is "
            f"clique={fc.get('clique', 0)}, ring={fc.get('ring', 0)}, "
            f"mixed={fc.get('mixed', 0)}. Each day independently sampled "
            f"manipulator parameters from one of five regimes so the model "
            f"sees a wide distribution. Explain in plain English why this "
            f"matters for generalization, in two sentences.",
            fallback=f"Cohort plan complete: {tm.get('total_runs', 0)} "
                     f"training runs across {tm.get('days', 0)} days plus "
                     f"{em.get('n_runs', 0)} held-out OOD test runs. Family "
                     f"balance: {fc.get('clique', 0)} clique, "
                     f"{fc.get('ring', 0)} ring, {fc.get('mixed', 0)} mixed.",
        )
    )
else:
    st.info("Waiting for Stage 1 outputs. Run the command above, then click "
            "**Refresh status**.")

st.divider()


# ---------------------------------------------------------------------------
# Stage 2 — synthesize runs
# ---------------------------------------------------------------------------
st.header("Stage 2 — synthesize the runs")
command_block(
    "Run on host (8-12 hours for the pilot, resumable):",
    "docker compose run --rm trainer python /app/scripts/run_phase_g.py --stage synth",
)
st.caption("Each `scenario_config.json` becomes a full simulation: "
           "`orders.csv`, `trades.csv`, `traders.csv`, `manipulator_labels.csv`. "
           "The orchestrator skips runs that already have `orders.csv`, so "
           "interruptions are safe.")
stage_controls("synth", "Stage 2")


if cs["done"]:
    train_pct = (ss["train_done"] / ss["train_total"] * 100.0
                 if ss["train_total"] else 0.0)
    test_pct = (ss["test_done"] / ss["test_total"] * 100.0
                if ss["test_total"] else 0.0)
    c1, c2 = st.columns(2)
    c1.metric("Training runs synthesized",
              f"{ss['train_done']} / {ss['train_total']}",
              f"{train_pct:.1f}%")
    c2.metric("OOD test runs synthesized",
              f"{ss['test_done']} / {ss['test_total']}",
              f"{test_pct:.1f}%")
    st.progress(min(1.0, (ss["train_done"] + ss["test_done"]) /
                    max(1, ss["train_total"] + ss["test_total"])),
                text="Overall synth progress")

    # If at least one orders.csv has materialized, show a sample
    sample_run = None
    if COHORT_DIR.is_dir():
        for day in sorted(COHORT_DIR.iterdir()):
            if not (day.is_dir() and day.name.startswith("DAY_")):
                continue
            for r in sorted(day.iterdir()):
                if r.is_dir() and (r / "orders.csv").is_file():
                    sample_run = r
                    break
            if sample_run:
                break

    if sample_run:
        with st.expander(f"Sample: orders.csv from {sample_run.name}",
                         expanded=False):
            try:
                df = pd.read_csv(sample_run / "orders.csv", nrows=10)
                st.caption(f"First 10 rows. Full file has "
                           f"{sum(1 for _ in open(sample_run / 'orders.csv', 'r')) - 1:,} "
                           f"orders.")
                st.dataframe(df, use_container_width=True)
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Could not load orders.csv: {exc!r}")

        with st.expander(f"Sample: manipulator_labels.csv from "
                         f"{sample_run.name}", expanded=False):
            mlbl = sample_run / "manipulator_labels.csv"
            if mlbl.is_file():
                try:
                    df = pd.read_csv(mlbl)
                    st.caption(f"{len(df)} manipulator traders in this run.")
                    st.dataframe(df, use_container_width=True)
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"Could not load labels: {exc!r}")
            else:
                st.info("manipulator_labels.csv not found in this run yet.")

    if ss["done"]:
        st.success(
            ollama_explain(
                f"Synthesis is complete: {ss['train_done']} training runs and "
                f"{ss['test_done']} OOD test runs all have orders, trades, "
                f"and manipulator labels generated. Explain in two plain "
                f"sentences what these files contain and how they feed into "
                f"the GraphSAGE model training.",
                fallback=f"Synthesis complete. {ss['train_done']} training "
                         f"+ {ss['test_done']} OOD runs produced. Each run "
                         f"contributes one trader-trader interaction graph "
                         f"to the GraphSAGE training set.",
            )
        )
elif ss["train_done"] > 0:
    st.warning("Stage 2 partially done. Re-run the synth command to resume.")
else:
    st.info("Run Stage 1 first.")

st.divider()


# ---------------------------------------------------------------------------
# Stage 3 — continual training
# ---------------------------------------------------------------------------
st.header("Stage 3 — continual training (sequential warm-start)")
command_block(
    "Run on host (≈30-60 minutes on GPU):",
    "docker compose run --rm trainer-gpu python /app/scripts/run_phase_g.py --stage train",
)
st.caption("Day 0 trains from scratch. Days 1+ load the previous day's "
           "checkpoint and fine-tune. Uses focal loss + validation-trader-recall "
           "early stopping. Resumable — skip days already checkpointed.")

if ss["done"]:
    c1, c2 = st.columns(2)
    c1.metric("Days trained",
              f"{ts['checkpoints']} / {ts['n_days_total']}")
    if ts["history"]:
        last = ts["history"][-1]
        c2.metric("Latest val_recall_mean",
                  f"{last.get('val_recall_mean', 0):.3f}",
                  f"day {last.get('day_idx', '?')}")

    if ts["history"]:
        df_hist = pd.DataFrame([
            {
                "day":          h.get("day_idx", -1),
                "n_runs":       h.get("n_runs", 0),
                "fit_seconds":  round(h.get("fit_seconds", 0), 1),
                "epochs":       h.get("epochs_used", 0),
                "val_recall":   round(h.get("val_recall_mean", 0), 4),
                "warm_started": h.get("warm_started", False),
            }
            for h in ts["history"]
        ])
        st.subheader("Per-day training trajectory")
        st.dataframe(df_hist, use_container_width=True, hide_index=True)
        st.line_chart(df_hist.set_index("day")["val_recall"],
                      height=200, use_container_width=True)
        st.caption("`val_recall` should trend upward across days. Oscillation "
                   "is normal; a monotonic crash to 0 means the model is "
                   "diverging — abort and recheck cohort balance.")

    if ts["done"]:
        st.success(
            ollama_explain(
                f"Continual training finished {ts['checkpoints']} days. The "
                f"final val_recall_mean was "
                f"{ts['history'][-1].get('val_recall_mean', 0):.3f} after "
                f"{sum(h.get('epochs_used', 0) for h in ts['history'])} "
                f"total epochs. Explain in two plain sentences what 'sequential "
                f"warm-start' means and why this trajectory matters.",
                fallback=f"Continual training complete: {ts['checkpoints']} "
                         f"days, final val recall "
                         f"{ts['history'][-1].get('val_recall_mean', 0):.3f}.",
            )
        )
else:
    st.info("Run Stage 2 first.")

st.divider()


# ---------------------------------------------------------------------------
# Stage 4 — OOD evaluation
# ---------------------------------------------------------------------------
st.header("Stage 4 — OOD evaluation")
command_block(
    "Run on host (≈5 minutes):",
    "docker compose run --rm trainer-gpu python /app/scripts/run_phase_g.py --stage eval",
)
st.caption("Loads the final checkpoint, predicts on all OOD test runs, "
           "computes per-family recall + purity + AUC + benign alarm. "
           "Registers the result in `/data/model_registry` so the Metric "
           "Timeline page picks it up.")
stage_controls("eval", "Stage 4")


if ts["done"]:
    if es["done"]:
        payload = es["payload"]
        metrics = payload.get("metrics", {})
        st.subheader("Headline OOD numbers")
        c1, c2, c3 = st.columns(3)
        c1.metric("CV AUC (edge-level)",
                  f"{payload.get('cv_auc', 0):.4f}")
        _mean_r = (
            metrics.get("locked_clique_recall", 0)
            + metrics.get("locked_ring_recall", 0)
            + metrics.get("locked_mixed_recall", 0)
        ) / 3.0
        c2.metric("Mean trader recall", f"{_mean_r:.3f}")
        c3.metric("Benign alarm",
                  f"{metrics.get('locked_benign_alarm', 0):.4f}")

        st.subheader("Per-family breakdown")
        df_fam = pd.DataFrame([
            {
                "family":  "clique",
                "recall":  round(metrics.get("locked_clique_recall", 0), 3),
                "purity":  round(metrics.get("locked_clique_purity", 0), 3),
            },
            {
                "family":  "ring",
                "recall":  round(metrics.get("locked_ring_recall", 0), 3),
                "purity":  round(metrics.get("locked_ring_purity", 0), 3),
            },
            {
                "family":  "mixed",
                "recall":  round(metrics.get("locked_mixed_recall", 0), 3),
                "purity":  round(metrics.get("locked_mixed_purity", 0), 3),
            },
        ])
        st.dataframe(df_fam, use_container_width=True, hide_index=True)

        st.success(
            ollama_explain(
                f"OOD evaluation complete. The model was trained on "
                f"{ts['n_days_total']} days of varied manipulator parameters "
                f"and tested on {payload.get('n_test_runs', '?')} runs with "
                f"strictly disjoint parameter ranges. Per-family OOD trader "
                f"recall: clique={metrics.get('locked_clique_recall', 0):.3f}, "
                f"ring={metrics.get('locked_ring_recall', 0):.3f}, "
                f"mixed={metrics.get('locked_mixed_recall', 0):.3f}. Per-family "
                f"OOD purity: clique={metrics.get('locked_clique_purity', 0):.3f}, "
                f"ring={metrics.get('locked_ring_purity', 0):.3f}, "
                f"mixed={metrics.get('locked_mixed_purity', 0):.3f}. Benign "
                f"alarm: {metrics.get('locked_benign_alarm', 0):.4f}. In two "
                f"plain sentences, summarize whether this is good or bad news "
                f"for the dissertation's generalization claim.",
                fallback="OOD eval complete. See per-family numbers above.",
            )
        )

        with st.expander("Full eval JSON", expanded=False):
            st.json(payload, expanded=False)
    else:
        st.info("Run Stage 3 first, then the eval command.")
else:
    st.info("Run Stage 3 first.")

st.divider()
st.markdown(
    f"<div style='font-size:11px;color:{PALETTE['muted']};margin-top:8px;'>"
    f"Page reads from <code>/outputs</code> (read-only). The 4 commands "
    f"above run in the trainer/trainer-gpu containers. Refresh after each "
    f"stage to see live progress.</div>",
    unsafe_allow_html=True,
)
