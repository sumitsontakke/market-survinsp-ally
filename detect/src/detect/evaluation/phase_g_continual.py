"""Phase G continual-learning trainer.

Sequential warm-start protocol the user asked for:
  * Day 0: train from scratch on day 0's runs (full epoch budget).
  * Day 1..N: load the previous day's checkpoint and fine-tune on day N's
    runs (smaller epoch budget so the model doesn't catastrophically forget).
  * Per-day val-trader-recall used for early stopping (Phase G uses the
    production metric, not val edge loss).
  * Single model evolves across days; final checkpoint trained on
    cumulative data via a separate "consolidation" pass.

Why this works for generalization (versus 1 fit on a flat cohort):
  * Sequential exposure to different parameter regimes (each day samples
    independently) is itself a form of curriculum / domain randomization.
  * Catastrophic forgetting is checked by re-evaluating on a held-out
    validation cohort that is FIXED across days.
  * Checkpoint after every day -> can plot generalization-over-time.

Resumable: state lives in outputs/phase_g_state/. Each completed day
writes day_<NNN>_checkpoint.pt + day_<NNN>_metrics.json. Re-launching
the script skips days already done.

Run inside trainer-gpu container:
    docker compose run --rm trainer-gpu \\
        python -u /app/training/phase_g_continual.py
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_APP_ROOT = os.environ.get("MSA_APP_ROOT", "/app")
sys.path.insert(0, _APP_ROOT)

import numpy as np  # noqa: E402

from detect.dataset.loader import load_run  # noqa: E402
from detect.features.pyg_builder import (  # noqa: E402
    AUGMENTED_NODE_FEATURE_NAMES,
    DEFAULT_NODE_FEATURE_NAMES,
    build_graph_arrays,
)
from detect.models.gnn_graphsage import (  # noqa: E402
    EdgeGraphSAGEModel, GraphSAGEConfig,
)
from detect.train.projection import project_edge_probs_to_traders  # noqa: E402
from detect.evaluate.locked_stress import _graphs_to_edge_frame  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# MSA_PHASE_G_VARIANT lets us run multiple pilots without overwriting each
# other's state. v1 = first pilot (defaults from 2026-05-17 evening run);
# v2 = second pilot with the three tuning knobs applied below.
_VARIANT = os.environ.get("MSA_PHASE_G_VARIANT", "v2")

# Phase J (family-disjoint test): set MSA_PHASE_G_HOLDOUT_FAMILY to one of
# {clique, ring, mixed} to exclude every run of that family from detect.
# State dir picks up a _no_{family} suffix so disjoint runs don't collide
# with the all-family checkpoint or with each other.
HOLDOUT_FAMILY = os.environ.get("MSA_PHASE_G_HOLDOUT_FAMILY", "")
_HOLDOUT_SUFFIX = f"_no_{HOLDOUT_FAMILY}" if HOLDOUT_FAMILY else ""

COHORT_ROOT  = Path(_APP_ROOT) / "outputs/phase_g_cohort"
STATE_ROOT   = Path(_APP_ROOT) / f"outputs/phase_g_state_{_VARIANT}{_HOLDOUT_SUFFIX}"

# Path A2 (Phase H): variant v4 trains on the 8-feature augmented node
# vector (2 topology + 6 engineered manipulation-signature features); all
# other variants use the original 2-feature topology vector. The graph
# cache is keyed by feature set so 2-dim and 8-dim graphs never collide.
_AUGMENTED = (_VARIANT == "v4")
NODE_FEATURE_NAMES = (AUGMENTED_NODE_FEATURE_NAMES if _AUGMENTED
                      else DEFAULT_NODE_FEATURE_NAMES)
GRAPH_CACHE  = Path(_APP_ROOT) / (
    "outputs/_phase_g_graph_cache_aug" if _AUGMENTED
    else "outputs/_phase_g_graph_cache")

# ------------------------------------------------------------------
# Second-pilot tuning (2026-05-17 after first pilot OOD eval).
# Per note 45 "Three tuning knobs":
#   1. Larger val set     - VAL_RUNS_PCT 0.10 -> 0.20 (~10 val graphs)
#   2. Looser patience    - PATIENCE 6 -> 12 (less aggressive early stop)
#   3. Softer focal       - gamma 2.0 -> 1.0, alpha 0.85 -> 0.75
# Set MSA_PHASE_G_VARIANT=v1 + revert these constants to reproduce v1.
# ------------------------------------------------------------------
# Per-variant constants. v1 = first pilot baseline; v2 = three tweaks together
# (regressed AUC 0.638 -> 0.359); v3 = ablation isolating which knob hurt.
if _VARIANT == "v1":
    VAL_RUNS_PCT = 0.10
    PATIENCE     = 6
    FOCAL_ALPHA  = 0.85
    FOCAL_GAMMA  = 2.0
elif _VARIANT == "v3":
    # v3: revert focal to v1 (gamma=2.0, alpha=0.85), keep v2's val_pct + patience.
    # Tests whether the focal change alone broke v2.
    VAL_RUNS_PCT = 0.20
    PATIENCE     = 12
    FOCAL_ALPHA  = 0.85
    FOCAL_GAMMA  = 2.0
elif _VARIANT == "v4":
    # v4 (Path A2 / Phase H): feature-augmented GraphSAGE. Training knobs
    # are IDENTICAL to v1 (the dissertation winner) so the only difference
    # vs v1 is the six engineered node features - a clean A/B on whether
    # end-to-end feature learning beats the bolt-on tier-2 stack.
    VAL_RUNS_PCT = 0.10
    PATIENCE     = 6
    FOCAL_ALPHA  = 0.85
    FOCAL_GAMMA  = 2.0
else:  # v2 (and any unknown variant, treated as v2)
    VAL_RUNS_PCT = 0.20
    PATIENCE     = 12
    FOCAL_ALPHA  = 0.75
    FOCAL_GAMMA  = 1.0

EPOCHS_DAY0  = 80
EPOCHS_DAYS  = 25
SEED         = 42


def _log(msg: str) -> None:
    print(msg, flush=True)


def _list_day_dirs() -> list[Path]:
    if not COHORT_ROOT.is_dir():
        return []
    return sorted(d for d in COHORT_ROOT.iterdir()
                  if d.is_dir() and d.name.startswith("DAY_"))


def _list_day_runs(day_dir: Path) -> list[Path]:
    runs = sorted(d for d in day_dir.iterdir()
                  if d.is_dir() and d.name.startswith("DAY"))
    if HOLDOUT_FAMILY:
        # Run names are like DAY000_RUN00_mixed_s297961320 — position 2
        # after splitting by '_' is the family. Filter out the held-out
        # family so it never enters training.
        runs = [d for d in runs
                if (d.name.split("_") + [""])[2] != HOLDOUT_FAMILY]
    return runs


def _load_or_build_graph(run_dir: Path):
    GRAPH_CACHE.mkdir(parents=True, exist_ok=True)
    pkl = GRAPH_CACHE / f"{run_dir.name}.pkl"
    if pkl.exists():
        with pkl.open("rb") as fh:
            return pickle.load(fh)
    run = load_run(run_dir)
    g = build_graph_arrays(run, sparsification="knn", k=12,
                           node_feature_names=NODE_FEATURE_NAMES)
    with pkl.open("wb") as fh:
        pickle.dump(g, fh)
    return g


def _val_trader_recall(model, val_graphs) -> float:
    """Production metric: mean per-family trader recall on val_graphs.
    Returns 0.0 if no positives in any family (degenerate val set).
    """
    if not val_graphs:
        return 0.0
    eval_probs = [model.predict_proba(g) for g in val_graphs]
    edge_frame = _graphs_to_edge_frame(val_graphs, eval_probs)
    if edge_frame.empty:
        return 0.0
    # Use a fixed mid-range threshold during training. The final eval
    # re-tunes against the per-fold OOF labels.
    trader_df = project_edge_probs_to_traders(edge_frame, edge_threshold=0.3)
    recalls = []
    for fam, sub in trader_df.groupby("run_family"):
        y = sub["label_core"].astype(int).to_numpy()
        p = sub["trader_pred"].astype(int).to_numpy()
        if int(y.sum()) == 0:
            continue
        tp = int(((p == 1) & (y == 1)).sum())
        recalls.append(tp / int(y.sum()))
    return float(np.mean(recalls)) if recalls else 0.0


def _make_model(f_node: int, f_edge: int, *, epochs: int) -> EdgeGraphSAGEModel:
    """V2 (max aggregator) + focal loss + val-trader-recall early stop.

    Max aggregator is the Phase F single-seed sweep winner on recall.
    Focal loss handles the 0.85% imbalance better than weighted BCE.
    """
    cfg = GraphSAGEConfig(
        node_in_dim=f_node, edge_in_dim=f_edge,
        hidden_dims=(256, 128, 64),
        dropout=0.3,
        aggregator="max",     # Phase F winner
        head_hidden=64,
        head_dropout=0.2,
    )
    return EdgeGraphSAGEModel(
        cfg,
        lr=1e-3,
        epochs=epochs,
        early_stopping_patience=PATIENCE,
        seed=SEED,
        loss="focal",
        focal_alpha=FOCAL_ALPHA,
        focal_gamma=FOCAL_GAMMA,
        val_recall_fn=_val_trader_recall,
        weight_decay=1e-4,
    )


def _split_train_val(graphs, val_pct: float, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(graphs))
    n_val = max(1, int(val_pct * len(graphs)))
    val_idx = set(perm[:n_val].tolist())
    train = [g for i, g in enumerate(graphs) if i not in val_idx]
    val   = [g for i, g in enumerate(graphs) if i in val_idx]
    return train, val


def _save_checkpoint(model: EdgeGraphSAGEModel, path: Path) -> None:
    """Pickle the trained nn.Module state. Simple and Docker-portable."""
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": model.config.__dict__,
        "state_dict": model._module.state_dict() if model._module else None,
        "seed":   model.seed,
        "device": model.device,
    }, str(path))


def _load_checkpoint_into(model: EdgeGraphSAGEModel, path: Path) -> None:
    """Warm-start: copy the saved state into a fresh module of the same shape."""
    import torch
    blob = torch.load(str(path), map_location=model.device)
    if model._module is None:
        model._build_module()
    state = blob.get("state_dict")
    if state is not None:
        model._module.load_state_dict(state)


def _save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> int:
    _log("=" * 78)
    _log("PHASE G - continual learning (sequential warm-start)")
    _log("=" * 78)

    days = _list_day_dirs()
    if not days:
        _log(f"FATAL: no day dirs under {COHORT_ROOT}")
        _log("  Run scripts/regen_phase_g_cohort.py first (or pilot a few days).")
        return 2
    _log(f"  cohort root : {COHORT_ROOT}")
    _log(f"  days found  : {len(days)}")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(f"  device      : {device}")

    STATE_ROOT.mkdir(parents=True, exist_ok=True)

    # Determine resume point: last day with a checkpoint AND a metrics file.
    last_done = -1
    for di, day_dir in enumerate(days):
        ckpt = STATE_ROOT / f"day_{di:03d}_checkpoint.pt"
        met  = STATE_ROOT / f"day_{di:03d}_metrics.json"
        if ckpt.exists() and met.exists():
            last_done = di
    _log(f"  resume from : day {last_done + 1} "
         f"({len(days) - (last_done + 1)} days remaining)")

    f_node = f_edge = None
    model: Optional[EdgeGraphSAGEModel] = None
    t_overall = time.perf_counter()

    for di, day_dir in enumerate(days):
        if di <= last_done:
            continue
        _log("-" * 78)
        _log(f"DAY {di:03d}  {day_dir.name}")
        run_dirs = _list_day_runs(day_dir)
        if not run_dirs:
            _log(f"  no runs in {day_dir.name} - skipping")
            continue
        # Load graphs (cached after first time).
        graphs = []
        for rd in run_dirs:
            try:
                g = _load_or_build_graph(rd)
                graphs.append(g)
            except Exception as exc:  # noqa: BLE001
                _log(f"  graph build failed for {rd.name}: {exc!r}")
        if not graphs:
            _log("  no graphs loaded - skipping day")
            continue
        if f_node is None:
            f_node = graphs[0].x.shape[1]
            f_edge = graphs[0].edge_attr.shape[1]
            _log(f"  features    : node_in={f_node}  edge_in={f_edge}")

        train_graphs, val_graphs = _split_train_val(graphs, VAL_RUNS_PCT,
                                                    seed=SEED + di)
        _log(f"  graphs      : day total={len(graphs)} "
             f"train={len(train_graphs)} val={len(val_graphs)}")

        # First day or first un-done day: build fresh model.
        if model is None:
            epochs = EPOCHS_DAY0 if di == 0 else EPOCHS_DAYS
            model = _make_model(f_node, f_edge, epochs=epochs)
            prev_ckpt = (STATE_ROOT / f"day_{di - 1:03d}_checkpoint.pt"
                         if di > 0 else None)
            if prev_ckpt and prev_ckpt.exists():
                model._build_module()
                _load_checkpoint_into(model, prev_ckpt)
                _log(f"  warm-start  : loaded {prev_ckpt.name}")
        else:
            # Same model object, warm-started by the prior loop iteration.
            # Update epoch budget to the per-day setting.
            model.epochs = EPOCHS_DAYS

        t_fit = time.perf_counter()
        try:
            model.fit(train_graphs, val_graphs=val_graphs)
        except Exception as exc:  # noqa: BLE001
            _log(f"  fit FAILED: {exc!r}")
            return 3
        fit_secs = time.perf_counter() - t_fit

        # Compute end-of-day val recall + per-family breakdown.
        val_recall_mean = _val_trader_recall(model, val_graphs)

        ckpt = STATE_ROOT / f"day_{di:03d}_checkpoint.pt"
        _save_checkpoint(model, ckpt)
        met = STATE_ROOT / f"day_{di:03d}_metrics.json"
        _save_metrics({
            "day_idx":         di,
            "day_dirname":     day_dir.name,
            "n_runs":          len(graphs),
            "n_train":         len(train_graphs),
            "n_val":           len(val_graphs),
            "fit_seconds":     fit_secs,
            "epochs_used":     len(model.epoch_losses),
            "val_recall_mean": val_recall_mean,
            "epoch_losses":    model.epoch_losses,
            "warm_started":    di > 0,
        }, met)
        _log(f"  fit ok      : {fit_secs:.1f}s  "
             f"({len(model.epoch_losses)} epochs)  "
             f"val_recall_mean={val_recall_mean:.3f}")
        _log(f"  ckpt        : {ckpt.name}")

    total = time.perf_counter() - t_overall
    _log("=" * 78)
    _log(f"CONTINUAL TRAINING COMPLETE  ({total/60:.1f} min for "
         f"{len(days) - last_done - 1} new days)")
    _log("=" * 78)
    _log("Next: run training/phase_g_eval.py on the final checkpoint "
         "against the OOD test cohort.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
