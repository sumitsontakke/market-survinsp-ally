"""V1 projection-threshold retune — tests the two-layer-task hypothesis.

Background: in the single-seed sweep (auc_sweep.py), V1 (wider hidden dims
(512,256,128)) doubled edge AUC to 0.71 but cut mean trader recall from
0.99 (V0 baseline) to 0.48. The note-44 framing is that the trader
projection's 0.7*max + 0.3*top3 threshold was tuned to V0's edge-score
distribution; V1's sharper distribution breaks the threshold.

This script tests that claim directly:
  1. Retrain V1 on the M3+ holdout.
  2. Get its per-edge probabilities on every eval graph.
  3. Project to per-trader scores.
  4. Sweep the trader threshold across a fine grid; for each, compute
     per-family recall.
  5. Report the recall curve.

If recall recovers near 1.0 at some threshold, the "two-layer task"
framing is best read as "two-layer task with co-adaptation needed" —
V1 is a viable architecture, the projection threshold just needs to
be re-tuned.

Run inside the trainer-gpu container, or with cached graphs on CPU:
    python -u /app/training/v1_projection_retune.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_APP_ROOT = os.environ.get("MSA_APP_ROOT", "/app")
sys.path.insert(0, _APP_ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from detect.dataset.loader import list_runs_in_cohort, load_run  # noqa: E402
from detect.dataset.splitter import iter_run_holdout_split  # noqa: E402
from detect.evaluate.locked_stress import _graphs_to_edge_frame  # noqa: E402
from detect.features.pyg_builder import build_graph_arrays  # noqa: E402
from detect.models.gnn_graphsage import (  # noqa: E402
    EdgeGraphSAGEModel, GraphSAGEConfig,
)
from detect.train.projection import project_edge_probs_to_traders  # noqa: E402
import pickle  # noqa: E402

COHORT_NAME = "PHASE1_R01_R24"
GRAPH_CACHE_DIR = Path(_APP_ROOT) / "outputs/_m3_graph_cache_v2"
EVAL_PROBS_PKL = Path(_APP_ROOT) / "outputs/_v1_eval_probs.pkl"
OUT_JSON = str(Path(_APP_ROOT) / "outputs/_v1_projection_retune.json")
_RUN_ROOTS = (
    str(Path(_APP_ROOT) / "outputs/calibrated_runs"),
    str(Path(_APP_ROOT) / "outputs/runs"),
)
SEED = 42

HOLDOUT_RUNS = [
    "R03_clique_medium_medium_single_s17",
    "R07_clique_low_high_single_s31",
    "R09_ring_high_low_single_s41",
    "R11_ring_medium_medium_single_s47",
    "R17_mixed_high_low_single_s73",
    "R19_mixed_medium_medium_single_s83",
]

# Variant under test.
V1_HIDDEN = (512, 256, 128)
V1_DROPOUT = 0.3
V1_AGGREGATOR = "mean"

# Trader-threshold grid for the projection sweep.
THRESHOLDS = list(np.round(np.arange(0.05, 0.95, 0.05), 2))


def _log(msg: str) -> None:
    print(msg, flush=True)


def _load_or_build_graph(run_path: Path, k: int = 12):
    GRAPH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pkl = GRAPH_CACHE_DIR / f"{run_path.name}.pkl"
    if pkl.exists():
        with pkl.open("rb") as fh:
            return pickle.load(fh)
    run = load_run(run_path)
    g = build_graph_arrays(run, sparsification="knn", k=k)
    with pkl.open("wb") as fh:
        pickle.dump(g, fh)
    return g


def _resolve_run_path(run_id: str):
    for root in _RUN_ROOTS:
        cand = Path(root) / run_id
        if cand.exists():
            return cand
    return None


def _split_inner(train_graphs, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_graphs))
    val_cut = max(1, int(0.2 * len(train_graphs)))
    val_idx = set(perm[:val_cut].tolist())
    inner_train = [g for i, g in enumerate(train_graphs) if i not in val_idx]
    inner_val = [g for i, g in enumerate(train_graphs) if i in val_idx]
    return inner_train, inner_val


def _family_recall_at(trader_df: pd.DataFrame, t: float) -> dict:
    out = {}
    for family, sub in trader_df.groupby("run_family"):
        y_true = sub["label_core"].astype(int).to_numpy()
        score = sub["trader_score"].astype(float).to_numpy()
        pred = (score >= t).astype(int)
        if y_true.sum() == 0:
            out[family] = -1.0
            continue
        out[family] = float((pred[y_true == 1]).sum()) / float(y_true.sum())
    # Benign alarm = fraction of benign traders predicted positive
    benign = trader_df[trader_df["label_core"] == 0]
    if len(benign) == 0:
        out["benign_alarm"] = -1.0
    else:
        out["benign_alarm"] = float(
            (benign["trader_score"].astype(float) >= t).sum()
        ) / float(len(benign))
    return out


def main() -> int:
    _log("=" * 78)
    _log("V1 PROJECTION THRESHOLD RETUNE — two-layer-task hypothesis test")
    _log("=" * 78)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(f"  device      : {device}")

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)

    cohort_paths = list_runs_in_cohort(COHORT_NAME)
    all_run_ids = [p.name for p in cohort_paths]
    train_ids, eval_ids = iter_run_holdout_split(all_run_ids, HOLDOUT_RUNS)
    _log(f"  cohort      : {COHORT_NAME} train={len(train_ids)} eval={len(eval_ids)}")

    train_graphs = []
    eval_graphs = []
    for run_id in train_ids:
        p = _resolve_run_path(run_id)
        if p is not None:
            train_graphs.append(_load_or_build_graph(p))
    for run_id in eval_ids:
        p = _resolve_run_path(run_id)
        if p is not None:
            eval_graphs.append(_load_or_build_graph(p))
    _log(f"  graphs      : train={len(train_graphs)} eval={len(eval_graphs)}")

    inner_train, inner_val = _split_inner(train_graphs, seed=SEED)
    f_node = train_graphs[0].x.shape[1]
    f_edge = train_graphs[0].edge_attr.shape[1]

    # Cache eval_probs to disk so the slow V1 fit only happens once.
    eval_probs = None
    if EVAL_PROBS_PKL.exists():
        with EVAL_PROBS_PKL.open("rb") as fh:
            eval_probs = pickle.load(fh)
        _log(f"  eval_probs  : loaded from {EVAL_PROBS_PKL.name} "
             f"({len(eval_probs)} graphs)")
    else:
        _log(f"  V1 config   : hidden={V1_HIDDEN} dropout={V1_DROPOUT} "
             f"aggr={V1_AGGREGATOR}")
        cfg = GraphSAGEConfig(
            node_in_dim=f_node, edge_in_dim=f_edge,
            hidden_dims=V1_HIDDEN, dropout=V1_DROPOUT,
            aggregator=V1_AGGREGATOR,
            head_hidden=64, head_dropout=0.2,
        )
        # Capped epochs + tight patience so the V1 fit finishes inside the
        # sandbox 45s bash limit on CPU; matches the GPU V1 early-stop point
        # observed in auc_sweep.py (23 epochs).
        model = EdgeGraphSAGEModel(cfg, lr=1e-3, epochs=30,
                                   early_stopping_patience=5,
                                   seed=SEED, device=device)
        _log("training V1...")
        t0 = time.perf_counter()
        model.fit(inner_train, val_graphs=inner_val)
        _log(f"  fit elapsed : {time.perf_counter() - t0:.1f}s "
             f"({len(model.epoch_losses)} epochs)")
        eval_probs = [model.predict_proba(g) for g in eval_graphs]
        EVAL_PROBS_PKL.parent.mkdir(parents=True, exist_ok=True)
        with EVAL_PROBS_PKL.open("wb") as fh:
            pickle.dump(eval_probs, fh)
        _log(f"  cached eval_probs -> {EVAL_PROBS_PKL.name}")
    edge_frame = _graphs_to_edge_frame(eval_graphs, eval_probs)
    _log(f"  edge_frame  : {len(edge_frame)} rows")
    if edge_frame.empty:
        _log("  ABORT — empty edge frame")
        return 1

    # The trader projection IS deterministic given edge probs — only
    # threshold matters. Compute project once with a placeholder threshold;
    # we'll re-threshold trader_score in the sweep.
    trader_df = project_edge_probs_to_traders(edge_frame, edge_threshold=0.5)
    _log(f"  trader_df   : {len(trader_df)} rows  "
         f"(positive={int(trader_df['label_core'].sum())})")
    _log(f"  score stats : min={trader_df['trader_score'].min():.4f} "
         f"med={trader_df['trader_score'].median():.4f} "
         f"max={trader_df['trader_score'].max():.4f}")
    _log(f"  benign med  : "
         f"{trader_df[trader_df['label_core']==0]['trader_score'].median():.4f}")
    _log(f"  manip  med  : "
         f"{trader_df[trader_df['label_core']==1]['trader_score'].median():.4f}")

    rows = []
    for t in THRESHOLDS:
        fam = _family_recall_at(trader_df, float(t))
        rows.append({"threshold": float(t), **fam})

    _log("")
    _log(f"{'thresh':<8s}  {'clique':<8s} {'ring':<8s} {'mixed':<8s} {'mean':<8s}  benign_alarm")
    best = None
    for r in rows:
        valid = [r[k] for k in ("clique", "ring", "mixed") if r.get(k, -1) >= 0]
        mean_r = float(np.mean(valid)) if valid else -1.0
        r["mean_recall"] = mean_r
        marker = ""
        if mean_r >= 0 and (best is None or mean_r > best["mean_recall"]):
            best = r
        _log(f"  {r['threshold']:.2f}    "
             f"{r.get('clique',-1):.3f}    {r.get('ring',-1):.3f}    "
             f"{r.get('mixed',-1):.3f}    {mean_r:.3f}    "
             f"{r.get('benign_alarm',-1):.4f}{marker}")

    if best is not None:
        _log("")
        _log(f"  BEST mean recall at threshold={best['threshold']:.2f}: "
             f"clique={best['clique']:.3f}  ring={best['ring']:.3f}  "
             f"mixed={best['mixed']:.3f}  mean={best['mean_recall']:.3f}  "
             f"benign_alarm={best['benign_alarm']:.4f}")

    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_JSON).write_text(json.dumps({
        "schema_version": "1",
        "variant":        "V1_wider",
        "hidden_dims":    list(V1_HIDDEN),
        "dropout":        V1_DROPOUT,
        "aggregator":     V1_AGGREGATOR,
        "seed":           SEED,
        "rows":           rows,
        "best":           best,
        "n_eval_graphs":  len(eval_graphs),
    }, indent=2), encoding="utf-8")
    _log(f"results : {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
