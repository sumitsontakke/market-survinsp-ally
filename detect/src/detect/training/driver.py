"""M3+ boosted training run — uses GPU + bigger model + more epochs.

Differences vs run_m3.py:
  * Auto-detect CUDA (uses GPU when sm_120 nightly is present)
  * Bigger model: hidden = [256, 128, 64], 3-layer SAGE
  * Up to 200 epochs with patience 20
  * Pickles graph arrays to disk on first run; subsequent runs skip
    the ~9-minute graph build entirely
  * `python -u` recommended (flush=True on every print for live logs)
  * Writes metrics + loss curve to dedicated `_boosted_*` paths so the
    baseline M3 artifacts stay untouched

Reference
---------
Cont, R. (2001). Empirical properties of asset returns. Quantitative Finance, 1(2), 223-236.
Hamilton, W. L., Ying, R., Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. NeurIPS.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, "/app")

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from detect.dataset.loader import list_runs_in_cohort, load_run  # noqa: E402
from detect.dataset.splitter import iter_run_holdout_split  # noqa: E402
from detect.evaluate.locked_stress import evaluate_locked_stress  # noqa: E402
from detect.features.pyg_builder import (  # noqa: E402
    DEFAULT_EDGE_FEATURE_NAMES,
    DEFAULT_NODE_FEATURE_NAMES,
    build_graph_arrays,
)
from detect.models.gnn_graphsage import EdgeGraphSAGEModel, GraphSAGEConfig  # noqa: E402

COHORT_NAME = "PHASE1_R01_R24"
GRAPH_CACHE_DIR = Path("/app/outputs/_m3_graph_cache_v2")
OUT_METRICS = "/app/outputs/_m3_boosted_metrics.json"
OUT_LOSS = "/app/outputs/_m3_boosted_loss_curve.json"
# Per-trader prediction dump — read by webapp_v2's Demo Flow Step 7 to
# compute Rung-4 precision/F1/accuracy/specificity/purity. Without this,
# only aggregate recall is auditable.
OUT_PREDICTIONS = "/app/outputs/_m3_boosted_predictions.json"

# Boosted hyperparameters — leverage GPU
HIDDEN_DIMS = (256, 128, 64)     # 3-layer SAGE (was [128, 64])
DROPOUT = 0.3
EPOCHS = 200                     # was 50
PATIENCE = 20                    # was 8
LR = 1e-3
SEED = 42

# Balanced holdout: 2 from each family for proper coverage
# (vs M3 baseline which only held out ring + mixed)
HOLDOUT_RUNS = [
    "R03_clique_medium_medium_single_s17",   # clique
    "R07_clique_low_high_single_s31",        # clique
    "R09_ring_high_low_single_s41",          # ring
    "R11_ring_medium_medium_single_s47",     # ring
    "R17_mixed_high_low_single_s73",         # mixed
    "R19_mixed_medium_medium_single_s83",    # mixed
]


def _log(msg: str) -> None:
    print(msg, flush=True)


def _load_or_build_graph(run_path: Path, k: int = 12):
    """Cache graph arrays to disk so we skip ~7s build per run on subsequent calls."""
    GRAPH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pkl = GRAPH_CACHE_DIR / f"{run_path.name}.pkl"
    if pkl.exists():
        with pkl.open("rb") as fh:
            return pickle.load(fh)
    t = time.perf_counter()
    run = load_run(run_path)
    g = build_graph_arrays(run, sparsification="knn", k=k)
    with pkl.open("wb") as fh:
        pickle.dump(g, fh)
    _log(f"  built+cached {run_path.name}  fam={g.metadata['family']}  "
         f"nodes={g.num_nodes}  edges={g.num_edges}  pos={g.num_positives}  "
         f"({time.perf_counter()-t:.1f}s)")
    return g


def main() -> int:
    _log("=" * 72)
    _log("M3+ BOOSTED TRAINING RUN — Rung 4 GraphSAGE (GPU-capable)")
    _log("=" * 72)
    t0 = time.perf_counter()

    # Device detection
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            cap = torch.cuda.get_device_capability(0)
            _log(f"  torch       : {torch.__version__}")
            _log(f"  device      : cuda (capability sm_{cap[0]}{cap[1]})")
            _log(f"  gpu         : {torch.cuda.get_device_name(0)}")
        else:
            _log(f"  torch       : {torch.__version__}")
            _log(f"  device      : cpu (CUDA not available)")
    except ImportError:
        _log("  torch       : NOT INSTALLED")
        return 1

    np.random.seed(SEED)
    try:
        import torch
        torch.manual_seed(SEED)
        if device == "cuda":
            torch.cuda.manual_seed_all(SEED)
    except Exception:
        pass

    _log(f"  cohort      : {COHORT_NAME}")
    _log(f"  hidden_dims : {HIDDEN_DIMS}  ({len(HIDDEN_DIMS)}-layer)")
    _log(f"  epochs      : {EPOCHS} (patience {PATIENCE})")
    _log(f"  lr          : {LR}")
    _log(f"  seed        : {SEED}")
    _log(f"  holdout     : {HOLDOUT_RUNS}")
    _log("")

    # Resolve cohort + split
    cohort_paths = list_runs_in_cohort(COHORT_NAME)
    all_run_ids = [p.name for p in cohort_paths]
    train_ids, eval_ids = iter_run_holdout_split(all_run_ids, HOLDOUT_RUNS)
    _log(f"  train_runs  : {len(train_ids)}")
    _log(f"  eval_runs   : {len(eval_ids)}")
    _log("")

    # Build / load graphs
    _log("building graph arrays (knn, k=12) — using disk cache when available...")
    train_graphs = []
    eval_graphs = []
    for run_id in train_ids:
        for root in ("/app/outputs/calibrated_runs", "/app/outputs/runs"):
            cand = Path(root) / run_id
            if cand.exists():
                break
        else:
            continue
        train_graphs.append(_load_or_build_graph(cand))
    for run_id in eval_ids:
        for root in ("/app/outputs/calibrated_runs", "/app/outputs/runs"):
            cand = Path(root) / run_id
            if cand.exists():
                break
        else:
            continue
        eval_graphs.append(_load_or_build_graph(cand))

    _log("")
    _log(f"  graphs ready  : train={len(train_graphs)}  eval={len(eval_graphs)}")
    total_edges = sum(g.num_edges for g in train_graphs)
    total_pos = sum(g.num_positives for g in train_graphs)
    _log(f"  total edges (train) : {total_edges:,}   positives : {total_pos:,}  "
         f"({100*total_pos/max(total_edges,1):.2f}%)")
    _log("")

    # Inner train/val split for early stopping
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(train_graphs))
    val_cut = max(1, int(0.2 * len(train_graphs)))
    val_idx = set(perm[:val_cut].tolist())
    inner_train = [g for i, g in enumerate(train_graphs) if i not in val_idx]
    inner_val   = [g for i, g in enumerate(train_graphs) if i in val_idx]
    _log(f"  inner train : {len(inner_train)}   inner val : {len(inner_val)}")
    _log("")

    f_node = train_graphs[0].x.shape[1]
    f_edge = train_graphs[0].edge_attr.shape[1]
    model_cfg = GraphSAGEConfig(
        node_in_dim=f_node, edge_in_dim=f_edge,
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT, aggregator="mean",
    )
    model = EdgeGraphSAGEModel(
        model_cfg, lr=LR, epochs=EPOCHS,
        early_stopping_patience=PATIENCE,
        seed=SEED, device=device,
    )
    _log(f"=== FIT ({len(inner_train)} train, {len(inner_val)} val) on {device} ===")
    t_fit = time.perf_counter()
    model.fit(inner_train, val_graphs=inner_val)
    _log(f"  fit elapsed: {time.perf_counter()-t_fit:.1f}s")
    _log("")

    # CV scoring
    cv_probs = np.concatenate(
        [model.predict_proba(g) for g in inner_val] or [np.zeros(0, dtype=np.float32)]
    )
    cv_labels = np.concatenate(
        [g.y.astype(int) for g in inner_val] or [np.zeros(0, dtype=int)]
    )
    cv_auc = -1.0
    if cv_labels.size > 1 and len(np.unique(cv_labels)) > 1:
        cv_auc = float(roc_auc_score(cv_labels, cv_probs))
    _log(f"  cv_auc      : {cv_auc:.4f}")

    # Holdout
    eval_probs = [model.predict_proba(g) for g in eval_graphs]

    metrics = evaluate_locked_stress(
        train_graphs=train_graphs,
        eval_graphs=eval_graphs,
        cv_probs=cv_probs,
        cv_labels=cv_labels,
        eval_probs_per_graph=eval_probs,
        cv_auc=cv_auc,
        n_train_runs=len(train_graphs),
        n_eval_runs=len(eval_graphs),
        seed=SEED,
        predictions_out_path=OUT_PREDICTIONS,
    )

    os.makedirs(os.path.dirname(OUT_METRICS), exist_ok=True)
    metrics.write_json(OUT_METRICS)
    with open(OUT_LOSS, "w", encoding="utf-8") as fh:
        json.dump(model.epoch_losses, fh, indent=2)

    _log("")
    _log("=" * 72)
    _log("FINAL METRIC BUNDLE")
    _log("=" * 72)
    _log(metrics.summary_line())
    _log("")
    for k_, v in metrics.locked_per_run.items():
        _log(f"  per-run {k_:<45s} recall={v:.3f}")
    _log("")
    _log(f"  metrics      : {OUT_METRICS}")
    _log(f"  losses       : {OUT_LOSS}")
    _log(f"  predictions  : {OUT_PREDICTIONS}")
    _log(f"  total time : {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
