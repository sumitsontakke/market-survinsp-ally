"""Standalone M3 runner — bypasses the locked trainer.py.

Builds graphs for R01-R24, trains Rung 4 GraphSAGE end-to-end on the
calibrated configuration, writes MetricBundle JSON + loss curve JSON.

Invocation (inside the trainer container):
    python /app/training/run_m3.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# /app is the working dir inside the container. Ensure imports resolve.
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
from detect.registry.experiment import ExperimentConfig  # noqa: E402


CONFIG_PATH = "/app/training/configs/rung4_sage_v1_calibrated.yaml"
COHORT_NAME = "PHASE1_R01_R24"   # legacy R01-R24 (calibrated cohort would be R3_CALIBRATED_24)
OUT_METRICS = "/app/outputs/_m3_full_metrics.json"
OUT_LOSS = "/app/outputs/_m3_full_loss_curve.json"
# Per-trader prediction dump for full Rung-4 surveillance metrics.
OUT_PREDICTIONS = "/app/outputs/_m3_full_predictions.json"


def main() -> int:
    print("=" * 72)
    print("M3 FULL TRAINING RUN — Rung 4 GraphSAGE")
    print("=" * 72)
    t0 = time.perf_counter()

    config = ExperimentConfig.from_yaml(CONFIG_PATH)
    print(f"  config            : {config.experiment_id}")
    print(f"  seed              : {config.seed}")
    print(f"  cohort            : {COHORT_NAME}")
    print(f"  device            : auto (CUDA if available)")
    print()

    np.random.seed(config.seed)
    try:
        import torch
        torch.manual_seed(config.seed)
    except ImportError:
        pass

    # ---- Resolve cohort + split ------------------------------------
    cohort_paths = list_runs_in_cohort(COHORT_NAME)
    all_run_ids = [p.name for p in cohort_paths]
    holdout = list(config.split.get("holdout_runs", []))
    train_ids, eval_ids = iter_run_holdout_split(all_run_ids, holdout)
    print(f"  train_runs        : {len(train_ids)}")
    print(f"  eval_runs         : {len(eval_ids)}  -> {eval_ids}")
    print()

    sparsification_cfg = config.features.get("adaptive_sparsification") or {}
    method = sparsification_cfg.get("method", "knn")
    k = int(sparsification_cfg.get("k", 12))

    # ---- Build graphs ----------------------------------------------
    print(f"building graph arrays (sparsification={method}, k={k})...")
    train_graphs: list = []
    eval_graphs: list = []
    for run_id, target_list, label in [
        (train_ids, train_graphs, "train"),
        (eval_ids, eval_graphs, "eval"),
    ]:
        for rid in run_id:
            for root in ("/app/outputs/calibrated_runs", "/app/outputs/runs"):
                cand = Path(root) / rid
                if cand.exists():
                    break
            else:
                print(f"  warn: {rid} not found, skipping")
                continue
            t_b = time.perf_counter()
            run = load_run(cand)
            g = build_graph_arrays(run, sparsification=method, k=k)
            if g.num_edges == 0:
                print(f"  warn: {rid} empty after sparsify; skipping")
                continue
            target_list.append(g)
            print(f"  [{label}] {rid}  fam={g.metadata['family']}  "
                  f"nodes={g.num_nodes}  edges={g.num_edges}  "
                  f"pos={g.num_positives}  ({time.perf_counter()-t_b:.1f}s)")
    print()
    print(f"  graphs ready: train={len(train_graphs)}  eval={len(eval_graphs)}")

    if not train_graphs or not eval_graphs:
        print("ERROR: empty train or eval set; aborting")
        return 1

    # ---- Inner train/val split for early stopping ------------------
    rng = np.random.default_rng(config.seed)
    perm = rng.permutation(len(train_graphs))
    val_cut = max(1, int(0.2 * len(train_graphs)))
    val_idx = set(perm[:val_cut].tolist())
    inner_train = [g for i, g in enumerate(train_graphs) if i not in val_idx]
    inner_val   = [g for i, g in enumerate(train_graphs) if i in val_idx]
    print(f"  inner train : {len(inner_train)}   val : {len(inner_val)}")
    print()

    # ---- Build + fit model -----------------------------------------
    f_node = train_graphs[0].x.shape[1]
    f_edge = train_graphs[0].edge_attr.shape[1]
    model_cfg = GraphSAGEConfig(
        node_in_dim=f_node,
        edge_in_dim=f_edge,
        hidden_dims=tuple(config.model.get("layers", [128, 64])),
        dropout=float(config.model.get("dropout", 0.3)),
        aggregator=str(config.model.get("pooling", "mean")),
    )
    training_cfg = config.training or {}
    model = EdgeGraphSAGEModel(
        model_cfg,
        lr=float(training_cfg.get("lr", 1e-3)),
        epochs=int(training_cfg.get("epochs", 50)),
        early_stopping_patience=int(training_cfg.get("early_stopping_patience", 8)),
        seed=config.seed,
    )

    print(f"=== FIT ({len(inner_train)} train, {len(inner_val)} val, "
          f"epochs={model.epochs}, lr={model.lr}) ===")
    t_fit = time.perf_counter()
    model.fit(inner_train, val_graphs=inner_val)
    print(f"  fit elapsed: {time.perf_counter()-t_fit:.1f}s")
    print()

    # ---- OOF for CV metrics ---------------------------------------
    cv_probs = np.concatenate(
        [model.predict_proba(g) for g in inner_val] or [np.zeros(0, dtype=np.float32)]
    )
    cv_labels = np.concatenate(
        [g.y.astype(int) for g in inner_val] or [np.zeros(0, dtype=int)]
    )
    cv_auc = -1.0
    if cv_labels.size > 1 and len(np.unique(cv_labels)) > 1:
        cv_auc = float(roc_auc_score(cv_labels, cv_probs))

    # ---- Holdout predictions --------------------------------------
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
        seed=config.seed,
        predictions_out_path=OUT_PREDICTIONS,
    )

    # ---- Persist ---------------------------------------------------
    os.makedirs(os.path.dirname(OUT_METRICS), exist_ok=True)
    metrics.write_json(OUT_METRICS)
    with open(OUT_LOSS, "w", encoding="utf-8") as fh:
        json.dump(model.epoch_losses, fh, indent=2)

    print("=" * 72)
    print("FINAL METRIC BUNDLE")
    print("=" * 72)
    print(metrics.summary_line())
    print()
    for k_, v in metrics.locked_per_run.items():
        print(f"  per-run {k_:<45s} recall={v:.3f}")
    print()
    print(f"  metrics     : {OUT_METRICS}")
    print(f"  losses      : {OUT_LOSS}")
    print(f"  predictions : {OUT_PREDICTIONS}")
    print(f"  total elapsed: {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
