"""Full GraphSAGE training run for Rung 4.

Reads an ExperimentConfig, builds graph arrays, runs CV + final
training, evaluates on the holdout, returns a MetricBundle.

Invoked from ``training/train/trainer.py`` when ``model.family ==
"graphsage"``. Kept separate from the dispatcher for readability.

Reference
---------
Cont, R. (2001). Empirical properties of asset returns. Quantitative Finance, 1(2), 223-236.
Hamilton, W. L., Ying, R., Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. NeurIPS.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from detect.dataset.loader import list_runs_in_cohort, load_run
from detect.dataset.splitter import iter_run_holdout_split
from detect.evaluate.locked_stress import evaluate_locked_stress
from detect.evaluate.metrics import MetricBundle
from detect.features.pyg_builder import (
    DEFAULT_EDGE_FEATURE_NAMES,
    DEFAULT_NODE_FEATURE_NAMES,
    build_graph_arrays,
)
from detect.models.gnn_graphsage import EdgeGraphSAGEModel, GraphSAGEConfig
from detect.registry.experiment import ExperimentConfig

_log = logging.getLogger(__name__)


def _build_graphs(run_ids: list[str], sparsification: str, k: int) -> list:
    """Load and build graph arrays for a list of run_ids."""
    out: list = []
    for run_id in run_ids:
        # Resolve the run_id to a Path under one of the known cohorts.
        # We probe both phase-1 and calibrated locations.
        for root in ("outputs/calibrated_runs", "outputs/runs"):
            cand = Path(root) / run_id
            if cand.exists():
                break
        else:
            _log.warning("could not resolve run_id=%s", run_id)
            continue
        run = load_run(cand)
        g = build_graph_arrays(
            run,
            sparsification=sparsification,
            k=k,
        )
        if g.num_edges == 0:
            _log.warning("graph empty after sparsification: %s", run_id)
            continue
        out.append(g)
    return out


def train_graphsage(config: ExperimentConfig) -> MetricBundle:
    """Train GraphSAGE per the config; return a MetricBundle."""
    seed = int(config.seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

    cohort_paths = list_runs_in_cohort(config.runs_cohort)
    all_run_ids = [p.name for p in cohort_paths]
    if not all_run_ids:
        raise RuntimeError(f"cohort {config.runs_cohort!r} is empty")

    holdout = list(config.split.get("holdout_runs", []))
    train_ids, eval_ids = iter_run_holdout_split(all_run_ids, holdout)
    _log.info("train_ids=%s eval_ids=%s", train_ids, eval_ids)

    sparsification_cfg = config.features.get("adaptive_sparsification") or {}
    method = sparsification_cfg.get("method", "knn")
    k = int(sparsification_cfg.get("k", 12))

    train_graphs = _build_graphs(train_ids, sparsification=method, k=k)
    eval_graphs = _build_graphs(eval_ids, sparsification=method, k=k)
    if not train_graphs:
        raise RuntimeError("no training graphs after sparsification")
    if not eval_graphs:
        raise RuntimeError("no eval graphs after sparsification")

    edge_feature_names = tuple(
        config.features.get("edges") or DEFAULT_EDGE_FEATURE_NAMES
    )
    node_feature_names = tuple(
        config.features.get("nodes") or DEFAULT_NODE_FEATURE_NAMES
    )

    # All graphs have the same edge/node dims by construction.
    f_node = train_graphs[0].x.shape[1]
    f_edge = train_graphs[0].edge_attr.shape[1]
    model_cfg = GraphSAGEConfig(
        node_in_dim=f_node,
        edge_in_dim=f_edge,
        hidden_dims=tuple(config.model.get("layers", [128, 64])),
        dropout=float(config.model.get("dropout", 0.3)),
        aggregator=str(config.model.get("pooling", "mean")),
    )

    # Inner train/val split for early stopping.
    # Simple holdout: last 20% of train_graphs as val.
    n_train = len(train_graphs)
    val_cut = max(1, int(0.2 * n_train))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_train)
    val_idx = set(perm[:val_cut].tolist())
    inner_train = [g for i, g in enumerate(train_graphs) if i not in val_idx]
    inner_val   = [g for i, g in enumerate(train_graphs) if i in val_idx]

    training_cfg = config.training or {}
    model = EdgeGraphSAGEModel(
        model_cfg,
        lr=float(training_cfg.get("lr", 1e-3)),
        epochs=int(training_cfg.get("epochs", 50)),
        early_stopping_patience=int(training_cfg.get("early_stopping_patience", 8)),
        seed=seed,
    )
    model.fit(inner_train, val_graphs=inner_val)

    # Out-of-fold proxy: predict on the inner-val set; pool with the
    # full train predictions for threshold tuning. For a richer CV
    # signal, M3 could swap in StratifiedGroupKFold over graphs.
    cv_probs_list: list[np.ndarray] = []
    cv_labels_list: list[np.ndarray] = []
    for g in inner_val:
        cv_probs_list.append(model.predict_proba(g))
        cv_labels_list.append(g.y.astype(int))
    cv_probs = np.concatenate(cv_probs_list) if cv_probs_list else np.zeros(0, dtype=np.float32)
    cv_labels = np.concatenate(cv_labels_list) if cv_labels_list else np.zeros(0, dtype=int)
    cv_auc = -1.0
    try:
        if cv_labels.size > 1 and len(np.unique(cv_labels)) > 1:
            cv_auc = float(roc_auc_score(cv_labels, cv_probs))
    except Exception:  # noqa: BLE001
        cv_auc = -1.0

    eval_probs_per_graph = [model.predict_proba(g) for g in eval_graphs]

    n_pos = int(sum(int(g.y.sum()) for g in train_graphs))
    n_edges = int(sum(g.num_edges for g in train_graphs))

    metrics = evaluate_locked_stress(
        train_graphs=train_graphs,
        eval_graphs=eval_graphs,
        cv_probs=cv_probs,
        cv_labels=cv_labels,
        eval_probs_per_graph=eval_probs_per_graph,
        cv_auc=cv_auc,
        n_train_runs=len(train_graphs),
        n_eval_runs=len(eval_graphs),
        seed=seed,
    )

    # Stash the training loss curve as a side-artifact via the registry.
    metrics_payload = metrics.to_dict()
    metrics_payload["_epoch_losses"] = model.epoch_losses
    return metrics, model
