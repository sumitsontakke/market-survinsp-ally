"""Locked-stress evaluator - shared by Rung 3 and Rung 4.

Takes a trained model + a list of GraphArrays (the holdout runs) and
populates a MetricBundle. Both rungs use this exact code path so
clique / ring / mixed recall numbers are computed identically.

Reference
---------
Cont, R. (2001). Empirical properties of asset returns: stylized facts
and statistical implications. Quantitative Finance, 1(2), 223-236.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from detect.evaluate.metrics import MetricBundle, NOT_COMPUTED
from detect.train.projection import project_edge_probs_to_traders


# ---------------------------------------------------------------------------
# Helper: convert a list of GraphArrays + predicted probs into per-trader DF
# ---------------------------------------------------------------------------

def _graphs_to_edge_frame(
    graphs: Sequence,
    probs_per_graph: Sequence[np.ndarray],
) -> pd.DataFrame:
    """Stack per-graph edge predictions into one DataFrame for projection."""
    rows: list[dict] = []
    for g, probs in zip(graphs, probs_per_graph):
        if g.num_edges == 0 or probs.size == 0:
            continue
        node_ids = g.node_ids
        src = g.edge_index[0]
        dst = g.edge_index[1]
        run_id = g.metadata.get("run_id", "")
        family = g.metadata.get("family", "")
        for i in range(g.num_edges):
            rows.append({
                "run_id": run_id,
                "run_name": run_id,
                "run_family": family,
                "sell_trader_id": node_ids[int(src[i])],
                "buy_trader_id": node_ids[int(dst[i])],
                # ground-truth (for label-based bookkeeping; not used by projection itself)
                "label_any_edge": int(g.y[i]),
                # core-edge label proxy: same as any-edge in M3 (the calibrated
                # cohort uses label_any_edge as the supervision target).
                "label_core_edge": int(g.y[i]),
                "seller_core_label": int(g.y[i]),
                "buyer_core_label": int(g.y[i]),
                "edge_prob": float(probs[i]),
            })
    if not rows:
        return pd.DataFrame(columns=[
            "run_id", "run_name", "run_family", "sell_trader_id", "buy_trader_id",
            "label_any_edge", "label_core_edge", "seller_core_label",
            "buyer_core_label", "edge_prob",
        ])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def _select_edge_threshold(
    y_true: np.ndarray, probs: np.ndarray,
    candidates: Iterable[float],
) -> tuple[float, float]:
    """Pick the threshold that maximizes edge-level F1 on the OOF predictions."""
    best_t = 0.5
    best_f1 = -1.0
    for t in candidates:
        pred = (probs >= t).astype(int)
        f1 = float(f1_score(y_true, pred, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def evaluate_locked_stress(
    train_graphs: Sequence,
    eval_graphs: Sequence,
    *,
    predictions_out_path: "str | Path | None" = None,
    cv_probs: np.ndarray,
    cv_labels: np.ndarray,
    eval_probs_per_graph: Sequence[np.ndarray],
    cv_auc: float,
    n_train_runs: int,
    n_eval_runs: int,
    seed: int = 42,
    thresholds: Iterable[float] = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60),
) -> MetricBundle:
    """Compute a full MetricBundle from cross-validation + heldout predictions.

    ``cv_probs`` and ``cv_labels`` are the pooled out-of-fold predictions
    from detect (used to tune the threshold and report CV F1).

    ``eval_probs_per_graph`` is one numpy array per graph in ``eval_graphs``.
    """
    # ---- In-distribution metrics ----
    edge_threshold, cv_f1 = _select_edge_threshold(
        cv_labels, cv_probs, thresholds,
    )
    cv_pred = (cv_probs >= edge_threshold).astype(int)
    cv_precision = float(precision_score(cv_labels, cv_pred, zero_division=0))
    cv_recall = float(recall_score(cv_labels, cv_pred, zero_division=0))

    # ---- Locked-stress per-family recall ----
    edge_frame = _graphs_to_edge_frame(eval_graphs, eval_probs_per_graph)
    if edge_frame.empty:
        return MetricBundle(
            cv_f1=cv_f1, cv_auc=cv_auc,
            cv_precision=cv_precision, cv_recall=cv_recall,
            cv_threshold=edge_threshold,
            locked_clique_recall=NOT_COMPUTED,
            locked_ring_recall=NOT_COMPUTED,
            locked_mixed_recall=NOT_COMPUTED,
            locked_benign_alarm=NOT_COMPUTED,
            locked_per_run={},
            n_train_runs=n_train_runs,
            n_eval_runs=n_eval_runs,
            n_edges_train=int(cv_labels.size),
            n_pos_edges_train=int(cv_labels.sum()),
            seed=seed,
        )

    trader_df = project_edge_probs_to_traders(
        edge_frame, edge_threshold=edge_threshold,
    )

    # Per-family trader recall on the heldout runs (matches Phase 1 protocol).
    # Also per-family purity = TP/(TP+FP) — added Phase F for the
    # surveillance auditor's "of the traders you flagged, how many were
    # actually manipulating?" question. Recall alone is misleading on
    # imbalanced cohorts.
    family_recall: dict[str, float] = {}
    family_purity: dict[str, float] = {}
    for family, sub in trader_df.groupby("run_family"):
        y_true = sub["label_core"].astype(int).to_numpy()
        y_pred = sub["trader_pred"].astype(int).to_numpy()
        if int(y_true.sum()) == 0:
            family_recall[family] = NOT_COMPUTED
        else:
            family_recall[family] = float(recall_score(y_true, y_pred, zero_division=0))
        # Purity is well-defined whenever the model flagged ANY trader in
        # this family — uses TP / (TP + FP) on the per-family subset.
        flagged = int(y_pred.sum())
        if flagged == 0:
            family_purity[family] = NOT_COMPUTED
        else:
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            family_purity[family] = float(tp) / float(flagged)

    # Benign alarm rate (any-family false-positive rate on benign runs)
    benign_alarm = NOT_COMPUTED
    benign_sub = trader_df[trader_df["run_family"] == "benign"]
    if not benign_sub.empty:
        bn_true = benign_sub["label_core"].astype(int).to_numpy()
        bn_pred = benign_sub["trader_pred"].astype(int).to_numpy()
        negatives = int((bn_true == 0).sum())
        if negatives > 0:
            benign_alarm = float(((bn_pred == 1) & (bn_true == 0)).sum() / negatives)

    # Per-run trader recall for the locked_per_run breakdown.
    per_run: dict[str, float] = {}
    for run_id, sub in trader_df.groupby("run_id"):
        y_true = sub["label_core"].astype(int).to_numpy()
        y_pred = sub["trader_pred"].astype(int).to_numpy()
        if int(y_true.sum()) == 0:
            per_run[str(run_id)] = NOT_COMPUTED
            continue
        per_run[str(run_id)] = float(recall_score(y_true, y_pred, zero_division=0))

    # Optionally dump the per-trader predictions so downstream tooling can
    # compute precision, accuracy, specificity, purity, coverage — the
    # surveillance numbers an auditor needs that aggregate recall alone
    # cannot answer. We write a small JSON keyed by run_id, each value
    # being a list of {trader_id, label_core, trader_score, trader_pred,
    # max_edge_prob, top3_edge_prob}. The file is small (≤ 24 runs × 500
    # traders × ~100 bytes ≈ 1 MB).
    if predictions_out_path is not None:
        from pathlib import Path as _Path  # local import: avoid top-level cost
        out_p = _Path(predictions_out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        payload: dict = {
            "schema_version": "1",
            "edge_threshold": float(edge_threshold),
            "projection":     "trader_score = 0.7*max + 0.3*top3_mean",
            "per_run": {},
        }
        for run_id, sub in trader_df.groupby("run_id"):
            payload["per_run"][str(run_id)] = [
                {
                    "trader_id":      str(row["trader_id"]),
                    "label_core":     int(row["label_core"]),
                    "trader_score":   float(row["trader_score"]),
                    "trader_pred":    int(row["trader_pred"]),
                    "max_edge_prob":  float(row["max_edge_prob"]),
                    "top3_edge_prob": float(row["top3_edge_prob"]),
                }
                for _, row in sub.iterrows()
            ]
        import json as _json
        out_p.write_text(_json.dumps(payload, indent=2), encoding="utf-8")

    return MetricBundle(
        cv_f1=cv_f1, cv_auc=cv_auc,
        cv_precision=cv_precision, cv_recall=cv_recall,
        cv_threshold=edge_threshold,
        locked_clique_recall=family_recall.get("clique", NOT_COMPUTED),
        locked_ring_recall=family_recall.get("ring", NOT_COMPUTED),
        locked_mixed_recall=family_recall.get("mixed", NOT_COMPUTED),
        locked_benign_alarm=benign_alarm,
        locked_clique_purity=family_purity.get("clique", NOT_COMPUTED),
        locked_ring_purity=family_purity.get("ring", NOT_COMPUTED),
        locked_mixed_purity=family_purity.get("mixed", NOT_COMPUTED),
        locked_per_run=per_run,
        n_train_runs=n_train_runs,
        n_eval_runs=n_eval_runs,
        n_edges_train=int(cv_labels.size),
        n_pos_edges_train=int(cv_labels.sum()),
        seed=seed,
    )
