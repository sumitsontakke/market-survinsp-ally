"""Aggregate Rung-1 surveillance metrics from the on-disk analysis artifacts.

Every locked-cohort run already has ``analysis/detection_evaluation.json``
with the full Rung-1 confusion matrix (TP/FN/FP/TN) and per-scenario
purity + coverage. This module reads those JSONs and aggregates by
manipulation family (clique / ring / mixed), so Demo Flow Step 7 can
show real precision, accuracy, purity, coverage numbers — not the ``?``
placeholders we had before.

For Rung 4 the same numbers are blocked on per-trader prediction
recording; once `run_m3.py` / `run_m3_boosted.py` emit
``outputs/predictions_rung4_<run>.json`` we can do the same aggregation
against the same ground truth and produce a full Rung-4 confusion matrix.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

OUTPUTS_DIR = (
    Path("/outputs") if Path("/outputs").exists()
    else Path(os.environ.get("OUTPUTS_DIR", "")) or
         Path(__file__).resolve().parent.parent.parent / "outputs"
)


def _infer_family(run_name: str) -> str:
    n = run_name.lower()
    if "clique" in n:
        return "clique"
    if "ring" in n:
        return "ring"
    if "mixed" in n:
        return "mixed"
    return "unknown"


def load_rung1_per_run() -> list[dict[str, Any]]:
    """One row per locked R01-R24 run with that run's Rung-1 numbers."""
    rows: list[dict[str, Any]] = []
    runs_root = OUTPUTS_DIR / "runs"
    if not runs_root.exists():
        return rows
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith("R"):
            continue
        de = run_dir / "analysis" / "detection_evaluation.json"
        if not de.exists():
            continue
        try:
            payload = json.loads(de.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        family = _infer_family(run_dir.name)
        # Pull the scenario_evaluations purity/coverage for the manipulative
        # scenarios in this run (skip "normal" background scenario).
        scenario_evals = payload.get("scenario_evaluations", []) or []
        purities  = [float(s.get("purity",  0.0)) for s in scenario_evals]
        coverages = [float(s.get("coverage", 0.0)) for s in scenario_evals]
        rows.append({
            "run":      run_dir.name,
            "family":   family,
            "total":    int(payload.get("total_traders",  0)),
            "n_manip":  int(payload.get("injected_trader_count", 0)),
            "n_flag":   int(payload.get("flagged_trader_count", 0)),
            "tp":       int(payload.get("true_positive_count",  0)),
            "fn":       int(payload.get("false_negative_count", 0)),
            "fp":       int(payload.get("false_positive_count", 0)),
            "tn":       int(payload.get("true_negative_count",  0)),
            "precision": float(payload.get("precision", 0.0)),
            "recall":    float(payload.get("recall",    0.0)),
            "f1":        float(payload.get("f1_score",  0.0)),
            "fpr":       float(payload.get("false_positive_rate", 0.0)),
            "fnr":       float(payload.get("false_negative_rate", 0.0)),
            "purity_per_scenario":   purities,
            "coverage_per_scenario": coverages,
            "verdict":   payload.get("verdict", "?"),
        })
    return rows


def load_rung4_per_run(predictions_path: str = "_m3_boosted_predictions.json"
                       ) -> dict[str, list[dict[str, Any]]] | None:
    """Read the per-trader prediction dump if `run_m3_boosted.py` has been
    re-run with the new ``predictions_out_path`` argument. Returns
    ``{run_id: [trader_row, ...]}`` or ``None`` if the file isn't there yet.
    """
    p = OUTPUTS_DIR / predictions_path
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload.get("per_run", {})


def aggregate_rung4_by_family(predictions: dict[str, list[dict[str, Any]]]
                              ) -> dict[str, dict[str, Any]]:
    """Same shape as ``aggregate_by_family`` but computed from the Rung-4
    per-trader prediction dump. Once `run_m3_boosted.py` has been re-run
    after the per-trader recording change, this powers the full Rung-4
    column on Demo Flow Step 7.
    """
    by_fam: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "n_runs": 0}
    )
    for run_id, rows in predictions.items():
        fam = _infer_family(run_id)
        by_fam[fam]["n_runs"] += 1
        for row in rows:
            label = int(row["label_core"])
            pred  = int(row["trader_pred"])
            if label == 1 and pred == 1:
                by_fam[fam]["tp"] += 1
            elif label == 1 and pred == 0:
                by_fam[fam]["fn"] += 1
            elif label == 0 and pred == 1:
                by_fam[fam]["fp"] += 1
            else:
                by_fam[fam]["tn"] += 1
    out: dict[str, dict[str, Any]] = {}
    for fam, c in by_fam.items():
        tp, fn, fp, tn = c["tp"], c["fn"], c["fp"], c["tn"]
        total_pos = tp + fn
        flagged   = tp + fp
        total_traders = tp + fn + fp + tn
        precision = (tp / flagged)   if flagged   > 0 else 0.0
        recall    = (tp / total_pos) if total_pos > 0 else 0.0
        f1        = ((2 * precision * recall) / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        accuracy  = ((tp + tn) / total_traders) if total_traders > 0 else 0.0
        specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        # Purity per family = TP / (TP + FP) = precision; coverage = recall.
        # The richer per-scenario purity/coverage would need scenario IDs
        # in the prediction dump — left as the next enhancement.
        out[fam] = {
            "n_runs":     c["n_runs"],
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "precision":  precision,
            "recall":     recall,
            "f1":         f1,
            "accuracy":   accuracy,
            "specificity": specificity,
            "purity":     precision,   # cluster-level == flat precision here
            "coverage":   recall,
        }
    return out


def aggregate_by_family(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate the per-run rows by family.

    For TP/FN/FP/TN we sum across runs in the family (so the aggregate
    confusion matrix reflects total counts, not means). For
    precision/recall/F1 we compute them from those summed counts, which
    is the correct way to aggregate per-trader metrics (NOT the mean of
    per-run rates). Purity and coverage are micro-averaged over scenarios.
    """
    by_fam: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append(r)

    out: dict[str, dict[str, Any]] = {}
    for fam, group in by_fam.items():
        tp = sum(r["tp"] for r in group)
        fn = sum(r["fn"] for r in group)
        fp = sum(r["fp"] for r in group)
        tn = sum(r["tn"] for r in group)
        total_pos = tp + fn
        flagged   = tp + fp
        total_traders = tp + fn + fp + tn
        precision = (tp / flagged)    if flagged    > 0 else 0.0
        recall    = (tp / total_pos)  if total_pos  > 0 else 0.0
        f1        = ((2 * precision * recall) / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        accuracy  = ((tp + tn) / total_traders) if total_traders > 0 else 0.0
        # Specificity (true-negative rate) — surveillance auditors care
        # about how often we leave honest traders alone.
        specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        # Aggregate purity + coverage across scenarios (micro-average).
        purities   = [p for r in group for p in r["purity_per_scenario"]]
        coverages  = [c for r in group for c in r["coverage_per_scenario"]]
        purity_mean   = (sum(purities)  / len(purities))  if purities  else 0.0
        coverage_mean = (sum(coverages) / len(coverages)) if coverages else 0.0
        out[fam] = {
            "n_runs":     len(group),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "precision":  precision,
            "recall":     recall,
            "f1":         f1,
            "accuracy":   accuracy,
            "specificity": specificity,
            "purity":     purity_mean,
            "coverage":   coverage_mean,
            "n_scenarios": len(purities),
        }
    return out
