"""Engineered features for manipulation detection - analysis CLI.

The feature MATH lives in ``training/features/engineered_core.py`` and is
shared with the Path A2 GNN node-feature injection
(``training/features/node_engineered.py``), so the features analysed here
and the features fed to the feature-augmented GraphSAGE are the same code.

This script is the thin analysis layer: per-run featurisation with
ground-truth labels, plus the pooled per-feature ROC AUC report.

Features (per trader, scoped to their busiest 5-minute window):
  1. burst_concentration         - fraction of orders in peak 5-min window
  2. side_entropy_in_burst       - Shannon entropy of buy/sell in burst
  3. counterparty_hhi_burst      - Herfindahl-Hirschman of CP shares in burst
  4. order_qty_cov               - coefficient of variation of order qty
  5. top_partner_trade_share     - fraction of burst trades vs top-1 CP
  6. co_active_top_count         - count of co-active top-quantile traders

For each OOD run we save:
  outputs/_phase_g_features/<run_name>.csv
      columns: trader_id, label_core, the six features, burst metadata

Aggregate per-feature ROC AUC vs label_core, saved to
  outputs/_phase_g_features_auc.json

If any single feature has AUC >= 0.7 on its own, feature-augmented
retraining is justified.

Pure CPU pandas. Should finish under 2 minutes on a 50-run cohort.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Feature math is shared with the GNN node-feature path. Add the repo
# root so ``training`` is importable when this script is run from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from detect.features.engineered_core import (  # noqa: E402
    FEATURE_DESCRIPTIONS,
    FEATURE_NAMES,
    compute_features_frame,
)


def compute_features_for_run(run_dir: Path,
                             keep_top_active: int = 200) -> pd.DataFrame:
    """One row per (trader_id, label_core) with the six features.

    Analysis mode: featurises the ``keep_top_active`` most-active traders
    plus all manipulators (see engineered_core.compute_features_frame).
    The label is looked up from orders.is_manipulative.
    """
    orders = pd.read_csv(run_dir / "orders.csv")
    trades_path = run_dir / "trades.csv"
    trades = (pd.read_csv(trades_path)
              if trades_path.is_file() else pd.DataFrame())

    feats = compute_features_frame(orders, trades,
                                   keep_top_active=keep_top_active)
    if feats.empty:
        return feats

    orders["trader_id"] = orders["trader_id"].astype(str)
    truth = (orders.groupby("trader_id")["is_manipulative"].any()
             .astype(int))
    feats = feats.copy()
    feats["label_core"] = feats["trader_id"].map(truth).fillna(0).astype(int)
    cols = ["trader_id", "label_core", *FEATURE_NAMES,
            "burst_start", "burst_end", "n_orders", "n_burst_orders"]
    return feats[cols]


def main() -> int:
    # Lazy import: engineered_core (and the GNN path) must not require
    # sklearn, so it is imported only when the analysis CLI actually runs.
    from sklearn.metrics import roc_auc_score

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort-dir", type=Path,
                   default=Path("outputs/phase_g_test_ood"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("outputs/_phase_g_features"))
    p.add_argument("--auc-out", type=Path,
                   default=Path("outputs/_phase_g_features_auc.json"))
    p.add_argument("--keep-top-active", type=int, default=200)
    p.add_argument("--run-prefix", default="OOD_RUN",
                   help="directory-name prefix that marks a run")
    p.add_argument("--limit-runs", type=int, default=0,
                   help="Set >0 to dry-run on first N runs")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(d for d in args.cohort_dir.iterdir()
                      if d.is_dir() and d.name.startswith(args.run_prefix))
    if args.limit_runs > 0:
        run_dirs = run_dirs[:args.limit_runs]

    print(f"Computing 6 features over {len(run_dirs)} runs "
          f"(keep_top_active={args.keep_top_active}) ...", flush=True)

    all_rows: list[pd.DataFrame] = []
    t_start = time.perf_counter()
    for i, d in enumerate(run_dirs):
        if not (d / "orders.csv").is_file():
            continue
        t0 = time.perf_counter()
        df = compute_features_for_run(d, keep_top_active=args.keep_top_active)
        elapsed = time.perf_counter() - t0
        if df.empty:
            continue
        df.to_csv(args.out_dir / f"{d.name}.csv", index=False)
        df = df.copy()
        df["run"] = d.name
        all_rows.append(df)
        if i % 5 == 0 or i == len(run_dirs) - 1:
            print(f"  [{i+1:>2}/{len(run_dirs)}] {d.name}  "
                  f"rows={len(df)}  pos={int(df['label_core'].sum())}  "
                  f"({elapsed:.1f}s)", flush=True)

    if not all_rows:
        print("no runs produced features", file=sys.stderr)
        return 1

    pooled = pd.concat(all_rows, ignore_index=True)
    print(f"\nPooled per-trader rows: {len(pooled)}  "
          f"(manipulators: {int(pooled['label_core'].sum())})  "
          f"in {time.perf_counter() - t_start:.1f}s")

    print("\nPer-feature ROC AUC vs ground truth:")
    per_feat: dict[str, dict] = {}
    for f in FEATURE_NAMES:
        vals = pooled[f].astype(float).fillna(0.0).to_numpy()
        truth = pooled["label_core"].astype(int).to_numpy()
        try:
            auc_pos = roc_auc_score(truth, vals)
        except Exception:  # noqa: BLE001
            auc_pos = float("nan")
        auc = (max(auc_pos, 1.0 - auc_pos)
               if not math.isnan(auc_pos) else float("nan"))
        direction = ("+" if (not math.isnan(auc_pos)) and auc_pos >= 0.5
                     else "-")
        per_feat[f] = {
            "auc":        float(auc),
            "auc_raw":    float(auc_pos),
            "direction":  direction,
            "mean_pos":   (float(vals[truth == 1].mean())
                           if int(truth.sum()) else float("nan")),
            "mean_neg":   (float(vals[truth == 0].mean())
                           if int((1 - truth).sum()) else float("nan")),
            "median_pos": (float(np.median(vals[truth == 1]))
                           if int(truth.sum()) else float("nan")),
            "median_neg": (float(np.median(vals[truth == 0]))
                           if int((1 - truth).sum()) else float("nan")),
        }
        print(f"  {f:<32s}  AUC={auc:.3f} ({direction})  "
              f"mean_pos={per_feat[f]['mean_pos']:.3f}  "
              f"mean_neg={per_feat[f]['mean_neg']:.3f}")

    payload = {
        "n_traders_pooled":      int(len(pooled)),
        "n_manipulators_pooled": int(pooled["label_core"].sum()),
        "n_runs":                len(all_rows),
        "feature_descriptions":  FEATURE_DESCRIPTIONS,
        "per_feature_auc":       per_feat,
        "best_single_feature":   max(per_feat.items(),
                                     key=lambda kv: kv[1]["auc"])[0],
        "best_single_auc":       max(p["auc"] for p in per_feat.values()),
    }
    args.auc_out.parent.mkdir(parents=True, exist_ok=True)
    args.auc_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.auc_out}")
    print(f"Best single feature: {payload['best_single_feature']}  "
          f"AUC={payload['best_single_auc']:.3f}")
    best_auc = payload['best_single_auc']
    if best_auc >= 0.7:
        print("DECISION: feature-augmented retraining is justified "
              "(best single feature AUC >= 0.7).")
    elif best_auc >= 0.6:
        print("DECISION: partial signal; features may help in combination.")
    else:
        print("DECISION: weak per-feature signal; group-level features needed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
