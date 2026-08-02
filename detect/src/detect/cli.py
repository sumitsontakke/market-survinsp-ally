"""Command-line entry point for the ``detect`` module.

Three subcommands, wrapping the existing per-run feature and model code:

* ``detect features --cohort <dir>`` — extract the six engineered
  manipulation-detection features (φ₁ … φ₆) per trader per run, write
  labelled feature CSVs to ``<cohort>/_features/<run_label>.csv``.

* ``detect train --cohort <dir> --model {tier2}`` — train a detector
  on the extracted features. v0.2.0 supports the tier-2 Gradient
  Boosting Machine only (fast, CPU-only, no GPU or checkpoint dependency).
  v1 / v4 GraphSAGE need GPU + PyTorch and are targeted for v0.3.0.

* ``detect evaluate --cohort <dir> --model {tier2}`` — leave-one-run-out
  cross-validation. For each run, train tier-2 on the other N-1 runs
  and report per-run AUC + summary. Writes a JSON report to
  ``<cohort>/_detect_tier2/report.json``.

Design decisions:

* Feature math lives in :mod:`detect.features.engineered_core`. This CLI
  is the on-cohort orchestrator, not the feature implementation.
* Trader labels are derived from ``orders.csv#is_manipulative`` per
  SCHEMA.md § "Deriving per-trader labels" — no separate labels file
  needed.
* All heavy dependencies (sklearn) are imported lazily inside the
  functions that need them, so the module imports cheaply for tests.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

__all__ = [
    "main",
    "cmd_features",
    "cmd_train",
    "cmd_evaluate",
    "discover_runs",
    "derive_trader_labels",
    "FEATURE_COLUMNS",
]


# Six engineered features, in the canonical order (matches engineered_core.FEATURE_NAMES).
FEATURE_COLUMNS: tuple[str, ...] = (
    "burst_concentration",
    "side_entropy_in_burst",
    "counterparty_hhi_burst",
    "order_qty_cov",
    "top_partner_trade_share",
    "co_active_top_count",
)


# ── Cohort discovery ────────────────────────────────────────────────


def discover_runs(cohort_dir: Path) -> list[Path]:
    """Return sorted list of run subdirectories (those with a manifest.json).

    Skips ``_features`` and other underscore-prefixed sibling dirs
    (they're detector output, not source runs).
    """
    return sorted(
        d for d in cohort_dir.iterdir()
        if d.is_dir()
        and not d.name.startswith("_")
        and (d / "manifest.json").exists()
    )


# ── Trader-label derivation (SCHEMA.md § "Deriving per-trader labels") ─


def derive_trader_labels(orders_df) -> "pandas.DataFrame":  # noqa: F821
    """Given orders.csv as a DataFrame, return a two-column frame
    (trader_id, is_manipulator) with one row per unique trader.

    ``is_manipulator`` is True iff any of the trader's orders has
    ``is_manipulative == True``.
    """
    import pandas as pd

    # Accept both boolean and string encodings
    is_manip = orders_df["is_manipulative"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )
    manip_trader_ids = set(orders_df.loc[is_manip, "trader_id"].unique())

    all_trader_ids = orders_df["trader_id"].unique()
    return pd.DataFrame({
        "trader_id": all_trader_ids,
        "is_manipulator": [tid in manip_trader_ids for tid in all_trader_ids],
    })


# ── features subcommand ────────────────────────────────────────────


def cmd_features(args: argparse.Namespace) -> int:
    """Extract engineered features for every run in the cohort."""
    import pandas as pd
    from detect.features.engineered_core import compute_features_frame

    cohort_dir = Path(args.cohort)
    if not cohort_dir.is_dir():
        print(f"error: cohort directory not found: {cohort_dir}", file=sys.stderr)
        return 1

    out_dir = Path(args.out) if args.out else (cohort_dir / "_features")
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(cohort_dir)
    if not runs:
        print(f"error: no runs found in {cohort_dir}", file=sys.stderr)
        return 1

    for run_dir in runs:
        orders = pd.read_csv(run_dir / "orders.csv")
        trades = (
            pd.read_csv(run_dir / "trades.csv")
            if (run_dir / "trades.csv").exists() else None
        )
        # Compute all-trader features (trader_ids=None means "auto-select busy
        # traders + all manipulators" per engineered_core doc). For a demo
        # cohort this is small; for large cohorts users can pass
        # --keep-top-active elsewhere.
        features = compute_features_frame(orders, trades)

        # Attach labels
        labels = derive_trader_labels(orders)
        merged = features.merge(labels, on="trader_id", how="left")
        merged["is_manipulator"] = merged["is_manipulator"].fillna(False)

        out_csv = out_dir / f"{run_dir.name}.csv"
        merged.to_csv(out_csv, index=False)
        if args.verbose:
            n_manip = int(merged["is_manipulator"].sum())
            print(f"[features] {run_dir.name}: {len(merged)} traders "
                  f"({n_manip} manipulators)", flush=True)

    print(f"OK: features for {len(runs)} runs at {out_dir}")
    return 0


# ── train subcommand (tier-2 GBM) ──────────────────────────────────


def _load_feature_bundle(features_dir: Path):
    """Concatenate all per-run feature CSVs into one DataFrame with
    a ``run`` column for LOO-CV grouping. Returns (X, y, groups)."""
    import pandas as pd

    csvs = sorted(features_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"no feature CSVs in {features_dir}")
    frames = []
    for p in csvs:
        df = pd.read_csv(p)
        df["run"] = p.stem
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    X = all_df[list(FEATURE_COLUMNS)]
    y = all_df["is_manipulator"].astype(int)
    groups = all_df["run"]
    return X, y, groups


def cmd_train(args: argparse.Namespace) -> int:
    """Train tier-2 GBM on all runs, save pickle. Evaluation is separate."""
    if args.model != "tier2":
        print(
            f"error: v0.2.0 supports --model tier2 only; got {args.model!r}. "
            f"v1 / v4 GraphSAGE training targeted for v0.3.0.",
            file=sys.stderr,
        )
        return 1

    import pickle
    from sklearn.ensemble import GradientBoostingClassifier

    cohort_dir = Path(args.cohort)
    features_dir = cohort_dir / "_features"
    if not features_dir.is_dir():
        print(
            f"error: features dir not found ({features_dir}). "
            f"Run `detect features --cohort {cohort_dir}` first.",
            file=sys.stderr,
        )
        return 1

    try:
        X, y, _groups = _load_feature_bundle(features_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if y.nunique() < 2:
        print(
            f"error: labels have only one class (all {y.iloc[0]!r}); "
            f"can't train a classifier",
            file=sys.stderr,
        )
        return 1

    model = GradientBoostingClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        random_state=args.random_state,
    )
    model.fit(X, y)

    out_dir = cohort_dir / "_detect_tier2"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "model.pkl"
    with ckpt.open("wb") as h:
        pickle.dump({"model": model, "features": list(FEATURE_COLUMNS)}, h)

    print(f"OK: trained tier2 on {len(X)} rows, {int(y.sum())} positives; "
          f"saved to {ckpt}")
    return 0


# ── evaluate subcommand (LOO-CV) ───────────────────────────────────


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Leave-one-run-out CV: train on N-1 runs, evaluate on held-out."""
    if args.model != "tier2":
        print(
            f"error: v0.2.0 supports --model tier2 only; got {args.model!r}.",
            file=sys.stderr,
        )
        return 1

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    cohort_dir = Path(args.cohort)
    features_dir = cohort_dir / "_features"
    if not features_dir.is_dir():
        print(
            f"error: features dir not found ({features_dir}). "
            f"Run `detect features --cohort {cohort_dir}` first.",
            file=sys.stderr,
        )
        return 1

    try:
        X, y, groups = _load_feature_bundle(features_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    unique_runs = sorted(groups.unique())
    if len(unique_runs) < 2:
        print(
            f"error: need at least 2 runs for LOO-CV; got {len(unique_runs)}.",
            file=sys.stderr,
        )
        return 1

    per_run: list[dict] = []
    for held_out in unique_runs:
        train_mask = groups != held_out
        test_mask  = ~train_mask
        y_train = y[train_mask]
        y_test  = y[test_mask]

        # Skip folds where the training data has only one class
        if y_train.nunique() < 2:
            per_run.append({
                "run": held_out, "auc": None, "skipped": True,
                "reason": "training fold has only one label class",
                "n_train": int(train_mask.sum()),
                "n_test":  int(test_mask.sum()),
                "n_test_positives": int(y_test.sum()),
            })
            continue

        model = GradientBoostingClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            random_state=args.random_state,
        )
        model.fit(X[train_mask], y_train)
        y_score = model.predict_proba(X[test_mask])[:, 1]

        auc: float | None
        if y_test.nunique() < 2:
            auc = None
            reason = "test fold has only one label class"
        else:
            auc = float(roc_auc_score(y_test, y_score))
            reason = None

        per_run.append({
            "run": held_out,
            "auc": auc,
            "skipped": False,
            "reason": reason,
            "n_train": int(train_mask.sum()),
            "n_test":  int(test_mask.sum()),
            "n_test_positives": int(y_test.sum()),
        })

    aucs = [r["auc"] for r in per_run if r["auc"] is not None]
    mean_auc = sum(aucs) / len(aucs) if aucs else None

    report = {
        "model": args.model,
        "cohort": str(cohort_dir),
        "n_runs": len(unique_runs),
        "n_folds_scored": len(aucs),
        "mean_auc": mean_auc,
        "per_run": per_run,
        "features": list(FEATURE_COLUMNS),
    }

    out_dir = cohort_dir / "_detect_tier2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"OK: evaluated {args.model} — {len(aucs)}/{len(unique_runs)} folds scored")
    if mean_auc is not None:
        print(f"    mean AUC across folds: {mean_auc:.4f}")
    else:
        print("    mean AUC unavailable (no folds could be scored)")
    print(f"    report: {out_path}")
    return 0


# ── Argument parsing + main ────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="detect",
        description="Extract features, train, and evaluate manipulation detectors.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # features
    feat = sub.add_parser(
        "features",
        help="Extract 6 engineered features per trader per run.",
        description=(
            "Runs `compute_features_frame` on every run in the cohort. "
            "Attaches ground-truth per-trader labels derived from "
            "orders.csv#is_manipulative."
        ),
    )
    feat.add_argument("--cohort", required=True, help="Cohort directory.")
    feat.add_argument(
        "--out", default=None,
        help="Output dir (default: <cohort>/_features)",
    )
    feat.add_argument("--verbose", action="store_true")

    # train
    train = sub.add_parser(
        "train",
        help="Train a detector on extracted features.",
        description=(
            "v0.2.0: tier-2 Gradient Boosting Machine only. "
            "Requires `detect features` to have run first."
        ),
    )
    train.add_argument("--cohort", required=True)
    train.add_argument(
        "--model", choices=("tier2",), default="tier2",
        help="Model family (default: tier2). v1/v4 GraphSAGE in v0.3.0.",
    )
    train.add_argument("--n-estimators", type=int, default=100)
    train.add_argument("--max-depth", type=int, default=3)
    train.add_argument("--learning-rate", type=float, default=0.1)
    train.add_argument("--random-state", type=int, default=42)

    # evaluate
    ev = sub.add_parser(
        "evaluate",
        help="Leave-one-run-out CV evaluation.",
        description=(
            "For each run: train on the other N-1 runs, score the held-out "
            "run, report per-run AUC + mean. Writes JSON to "
            "<cohort>/_detect_tier2/report.json."
        ),
    )
    ev.add_argument("--cohort", required=True)
    ev.add_argument("--model", choices=("tier2",), default="tier2")
    ev.add_argument("--n-estimators", type=int, default=100)
    ev.add_argument("--max-depth", type=int, default=3)
    ev.add_argument("--learning-rate", type=float, default=0.1)
    ev.add_argument("--random-state", type=int, default=42)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "features":
        return cmd_features(args)
    if args.cmd == "train":
        return cmd_train(args)
    if args.cmd == "evaluate":
        return cmd_evaluate(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
