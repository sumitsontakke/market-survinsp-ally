"""Tests for :mod:`detect.cli`.

Strategy:

* Pure-function tests (label derivation, run discovery, feature bundling)
  use tiny synthetic DataFrames — fast, no filesystem.
* Argument parsing tests use ``build_parser`` directly, no execution.
* End-to-end tests for ``train`` and ``evaluate`` write a synthetic
  cohort with pre-computed feature CSVs (skipping feature extraction),
  then let the sklearn training run for real on ~10 rows. This is still
  fast (<1 second) because sklearn on 10 rows is trivial.
* End-to-end test for ``features`` requires real orders.csv shape, so
  it's marked with a fixture that builds a minimal orders + trades pair
  and calls the real ``compute_features_frame``.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from detect.cli import (
    FEATURE_COLUMNS,
    build_parser,
    cmd_evaluate,
    cmd_features,
    cmd_train,
    derive_trader_labels,
    discover_runs,
    main,
)


# ── discover_runs ──────────────────────────────────────────────────


def test_discover_runs_finds_run_subdirs(tmp_path):
    (tmp_path / "R01_msa_clique_s42_20260314").mkdir()
    (tmp_path / "R01_msa_clique_s42_20260314" / "manifest.json").write_text("{}")
    (tmp_path / "R02_msa_ring_s43_20260314").mkdir()
    (tmp_path / "R02_msa_ring_s43_20260314" / "manifest.json").write_text("{}")

    runs = discover_runs(tmp_path)
    assert len(runs) == 2
    assert runs[0].name == "R01_msa_clique_s42_20260314"


def test_discover_runs_skips_underscore_prefixed(tmp_path):
    (tmp_path / "R01_a").mkdir()
    (tmp_path / "R01_a" / "manifest.json").write_text("{}")
    (tmp_path / "_features").mkdir()
    (tmp_path / "_features" / "manifest.json").write_text("{}")  # even if present

    runs = discover_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0].name == "R01_a"


def test_discover_runs_skips_dirs_without_manifest(tmp_path):
    (tmp_path / "R01_a").mkdir()  # no manifest
    (tmp_path / "R02_b").mkdir()
    (tmp_path / "R02_b" / "manifest.json").write_text("{}")

    runs = discover_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0].name == "R02_b"


# ── derive_trader_labels ───────────────────────────────────────────


def test_derive_labels_basic():
    import pandas as pd
    orders = pd.DataFrame({
        "trader_id": ["t1", "t1", "t2", "t3"],
        "is_manipulative": ["False", "False", "True", "False"],
    })
    labels = derive_trader_labels(orders)
    assert set(labels.columns) == {"trader_id", "is_manipulator"}
    assert len(labels) == 3
    row_t2 = labels[labels["trader_id"] == "t2"].iloc[0]
    assert bool(row_t2["is_manipulator"]) is True
    row_t1 = labels[labels["trader_id"] == "t1"].iloc[0]
    assert bool(row_t1["is_manipulator"]) is False


def test_derive_labels_any_manip_flags_trader():
    import pandas as pd
    # Trader with a mix of manip + benign orders should be flagged
    orders = pd.DataFrame({
        "trader_id": ["t1", "t1", "t1"],
        "is_manipulative": ["False", "True", "False"],
    })
    labels = derive_trader_labels(orders)
    assert bool(labels.iloc[0]["is_manipulator"]) is True


def test_derive_labels_accepts_various_encodings():
    import pandas as pd
    orders = pd.DataFrame({
        "trader_id": ["t1", "t2", "t3", "t4"],
        "is_manipulative": ["true", "TRUE", "1", "yes"],
    })
    labels = derive_trader_labels(orders)
    assert all(labels["is_manipulator"])


# ── argument parsing ───────────────────────────────────────────────


def test_parser_features_requires_cohort():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["features"])


def test_parser_train_defaults_to_tier2():
    args = build_parser().parse_args(["train", "--cohort", "x"])
    assert args.model == "tier2"


def test_parser_evaluate_accepts_hyperparams():
    args = build_parser().parse_args([
        "evaluate", "--cohort", "x", "--n-estimators", "50",
        "--max-depth", "5", "--learning-rate", "0.05",
    ])
    assert args.n_estimators == 50
    assert args.max_depth == 5
    assert args.learning_rate == 0.05


# ── Helper: write pre-computed feature CSVs so train/evaluate can be tested ─


def _write_feature_csv(path: Path, n_traders: int, n_manip: int, seed: int):
    """Write a per-run feature CSV with plausible values. Manipulator rows
    have higher burst_concentration + counterparty_hhi, so tier-2 GBM can
    actually learn something."""
    import numpy as np
    rng = np.random.default_rng(seed)

    rows = []
    for i in range(n_traders):
        is_manip = i < n_manip
        # Manipulators: high burst concentration, high hhi
        # Benign: spread out
        if is_manip:
            row = {
                "trader_id": f"trader_{i:03d}",
                "burst_concentration": float(rng.uniform(0.55, 0.9)),
                "side_entropy_in_burst": float(rng.uniform(0.0, 0.4)),
                "counterparty_hhi_burst": float(rng.uniform(0.5, 0.9)),
                "order_qty_cov": float(rng.uniform(0.0, 0.2)),
                "top_partner_trade_share": float(rng.uniform(0.5, 0.9)),
                "co_active_top_count": int(rng.integers(3, 8)),
                "is_manipulator": True,
            }
        else:
            row = {
                "trader_id": f"trader_{i:03d}",
                "burst_concentration": float(rng.uniform(0.05, 0.3)),
                "side_entropy_in_burst": float(rng.uniform(0.6, 1.0)),
                "counterparty_hhi_burst": float(rng.uniform(0.05, 0.3)),
                "order_qty_cov": float(rng.uniform(0.3, 0.9)),
                "top_partner_trade_share": float(rng.uniform(0.05, 0.4)),
                "co_active_top_count": int(rng.integers(0, 3)),
                "is_manipulator": False,
            }
        rows.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as h:
        writer = csv.DictWriter(
            h, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def cohort_with_features(tmp_path):
    """Build a 3-run cohort with pre-computed feature CSVs. Enough separable
    signal that tier-2 should get AUC well above 0.5 on LOO folds."""
    features_dir = tmp_path / "_features"
    _write_feature_csv(features_dir / "R01_msa_clique_s42_20260314.csv",
                       n_traders=20, n_manip=3, seed=1)
    _write_feature_csv(features_dir / "R02_msa_ring_s43_20260314.csv",
                       n_traders=20, n_manip=3, seed=2)
    _write_feature_csv(features_dir / "R03_msa_benign_s44_20260314.csv",
                       n_traders=20, n_manip=0, seed=3)
    return tmp_path


# ── cmd_train ──────────────────────────────────────────────────────


def test_train_writes_pickle(cohort_with_features):
    args = build_parser().parse_args([
        "train", "--cohort", str(cohort_with_features),
    ])
    exit_code = cmd_train(args)
    assert exit_code == 0
    ckpt = cohort_with_features / "_detect_tier2" / "model.pkl"
    assert ckpt.exists()

    import pickle
    with ckpt.open("rb") as h:
        bundle = pickle.load(h)
    assert "model" in bundle
    assert bundle["features"] == list(FEATURE_COLUMNS)


def test_train_missing_features_dir_errors(tmp_path):
    args = build_parser().parse_args([
        "train", "--cohort", str(tmp_path),
    ])
    exit_code = cmd_train(args)
    assert exit_code == 1


def test_train_rejects_unsupported_model(cohort_with_features):
    # Argparse restricts choices, so this hits at parse time
    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "train", "--cohort", str(cohort_with_features), "--model", "v4",
        ])


def test_train_errors_on_single_class(tmp_path):
    """If all rows have the same label, GBM can't be trained — should error cleanly."""
    features_dir = tmp_path / "_features"
    _write_feature_csv(features_dir / "R01.csv", n_traders=10, n_manip=0, seed=1)
    args = build_parser().parse_args(["train", "--cohort", str(tmp_path)])
    exit_code = cmd_train(args)
    assert exit_code == 1


# ── cmd_evaluate ───────────────────────────────────────────────────


def test_evaluate_produces_report(cohort_with_features):
    args = build_parser().parse_args([
        "evaluate", "--cohort", str(cohort_with_features),
    ])
    exit_code = cmd_evaluate(args)
    assert exit_code == 0

    report_path = cohort_with_features / "_detect_tier2" / "report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["model"] == "tier2"
    assert report["n_runs"] == 3
    assert len(report["per_run"]) == 3
    # At least some folds should have scored; separable synthetic data
    # means we should get real AUCs on the two positive-class folds.
    scored = [r for r in report["per_run"] if r["auc"] is not None]
    assert len(scored) >= 1


def test_evaluate_needs_features_dir(tmp_path):
    args = build_parser().parse_args(["evaluate", "--cohort", str(tmp_path)])
    exit_code = cmd_evaluate(args)
    assert exit_code == 1


def test_evaluate_needs_multiple_runs(tmp_path):
    features_dir = tmp_path / "_features"
    _write_feature_csv(features_dir / "R01.csv", n_traders=10, n_manip=2, seed=1)
    # Only 1 run → can't do LOO-CV
    args = build_parser().parse_args(["evaluate", "--cohort", str(tmp_path)])
    exit_code = cmd_evaluate(args)
    assert exit_code == 1


# ── cmd_features (end-to-end using engineered_core on synthetic orders) ─


def test_features_end_to_end(tmp_path):
    """Build a minimal cohort with 1 run, write orders + trades, run
    the real feature extractor."""
    run_dir = tmp_path / "R01_msa_clique_s42_20260314"
    run_dir.mkdir()

    # A tiny orders.csv the feature extractor can consume.
    # Two traders: one bursty manipulator, one spread-out benign.
    orders_rows = [
        # Trader t_manip: 4 rapid orders in 09:30-09:31 window
        {"order_id": "1", "timestamp": "2026-03-14T09:30:00", "trader_id": "t_manip",
         "instrument_id": "ALPHA", "side": "buy", "quantity": "10", "price": "100.0",
         "is_manipulative": "True"},
        {"order_id": "2", "timestamp": "2026-03-14T09:30:10", "trader_id": "t_manip",
         "instrument_id": "ALPHA", "side": "buy", "quantity": "10", "price": "100.0",
         "is_manipulative": "True"},
        {"order_id": "3", "timestamp": "2026-03-14T09:30:20", "trader_id": "t_manip",
         "instrument_id": "ALPHA", "side": "buy", "quantity": "10", "price": "100.0",
         "is_manipulative": "True"},
        {"order_id": "4", "timestamp": "2026-03-14T09:30:30", "trader_id": "t_manip",
         "instrument_id": "ALPHA", "side": "buy", "quantity": "10", "price": "100.0",
         "is_manipulative": "True"},
        # Trader t_benign: 4 spread out over the whole day
        {"order_id": "5", "timestamp": "2026-03-14T09:35:00", "trader_id": "t_benign",
         "instrument_id": "ALPHA", "side": "buy", "quantity": "5", "price": "100.0",
         "is_manipulative": "False"},
        {"order_id": "6", "timestamp": "2026-03-14T10:00:00", "trader_id": "t_benign",
         "instrument_id": "ALPHA", "side": "sell", "quantity": "5", "price": "100.0",
         "is_manipulative": "False"},
        {"order_id": "7", "timestamp": "2026-03-14T11:00:00", "trader_id": "t_benign",
         "instrument_id": "ALPHA", "side": "buy", "quantity": "5", "price": "100.0",
         "is_manipulative": "False"},
        {"order_id": "8", "timestamp": "2026-03-14T12:00:00", "trader_id": "t_benign",
         "instrument_id": "ALPHA", "side": "sell", "quantity": "5", "price": "100.0",
         "is_manipulative": "False"},
    ]
    orders_fields = list(orders_rows[0].keys())
    with (run_dir / "orders.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=orders_fields)
        w.writeheader()
        w.writerows(orders_rows)

    # Minimal trades.csv (can be empty of matches)
    trades_fields = ["trade_id", "timestamp", "buy_order_id", "sell_order_id",
                     "buy_trader_id", "sell_trader_id", "instrument_id",
                     "price", "quantity", "is_manipulative"]
    with (run_dir / "trades.csv").open("w", newline="") as h:
        csv.DictWriter(h, fieldnames=trades_fields).writeheader()

    # Minimal manifest.json so discover_runs picks it up
    (run_dir / "manifest.json").write_text("{}")

    args = build_parser().parse_args([
        "features", "--cohort", str(tmp_path),
    ])
    exit_code = cmd_features(args)
    assert exit_code == 0

    out_csv = tmp_path / "_features" / "R01_msa_clique_s42_20260314.csv"
    assert out_csv.exists()

    import pandas as pd
    df = pd.read_csv(out_csv)
    assert "is_manipulator" in df.columns
    # Every feature column should be present
    for col in FEATURE_COLUMNS:
        assert col in df.columns


def test_features_missing_cohort_errors(tmp_path):
    args = build_parser().parse_args([
        "features", "--cohort", str(tmp_path / "nope"),
    ])
    exit_code = cmd_features(args)
    assert exit_code == 1


def test_features_empty_cohort_errors(tmp_path):
    args = build_parser().parse_args([
        "features", "--cohort", str(tmp_path),
    ])
    exit_code = cmd_features(args)
    assert exit_code == 1


# ── main dispatch ──────────────────────────────────────────────────


def test_main_no_subcommand_exits_two():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_main_dispatches_evaluate(tmp_path):
    exit_code = main(["evaluate", "--cohort", str(tmp_path)])
    assert exit_code == 1  # no features dir
