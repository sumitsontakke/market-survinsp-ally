"""Tests for :mod:`synth.validate`.

Uses tmp_path fixtures to build minimal valid + intentionally-broken
cohorts on disk, then checks that the validator flags the right things.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from synth.validate import (
    ValidationIssue,
    ValidationReport,
    validate,
    validate_cohort,
    validate_run,
)


# ── Fixture: builds a minimal valid run on disk ──────────────────────


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _build_valid_run(root: Path, run_label: str = "R01_msa_clique_s42_20260314") -> Path:
    """Write a minimal-but-conformant single-run directory. Returns its path."""
    run_dir = root / run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    # Uses the generator's actual PK column (`beneficial_owner_id`).
    # SCHEMA.md aspirationally documents `owner_id`; reconciled in v0.3.0.
    _write_csv(
        run_dir / "beneficial_owners.csv",
        [
            {"beneficial_owner_id": "owner_001", "name": "Alice",
             "kyc_status": "verified", "region": "IN",
             "created_at": "2026-01-01T00:00:00"},
            {"beneficial_owner_id": "owner_002", "name": "Bob",
             "kyc_status": "verified", "region": "IN",
             "created_at": "2026-01-01T00:00:00"},
        ],
        ["beneficial_owner_id", "name", "kyc_status", "region", "created_at"],
    )

    _write_csv(
        run_dir / "accounts.csv",
        [
            {"account_id": "account_001", "beneficial_owner_id": "owner_001",
             "opened_at": "2026-01-02T00:00:00", "status": "active"},
            {"account_id": "account_002", "beneficial_owner_id": "owner_002",
             "opened_at": "2026-01-02T00:00:00", "status": "active"},
        ],
        ["account_id", "beneficial_owner_id", "opened_at", "status"],
    )

    _write_csv(
        run_dir / "brokers.csv",
        [{"broker_id": "broker_001", "name": "BrokerCo", "region": "IN",
          "registered_at": "2026-01-01T00:00:00", "status": "active"}],
        ["broker_id", "name", "region", "registered_at", "status"],
    )

    _write_csv(
        run_dir / "instruments.csv",
        [{"instrument_id": "instrument_001", "symbol": "ALPHA", "asset_class": "equity",
          "listing_venue": "NSE", "currency": "INR"}],
        ["instrument_id", "symbol", "asset_class", "listing_venue", "currency"],
    )

    # Uses generator's actual column names (trade_date/open_time/close_time).
    # SCHEMA.md documents session_date/open_ts/close_ts; reconciled in v0.3.0.
    _write_csv(
        run_dir / "sessions.csv",
        [{"session_id": "session_001", "trade_date": "2026-03-14",
          "open_time": "09:30:00", "close_time": "15:30:00",
          "auction_windows": "", "timezone": "UTC"}],
        ["session_id", "trade_date", "open_time", "close_time",
         "auction_windows", "timezone"],
    )

    _write_csv(
        run_dir / "traders.csv",
        [
            {"trader_id": "trader_001", "account_id": "account_001",
             "beneficial_owner_id": "owner_001", "broker_id": "broker_001",
             "trader_profile_id": "noise_trader", "risk_tier": "medium",
             "region": "IN", "created_at": "2026-01-02T00:00:00", "status": "active"},
            {"trader_id": "trader_002", "account_id": "account_002",
             "beneficial_owner_id": "owner_002", "broker_id": "broker_001",
             "trader_profile_id": "clique_alpha", "risk_tier": "high",
             "region": "IN", "created_at": "2026-01-02T00:00:00", "status": "active"},
        ],
        ["trader_id", "account_id", "beneficial_owner_id", "broker_id",
         "trader_profile_id", "risk_tier", "region", "created_at", "status"],
    )

    _write_csv(
        run_dir / "scenarios.csv",
        [
            {"scenario_id": "normal", "scenario_label": "normal",
             "scenario_type": "generic_background",
             "start_ts": "2026-03-14T09:30:00", "end_ts": "2026-03-14T15:30:00",
             "manipulator_count": "0"},
            {"scenario_id": "scenario_clique_001", "scenario_label": "clique_alpha_0",
             "scenario_type": "collusive_clique",
             "start_ts": "2026-03-14T10:00:00", "end_ts": "2026-03-14T11:00:00",
             "manipulator_count": "1"},
        ],
        ["scenario_id", "scenario_label", "scenario_type", "start_ts", "end_ts",
         "manipulator_count"],
    )

    _write_csv(
        run_dir / "orders.csv",
        [
            {"order_id": "1", "timestamp": "2026-03-14T09:30:00",
             "trader_id": "trader_001", "account_id": "account_001",
             "broker_id": "broker_001", "instrument_id": "instrument_001",
             "side": "buy", "order_type": "limit", "price": "100.00",
             "quantity": "10", "time_in_force": "day", "scenario_id": "normal",
             "scenario_label": "normal", "scenario_type": "generic_background",
             "is_manipulative": "False", "remaining_quantity": "0"},
            {"order_id": "2", "timestamp": "2026-03-14T10:05:00",
             "trader_id": "trader_002", "account_id": "account_002",
             "broker_id": "broker_001", "instrument_id": "instrument_001",
             "side": "sell", "order_type": "limit", "price": "100.00",
             "quantity": "10", "time_in_force": "day",
             "scenario_id": "scenario_clique_001",
             "scenario_label": "clique_alpha_0",
             "scenario_type": "collusive_clique",
             "is_manipulative": "True", "remaining_quantity": "0"},
        ],
        ["order_id", "timestamp", "trader_id", "account_id", "broker_id",
         "instrument_id", "side", "order_type", "price", "quantity",
         "time_in_force", "scenario_id", "scenario_label", "scenario_type",
         "is_manipulative", "remaining_quantity"],
    )

    _write_csv(
        run_dir / "trades.csv",
        [{"trade_id": "1", "timestamp": "2026-03-14T10:05:00",
          "buy_order_id": "1", "sell_order_id": "2",
          "buy_trader_id": "trader_001", "sell_trader_id": "trader_002",
          "instrument_id": "instrument_001", "price": "100.00", "quantity": "10",
          "scenario_id": "scenario_clique_001",
          "scenario_label": "clique_alpha_0",
          "scenario_type": "collusive_clique",
          "is_manipulative": "True"}],
        ["trade_id", "timestamp", "buy_order_id", "sell_order_id",
         "buy_trader_id", "sell_trader_id", "instrument_id", "price",
         "quantity", "scenario_id", "scenario_label", "scenario_type",
         "is_manipulative"],
    )

    _write_json(
        run_dir / "manifest.json",
        {
            "schema_version": "0.1.0",
            "generator_version": "msa-synth-0.1.0",
            "config_hash": "abc123",
            "generated_at": "2026-03-14T00:00:00Z",
            "run_label": run_label,
            "data_source": "msa",
            "counts": {
                "brokers": 1, "beneficial_owners": 2, "accounts": 2,
                "traders": 2, "instruments": 1, "sessions": 1,
                "orders": 2, "trades": 1, "scenarios": 2,
            },
            "scenario_types": ["generic_background", "collusive_clique"],
            "scenario_ids": ["normal", "scenario_clique_001"],
            "manipulative_order_count": 1,
            "manipulative_trade_count": 1,
        },
    )

    return run_dir


# ── Report dataclass tests ──────────────────────────────────────────


def test_validation_report_empty_ok():
    r = ValidationReport(target=Path("/tmp/x"))
    assert r.ok
    assert r.errors == []
    assert r.warnings == []


def test_validation_report_error_flips_ok():
    r = ValidationReport(target=Path("/tmp/x"))
    r.add("error", "foo", "bad thing happened")
    assert not r.ok
    assert len(r.errors) == 1


def test_validation_report_warning_stays_ok():
    r = ValidationReport(target=Path("/tmp/x"))
    r.add("warning", "foo", "iffy thing noticed")
    assert r.ok
    assert len(r.warnings) == 1


def test_validation_report_extend_prefixes_scope():
    outer = ValidationReport(target=Path("/tmp/x"))
    inner = ValidationReport(target=Path("/tmp/x/run"))
    inner.add("error", "orders.csv", "bang")
    outer.extend(inner, scope_prefix="run:R01/")
    assert outer.issues[0].scope == "run:R01/orders.csv"


def test_validation_report_to_dict_shape():
    r = ValidationReport(target=Path("/tmp/x"))
    r.add("error", "foo", "bar")
    d = r.to_dict()
    assert d["ok"] is False
    assert d["counts"]["errors"] == 1
    assert d["issues"][0]["message"] == "bar"


# ── Run validation tests ────────────────────────────────────────────


def test_validate_run_happy_path(tmp_path):
    run_dir = _build_valid_run(tmp_path)
    report = validate_run(run_dir)
    assert report.ok, "\n".join(i.format() for i in report.issues)
    assert report.errors == []


def test_validate_run_missing_dir(tmp_path):
    report = validate_run(tmp_path / "does_not_exist")
    assert not report.ok
    assert any("does not exist" in i.message for i in report.errors)


def test_validate_run_missing_manifest(tmp_path):
    run_dir = _build_valid_run(tmp_path)
    (run_dir / "manifest.json").unlink()
    report = validate_run(run_dir)
    assert not report.ok
    assert any("manifest.json" in i.scope for i in report.errors)


def test_validate_run_missing_orders(tmp_path):
    run_dir = _build_valid_run(tmp_path)
    (run_dir / "orders.csv").unlink()
    report = validate_run(run_dir)
    assert not report.ok
    assert any(i.scope == "orders.csv" for i in report.errors)


def test_validate_run_bad_manifest_count(tmp_path):
    run_dir = _build_valid_run(tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["counts"]["orders"] = 999  # actual is 2
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    report = validate_run(run_dir)
    assert not report.ok
    assert any("counts.orders" in i.scope for i in report.errors)


def test_validate_run_bad_manipulative_count(tmp_path):
    run_dir = _build_valid_run(tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["manipulative_order_count"] = 42  # actual is 1
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    report = validate_run(run_dir)
    assert not report.ok
    assert any("manipulative_order_count" in i.scope for i in report.errors)


def test_validate_run_fk_dangling_trader(tmp_path):
    run_dir = _build_valid_run(tmp_path)
    orders_path = run_dir / "orders.csv"
    text = orders_path.read_text()
    orders_path.write_text(text.replace("trader_001", "trader_GHOST"))
    report = validate_run(run_dir)
    assert not report.ok
    assert any("unknown trader_id" in i.message for i in report.errors)


def test_validate_run_bad_enum_side(tmp_path):
    run_dir = _build_valid_run(tmp_path)
    orders_path = run_dir / "orders.csv"
    text = orders_path.read_text()
    orders_path.write_text(text.replace(",buy,", ",BUY,"))
    report = validate_run(run_dir)
    assert not report.ok
    assert any("side" in i.message and "BUY" in i.message for i in report.errors)


def test_validate_run_missing_column(tmp_path):
    run_dir = _build_valid_run(tmp_path)
    # Overwrite traders.csv without a still-required column (`broker_id`).
    # The permissive v0.2.0 required-set is: trader_id, account_id,
    # beneficial_owner_id, broker_id — drop one of those to trigger.
    _write_csv(
        run_dir / "traders.csv",
        [{"trader_id": "trader_001", "account_id": "account_001",
          "beneficial_owner_id": "owner_001",
          "trader_profile_id": "noise_trader", "risk_tier": "medium",
          "region": "IN", "created_at": "2026-01-02T00:00:00",
          "status": "active"}],
        ["trader_id", "account_id", "beneficial_owner_id",
         "trader_profile_id", "risk_tier", "region", "created_at",
         "status"],  # no `broker_id`
    )
    report = validate_run(run_dir)
    assert not report.ok
    assert any("broker_id" in i.message for i in report.errors)


def test_validate_run_unknown_scenario_type_warns(tmp_path):
    run_dir = _build_valid_run(tmp_path)
    scen = run_dir / "scenarios.csv"
    text = scen.read_text()
    scen.write_text(text.replace("collusive_clique", "quantum_squeeze"))
    report = validate_run(run_dir)
    # Unknown scenario type is a warning, not an error
    assert any("quantum_squeeze" in i.message for i in report.warnings)


# ── Cohort validation tests ─────────────────────────────────────────


def test_validate_cohort_happy_path(tmp_path):
    cohort_root = tmp_path / "cohort"
    cohort_root.mkdir()
    run_dir = _build_valid_run(cohort_root, run_label="R01_msa_clique_s42_20260314")
    _write_json(
        cohort_root / "cohort_manifest.json",
        {
            "spec": {
                "cohort_name": "test_cohort",
                "families": ["clique"],
                "seeds": [42],
                "calibration_dates": ["2026-03-14"],
                "num_traders": 2,
                "manipulators_per_run": 1,
            },
            "runs": ["R01_msa_clique_s42_20260314"],
        },
    )
    report = validate_cohort(cohort_root)
    assert report.ok, "\n".join(i.format() for i in report.issues)


def test_validate_cohort_missing_run(tmp_path):
    cohort_root = tmp_path / "cohort"
    cohort_root.mkdir()
    _build_valid_run(cohort_root, run_label="R01_msa_clique_s42_20260314")
    _write_json(
        cohort_root / "cohort_manifest.json",
        {
            "spec": {"cohort_name": "c", "families": ["clique"], "seeds": [42],
                     "calibration_dates": ["2026-03-14"], "num_traders": 2,
                     "manipulators_per_run": 1},
            "runs": ["R01_msa_clique_s42_20260314", "R99_msa_missing_s99_20260101"],
        },
    )
    report = validate_cohort(cohort_root)
    assert not report.ok
    assert any("R99" in i.message and "not found" in i.message for i in report.errors)


def test_validate_cohort_loose_mode_warns(tmp_path):
    cohort_root = tmp_path / "cohort"
    cohort_root.mkdir()
    _build_valid_run(cohort_root)
    # No cohort_manifest.json — should still validate but warn
    report = validate_cohort(cohort_root)
    assert report.ok  # loose mode is a warning, not an error
    assert any("cohort_manifest.json" in i.scope and "missing" in i.message
               for i in report.warnings)


def test_validate_cohort_empty_errors(tmp_path):
    cohort_root = tmp_path / "cohort"
    cohort_root.mkdir()
    report = validate_cohort(cohort_root)
    assert not report.ok


# ── Auto-dispatch tests ────────────────────────────────────────────


def test_validate_autodispatches_to_run(tmp_path):
    run_dir = _build_valid_run(tmp_path)
    report = validate(run_dir)
    assert report.ok


def test_validate_autodispatches_to_cohort(tmp_path):
    cohort_root = tmp_path / "cohort"
    cohort_root.mkdir()
    _build_valid_run(cohort_root)
    _write_json(
        cohort_root / "cohort_manifest.json",
        {
            "spec": {"cohort_name": "c", "families": ["clique"], "seeds": [42],
                     "calibration_dates": ["2026-03-14"], "num_traders": 2,
                     "manipulators_per_run": 1},
            "runs": ["R01_msa_clique_s42_20260314"],
        },
    )
    report = validate(cohort_root)
    assert report.ok


def test_validate_bad_path_errors(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    report = validate(empty)
    assert not report.ok


# ── CLI smoke test ─────────────────────────────────────────────────


def test_cli_main_returns_zero_on_valid(tmp_path, capsys):
    from synth.validate import main
    _build_valid_run(tmp_path, run_label="R01_msa_clique_s42_20260314")
    exit_code = main([str(tmp_path)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PASS" in captured.out


def test_cli_main_returns_one_on_invalid(tmp_path, capsys):
    from synth.validate import main
    exit_code = main([str(tmp_path)])  # empty dir
    assert exit_code == 1


def test_cli_main_json_format(tmp_path, capsys):
    from synth.validate import main
    _build_valid_run(tmp_path, run_label="R01_msa_clique_s42_20260314")
    exit_code = main([str(tmp_path), "--format", "json"])
    captured = capsys.readouterr()
    assert exit_code == 0
    parsed = json.loads(captured.out)
    assert parsed["ok"] is True


def test_cli_main_strict_flags_warnings(tmp_path):
    from synth.validate import main
    # Build a loose cohort (has warning about missing cohort_manifest.json)
    _build_valid_run(tmp_path, run_label="R01_msa_clique_s42_20260314")
    exit_code_normal = main([str(tmp_path)])
    exit_code_strict = main([str(tmp_path), "--strict"])
    assert exit_code_normal == 0     # warnings don't fail without --strict
    assert exit_code_strict == 2      # warnings fail with --strict
