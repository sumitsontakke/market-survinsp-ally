"""Tests for :mod:`synth.cli`.

Uses a fake orchestrator factory to avoid running the real simulation.
This keeps tests fast (<1s) and focused on CLI logic:

* Argument parsing + subcommand dispatch
* Run-label naming convention
* Family detection from scenarios
* Cohort YAML → per-run config extraction
* Cohort manifest emission
* Error paths (missing config, empty runs, existing dir without --force)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from synth.cli import (
    build_cohort_manifest,
    build_parser,
    cmd_generate,
    cmd_validate,
    detect_family,
    extract_run_config,
    main,
    make_run_label,
)


# ── Fake orchestrator ──────────────────────────────────────────────


class _FakeOrchestrator:
    """Stand-in for SimulationOrchestrator. Records the config + writes a
    minimal-but-plausible run directory that :mod:`synth.validate` could
    at least attempt to parse."""

    calls: list[tuple[dict, int]] = []

    def __init__(self, config: dict, seed: int):
        self.config = config
        self.seed = seed
        _FakeOrchestrator.calls.append((config, seed))

    def run(self) -> dict:
        return {"orders": [], "trades": [], "traders": []}

    def export(self, out_dir: str, dataset: dict) -> dict:
        # Write a marker so we can prove export ran on the right dir.
        (Path(out_dir) / "marker.txt").write_text(f"seed={self.seed}\n")
        return {"exported_to": out_dir}


@pytest.fixture(autouse=True)
def _reset_fake_orch_calls():
    _FakeOrchestrator.calls.clear()
    yield
    _FakeOrchestrator.calls.clear()


def _fake_factory(config: dict, seed: int) -> _FakeOrchestrator:
    return _FakeOrchestrator(config, seed)


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def minimal_cohort_yaml(tmp_path: Path) -> Path:
    """Write a 2-run cohort YAML and return its path."""
    cfg = {
        "cohort": {
            "name": "test_cohort",
            "schema_version": "0.1.0",
            "generator_version": "0.1.0",
        },
        "runs": [
            {
                "run_id": "run_001_clique",
                "seed": 42,
                "wrapper": "run_manipulation",
                "session": {"trade_date": "2026-03-14"},
                "beneficial_owners": {"count": 5},
                "accounts": {"per_owner_max": 2},
                "scenarios": [
                    {"scenario_type": "collusive_clique", "participant_count": 3},
                ],
            },
            {
                "run_id": "run_002_control",
                "seed": 43,
                "wrapper": "run_generic",
                "session": {"trade_date": "2026-03-14"},
                "beneficial_owners": {"count": 5},
                "accounts": {"per_owner_max": 2},
            },
        ],
    }
    path = tmp_path / "cohort.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


# ── detect_family ─────────────────────────────────────────────────


def test_family_no_scenarios_is_benign():
    assert detect_family({}) == "benign"
    assert detect_family({"scenarios": []}) == "benign"


def test_family_only_background_is_benign():
    cfg = {"scenarios": [{"scenario_type": "generic_background"}]}
    assert detect_family(cfg) == "benign"


def test_family_clique():
    cfg = {"scenarios": [{"scenario_type": "collusive_clique"}]}
    assert detect_family(cfg) == "clique"


def test_family_ring_from_circular_trading_ring():
    cfg = {"scenarios": [{"scenario_type": "circular_trading_ring"}]}
    assert detect_family(cfg) == "ring"


def test_family_ring_from_ring_trader():
    cfg = {"scenarios": [{"scenario_type": "ring_trader"}]}
    assert detect_family(cfg) == "ring"


def test_family_front():
    cfg = {"scenarios": [{"scenario_type": "front_account"}]}
    assert detect_family(cfg) == "front"


def test_family_mixed_when_multiple_types():
    cfg = {"scenarios": [
        {"scenario_type": "collusive_clique"},
        {"scenario_type": "ring_trader"},
    ]}
    assert detect_family(cfg) == "mixed"


def test_family_unknown_type_falls_back_to_benign():
    cfg = {"scenarios": [{"scenario_type": "quantum_squeeze"}]}
    assert detect_family(cfg) == "benign"


# ── make_run_label ────────────────────────────────────────────────


def test_run_label_basic_clique():
    cfg = {
        "seed": 42, "session": {"trade_date": "2026-03-14"},
        "scenarios": [{"scenario_type": "collusive_clique"}],
    }
    assert make_run_label(1, cfg) == "R01_msa_clique_s42_20260314"


def test_run_label_zero_padded_to_two_digits():
    cfg = {"seed": 1, "session": {"trade_date": "2026-01-01"}, "scenarios": []}
    assert make_run_label(3, cfg).startswith("R03_")


def test_run_label_grows_past_99():
    cfg = {"seed": 1, "session": {"trade_date": "2026-01-01"}, "scenarios": []}
    assert make_run_label(100, cfg).startswith("R100_")


def test_run_label_custom_generator():
    cfg = {"seed": 42, "session": {"trade_date": "2026-03-14"},
           "scenarios": [{"scenario_type": "collusive_clique"}]}
    assert make_run_label(1, cfg, generator="abides") == "R01_abides_clique_s42_20260314"


def test_run_label_benign_when_no_scenarios():
    cfg = {"seed": 42, "session": {"trade_date": "2026-03-14"}}
    assert "_benign_" in make_run_label(1, cfg)


def test_run_label_handles_compact_date():
    cfg = {"seed": 42, "session": {"trade_date": "20260314"},
           "scenarios": [{"scenario_type": "collusive_clique"}]}
    assert make_run_label(1, cfg) == "R01_msa_clique_s42_20260314"


# ── extract_run_config ────────────────────────────────────────────


def test_extract_run_config_strips_meta():
    cohort = {"cohort": {}, "runs": []}
    run_entry = {"run_id": "R1", "wrapper": "run_generic", "seed": 1, "brokers": {"count": 2}}
    cfg = extract_run_config(cohort, run_entry)
    assert "run_id" not in cfg
    assert "wrapper" not in cfg
    assert cfg["seed"] == 1
    assert cfg["brokers"] == {"count": 2}


def test_extract_run_config_inherits_versions_from_cohort():
    cohort = {"cohort": {"schema_version": "0.1.0", "generator_version": "0.1.0"}, "runs": []}
    run_entry = {"seed": 1}
    cfg = extract_run_config(cohort, run_entry)
    assert cfg["schema_version"] == "0.1.0"
    assert cfg["generator_version"] == "0.1.0"


def test_extract_run_config_run_can_override_versions():
    cohort = {"cohort": {"schema_version": "0.1.0"}, "runs": []}
    run_entry = {"seed": 1, "schema_version": "0.2.0"}
    cfg = extract_run_config(cohort, run_entry)
    assert cfg["schema_version"] == "0.2.0"


# ── build_cohort_manifest ─────────────────────────────────────────


def test_cohort_manifest_shape():
    cohort_yaml = {"cohort": {"name": "demo"}}
    run_labels = ["R01_msa_clique_s42_20260314", "R02_msa_benign_s43_20260314"]
    run_configs = [
        {"seed": 42, "session": {"trade_date": "2026-03-14"},
         "beneficial_owners": {"count": 5}, "accounts": {"per_owner_max": 2},
         "scenarios": [{"scenario_type": "collusive_clique", "participant_count": 3}]},
        {"seed": 43, "session": {"trade_date": "2026-03-14"},
         "beneficial_owners": {"count": 5}, "accounts": {"per_owner_max": 2}},
    ]
    m = build_cohort_manifest(cohort_yaml, run_labels, run_configs)
    assert m["spec"]["cohort_name"] == "demo"
    assert set(m["spec"]["families"]) == {"clique", "benign"}
    assert m["spec"]["seeds"] == [42, 43]
    assert m["spec"]["calibration_dates"] == ["2026-03-14"]
    assert m["spec"]["num_traders"] == 10  # 5 owners * 2 per_owner_max
    assert m["spec"]["manipulators_per_run"] == 3
    assert m["runs"] == run_labels
    assert "generated_at" in m["spec"]


def test_cohort_manifest_defaults_cohort_name():
    m = build_cohort_manifest({}, [], [])
    assert m["spec"]["cohort_name"] == "unnamed"


# ── cmd_generate (end-to-end with fake orchestrator) ─────────────


def test_generate_writes_runs_and_manifest(minimal_cohort_yaml, tmp_path):
    out = tmp_path / "cohort_out"
    args = build_parser().parse_args([
        "generate", "--config", str(minimal_cohort_yaml), "--out", str(out),
    ])
    exit_code = cmd_generate(args, orchestrator_factory=_fake_factory)

    assert exit_code == 0
    assert (out / "cohort_manifest.json").exists()

    manifest = json.loads((out / "cohort_manifest.json").read_text())
    assert manifest["spec"]["cohort_name"] == "test_cohort"
    assert manifest["runs"] == [
        "R01_msa_clique_s42_20260314",
        "R02_msa_benign_s43_20260314",
    ]

    # Fake orchestrator was called twice (one per run), got right seeds
    assert len(_FakeOrchestrator.calls) == 2
    seeds = [seed for _, seed in _FakeOrchestrator.calls]
    assert seeds == [42, 43]

    # Marker files landed in the right run subdirs
    assert (out / "R01_msa_clique_s42_20260314" / "marker.txt").exists()
    assert (out / "R02_msa_benign_s43_20260314" / "marker.txt").exists()


def test_generate_missing_config_errors(tmp_path):
    args = build_parser().parse_args([
        "generate", "--config", str(tmp_path / "nope.yaml"),
        "--out", str(tmp_path / "out"),
    ])
    exit_code = cmd_generate(args, orchestrator_factory=_fake_factory)
    assert exit_code == 1


def test_generate_empty_runs_errors(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("cohort:\n  name: x\nruns: []\n")
    args = build_parser().parse_args([
        "generate", "--config", str(bad), "--out", str(tmp_path / "out"),
    ])
    exit_code = cmd_generate(args, orchestrator_factory=_fake_factory)
    assert exit_code == 1


def test_generate_refuses_overwrite_without_force(minimal_cohort_yaml, tmp_path):
    out = tmp_path / "cohort_out"
    # Pre-create the first run dir
    (out / "R01_msa_clique_s42_20260314").mkdir(parents=True)

    args = build_parser().parse_args([
        "generate", "--config", str(minimal_cohort_yaml), "--out", str(out),
    ])
    exit_code = cmd_generate(args, orchestrator_factory=_fake_factory)
    assert exit_code == 1


def test_generate_force_overwrites(minimal_cohort_yaml, tmp_path):
    out = tmp_path / "cohort_out"
    (out / "R01_msa_clique_s42_20260314").mkdir(parents=True)

    args = build_parser().parse_args([
        "generate", "--config", str(minimal_cohort_yaml),
        "--out", str(out), "--force",
    ])
    exit_code = cmd_generate(args, orchestrator_factory=_fake_factory)
    assert exit_code == 0


def test_generate_custom_generator_label(minimal_cohort_yaml, tmp_path):
    out = tmp_path / "cohort_out"
    args = build_parser().parse_args([
        "generate", "--config", str(minimal_cohort_yaml),
        "--out", str(out), "--generator", "abides",
    ])
    exit_code = cmd_generate(args, orchestrator_factory=_fake_factory)
    assert exit_code == 0
    manifest = json.loads((out / "cohort_manifest.json").read_text())
    assert manifest["runs"][0] == "R01_abides_clique_s42_20260314"


# ── cmd_validate (delegation to synth.validate) ──────────────────


def test_validate_delegates_and_returns_one_for_invalid(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    args = build_parser().parse_args(["validate", str(empty)])
    exit_code = cmd_validate(args)
    assert exit_code == 1


# ── main dispatch ─────────────────────────────────────────────────


def test_main_help_returns_two_on_no_subcommand(capsys):
    # argparse with `required=True` on subparsers exits 2 without a subcommand
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_main_dispatches_validate(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    exit_code = main(["validate", str(empty)])
    assert exit_code == 1  # empty dir → invalid
