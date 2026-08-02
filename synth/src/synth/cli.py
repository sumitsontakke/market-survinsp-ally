"""Command-line entry point for the ``synth`` module.

Two subcommands:

* ``synth generate --config <cohort.yaml> --out <cohort_dir>`` — read a
  cohort spec, invoke :class:`SimulationOrchestrator` for each run, and
  emit a SCHEMA.md-conformant cohort directory (one subdirectory per
  run, plus a top-level ``cohort_manifest.json``).

* ``synth validate <path> [--strict] [--format {human,json}]`` — delegate
  to :func:`synth.validate.main`. Thin wrapper so users don't have to
  remember two module paths.

The ``generate`` subcommand is intentionally a thin orchestrator that
reuses the existing per-run wrappers' implementation
(``SimulationOrchestrator``). It adds cohort-level concerns on top:

* Run-label naming (``R<NN>_<generator>_<family>_s<seed>_<yyyymmdd>``)
* Family detection from the scenarios present in each run
* Cohort-level manifest emission
* Deterministic ordering (runs generated in the order the YAML lists them)

Design constraints:

* No new dependencies. Uses the stdlib ``argparse`` and ``yaml`` (already
  a synth dependency).
* Orchestrator factory is injectable so tests can substitute a fake and
  avoid running actual simulations.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

__all__ = [
    "main",
    "cmd_generate",
    "cmd_validate",
    "make_run_label",
    "detect_family",
    "extract_run_config",
    "build_cohort_manifest",
]


# ── Family detection (SCHEMA.md § Scenario-type vocabulary) ────────────


_FAMILY_MAP: dict[str, str] = {
    # scenario_type as it appears in configs and scenarios.csv → high-level family
    "collusive_clique":  "clique",
    "circular_trading_ring": "ring",
    "ring_trader":       "ring",
    "front_account":     "front",
    "generic_background": "benign",
}


def detect_family(run_config: dict) -> str:
    """Infer the high-level manipulation family for a run.

    Returns one of ``"clique"``, ``"ring"``, ``"front"``, ``"mixed"``,
    or ``"benign"``.

    * No ``scenarios`` block or empty scenarios → ``"benign"``.
    * One manipulative scenario type → its family name.
    * Two or more distinct manipulative scenario types → ``"mixed"``.
    """
    scenarios = run_config.get("scenarios") or []
    if not scenarios:
        return "benign"

    families = {
        _FAMILY_MAP.get(s.get("scenario_type"), "unknown")
        for s in scenarios
        if s.get("scenario_type") and s.get("scenario_type") != "generic_background"
    }
    families.discard("unknown")

    if not families:
        return "benign"
    if len(families) == 1:
        return next(iter(families))
    return "mixed"


# ── Run-label naming (SCHEMA.md § Cohort filesystem layout) ────────────


def make_run_label(
    index: int,
    run_config: dict,
    generator: str = "msa",
) -> str:
    """Build the canonical run directory name.

    Format: ``R<NN>_<generator>_<family>_s<seed>_<yyyymmdd>``

    ``NN`` is zero-padded to 2 digits (matches every cohort in the
    reference data). If ``index`` exceeds 99 it grows naturally
    (``R100_...``) — no truncation.
    """
    seed = int(run_config.get("seed", 0))
    family = detect_family(run_config)
    trade_date = run_config.get("session", {}).get("trade_date", "19700101")
    # Accept both "2026-03-14" and "20260314" input
    date_compact = str(trade_date).replace("-", "")
    return f"R{index:02d}_{generator}_{family}_s{seed}_{date_compact}"


# ── Cohort YAML → per-run config extraction ────────────────────────────


# Keys we strip when building the per-run config passed to
# SimulationOrchestrator (they are cohort/runtime metadata, not sim inputs).
_RUN_META_KEYS: frozenset[str] = frozenset({"run_id", "wrapper"})


def extract_run_config(cohort_yaml: dict, run_entry: dict) -> dict:
    """Build the config dict handed to SimulationOrchestrator for one run.

    Inherits ``schema_version`` and ``generator_version`` from the
    top-level ``cohort:`` block if the run doesn't set them explicitly.
    Strips cohort-only metadata keys.
    """
    cohort_block = cohort_yaml.get("cohort", {})
    run_config: dict[str, Any] = {}

    # Inherit from cohort block
    for k in ("schema_version", "generator_version"):
        if k in cohort_block:
            run_config[k] = cohort_block[k]

    # Copy run fields, stripping meta
    for k, v in run_entry.items():
        if k not in _RUN_META_KEYS:
            run_config[k] = v

    return run_config


# ── Cohort manifest emission (SCHEMA.md § Cohort manifest) ─────────────


def build_cohort_manifest(
    cohort_yaml: dict,
    run_labels: list[str],
    run_configs: list[dict],
) -> dict:
    """Assemble the cohort_manifest.json body from the cohort YAML.

    Fields per SCHEMA.md § "Cohort manifest":

    * ``spec.cohort_name`` — from ``cohort.name``, else ``"unnamed"``.
    * ``spec.families`` — union of detected families across runs.
    * ``spec.seeds`` — sorted union of run seeds.
    * ``spec.calibration_dates`` — sorted union of run trade_dates.
    * ``spec.num_traders`` — inferred from first run (or 0 if unknowable
      at spec time; the true per-run count is authoritatively in each
      run's ``manifest.json#counts.traders``).
    * ``spec.manipulators_per_run`` — sum of scenario ``participant_count``
      across manipulative scenarios, on the first non-benign run.
    * ``runs`` — the list of run directory names, in generation order.
    """
    cohort_block = cohort_yaml.get("cohort", {})
    cohort_name = cohort_block.get("name", "unnamed")

    families = sorted({detect_family(rc) for rc in run_configs})
    seeds = sorted({int(rc.get("seed", 0)) for rc in run_configs})
    calibration_dates = sorted({
        str(rc.get("session", {}).get("trade_date", ""))
        for rc in run_configs
        if rc.get("session", {}).get("trade_date")
    })

    # num_traders: infer from first run's trader profiles + beneficial_owners
    # Real count depends on per_owner_min/max; report the spec-level target
    # if present, else 0.
    num_traders = 0
    if run_configs:
        first = run_configs[0]
        owners = first.get("beneficial_owners", {}).get("count", 0)
        per_owner_max = first.get("accounts", {}).get("per_owner_max", 1)
        # Ceiling estimate: owners * max_accounts * 1 (trader-per-account)
        num_traders = int(owners) * int(per_owner_max)

    # manipulators_per_run: sum participant_count on non-benign runs
    manipulators_per_run = 0
    for rc in run_configs:
        if detect_family(rc) == "benign":
            continue
        total = sum(
            int(s.get("participant_count", 0))
            for s in (rc.get("scenarios") or [])
        )
        if total:
            manipulators_per_run = total
            break

    return {
        "spec": {
            "cohort_name": cohort_name,
            "families": families,
            "seeds": seeds,
            "calibration_dates": calibration_dates,
            "num_traders": num_traders,
            "manipulators_per_run": manipulators_per_run,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "runs": list(run_labels),
    }


# ── generate subcommand ────────────────────────────────────────────────


# Type alias for the orchestrator factory (kept injectable for tests).
OrchestratorFactory = Callable[[dict, int], Any]


def _default_orchestrator_factory(config: dict, seed: int) -> Any:
    """Import and construct the real SimulationOrchestrator.

    Imported lazily inside the function so unit tests that inject a fake
    factory don't pay the import cost (and don't accidentally exercise
    the real simulator).
    """
    from synth.generator.simulation.orchestrator import SimulationOrchestrator
    from synth.generator.utils.seed import build_rng
    return SimulationOrchestrator(config=config, rng=build_rng(seed))


def cmd_generate(
    args: argparse.Namespace,
    orchestrator_factory: OrchestratorFactory | None = None,
) -> int:
    """Execute ``synth generate``.

    ``orchestrator_factory`` defaults to the real
    :class:`SimulationOrchestrator`. Tests inject a fake here to avoid
    running actual simulations.
    """
    factory = orchestrator_factory or _default_orchestrator_factory

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"error: cohort config not found: {config_path}", file=sys.stderr)
        return 1

    with config_path.open("r", encoding="utf-8") as h:
        cohort_yaml = yaml.safe_load(h) or {}

    runs = cohort_yaml.get("runs")
    if not isinstance(runs, list) or not runs:
        print(
            "error: cohort config must contain a non-empty 'runs' list",
            file=sys.stderr,
        )
        return 1

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    run_labels: list[str] = []
    run_configs: list[dict] = []

    for idx, run_entry in enumerate(runs, start=1):
        run_config = extract_run_config(cohort_yaml, run_entry)
        run_label = make_run_label(idx, run_config, generator=args.generator)
        run_dir = out_root / run_label

        if run_dir.exists() and not args.force:
            print(
                f"error: run directory already exists: {run_dir} "
                f"(pass --force to overwrite)",
                file=sys.stderr,
            )
            return 1
        run_dir.mkdir(parents=True, exist_ok=True)

        if args.verbose:
            print(f"[generate] run {idx}/{len(runs)}: {run_label}", flush=True)

        seed = int(run_config.get("seed", 0))
        orchestrator = factory(run_config, seed)
        dataset = orchestrator.run()
        orchestrator.export(str(run_dir), dataset=dataset)

        run_labels.append(run_label)
        run_configs.append(run_config)

    manifest = build_cohort_manifest(cohort_yaml, run_labels, run_configs)
    (out_root / "cohort_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    if args.verbose:
        print(f"[generate] wrote cohort_manifest.json ({len(run_labels)} runs)")
    print(f"OK: {len(run_labels)} runs generated at {out_root}")
    return 0


# ── validate subcommand (thin wrapper) ─────────────────────────────────


def cmd_validate(args: argparse.Namespace) -> int:
    """Delegate to :func:`synth.validate.main`."""
    from synth.validate import main as validate_main
    forwarded = [args.path]
    if args.strict:
        forwarded.append("--strict")
    if args.format != "human":
        forwarded.extend(["--format", args.format])
    return validate_main(forwarded)


# ── Argument parsing + main ───────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synth",
        description="Generate and validate synthetic-market cohorts.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # generate
    gen = sub.add_parser(
        "generate",
        help="Generate a cohort from a cohort YAML spec.",
        description=(
            "Read a cohort YAML, invoke SimulationOrchestrator per run, "
            "emit SCHEMA.md-conformant output."
        ),
    )
    gen.add_argument("--config", required=True, help="Path to cohort YAML.")
    gen.add_argument("--out", required=True, help="Cohort output directory.")
    gen.add_argument(
        "--generator", default="msa",
        help="Generator identifier used in run labels (default: msa).",
    )
    gen.add_argument(
        "--force", action="store_true",
        help="Overwrite existing run directories.",
    )
    gen.add_argument("--verbose", action="store_true", help="Verbose logging.")

    # validate
    val = sub.add_parser(
        "validate",
        help="Validate a cohort or single run against SCHEMA.md.",
        description="Delegates to `python -m synth.validate`.",
    )
    val.add_argument("path", help="Cohort root or single run directory.")
    val.add_argument("--strict", action="store_true", help="Fail on warnings.")
    val.add_argument(
        "--format", choices=("human", "json"), default="human",
        help="Output format (default: human).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "generate":
        return cmd_generate(args)
    if args.cmd == "validate":
        return cmd_validate(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
