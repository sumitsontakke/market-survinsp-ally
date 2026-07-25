"""Phase D STUB: orchestrate a cohort of ABIDES runs equivalent to MSA's R01-R24.

Produces, per cohort spec:
  - clique runs (R01-R08 equivalents)
  - ring runs (R09-R16 equivalents)
  - mixed runs (R17-R24 equivalents)
  - benign-only runs (Phase 3 false-alarm cohort)
  - scaled runs (5000 traders, 0.1% manipulators — Phase 3 scale cohort)
  - temporal-shift runs (multiple calibration dates)

For each run:
  1. Resolve calibration params for the date.
  2. Build manipulator config JSON (which cliques/rings/fronts to inject).
  3. Invoke ABIDES via configs/msa_rmsc.py (subprocess).
  4. Run schema adapter on the ABIDES output dir, writing to
     <msa_runs_root>/R{NN}_abides_{family}_{params}_s{seed}/
  5. Emit run-level manifest with config_hash.

Status: orchestration plumbing stubbed; actual ABIDES invocation depends on
Phase C/D completion.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

SERVICE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVICE_DIR))

from src.adapters.exchange_log_to_msa import adapt_abides_run_to_msa  # noqa: E402


@dataclass
class CohortSpec:
    """Declarative cohort definition. One CohortSpec → multiple runs."""

    cohort_name: str                # e.g. "abides_R01_24"
    families: list[str]             # subset of {"clique","ring","mixed","benign"}
    seeds: list[int]                # one run per (family, seed) combination
    calibration_dates: list[str]    # ISO dates; one run per (family, seed, date)
    num_traders: int = 500
    manipulators_per_run: int = 100
    out_root: Path = Path("/srv/output/cohorts")


def _manipulator_config_for(family: str, manipulators: int) -> dict:
    """Build the JSON config the rmsc-derived script consumes."""
    if family == "clique":
        return {
            "cliques": [{"size": manipulators, "target_pct_move": 0.005, "num_actions": 3}],
            "rings": [],
            "fronts": [],
        }
    if family == "ring":
        return {
            "cliques": [],
            "rings": [{"size": manipulators, "rotation_window_ms": 60_000}],
            "fronts": [],
        }
    if family == "mixed":
        half = manipulators // 2
        return {
            "cliques": [{"size": half, "target_pct_move": 0.005, "num_actions": 2}],
            "rings": [{"size": manipulators - half, "rotation_window_ms": 60_000}],
            "fronts": [{"size": max(5, manipulators // 10), "subscribes_to": "clique_0"}],
        }
    if family == "benign":
        return {"cliques": [], "rings": [], "fronts": []}
    raise ValueError(f"unknown family: {family}")


def _run_label(family: str, seed: int, date: str, idx: int) -> str:
    """Mirror MSA naming: R<NN>_abides_<family>_s<seed>_<date>."""
    return f"R{idx:02d}_abides_{family}_s{seed}_{date.replace('-', '')}"


def execute_cohort(spec: CohortSpec) -> list[Path]:
    """Run every (family, seed, date) combo, return list of produced run dirs."""
    spec.out_root.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []
    idx = 0
    for family in spec.families:
        for date in spec.calibration_dates:
            for seed in spec.seeds:
                idx += 1
                run_label = _run_label(family, seed, date, idx)
                manip_cfg = _manipulator_config_for(family, spec.manipulators_per_run)

                manip_cfg_path = spec.out_root / f"{run_label}_manipulator.json"
                manip_cfg_path.write_text(json.dumps(manip_cfg, indent=2))

                abides_out = spec.out_root / f"{run_label}_abides_raw"
                msa_out = spec.out_root / run_label

                _invoke_abides(
                    seed=seed,
                    calibration_date=date,
                    num_traders=spec.num_traders,
                    manipulator_config=manip_cfg_path,
                    out_dir=abides_out,
                )
                adapt_abides_run_to_msa(
                    abides_run_dir=abides_out,
                    out_dir=msa_out,
                    run_label=run_label,
                )
                produced.append(msa_out)
    (spec.out_root / "cohort_manifest.json").write_text(
        json.dumps({"spec": _serialise_spec(spec), "runs": [p.name for p in produced]}, indent=2)
    )
    return produced


def _invoke_abides(
    seed: int,
    calibration_date: str,
    num_traders: int,
    manipulator_config: Path,
    out_dir: Path,
    end_time: str = "10:30:00",
) -> None:
    """Spawn ``configs/msa_rmsc.py`` as a subprocess.

    Isolating each run in its own subprocess keeps ABIDES' global numpy
    seed clean between runs and means a single bad scenario can't crash
    the whole cohort. The child writes its trade tape, orders, and
    manipulator-labels sidecar into ``out_dir`` (the adapter consumes the
    same directory afterward).
    """
    config_script = Path(__file__).resolve().parent / "configs" / "msa_rmsc.py"
    cmd = [
        sys.executable, str(config_script),
        "--seed", str(seed),
        "--calibration-date", str(calibration_date),
        "--num-traders", str(num_traders),
        "--manipulator-config", str(manipulator_config),
        "--out-dir", str(out_dir),
        "--end-time", str(end_time),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"msa_rmsc.py exited non-zero for seed={seed} date={calibration_date}.\n"
            f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )
    # Forward the child's stdout so cohort logs show per-run trade counts.
    if res.stdout.strip():
        for line in res.stdout.splitlines():
            print(f"  [seed={seed}] {line}")


def _serialise_spec(spec: CohortSpec) -> dict:
    d = asdict(spec)
    d["out_root"] = str(spec.out_root)
    return d


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run an ABIDES cohort.")
    p.add_argument("--cohort-name", required=True)
    p.add_argument("--families", nargs="+", default=["clique", "ring", "mixed"])
    p.add_argument("--seeds", nargs="+", type=int, default=[11, 13, 17])
    p.add_argument("--calibration-dates", nargs="+", default=["2026-03-21"])
    p.add_argument("--num-traders", type=int, default=500)
    p.add_argument("--manipulators-per-run", type=int, default=100)
    p.add_argument("--out-root", type=Path, default=Path("/srv/output/cohorts"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    spec = CohortSpec(
        cohort_name=args.cohort_name,
        families=args.families,
        seeds=args.seeds,
        calibration_dates=args.calibration_dates,
        num_traders=args.num_traders,
        manipulators_per_run=args.manipulators_per_run,
        out_root=args.out_root / args.cohort_name,
    )
    runs = execute_cohort(spec)
    print(f"Produced {len(runs)} runs under {spec.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
