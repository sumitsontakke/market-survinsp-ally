from __future__ import annotations

import argparse
import json
from pathlib import Path

from synth.generator.simulation.orchestrator import SimulationOrchestrator
from synth.generator.utils.config import load_config
from synth.generator.utils.seed import build_rng


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a batch of generic synthetic market simulations.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--output-root", required=True, help="Root directory for batch outputs.")
    parser.add_argument("--seeds", nargs="+", required=True, type=int, help="Seed list.")
    args = parser.parse_args()

    base_config = load_config(args.config)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in args.seeds:
        config = dict(base_config)
        config["seed"] = seed
        orchestrator = SimulationOrchestrator(config=config, rng=build_rng(seed))
        run_dir = output_root / f"seed_{seed}"
        results.append(orchestrator.export(str(run_dir)))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
