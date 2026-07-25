from __future__ import annotations

import argparse
import json

from synth.generator.simulation.orchestrator import SimulationOrchestrator
from synth.generator.utils.config import load_config
from synth.generator.utils.logging import configure_logging
from synth.generator.utils.seed import build_rng


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a generic synthetic market simulation.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--output-dir", required=True, help="Output directory for generated files.")
    parser.add_argument("--seed", type=int, help="Optional seed override.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.verbose)
    config = load_config(args.config)
    seed = args.seed if args.seed is not None else int(config["seed"])
    config["seed"] = seed
    orchestrator = SimulationOrchestrator(config=config, rng=build_rng(seed))
    result = orchestrator.export(args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
