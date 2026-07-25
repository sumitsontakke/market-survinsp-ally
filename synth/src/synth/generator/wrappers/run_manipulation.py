from __future__ import annotations

import argparse
import json

from synth.generator.simulation.orchestrator import SimulationOrchestrator
from synth.generator.utils.config import load_config
from synth.generator.utils.logging import configure_logging
from synth.generator.utils.seed import build_rng


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a manipulative synthetic market simulation.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--output-dir", required=True, help="Output directory for generated files.")
    parser.add_argument("--seed", type=int, help="Optional seed override.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def summarize_dataset(dataset: dict[str, list[dict]]) -> dict[str, object]:
    scenario_ids = sorted({row["scenario_id"] for row in dataset.get("scenarios", []) if row["scenario_id"] != "normal"})
    return {
        "counts": {name: len(rows) for name, rows in dataset.items()},
        "manipulative_orders": sum(1 for row in dataset.get("orders", []) if row.get("is_manipulative")),
        "manipulative_trades": sum(1 for row in dataset.get("trades", []) if row.get("is_manipulative")),
        "scenario_ids": scenario_ids,
    }


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.verbose)
    config = load_config(args.config)
    seed = args.seed if args.seed is not None else int(config["seed"])
    config["seed"] = seed
    if not config.get("scenarios"):
        raise ValueError("Manipulation run requires at least one scenario in config.")
    orchestrator = SimulationOrchestrator(config=config, rng=build_rng(seed))
    dataset = orchestrator.run()
    orchestrator.export(args.output_dir, dataset=dataset)
    print(json.dumps(summarize_dataset(dataset), indent=2))


if __name__ == "__main__":
    main()
