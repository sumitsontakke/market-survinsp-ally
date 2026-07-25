from __future__ import annotations

import json

from synthetic_market_sim.simulation.orchestrator import SimulationOrchestrator
from synthetic_market_sim.utils.config import load_config
from synthetic_market_sim.utils.seed import build_rng


def test_manifest_contains_expected_metadata(tmp_path) -> None:
    config = load_config("tests/fixtures/small_config.yaml")
    orchestrator = SimulationOrchestrator(config=config, rng=build_rng(config["seed"]))
    orchestrator.export(str(tmp_path))
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "0.1.0"
    assert manifest["counts"]["orders"] >= 0
    assert "config_hash" in manifest
