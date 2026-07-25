from __future__ import annotations

from synthetic_market_sim.simulation.orchestrator import SimulationOrchestrator
from synthetic_market_sim.utils.config import load_config
from synthetic_market_sim.utils.seed import build_rng


def test_generic_run_creates_linked_dataset(tmp_path) -> None:
    config = load_config("tests/fixtures/small_config.yaml")
    orchestrator = SimulationOrchestrator(config=config, rng=build_rng(config["seed"]))
    result = orchestrator.export(str(tmp_path))
    assert (tmp_path / "orders.csv").exists()
    assert (tmp_path / "traders.csv").exists()
    assert (tmp_path / "manifest.json").exists()
    assert result["manifest"].endswith("manifest.json")
