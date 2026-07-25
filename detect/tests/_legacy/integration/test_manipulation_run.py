from __future__ import annotations

import json

from synthetic_market_sim.simulation.orchestrator import SimulationOrchestrator
from synthetic_market_sim.utils.analysis import pairwise_correlations, signed_volume_by_bucket
from synthetic_market_sim.utils.config import load_config
from synthetic_market_sim.utils.seed import build_rng
from synthetic_market_sim.wrappers.run_manipulation import main
from synthetic_market_sim.wrappers.validate_output import validate_dataset


def test_manipulation_wrapper_runs_and_prints_summary(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_manipulation.py",
            "--config",
            "tests/fixtures/manipulation_config.yaml",
            "--output-dir",
            str(tmp_path),
        ],
    )
    main()
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["manipulative_orders"] > 0
    assert summary["scenario_ids"] == ["scenario_001"]
    assert (tmp_path / "orders.csv").exists()


def test_manipulative_output_valid_and_detectably_correlated(tmp_path) -> None:
    config = load_config("tests/fixtures/manipulation_config.yaml")
    orchestrator = SimulationOrchestrator(config=config, rng=build_rng(config["seed"]))
    dataset = orchestrator.run()
    orchestrator.export(str(tmp_path), dataset=dataset)

    issues = validate_dataset(tmp_path)
    assert issues == []

    manipulative_orders = [row for row in dataset["orders"] if row["is_manipulative"]]
    manipulative_trades = [row for row in dataset["trades"] if row["is_manipulative"]]
    assert manipulative_orders
    assert manipulative_trades

    scenario = next(row for row in dataset["scenarios"] if row["scenario_id"] == "scenario_001")
    participant_ids = list(scenario["participant_ids"])
    correlations = pairwise_correlations(signed_volume_by_bucket(manipulative_orders, participant_ids))
    assert correlations
    assert correlations[0][2] > 0.6
