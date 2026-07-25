from __future__ import annotations

import json

import pandas as pd

from synthetic_market_sim.analysis.pipeline import run_analysis
from synthetic_market_sim.simulation.orchestrator import SimulationOrchestrator
from synthetic_market_sim.utils.config import load_config
from synthetic_market_sim.utils.seed import build_rng
from synthetic_market_sim.wrappers.run_manipulation import main
from synthetic_market_sim.wrappers.validate_output import validate_dataset


def test_ring_manipulation_wrapper_runs_and_prints_summary(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_manipulation.py",
            "--config",
            "tests/fixtures/ring_config.yaml",
            "--output-dir",
            str(tmp_path),
        ],
    )
    main()
    summary = json.loads(capsys.readouterr().out)
    assert summary["manipulative_orders"] > 0
    assert summary["scenario_ids"] == ["scenario_002"]


def test_ring_output_validates_and_shows_repeated_ring_edges(tmp_path) -> None:
    config = load_config("tests/fixtures/ring_config.yaml")
    dataset_dir = tmp_path / "dataset"
    analysis_dir = tmp_path / "analysis"
    orchestrator = SimulationOrchestrator(config=config, rng=build_rng(config["seed"]))
    dataset = orchestrator.run()
    orchestrator.export(str(dataset_dir), dataset=dataset)

    issues = validate_dataset(dataset_dir)
    assert issues == []

    scenario = next(row for row in dataset["scenarios"] if row["scenario_id"] == "scenario_002")
    ring_order = list(scenario["ring_order"])
    expected_edges = {
        (ring_order[index], ring_order[(index + 1) % len(ring_order)])
        for index in range(len(ring_order))
    }
    manipulative_trades = [row for row in dataset["trades"] if row["scenario_id"] == "scenario_002"]
    assert manipulative_trades

    ring_edge_count = 0
    edge_counts: dict[tuple[str, str], int] = {}
    observed_ring_edges: set[tuple[str, str]] = set()
    for trade in manipulative_trades:
        edge = (trade["sell_trader_id"], trade["buy_trader_id"])
        edge_counts[edge] = edge_counts.get(edge, 0) + 1
        if edge in expected_edges:
            ring_edge_count += 1
            observed_ring_edges.add(edge)

    assert ring_edge_count >= len(ring_order)
    assert observed_ring_edges == expected_edges
    top_edges = sorted(edge_counts.items(), key=lambda item: item[1], reverse=True)[: len(ring_order)]
    assert sum(1 for edge, _ in top_edges if edge in expected_edges) >= len(ring_order) - 1

    run_analysis(
        input_dir=str(dataset_dir),
        output_dir=str(analysis_dir),
        source="orders",
        bucket_minutes=1,
        corr_method="pearson",
        corr_threshold=0.6,
        min_active_buckets=1,
    )
    directed_trade_edges = pd.read_csv(analysis_dir / "directed_trade_edges.csv")
    observed_directed_edges = {
        (row["seller"], row["buyer"])
        for row in directed_trade_edges.to_dict("records")
        if "scenario_002" in str(row["scenario_ids"])
    }
    assert expected_edges & observed_directed_edges
