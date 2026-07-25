from __future__ import annotations

import json

import pandas as pd

from synthetic_market_sim.simulation.orchestrator import SimulationOrchestrator
from synthetic_market_sim.utils.config import load_config
from synthetic_market_sim.utils.seed import build_rng
from synthetic_market_sim.wrappers.run_analysis import main


def test_analysis_cli_creates_artifacts_and_finds_scenario_overlap(tmp_path, monkeypatch, capsys) -> None:
    config = load_config("tests/fixtures/manipulation_config.yaml")
    dataset_dir = tmp_path / "dataset"
    analysis_dir = tmp_path / "analysis"
    orchestrator = SimulationOrchestrator(config=config, rng=build_rng(config["seed"]))
    orchestrator.export(str(dataset_dir))

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_analysis.py",
            "--input",
            str(dataset_dir),
            "--source",
            "orders",
            "--bucket-minutes",
            "1",
            "--corr-method",
            "pearson",
            "--corr-threshold",
            "0.7",
            "--output",
            str(analysis_dir),
        ],
    )
    main()
    summary = json.loads(capsys.readouterr().out)

    assert summary["traders_analyzed"] > 0
    assert summary["suspicious_group_count"] > 0
    assert (analysis_dir / "signed_volume_matrix.csv").exists()
    assert (analysis_dir / "correlation_matrix.csv").exists()
    assert (analysis_dir / "correlation_heatmap.png").exists()
    assert (analysis_dir / "correlation_edges.csv").exists()
    assert (analysis_dir / "directed_trade_edges.csv").exists()
    assert (analysis_dir / "suspicious_groups.csv").exists()
    assert (analysis_dir / "detection_evaluation.json").exists()
    assert (analysis_dir / "confusion_matrix.json").exists()
    assert (analysis_dir / "confusion_matrix.csv").exists()
    assert (analysis_dir / "confusion_matrix.png").exists()
    assert (analysis_dir / "network_graph.png").exists()
    assert (analysis_dir / "network_graph_highlighted.png").exists()
    assert (analysis_dir / "network_graph_clique_context.png").exists()
    assert (analysis_dir / "signed_volume_timeseries.png").exists()
    assert (analysis_dir / "top_suspicious_group_timeseries.png").exists()
    assert (analysis_dir / "analysis_summary.md").exists()

    suspicious_groups = pd.read_csv(analysis_dir / "suspicious_groups.csv")
    assert not suspicious_groups.empty
    assert suspicious_groups["has_scenario_participants"].astype(str).str.lower().eq("true").any()
    assert suspicious_groups["dominant_scenario"].fillna("").ne("").any()
    assert suspicious_groups["coverage"].max() > 0.5
    assert suspicious_groups["purity"].max() > 0.0
    assert "unmatched_flagged_count" in suspicious_groups.columns

    detection_evaluation = json.loads((analysis_dir / "detection_evaluation.json").read_text())
    assert detection_evaluation["has_ground_truth"] is True
    assert detection_evaluation["true_positive_count"] >= 1
    assert detection_evaluation["recall"] > 0.0

    confusion_matrix = pd.read_csv(analysis_dir / "confusion_matrix.csv")
    assert set(confusion_matrix["metric"]) >= {"TP", "FP", "FN", "TN", "precision", "recall", "f1_score"}

    summary_text = (analysis_dir / "analysis_summary.md").read_text()
    assert "## Trader-Level Confusion Matrix" in summary_text
