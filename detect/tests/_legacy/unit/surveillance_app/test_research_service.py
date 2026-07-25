from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from surveillance_app.services.research_service import ResearchService
from surveillance_app.viewmodels.run_summary import RunSummary


def test_research_service_loads_phase_snapshots_and_handles_missing_files(tmp_path: Path) -> None:
    (tmp_path / "outputs" / "ml").mkdir(parents=True)

    pd.DataFrame(
        [
            {"run_id": "R01", "scenario_type": "collusive_clique", "concealment": "low", "instrument_scope": "single", "precision": 1.0, "recall": 1.0, "F1": 1.0, "retained_edges_count": 5, "failure_type": "SUCCESS"},
            {"run_id": "R02", "scenario_type": "circular_trading_ring", "concealment": "high", "instrument_scope": "multi", "precision": 0.0, "recall": 0.0, "F1": 0.0, "retained_edges_count": 0, "failure_type": "NO_GRAPH_FORMED"},
        ]
    ).to_csv(tmp_path / "experiment_results.csv", index=False)
    pd.DataFrame(
        [
            {"run_id": "R01", "scenario_type": "collusive_clique", "concealment": "low", "instrument_scope": "single", "injected_core_count": 6},
            {"run_id": "R02", "scenario_type": "circular_trading_ring", "concealment": "high", "instrument_scope": "multi", "injected_core_count": 8},
        ]
    ).to_csv(tmp_path / "experiment_runs.csv", index=False)
    pd.DataFrame(
        [
            {"run_id": "R01", "scenario_type": "collusive_clique", "concealment": "low", "instrument_scope": "single", "precision": 1.0, "recall": 1.0, "F1": 1.0, "retained_edges_count": 5},
            {"run_id": "R02", "scenario_type": "circular_trading_ring", "concealment": "high", "instrument_scope": "multi", "precision": 0.0, "recall": 0.0, "F1": 0.0, "retained_edges_count": 0},
        ]
    ).to_csv(tmp_path / "combined_analysis.csv", index=False)
    pd.DataFrame(
        [
            {"scenario_type": "collusive_clique", "intensity": "high", "concealment": "low", "instrument_scope": "single", "run_count": 1, "avg_precision": 1.0, "avg_recall": 1.0, "avg_F1": 1.0, "avg_coverage": 1.0, "avg_purity": 1.0, "avg_retained_edges_count": 5, "avg_connected_components_count": 1, "avg_number_of_detected_groups": 1, "%_SUCCESS": 100.0, "%_NO_GRAPH_FORMED": 0.0, "%_PARTIAL_DETECTION": 0.0, "%_GRAPH_FRAGMENTATION": 0.0}
        ]
    ).to_csv(tmp_path / "cross_run_analysis.csv", index=False)
    pd.DataFrame(
        [
            {"run_id": "R01", "scenario_family": "collusive_clique", "group_id": "g1", "group_type": "scc", "group_size": 3, "generation_score": 10.0, "label_loose": 1},
            {"run_id": "R02", "scenario_family": "circular_trading_ring", "group_id": "g2", "group_type": "cycle", "group_size": 4, "generation_score": 9.0, "label_loose": 0},
        ]
    ).to_csv(tmp_path / "candidate_groups.csv", index=False)
    pd.DataFrame(
        [
            {"run_id": "R01", "scenario_family": "collusive_clique", "group_id": "g1", "group_type": "scc", "group_size": 3, "label_loose": 1, "label_partial": 0, "label_strict": 0, "label_high_confidence": 0},
            {"run_id": "R02", "scenario_family": "circular_trading_ring", "group_id": "g2", "group_type": "cycle", "group_size": 4, "label_loose": 0, "label_partial": 1, "label_strict": 0, "label_high_confidence": 0},
        ]
    ).to_csv(tmp_path / "ml_dataset_groups.csv", index=False)
    pd.DataFrame(
        [
            {"run_id": "R01", "scenario_family": "collusive_clique", "group_id": "g1", "group_type": "scc", "group_size": 3, "label_loose": 1, "label_partial": 0, "label_strict": 0, "label_high_confidence": 0},
        ]
    ).to_csv(tmp_path / "ml_dataset_groups_final.csv", index=False)
    pd.DataFrame([{"run_id": "R01", "trader_id": "t1"}]).to_csv(tmp_path / "ml_dataset_traders.csv", index=False)
    pd.DataFrame([{"run_id": "R01", "trader_id": "t1"}]).to_csv(tmp_path / "ml_dataset_traders_final.csv", index=False)
    (tmp_path / "dataset_refinement_report.md").write_text(
        "- Group rows after deduplication: `1`\n- Group rows after balancing: `1`\n- Dropped group count during deduplication: `1`\n## Label Redesign\nWeak supervision.\n",
        encoding="utf-8",
    )
    (tmp_path / "dataset_quality_report.md").write_text("## Leakage Risk Check\nNo leakage.\n", encoding="utf-8")
    (tmp_path / "research_findings.md").write_text("## Key Observations\nPhase 1 failed often.\n", encoding="utf-8")
    (tmp_path / "combined_analysis_summary.md").write_text("## High-level Findings\nRetained edges matter.\n", encoding="utf-8")
    (tmp_path / "feature_schema_final.json").write_text(
        json.dumps({"datasets": {"ml_dataset_groups_final.csv": [{"name": "group_size", "role": "feature"}], "ml_dataset_traders_final.csv": [{"name": "trader_id", "role": "metadata"}]}}),
        encoding="utf-8",
    )
    (tmp_path / "feature_schema.json").write_text(json.dumps({"datasets": {}}), encoding="utf-8")
    (tmp_path / "outputs" / "ml" / "group_model_metrics.json").write_text(
        json.dumps(
            {
                "test_metrics": {"accuracy": 0.8, "precision": 0.75, "recall": 0.7, "f1": 0.72, "roc_auc": 0.9},
                "baseline_test_metrics": {"f1": 0.0},
                "split": {"train_rows": 10, "test_rows": 2, "train_runs": ["R01"], "test_runs": ["R02"]},
                "model": {"selected_model": "RandomForestClassifier"},
                "test_scenario_family_metrics": {"collusive_clique": {"rows": 1, "precision": 1.0, "recall": 1.0, "f1": 1.0, "roc_auc": 1.0}},
                "top_features": [{"feature": "max_in_degree", "importance": 0.1}],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame([{"Unnamed: 0": "actual_0", "pred_0": 1, "pred_1": 0}, {"Unnamed: 0": "actual_1", "pred_0": 0, "pred_1": 1}]).to_csv(
        tmp_path / "outputs" / "ml" / "group_confusion_matrix.csv",
        index=False,
    )
    pd.DataFrame([{"run_id": "R02", "scenario_family": "collusive_clique", "group_id": "g1", "group_type": "scc", "label_loose": 1, "prediction": 1, "probability": 0.9}]).to_csv(
        tmp_path / "outputs" / "ml" / "group_predictions_test.csv",
        index=False,
    )
    pd.DataFrame([{"feature": "max_in_degree", "importance": 0.1}]).to_csv(tmp_path / "outputs" / "ml" / "group_feature_importance.csv", index=False)
    pd.DataFrame([{"feature": "max_in_degree", "mean_abs_shap": 0.05}]).to_csv(tmp_path / "outputs" / "ml" / "group_shap_summary.csv", index=False)
    (tmp_path / "outputs" / "ml" / "group_model_report.md").write_text("## Results\nPromising.\n## Research Interpretation\nStill early.\n", encoding="utf-8")
    (tmp_path / "outputs" / "ml" / "group_shap_notes.md").write_text("SHAP generated.\n", encoding="utf-8")

    service = ResearchService(root=tmp_path)

    overview = service.overview_snapshot()
    phase1 = service.phase1_snapshot()
    candidates = service.candidate_groups_snapshot()
    refined = service.refined_dataset_snapshot()
    model = service.model_results_snapshot()

    assert overview["headline_metrics"]["no_graph_pct"] == 50.0
    assert phase1["available"] is True
    assert candidates["raw_count"] == 2
    assert refined["report_metrics"]["rows_after_deduplication"] == 1
    assert model["available"] is True


def test_research_service_degrades_gracefully_with_malformed_artifacts(tmp_path: Path) -> None:
    (tmp_path / "outputs" / "ml").mkdir(parents=True)
    (tmp_path / "experiment_results.csv").write_text("run_id,failure_type\nR01,SUCCESS\n", encoding="utf-8")
    (tmp_path / "candidate_groups.csv").write_text("not,a,valid,csv\n1,2\n", encoding="utf-8")
    (tmp_path / "feature_schema_final.json").write_text("{bad json", encoding="utf-8")
    (tmp_path / "outputs" / "ml" / "group_model_metrics.json").write_text("{bad json", encoding="utf-8")
    (tmp_path / "outputs" / "ml" / "group_confusion_matrix.csv").write_text("bad\n", encoding="utf-8")

    service = ResearchService(root=tmp_path)

    overview = service.overview_snapshot()
    phase1 = service.phase1_snapshot()
    candidates = service.candidate_groups_snapshot()
    refined = service.refined_dataset_snapshot()
    model = service.model_results_snapshot()

    assert overview["headline_metrics"]["success_rate_pct"] == 100.0
    assert phase1["available"] is True
    assert candidates["available"] is False
    assert refined["available"] is False
    assert model["available"] is False


def test_research_service_curated_runs_and_catalog_enrichment(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {"run_id": "R01_clique_high_low_single_s11", "scenario_type": "collusive_clique", "concealment": "low", "instrument_scope": "single", "precision": 1.0, "recall": 1.0, "F1": 1.0, "retained_edges_count": 13, "failure_type": "SUCCESS"},
            {"run_id": "R09_ring_high_low_single_s41", "scenario_type": "circular_trading_ring", "concealment": "low", "instrument_scope": "single", "precision": 0.0, "recall": 0.0, "F1": 0.0, "retained_edges_count": 0, "failure_type": "NO_GRAPH_FORMED"},
            {"run_id": "R17_mixed_high_low_single_s73", "scenario_type": "mixed", "concealment": "low", "instrument_scope": "single", "precision": 1.0, "recall": 0.4, "F1": 0.571, "retained_edges_count": 6, "failure_type": "SUCCESS"},
        ]
    ).to_csv(tmp_path / "experiment_results.csv", index=False)
    pd.DataFrame(
        [
            {"run_id": "R01_clique_high_low_single_s11", "scenario_type": "collusive_clique", "instrument_scope": "single"},
            {"run_id": "R09_ring_high_low_single_s41", "scenario_type": "circular_trading_ring", "instrument_scope": "single"},
            {"run_id": "R17_mixed_high_low_single_s73", "scenario_type": "mixed", "instrument_scope": "single"},
        ]
    ).to_csv(tmp_path / "experiment_runs.csv", index=False)
    (tmp_path / "outputs" / "ml").mkdir(parents=True)
    (tmp_path / "outputs" / "ml" / "group_model_metrics.json").write_text(
        json.dumps({"test_metrics": {"f1": 0.8, "roc_auc": 0.94}}),
        encoding="utf-8",
    )

    runs = [
        RunSummary(run_id="R01_clique_high_low_single_s11", run_name="best", created_at="2026-04-04T00:00:00+00:00", mode="manipulative", seed=1, output_path="/tmp/R01", status="completed", analysis_completed=True, metrics={"imported_from_generated_batch": True}),
        RunSummary(run_id="R09_ring_high_low_single_s41", run_name="failure", created_at="2026-04-04T00:00:00+00:00", mode="manipulative", seed=2, output_path="/tmp/R09", status="completed", analysis_completed=True, metrics={"imported_from_generated_batch": True}),
        RunSummary(run_id="R17_mixed_high_low_single_s73", run_name="progress", created_at="2026-04-04T00:00:00+00:00", mode="manipulative", seed=3, output_path="/tmp/R17", status="completed", analysis_completed=True, metrics={"imported_from_generated_batch": True}),
    ]

    service = ResearchService(root=tmp_path)
    catalog = service.run_catalog_snapshot(runs)
    curated = service.curated_demo_runs(runs)
    aggregate = service.aggregate_research_summary()

    assert list(catalog["run_id"]) == [run.run_id for run in runs]
    assert all(record["missing"] is False for record in curated)
    assert [record["selected"]["run_id"] for record in curated] == [
        "R01_clique_high_low_single_s11",
        "R09_ring_high_low_single_s41",
        "R17_mixed_high_low_single_s73",
    ]
    assert aggregate["candidate_counts"]["raw"] == 0
