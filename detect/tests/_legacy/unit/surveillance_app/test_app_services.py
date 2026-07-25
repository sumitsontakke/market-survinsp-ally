from __future__ import annotations

import json
from pathlib import Path
import shutil

import pandas as pd

from surveillance_app.repository.run_registry import RunRegistry
from surveillance_app.services.analysis_service import AnalysisService
from surveillance_app.services.artifact_service import ArtifactService
from surveillance_app.services.config_service import ConfigService
from surveillance_app.services.run_service import RunService
from surveillance_app.viewmodels.scenario_forms import CollusiveCliqueForm


def test_run_service_and_analysis_service_execute_end_to_end(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SURVEILLANCE_OUTPUT_ROOT", str(tmp_path / "outputs"))
    registry = RunRegistry(tmp_path / "runs.db")
    config_service = ConfigService()
    run_service = RunService(registry, config_service)
    analysis_service = AnalysisService(registry)

    run = run_service.launch_run(
        run_name="ui_generic",
        seed=5,
        trader_count=8,
        broker_count=2,
        instrument_count=1,
        session_duration_minutes=20,
        mode="manipulative",
        scenarios=[
            CollusiveCliqueForm(
                scenario_id="scenario_001",
                instrument_symbol="INST_1",
                participant_count=3,
                start_minute=3,
                duration_minutes=8,
                intensity="medium",
                concealment="low",
                side_pattern="synchronized_buy_sell",
            )
        ],
    )
    assert run.status == "completed"
    assert Path(run.output_path) == tmp_path / "outputs" / "ui_runs" / run.run_id
    assert (Path(run.output_path) / "manifest.json").exists()
    assert (Path(run.output_path) / "experiment_design_summary.json").exists()

    analysis = analysis_service.run_for_existing_run(
        run.run_id,
        source="orders",
        bucket_minutes=1,
        corr_method="pearson",
        corr_threshold=0.6,
        min_active_buckets=1,
    )
    refreshed = registry.get_run(run.run_id)
    assert refreshed is not None
    assert refreshed.analysis_completed is True
    assert (Path(refreshed.analysis_output_path) / "analysis_summary.md").exists()
    artifact_service = ArtifactService()
    chart_labels = [label for label, _ in artifact_service.analysis_chart_paths(refreshed)]
    assert "Signed Volume Time Series" in chart_labels
    assert "Correlation Heatmap" in chart_labels
    assert "Highlighted Suspicious Network" in chart_labels
    assert "Manipulative Context Network" in chart_labels
    assert "signed_volume_timeseries" in refreshed.artifacts
    assert "decision_queue" in refreshed.artifacts
    assert "sensitivity_summary" in refreshed.artifacts
    assert "matrix_shape" in analysis
    assert "decision_queue" in analysis
    assert "sensitivity_summary" in analysis
    assert Path(refreshed.artifacts["decision_queue"]).exists()
    assert Path(refreshed.artifacts["sensitivity_summary"]).exists()


def test_single_run_analysis_variants_are_parameterized_and_legacy_still_loads(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SURVEILLANCE_OUTPUT_ROOT", str(tmp_path / "outputs"))
    registry = RunRegistry(tmp_path / "runs.db")
    config_service = ConfigService()
    run_service = RunService(registry, config_service)
    analysis_service = AnalysisService(registry)

    run = run_service.launch_run(
        run_name="variant_run",
        seed=13,
        trader_count=10,
        broker_count=3,
        instrument_count=2,
        session_duration_minutes=30,
        mode="manipulative",
        scenarios=[
            CollusiveCliqueForm(
                scenario_id="scenario_001",
                instrument_symbol="INST_1",
                participant_count=3,
                start_minute=4,
                duration_minutes=10,
                intensity="medium",
                concealment="low",
                side_pattern="synchronized_buy_sell",
            )
        ],
    )

    instruments = pd.read_csv(Path(run.output_path) / "instruments.csv")
    instrument_1 = str(instruments.iloc[0]["instrument_id"])

    root_a = analysis_service.resolve_analysis_output_path(
        [run.run_id],
        source="orders",
        bucket_minutes=1,
        corr_method="pearson",
        corr_threshold=0.60,
        min_active_buckets=1,
        instrument_id=None,
    )
    root_b = analysis_service.resolve_analysis_output_path(
        [run.run_id],
        source="orders",
        bucket_minutes=1,
        corr_method="pearson",
        corr_threshold=0.70,
        min_active_buckets=1,
        instrument_id=None,
    )
    root_c = analysis_service.resolve_analysis_output_path(
        [run.run_id],
        source="orders",
        bucket_minutes=1,
        corr_method="pearson",
        corr_threshold=0.60,
        min_active_buckets=1,
        instrument_id=instrument_1,
    )
    root_a_repeat = analysis_service.resolve_analysis_output_path(
        [run.run_id],
        source="orders",
        bucket_minutes=1,
        corr_method="pearson",
        corr_threshold=0.60,
        min_active_buckets=1,
        instrument_id=None,
    )

    assert root_a != root_b
    assert root_a != root_c
    assert root_a == root_a_repeat
    assert root_a.parent.name == "analyses"

    analysis_service.run_for_existing_run(
        run.run_id,
        source="orders",
        bucket_minutes=1,
        corr_method="pearson",
        corr_threshold=0.60,
        min_active_buckets=1,
    )
    analysis_service.run_for_existing_run(
        run.run_id,
        source="orders",
        bucket_minutes=1,
        corr_method="pearson",
        corr_threshold=0.70,
        min_active_buckets=1,
    )
    analysis_service.run_for_existing_run(
        run.run_id,
        source="orders",
        bucket_minutes=1,
        corr_method="pearson",
        corr_threshold=0.60,
        min_active_buckets=1,
        instrument_id=instrument_1,
    )

    legacy_root = Path(run.output_path) / "analysis"
    legacy_root.mkdir(parents=True, exist_ok=True)
    (legacy_root / "analysis_summary.md").write_text("# legacy\n", encoding="utf-8")

    variants = analysis_service.list_saved_single_run_analyses(run.run_id)
    analysis_ids = {record["analysis_id"] for record in variants}
    assert len([record for record in variants if not record.get("legacy")]) == 3
    assert "legacy_analysis" in analysis_ids

    dashboard_snapshot = analysis_service.dashboard_comparison_snapshot()
    matching_labels = [row for row in dashboard_snapshot["best_performing_analyses"] if "variant_run" in row["label"]]
    assert matching_labels


def test_instrument_scoped_analysis_filters_trade_flow_and_ground_truth(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SURVEILLANCE_OUTPUT_ROOT", str(tmp_path / "outputs"))
    registry = RunRegistry(tmp_path / "runs.db")
    config_service = ConfigService()
    run_service = RunService(registry, config_service)
    analysis_service = AnalysisService(registry)

    run = run_service.launch_run(
        run_name="instrument_scope_run",
        seed=17,
        trader_count=12,
        broker_count=3,
        instrument_count=2,
        session_duration_minutes=40,
        mode="manipulative",
        scenarios=[
            CollusiveCliqueForm(
                scenario_id="scenario_001",
                instrument_symbol="INST_1",
                participant_count=4,
                start_minute=5,
                duration_minutes=12,
                intensity="medium",
                concealment="low",
                side_pattern="synchronized_buy_sell",
            )
        ],
    )

    instruments = pd.read_csv(Path(run.output_path) / "instruments.csv")
    inst1 = str(instruments.loc[instruments["symbol"] == "INST_1", "instrument_id"].iloc[0])
    inst2 = str(instruments.loc[instruments["symbol"] == "INST_2", "instrument_id"].iloc[0])

    result_inst1 = analysis_service.run_for_existing_run(
        run.run_id,
        source="orders",
        bucket_minutes=1,
        corr_method="pearson",
        corr_threshold=0.60,
        min_active_buckets=1,
        instrument_id=inst1,
    )
    result_inst2 = analysis_service.run_for_existing_run(
        run.run_id,
        source="orders",
        bucket_minutes=1,
        corr_method="pearson",
        corr_threshold=0.60,
        min_active_buckets=1,
        instrument_id=inst2,
    )

    assert result_inst1["scope_type"] == "instrument_scoped"
    assert result_inst1["instrument_id"] == inst1
    assert result_inst1["detection_evaluation"]["has_ground_truth"] is True
    assert result_inst2["detection_evaluation"]["has_ground_truth"] is False
    assert result_inst2["scenario_count"] == 0
    assert all(row["benchmark_outcome"] == "Background-only" for row in result_inst2["sensitivity_summary"]["rows"])

    trade_edges_inst1 = pd.read_csv(Path(result_inst1["analysis_output_path"]) / "directed_trade_edges.csv")
    trade_edges_inst2 = pd.read_csv(Path(result_inst2["analysis_output_path"]) / "directed_trade_edges.csv")
    if not trade_edges_inst1.empty:
        assert trade_edges_inst1["scenario_ids"].astype(str).str.contains("scenario_001").any()
    if not trade_edges_inst2.empty:
        assert not trade_edges_inst2["scenario_ids"].astype(str).str.contains("scenario_001").any()

    detection_eval_inst2 = json.loads((Path(result_inst2["analysis_output_path"]) / "detection_evaluation.json").read_text(encoding="utf-8"))
    assert detection_eval_inst2["scope_type"] == "instrument_scoped"
    assert detection_eval_inst2["instrument_id"] == inst2


def test_generated_batch_import_registers_runs_and_promotes_legacy_analysis(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SURVEILLANCE_OUTPUT_ROOT", str(tmp_path / "outputs"))
    registry = RunRegistry(tmp_path / "runs.db")
    config_service = ConfigService()
    run_service = RunService(registry, config_service)
    analysis_service = AnalysisService(registry)

    run = run_service.launch_run(
        run_name="batch_seed_run",
        seed=23,
        trader_count=10,
        broker_count=3,
        instrument_count=1,
        session_duration_minutes=25,
        mode="manipulative",
        scenarios=[
            CollusiveCliqueForm(
                scenario_id="scenario_001",
                instrument_symbol="INST_1",
                participant_count=3,
                start_minute=3,
                duration_minutes=10,
                intensity="high",
                concealment="low",
                side_pattern="synchronized_buy_sell",
            )
        ],
    )
    analysis = analysis_service.run_for_existing_run(
        run.run_id,
        source="orders",
        bucket_minutes=1,
        corr_method="pearson",
        corr_threshold=0.7,
        min_active_buckets=2,
    )

    batch_root = tmp_path / "generated_batch" / "outputs" / "runs"
    batch_run_id = "R01_clique_high_low_single_s11"
    batch_run_root = batch_root / batch_run_id
    batch_run_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(Path(run.output_path), batch_run_root)
    shutil.rmtree(batch_run_root / "analyses", ignore_errors=True)
    shutil.copytree(Path(analysis["analysis_output_path"]), batch_run_root / "analysis", dirs_exist_ok=True)
    (batch_run_root / "scenario_config.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-04-04T10:30:00+00:00",
                "scenario_details": [
                    {
                        "scenario_id": "scenario_001",
                        "scenario_type": "collusive_clique",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    experiment_runs_csv = tmp_path / "generated_batch" / "experiment_runs.csv"
    pd.DataFrame(
        [
            {
                "run_id": batch_run_id,
                "seed": 23,
                "trader_count": 10,
                "injected_core_count": 3,
                "scenario_type": "collusive_clique",
                "intensity": "high",
                "concealment": "low",
                "instrument_scope": "single",
                "order_count": 100,
                "trade_count": 90,
                "manipulative_order_count": 12,
                "manipulative_trade_count": 10,
            }
        ]
    ).to_csv(experiment_runs_csv, index=False)

    combined_analysis_csv = tmp_path / "generated_batch" / "combined_analysis.csv"
    pd.DataFrame(
        [
            {
                "run_id": batch_run_id,
                "analysis_source": "orders",
                "bucket_minutes": 1,
                "corr_method": "pearson",
                "corr_threshold": 0.7,
                "min_active_buckets": 2,
            }
        ]
    ).to_csv(combined_analysis_csv, index=False)

    summary = run_service.import_generated_batch_runs(
        batch_root,
        experiment_runs_csv=experiment_runs_csv,
        combined_analysis_csv=combined_analysis_csv,
    )

    assert summary == {"imported_runs": 1, "imported_analyses": 1}
    imported_run = registry.get_run(batch_run_id)
    assert imported_run is not None
    assert imported_run.metrics["imported_from_generated_batch"] is True
    assert imported_run.metrics["import_source"] == "outputs/runs"
    assert imported_run.analysis_completed is True
    assert Path(imported_run.output_path) == tmp_path / "outputs" / "ui_runs" / batch_run_id
    assert Path(imported_run.analysis_output_path).name.startswith("analysis_")
    assert not (Path(imported_run.output_path) / "analysis").exists()

    variants = analysis_service.list_saved_single_run_analyses(batch_run_id)
    assert len(variants) == 1
    assert variants[0]["analysis_id"].startswith("analysis_")
    assert variants[0]["metadata"]["source"] == "orders"


def test_artifact_service_handles_malformed_csv_and_json(tmp_path: Path) -> None:
    artifact_service = ArtifactService()
    bad_csv = tmp_path / "broken.csv"
    bad_json = tmp_path / "broken.json"
    bad_csv.write_text('a,"unterminated\n1,2\n', encoding="utf-8")
    bad_json.write_text("{broken", encoding="utf-8")

    assert artifact_service.load_csv_preview(bad_csv).empty
    assert artifact_service.load_json(bad_json) == {}


def test_services_resolve_imported_workspace_paths(tmp_path: Path, monkeypatch) -> None:
    output_root = tmp_path / "outputs"
    run_root = output_root / "ui_runs" / "run_001"
    analysis_root = run_root / "analyses" / "analysis_abc123"
    analysis_root.mkdir(parents=True)
    (run_root / "manifest.json").write_text('{"counts": {"orders": 4, "trades": 2}}', encoding="utf-8")
    (run_root / "scenarios.csv").write_text("scenario_id,scenario_type\nscenario_001,collusive_clique\n", encoding="utf-8")
    (run_root / "instruments.csv").write_text("instrument_id,symbol\ninst_1,INST_1\n", encoding="utf-8")
    (analysis_root / "analysis_metadata.json").write_text(
        json.dumps(
            {
                "analysis_id": "analysis_abc123",
                "run_id": "run_001",
                "run_ids": ["run_001"],
                "input_dir": "/Users/s/PesFinal/market-survinsp-ally/outputs/ui_runs/run_001",
                "output_dir": "/Users/s/PesFinal/market-survinsp-ally/outputs/ui_runs/run_001/analyses/analysis_abc123",
                "source": "orders",
                "bucket_minutes": 1,
                "corr_method": "pearson",
                "corr_threshold": 0.7,
                "min_active_buckets": 2,
                "instrument_id": "",
                "scope_type": "full_dataset",
                "created_at": "2026-04-03T19:26:34.223206+00:00",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (analysis_root / "analysis_summary.md").write_text("# Summary\n", encoding="utf-8")
    (analysis_root / "correlation_matrix.csv").write_text("trader_id,a\nA,1.0\n", encoding="utf-8")
    (analysis_root / "signed_volume_matrix.csv").write_text("bucket,A\n0,10\n", encoding="utf-8")
    (analysis_root / "correlation_edges.csv").write_text("source,target,weight\n", encoding="utf-8")
    (analysis_root / "suspicious_groups.csv").write_text("group_id,trader_count\n", encoding="utf-8")
    (analysis_root / "directed_trade_edges.csv").write_text("seller,buyer,trade_count\n", encoding="utf-8")
    (analysis_root / "detection_evaluation.json").write_text("{}", encoding="utf-8")
    (analysis_root / "confusion_matrix.json").write_text("{}", encoding="utf-8")

    monkeypatch.setenv("SURVEILLANCE_OUTPUT_ROOT", str(output_root))
    registry = RunRegistry(tmp_path / "runs.db")
    registry.create_run(
        run_id="run_001",
        run_name="imported",
        created_at="2026-03-21T00:00:00Z",
        mode="manipulative",
        seed=7,
        output_path="/Users/s/PesFinal/market-survinsp-ally/outputs/ui_runs/run_001",
        status="completed",
        scenario_summary="scenario_001:collusive_clique",
        config_path="/Users/s/PesFinal/market-survinsp-ally/outputs/ui_runs/run_001/config.yaml",
        metrics={},
        artifacts={},
    )
    registry.update_run(
        "run_001",
        analysis_output_path="/Users/s/PesFinal/market-survinsp-ally/outputs/ui_runs/run_001/analyses/analysis_abc123",
        analysis_completed=True,
    )

    artifact_service = ArtifactService()
    analysis_service = AnalysisService(registry)
    run = registry.get_run("run_001")

    assert run is not None
    assert artifact_service.manifest_summary(run)["counts"]["orders"] == 4
    assert not artifact_service.scenarios_summary(run).empty
    variants = analysis_service.list_saved_single_run_analyses(run.run_id)
    assert len(variants) == 1
    assert variants[0]["analysis_output_path"] == str(analysis_root)
    assert variants[0]["result"]["orders_in_dataset"] == 4


def test_analysis_service_can_combine_multiple_runs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SURVEILLANCE_OUTPUT_ROOT", str(tmp_path / "outputs"))
    registry = RunRegistry(tmp_path / "runs.db")
    config_service = ConfigService()
    run_service = RunService(registry, config_service)
    analysis_service = AnalysisService(registry)

    generic_run = run_service.launch_run(
        run_name="ui_generic",
        seed=3,
        trader_count=8,
        broker_count=2,
        instrument_count=1,
        session_duration_minutes=20,
        mode="generic",
        scenarios=[],
    )
    manip_run = run_service.launch_run(
        run_name="ui_manip",
        seed=5,
        trader_count=8,
        broker_count=2,
        instrument_count=1,
        session_duration_minutes=20,
        mode="manipulative",
        scenarios=[
            CollusiveCliqueForm(
                scenario_id="scenario_001",
                instrument_symbol="INST_1",
                participant_count=3,
                start_minute=3,
                duration_minutes=8,
                intensity="medium",
                concealment="low",
                side_pattern="synchronized_buy_sell",
            )
        ],
    )

    result = analysis_service.run_for_run_group(
        [generic_run.run_id, manip_run.run_id],
        source="orders",
        bucket_minutes=1,
        corr_method="pearson",
        corr_threshold=0.6,
        min_active_buckets=1,
    )

    analysis_output = Path(result["analysis_output_path"])
    combined_input = Path(result["combined_input_path"])
    assert analysis_output.exists()
    assert combined_input.exists()
    assert (analysis_output / "analysis_summary.md").exists()
    assert (analysis_output / "correlation_matrix.csv").exists()
    assert (analysis_output / "analysis_metadata.json").exists()
    assert (combined_input / "traders.csv").exists()
    assert len(result["source_run_ids"]) == 2

    saved_analyses = analysis_service.list_saved_combined_analyses()
    assert len(saved_analyses) == 1
    assert saved_analyses[0]["source_run_ids"] == [generic_run.run_id, manip_run.run_id]
    assert saved_analyses[0]["metadata"]["source"] == "orders"
    assert saved_analyses[0]["result"]["suspicious_group_count"] >= 0
    assert saved_analyses[0]["result"]["analysis_source"] == "orders"
    assert saved_analyses[0]["result"]["bucket_minutes"] == 1
    assert "retained_correlation_edges" in saved_analyses[0]["result"]
    assert "orders_in_dataset" in saved_analyses[0]["result"]
    assert "trades_in_dataset" in saved_analyses[0]["result"]
    assert "scenario_count" in saved_analyses[0]["result"]
    assert "highest_pair_correlation" in saved_analyses[0]["result"]
    assert "decision_queue" in saved_analyses[0]["result"]
    assert "sensitivity_summary" in saved_analyses[0]["result"]
    assert "injected_trader_count" in saved_analyses[0]["result"]
    assert "true_positive_count" in saved_analyses[0]["result"]
    assert "false_positive_count" in saved_analyses[0]["result"]

    dashboard_snapshot = analysis_service.dashboard_comparison_snapshot()
    assert "best_performing_analyses" in dashboard_snapshot
    assert "best_balanced_detector_setup" in dashboard_snapshot

    run_variants = analysis_service.list_saved_single_run_analyses(manip_run.run_id)
    assert isinstance(run_variants, list)

    deleted = analysis_service.delete_saved_combined_analysis(saved_analyses[0]["analysis_id"])
    assert deleted is True
    assert analysis_service.list_saved_combined_analyses() == []


def test_run_service_can_import_research_workspace_runs(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "research_workspace"
    source_outputs = source_root / "outputs"
    source_app_data = source_root / "app_data"
    monkeypatch.setenv("SURVEILLANCE_OUTPUT_ROOT", str(source_outputs))
    monkeypatch.setenv("SURVEILLANCE_APP_DATA_DIR", str(source_app_data))

    source_registry = RunRegistry(source_app_data / "runs.db")
    source_run_service = RunService(source_registry, ConfigService())
    run = source_run_service.launch_run(
        run_name="import_me",
        seed=23,
        trader_count=10,
        broker_count=3,
        instrument_count=1,
        session_duration_minutes=25,
        mode="manipulative",
        scenarios=[
            CollusiveCliqueForm(
                scenario_id="scenario_001",
                instrument_symbol="INST_1",
                participant_count=3,
                start_minute=4,
                duration_minutes=10,
                intensity="medium",
                concealment="low",
                side_pattern="synchronized_buy_sell",
            )
        ],
    )
    source_analysis_service = AnalysisService(source_registry)
    source_analysis_service.run_for_existing_run(
        run.run_id,
        source="orders",
        bucket_minutes=1,
        corr_method="pearson",
        corr_threshold=0.60,
        min_active_buckets=1,
    )

    monkeypatch.setenv("SURVEILLANCE_OUTPUT_ROOT", str(tmp_path / "outputs"))
    monkeypatch.setenv("SURVEILLANCE_APP_DATA_DIR", str(tmp_path / "app_data"))
    target_registry = RunRegistry(tmp_path / "app_data" / "runs.db")
    target_run_service = RunService(target_registry, ConfigService())
    summary = target_run_service.import_research_workspace(source_root)

    assert summary["imported_runs"] == 1
    assert summary["imported_analyses"] >= 1

    imported = target_registry.get_run(run.run_id)
    assert imported is not None
    assert imported.metrics.get("imported_from_research_workspace") is True
    assert Path(imported.output_path).exists()

    target_analysis_service = AnalysisService(target_registry)
    variants = target_analysis_service.list_saved_single_run_analyses(run.run_id)
    assert len(variants) >= 1
