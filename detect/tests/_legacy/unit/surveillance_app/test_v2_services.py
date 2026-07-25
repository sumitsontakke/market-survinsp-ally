from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from surveillance_app.repository.run_registry import RunRegistry
from surveillance_app_v2.contracts import GRAPH_PACKAGE_SCHEMA_VERSION
from surveillance_app_v2.services.catalog_service import DatasetCatalogService
from surveillance_app_v2.services.comparison_service import ComparisonService
from surveillance_app_v2.services.scoring_service import TabularGroupScorer


def test_v2_catalog_service_seeds_curated_run_into_isolated_storage(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "source_outputs" / "ui_runs"
    run_root = source_root / "R01_clique_high_low_single_s11"
    analysis_root = run_root / "analyses" / "analysis_001"
    analysis_root.mkdir(parents=True)
    (run_root / "scenario_config.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-04-03T19:25:09.982378+00:00",
                "seed": 11,
                "trader_count": 500,
                "scenario_type": "collusive_clique",
                "instrument_scope": "single",
                "market_quality_summary": {"trade_count": 9445},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (analysis_root / "analysis_metadata.json").write_text("{}", encoding="utf-8")

    monkeypatch.setenv("SURVEILLANCE_V2_OUTPUT_ROOT", str(tmp_path / "outputs_v2"))
    monkeypatch.setenv("SURVEILLANCE_V2_APP_DATA_DIR", str(tmp_path / "app_data_v2"))
    monkeypatch.setattr("surveillance_app_v2.services.catalog_service.v2_source_demo_root", lambda: source_root)

    registry = RunRegistry(tmp_path / "app_data_v2" / "runs_v2.db")
    service = DatasetCatalogService(registry)
    service.ensure_seeded()

    run = registry.get_run("R01_clique_high_low_single_s11")
    assert run is not None
    assert Path(run.output_path) == tmp_path / "outputs_v2" / "ui_runs" / "R01_clique_high_low_single_s11"
    assert Path(run.analysis_output_path).name == "analysis_001"
    assert run.metrics["seeded_for_v2"] is True
    assert (Path(run.output_path) / "scenario_config.json").exists()


def test_v2_scoring_validation_and_comparison_summary(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs_v2" / "ui_runs" / "run_001"
    analysis_root = run_root / "analyses" / "analysis_001"
    scoring_root = run_root / "scoring"
    analysis_root.mkdir(parents=True)
    scoring_root.mkdir(parents=True)

    pd.DataFrame(
        [
            {"group_id": "group_a", "group_type": "scc"},
            {"group_id": "group_b", "group_type": "cycle_3"},
        ]
    ).to_csv(analysis_root / "suspicious_groups.csv", index=False)
    pd.DataFrame(
        [
            {"group_id": "group_a", "group_type": "scc", "score": 0.91, "predicted_label": 1},
            {"group_id": "group_c", "group_type": "window_scc", "score": 0.82, "predicted_label": 1},
        ]
    ).to_csv(scoring_root / "tabular_group_predictions.csv", index=False)

    scoring_payload = {
        "artifacts": {
            "predictions": str(scoring_root / "tabular_group_predictions.csv"),
        }
    }
    summary = ComparisonService().build_summary(
        run_root,
        analysis_root=analysis_root,
        scoring_payload=scoring_payload,
    )

    assert summary["overlap"]["shared_group_ids"] == ["group_a"]
    assert summary["overlap"]["statistical_only_group_ids"] == ["group_b"]
    assert summary["overlap"]["ml_only_group_ids"] == ["group_c"]

    scorer = TabularGroupScorer()
    compatibility = scorer._validate_manifest({"schema_version": GRAPH_PACKAGE_SCHEMA_VERSION})
    mismatch = scorer._validate_manifest({"schema_version": "older-schema"})
    assert compatibility["status"] == "compatible"
    assert mismatch["status"] == "incompatible"
