from __future__ import annotations

from pathlib import Path

from surveillance_app.repository.run_registry import RunRegistry


def test_run_registry_create_update_and_stats(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "runs.db")
    registry.create_run(
        run_id="run_001",
        run_name="first",
        created_at="2026-03-21T00:00:00Z",
        mode="manipulative",
        seed=7,
        output_path=str(tmp_path / "outputs" / "run_001"),
        status="completed",
        scenario_summary="scenario_001:collusive_clique",
        config_path=str(tmp_path / "outputs" / "run_001" / "config.yaml"),
        metrics={"counts": {"orders": 10}},
        artifacts={"manifest": "manifest.json"},
    )
    registry.update_run(
        "run_001",
        analysis_output_path=str(tmp_path / "outputs" / "run_001" / "analysis"),
        analysis_completed=True,
        metrics={"analysis": {"suspicious_group_count": 1}},
    )
    run = registry.get_run("run_001")
    assert run is not None
    assert run.analysis_completed is True
    assert run.metrics["counts"]["orders"] == 10
    assert run.metrics["analysis"]["suspicious_group_count"] == 1

    stats = registry.stats()
    assert stats["total_runs"] == 1
    assert stats["successful_runs"] == 1
    assert stats["manipulative_runs"] == 1
    assert stats["analyzed_runs"] == 1
