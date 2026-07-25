from __future__ import annotations

import pytest

from surveillance_app.services.config_service import ConfigService
from surveillance_app.viewmodels.scenario_forms import CircularTradingRingForm, CollusiveCliqueForm


def test_config_service_builds_manipulative_config() -> None:
    service = ConfigService()
    config = service.build_config(
        seed=11,
        trader_count=10,
        broker_count=3,
        instrument_count=2,
        session_duration_minutes=45,
        mode="manipulative",
        scenarios=[
            CollusiveCliqueForm(
                scenario_id="scenario_001",
                instrument_symbol="INST_1",
                participant_count=4,
                start_minute=5,
                duration_minutes=10,
                intensity="medium",
                concealment="low",
                side_pattern="synchronized_buy_sell",
            ),
            CircularTradingRingForm(
                scenario_id="scenario_002",
                instrument_symbol="INST_2",
                participant_count=4,
                start_minute=20,
                duration_minutes=12,
                cycles=4,
                intensity="medium",
                concealment="low",
            ),
        ],
    )
    assert config["seed"] == 11
    assert len(config["instruments"]) == 2
    assert len(config["scenarios"]) == 2
    assert config["scenarios"][1]["scenario_type"] == "circular_trading_ring"


def test_config_service_rejects_invalid_scenario_window() -> None:
    service = ConfigService()
    with pytest.raises(ValueError):
        service.build_config(
            seed=1,
            trader_count=6,
            broker_count=2,
            instrument_count=1,
            session_duration_minutes=30,
            mode="manipulative",
            scenarios=[
                CollusiveCliqueForm(
                    scenario_id="scenario_001",
                    instrument_symbol="INST_1",
                    participant_count=3,
                    start_minute=25,
                    duration_minutes=10,
                    intensity="medium",
                    concealment="low",
                    side_pattern="synchronized_buy_sell",
                )
            ],
        )


def test_config_service_exposes_templates_presets_and_experiment_summary() -> None:
    service = ConfigService()

    templates = service.run_templates()
    presets = service.analysis_presets()
    summary = service.summarize_experiment_design(
        trader_count=12,
        session_duration_minutes=60,
        mode="manipulative",
        scenarios=[
            CollusiveCliqueForm(
                scenario_id="scenario_001",
                instrument_symbol="INST_1",
                participant_count=4,
                start_minute=10,
                duration_minutes=15,
                intensity="medium",
                concealment="high",
                side_pattern="synchronized_buy_sell",
            ),
            CircularTradingRingForm(
                scenario_id="scenario_002",
                instrument_symbol="INST_1",
                participant_count=4,
                start_minute=18,
                duration_minutes=14,
                cycles=5,
                intensity="medium",
                concealment="medium",
            ),
        ],
    )

    assert "Balanced" in presets
    assert "Mixed Dual Scenario" in templates
    assert summary["scenario_count"] == 2
    assert summary["injected_participant_count"] == 8
    assert summary["timing_overlap_pairs"] == 1
    assert summary["expected_difficulty"] in {"Moderate", "Hard"}
