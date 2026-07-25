from __future__ import annotations

import json

import pandas as pd

from synthetic_market_sim.analysis.dataset_loader import LoadedDataset
from synthetic_market_sim.analysis.detection_evaluation import evaluate_detection_quality


def _dataset(scenarios: pd.DataFrame, traders: list[str]) -> LoadedDataset:
    return LoadedDataset(
        root=None,
        traders=pd.DataFrame({"trader_id": traders}),
        orders=pd.DataFrame(),
        trades=pd.DataFrame(),
        scenarios=scenarios,
        instruments=pd.DataFrame(),
        manifest={},
    )


def _groups(groups: list[list[str]]) -> pd.DataFrame:
    if not groups:
        return pd.DataFrame(
            columns=[
                "group_id",
                "participant_ids",
                "trader_count",
            ]
        )
    return pd.DataFrame(
        [
            {
                "group_id": "group_{0:03d}".format(index + 1),
                "participant_ids": json.dumps(group),
                "trader_count": len(group),
            }
            for index, group in enumerate(groups)
        ]
    )


def test_detection_evaluation_reports_perfect_detection() -> None:
    scenarios = pd.DataFrame(
        [
            {
                "scenario_id": "scenario_001",
                "scenario_type": "collusive_clique",
                "participant_ids": ["trader_a", "trader_b"],
                "is_manipulative": "True",
            }
        ]
    )
    evaluation = evaluate_detection_quality(_dataset(scenarios, ["trader_a", "trader_b", "trader_c"]), _groups([["trader_a", "trader_b"]]))

    assert evaluation["has_ground_truth"] is True
    assert evaluation["true_positive_count"] == 2
    assert evaluation["false_positive_count"] == 0
    assert evaluation["false_negative_count"] == 0
    assert evaluation["true_negative_count"] == 1
    assert evaluation["precision"] == 1.0
    assert evaluation["recall"] == 1.0
    assert evaluation["f1_score"] == 1.0
    assert evaluation["verdict"] == "exact_match"


def test_detection_evaluation_reports_partial_detection_with_fp_and_fn() -> None:
    scenarios = pd.DataFrame(
        [
            {
                "scenario_id": "scenario_001",
                "scenario_type": "collusive_clique",
                "participant_ids": ["trader_a", "trader_b", "trader_c"],
                "is_manipulative": "True",
            }
        ]
    )
    evaluation = evaluate_detection_quality(
        _dataset(scenarios, ["trader_a", "trader_b", "trader_c", "trader_x", "trader_y"]),
        _groups([["trader_a", "trader_x"]]),
    )

    assert evaluation["true_positive_count"] == 1
    assert evaluation["false_positive_count"] == 1
    assert evaluation["false_negative_count"] == 2
    assert evaluation["true_negative_count"] == 1
    assert round(evaluation["precision"], 3) == 0.5
    assert round(evaluation["recall"], 3) == 0.333
    assert round(evaluation["false_positive_rate"], 3) == 0.5
    assert round(evaluation["false_negative_rate"], 3) == 0.667
    assert evaluation["verdict"] == "partial_detection"


def test_detection_evaluation_handles_no_suspicious_groups() -> None:
    scenarios = pd.DataFrame(
        [
            {
                "scenario_id": "scenario_001",
                "scenario_type": "collusive_clique",
                "participant_ids": ["trader_a", "trader_b"],
                "is_manipulative": "True",
            }
        ]
    )
    evaluation = evaluate_detection_quality(_dataset(scenarios, ["trader_a", "trader_b", "trader_c"]), _groups([]))

    assert evaluation["flagged_trader_count"] == 0
    assert evaluation["true_positive_count"] == 0
    assert evaluation["false_negative_count"] == 2
    assert evaluation["verdict"] == "missed"


def test_detection_evaluation_handles_no_ground_truth() -> None:
    scenarios = pd.DataFrame(
        [
            {
                "scenario_id": "normal",
                "scenario_type": "generic_background",
                "participant_ids": ["trader_a", "trader_b"],
                "is_manipulative": "False",
            }
        ]
    )
    evaluation = evaluate_detection_quality(_dataset(scenarios, ["trader_a", "trader_b", "trader_c"]), _groups([["trader_a", "trader_c"]]))

    assert evaluation["has_ground_truth"] is False
    assert evaluation["injected_trader_count"] == 0
    assert evaluation["true_positive_count"] == 0
    assert evaluation["false_positive_count"] == 2
    assert evaluation["verdict"] == "no_ground_truth"


def test_detection_evaluation_handles_multiple_scenarios() -> None:
    scenarios = pd.DataFrame(
        [
            {
                "scenario_id": "scenario_001",
                "scenario_type": "collusive_clique",
                "participant_ids": ["trader_a", "trader_b"],
                "is_manipulative": "True",
            },
            {
                "scenario_id": "scenario_002",
                "scenario_type": "circular_trading_ring",
                "participant_ids": ["trader_c", "trader_d"],
                "is_manipulative": "True",
            },
        ]
    )
    evaluation = evaluate_detection_quality(
        _dataset(scenarios, ["trader_a", "trader_b", "trader_c", "trader_d", "trader_x"]),
        _groups([["trader_a", "trader_b"], ["trader_c", "trader_x"]]),
    )

    assert evaluation["true_positive_count"] == 3
    assert evaluation["false_positive_count"] == 1
    assert evaluation["false_negative_count"] == 1
    assert evaluation["detected_scenario_count"] == 2
    assert len(evaluation["scenario_evaluations"]) == 2


def test_detection_evaluation_uses_manifest_backed_scenario_details_and_activity_fallback() -> None:
    dataset = LoadedDataset(
        root=None,
        traders=pd.DataFrame({"trader_id": ["trader_a", "trader_b", "trader_x"]}),
        orders=pd.DataFrame(
            [
                {"trader_id": "trader_a", "scenario_id": "scenario_001"},
                {"trader_id": "trader_b", "scenario_id": "scenario_001"},
            ]
        ),
        trades=pd.DataFrame(),
        scenarios=pd.DataFrame(),
        instruments=pd.DataFrame(),
        manifest={"scenario_ids": ["scenario_001"]},
    )

    evaluation = evaluate_detection_quality(dataset, _groups([["trader_a", "trader_x"]]))

    assert evaluation["has_ground_truth"] is True
    assert evaluation["injected_trader_count"] == 2
    assert evaluation["true_positive_count"] == 1
    assert evaluation["false_positive_count"] == 1
    assert evaluation["false_negative_count"] == 1
    assert evaluation["total_scenarios"] == 1
