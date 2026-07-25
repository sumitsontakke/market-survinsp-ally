from __future__ import annotations

import networkx as nx

from synthetic_market_sim.analysis.dataset_loader import ScenarioAttribution, ScenarioAttributionIndex
from synthetic_market_sim.analysis.subgraph_detection import detect_suspicious_groups


def _index(mapping: dict[str, tuple[list[str], str]]) -> ScenarioAttributionIndex:
    scenario_to_traders = {scenario_id: participants for scenario_id, (participants, _) in mapping.items()}
    scenario_details = {
        scenario_id: ScenarioAttribution(
            scenario_id=scenario_id,
            scenario_type=scenario_type,
            is_manipulative=True,
        )
        for scenario_id, (_, scenario_type) in mapping.items()
    }
    trader_to_scenarios: dict[str, list[ScenarioAttribution]] = {}
    for scenario_id, traders in scenario_to_traders.items():
        for trader_id in traders:
            trader_to_scenarios.setdefault(trader_id, []).append(scenario_details[scenario_id])
    return ScenarioAttributionIndex(
        trader_to_scenarios=trader_to_scenarios,
        scenario_to_traders=scenario_to_traders,
        scenario_details=scenario_details,
    )


def test_suspicious_group_detection_reports_full_coverage_for_single_scenario() -> None:
    graph = nx.Graph()
    graph.add_edge("trader_a", "trader_b", weight=0.91)
    graph.add_edge("trader_b", "trader_c", weight=0.89)
    graph.add_edge("trader_a", "trader_c", weight=0.93)
    graph.add_edge("trader_d", "trader_e", weight=0.75)

    groups = detect_suspicious_groups(graph, _index({"scenario_001": (["trader_a", "trader_b", "trader_c"], "collusive_clique")}))
    assert not groups.empty
    best = groups.sort_values("coverage", ascending=False).iloc[0]
    assert best["dominant_scenario"] == "scenario_001"
    assert best["scenario_type"] == "collusive_clique"
    assert best["group_scenario_member_count"] == 3
    assert best["total_scenario_participants"] == 3
    assert best["coverage"] == 1.0
    assert best["purity"] == 1.0
    assert bool(best["is_detected"]) is True
    assert bool(best["exact_match"]) is True
    assert bool(best["is_mixed"]) is False


def test_suspicious_group_detection_marks_mixed_groups_and_partial_coverage() -> None:
    graph = nx.Graph()
    graph.add_edge("trader_a", "trader_b", weight=0.91)
    graph.add_edge("trader_b", "trader_c", weight=0.89)
    graph.add_edge("trader_a", "trader_c", weight=0.93)
    graph.add_edge("trader_c", "trader_x", weight=0.82)

    groups = detect_suspicious_groups(
        graph,
        _index(
            {
                "scenario_001": (["trader_a", "trader_b", "trader_d"], "collusive_clique"),
                "scenario_002": (["trader_c", "trader_e"], "circular_trading_ring"),
            }
        ),
    )

    best = groups.sort_values("coverage", ascending=False).iloc[0]
    assert best["dominant_scenario"] == "scenario_001"
    assert best["scenario_type"] == "collusive_clique"
    assert best["group_scenario_member_count"] == 2
    assert best["total_scenario_participants"] == 3
    assert round(float(best["coverage"]), 3) == 0.667
    assert round(float(best["purity"]), 3) == 0.667
    assert best["best_matching_scenario_id"] == "scenario_001"
    assert best["overlap_count"] == 2
    assert best["group_size"] == 3
    assert best["ground_truth_size"] == 3
    assert best["unmatched_flagged_count"] == 1
    assert bool(best["exact_match"]) is False
    assert bool(best["is_mixed"]) is True


def test_suspicious_group_detection_reports_no_dominant_scenario_for_normal_only_groups() -> None:
    graph = nx.Graph()
    graph.add_edge("trader_a", "trader_b", weight=0.91)
    graph.add_edge("trader_b", "trader_c", weight=0.89)

    groups = detect_suspicious_groups(graph, _index({}))
    assert not groups.empty
    best = groups.iloc[0]
    assert best["dominant_scenario"] == ""
    assert best["scenario_type"] == ""
    assert best["coverage"] == 0.0
    assert best["purity"] == 0.0
    assert bool(best["is_detected"]) is False
    assert bool(best["exact_match"]) is False
    assert bool(best["is_mixed"]) is False
