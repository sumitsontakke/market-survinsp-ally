from __future__ import annotations

import json
from dataclasses import dataclass

import networkx as nx
import pandas as pd

from synth.generator.analysis.dataset_loader import ScenarioAttributionIndex


@dataclass(frozen=True)
class ScenarioOverlap:
    scenario_id: str
    scenario_type: str
    overlap_count: int
    total_scenario_participants: int
    coverage: float
    purity: float
    is_mixed: bool


def detect_suspicious_groups(
    graph: nx.Graph,
    scenario_attribution: ScenarioAttributionIndex | dict[str, list[str]],
) -> pd.DataFrame:
    attribution_index = _coerce_scenario_attribution(scenario_attribution)
    groups: list[dict] = []
    group_counter = 0
    seen_member_sets: set[tuple[str, ...]] = set()
    for component in nx.connected_components(graph):
        members = sorted(component)
        if len(members) < 2:
            continue
        signature = tuple(members)
        seen_member_sets.add(signature)
        group_counter += 1
        groups.append(_group_record("component", group_counter, graph.subgraph(members).copy(), attribution_index))
    for clique in nx.find_cliques(graph):
        if len(clique) < 3:
            continue
        signature = tuple(sorted(clique))
        if signature in seen_member_sets:
            continue
        seen_member_sets.add(signature)
        group_counter += 1
        groups.append(_group_record("clique", group_counter, graph.subgraph(signature).copy(), attribution_index))
    if not groups:
        return pd.DataFrame(
            columns=[
                "group_id",
                "group_type",
                "trader_count",
                "edge_count",
                "average_edge_weight",
                "density",
                "participant_ids",
                "has_scenario_participants",
                "dominant_scenario",
                "scenario_type",
                "group_scenario_member_count",
                "total_scenario_participants",
                "coverage",
                "is_mixed",
                "unmatched_flagged_count",
                "best_scenario_id",
                "best_matching_scenario_id",
                "best_overlap_count",
                "overlap_count",
                "group_size",
                "ground_truth_size",
                "participant_coverage",
                "purity",
                "is_detected",
                "exact_match",
                "suspicious_score",
            ]
        )
    frame = pd.DataFrame(groups).sort_values(["suspicious_score", "trader_count"], ascending=False).reset_index(drop=True)
    return frame


def _group_record(
    group_type: str,
    group_counter: int,
    subgraph: nx.Graph,
    scenario_attribution: ScenarioAttributionIndex,
) -> dict:
    members = sorted(subgraph.nodes())
    edge_weights = [data["weight"] for _, _, data in subgraph.edges(data=True)]
    overlap = best_scenario_overlap(members, scenario_attribution)
    average_edge_weight = float(sum(edge_weights) / len(edge_weights)) if edge_weights else 0.0
    density = nx.density(subgraph) if subgraph.number_of_nodes() > 1 else 0.0
    suspicious_score = average_edge_weight * density * len(members)
    return {
        "group_id": "{0}_{1:03d}".format(group_type, group_counter),
        "group_type": group_type,
        "trader_count": subgraph.number_of_nodes(),
        "edge_count": subgraph.number_of_edges(),
        "average_edge_weight": average_edge_weight,
        "density": density,
        "participant_ids": json.dumps(members),
        "has_scenario_participants": overlap.overlap_count > 0 if overlap else False,
        "dominant_scenario": overlap.scenario_id if overlap else "",
        "scenario_type": overlap.scenario_type if overlap else "",
        "group_scenario_member_count": overlap.overlap_count if overlap else 0,
        "total_scenario_participants": overlap.total_scenario_participants if overlap else 0,
        "coverage": overlap.coverage if overlap else 0.0,
        "is_mixed": overlap.is_mixed if overlap else False,
        "unmatched_flagged_count": len(members) - (overlap.overlap_count if overlap else 0),
        "best_scenario_id": overlap.scenario_id if overlap else "",
        "best_matching_scenario_id": overlap.scenario_id if overlap else "",
        "best_overlap_count": overlap.overlap_count if overlap else 0,
        "overlap_count": overlap.overlap_count if overlap else 0,
        "group_size": len(members),
        "ground_truth_size": overlap.total_scenario_participants if overlap else 0,
        "participant_coverage": overlap.coverage if overlap else 0.0,
        "purity": overlap.purity if overlap else 0.0,
        "is_detected": overlap.overlap_count > 0 if overlap else False,
        "exact_match": bool(overlap and overlap.overlap_count == len(members) == overlap.total_scenario_participants),
        "suspicious_score": suspicious_score,
    }


def best_scenario_overlap(members: list[str], scenario_attribution: ScenarioAttributionIndex) -> ScenarioOverlap | None:
    best: ScenarioOverlap | None = None
    member_set = set(members)
    mixed_scenarios = {
        scenario_id
        for scenario_id, participants in scenario_attribution.scenario_to_traders.items()
        if member_set & set(participants)
    }
    for scenario_id, participants in scenario_attribution.scenario_to_traders.items():
        participant_set = set(participants)
        overlap_count = len(member_set & participant_set)
        if overlap_count == 0:
            continue
        detail = scenario_attribution.scenario_details.get(scenario_id)
        coverage = overlap_count / max(len(participant_set), 1)
        purity = overlap_count / max(len(member_set), 1)
        candidate = ScenarioOverlap(
            scenario_id=scenario_id,
            scenario_type=detail.scenario_type if detail else "",
            overlap_count=overlap_count,
            total_scenario_participants=len(participant_set),
            coverage=coverage,
            purity=purity,
            is_mixed=len(mixed_scenarios) > 1,
        )
        if best is None or (candidate.coverage, candidate.purity, candidate.overlap_count) > (
            best.coverage,
            best.purity,
            best.overlap_count,
        ):
            best = candidate
    return best


def _coerce_scenario_attribution(
    scenario_attribution: ScenarioAttributionIndex | dict[str, list[str]],
) -> ScenarioAttributionIndex:
    if isinstance(scenario_attribution, ScenarioAttributionIndex):
        return scenario_attribution
    scenario_to_traders = {
        scenario_id: sorted(str(participant) for participant in participants)
        for scenario_id, participants in scenario_attribution.items()
    }
    return ScenarioAttributionIndex(
        trader_to_scenarios={},
        scenario_to_traders=scenario_to_traders,
        scenario_details={},
    )
