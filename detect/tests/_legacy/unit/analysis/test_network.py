from __future__ import annotations

import pandas as pd

from synthetic_market_sim.analysis.network import build_correlation_graph


def test_graph_threshold_keeps_only_strong_edges() -> None:
    correlation = pd.DataFrame(
        [[1.0, 0.85, 0.4], [0.85, 1.0, 0.2], [0.4, 0.2, 1.0]],
        index=["trader_a", "trader_b", "trader_c"],
        columns=["trader_a", "trader_b", "trader_c"],
    )
    graph = build_correlation_graph(correlation, threshold=0.7)
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 1
    assert graph.has_edge("trader_a", "trader_b")
