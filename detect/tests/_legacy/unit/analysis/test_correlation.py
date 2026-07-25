from __future__ import annotations

import pandas as pd

from synthetic_market_sim.analysis.correlation import compute_correlation_matrix, filter_active_traders


def test_correlation_matrix_finds_aligned_traders() -> None:
    matrix = pd.DataFrame(
        {
            "trader_a": [1, 2, 3, 4],
            "trader_b": [2, 4, 6, 8],
            "trader_c": [1, 0, 1, 0],
        }
    )
    filtered = filter_active_traders(matrix, min_active_buckets=2)
    correlation = compute_correlation_matrix(filtered, method="pearson")
    assert correlation.loc["trader_a", "trader_b"] > 0.99
    assert correlation.loc["trader_a", "trader_c"] < 0.0
