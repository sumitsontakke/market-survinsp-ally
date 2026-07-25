from __future__ import annotations

from pathlib import Path

import pandas as pd

from synthetic_market_sim.analysis.dataset_loader import LoadedDataset
from synthetic_market_sim.analysis.signed_volume import build_signed_volume_matrix


def test_signed_volume_matrix_aggregates_and_zero_fills() -> None:
    dataset = LoadedDataset(
        root=Path("."),
        traders=pd.DataFrame({"trader_id": ["trader_a", "trader_b", "trader_c"]}),
        orders=pd.DataFrame(
            [
                {"timestamp": pd.Timestamp("2026-03-14T09:30:10"), "trader_id": "trader_a", "instrument_id": "inst_1", "side": "buy", "quantity": 10},
                {"timestamp": pd.Timestamp("2026-03-14T09:32:00"), "trader_id": "trader_b", "instrument_id": "inst_1", "side": "sell", "quantity": 5},
            ]
        ),
        trades=pd.DataFrame(),
        scenarios=pd.DataFrame(),
        instruments=pd.DataFrame({"instrument_id": ["inst_1"]}),
        manifest={},
    )
    matrix = build_signed_volume_matrix(dataset=dataset, source="orders", bucket_minutes=1)
    assert list(matrix.columns) == ["trader_a", "trader_b", "trader_c"]
    assert list(matrix.index.astype(str)) == ["2026-03-14 09:30:00", "2026-03-14 09:31:00", "2026-03-14 09:32:00"]
    assert matrix.loc[pd.Timestamp("2026-03-14T09:30:00"), "trader_a"] == 10
    assert matrix.loc[pd.Timestamp("2026-03-14T09:32:00"), "trader_b"] == -5
    assert matrix.loc[pd.Timestamp("2026-03-14T09:31:00"), "trader_c"] == 0
