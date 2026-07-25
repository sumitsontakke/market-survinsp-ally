from __future__ import annotations

import pandas as pd

from synth.generator.analysis.dataset_loader import LoadedDataset


def build_directed_trade_edges(dataset: LoadedDataset) -> pd.DataFrame:
    if dataset.trades.empty:
        return pd.DataFrame(columns=["seller", "buyer", "trade_count", "total_quantity", "scenario_ids"])
    trades = dataset.trades.copy()
    grouped = (
        trades.groupby(["sell_trader_id", "buy_trader_id"], dropna=False)
        .agg(
            trade_count=("trade_id", "count"),
            total_quantity=("quantity", "sum"),
            scenario_ids=("scenario_id", lambda values: sorted(set(values))),
        )
        .reset_index()
        .rename(columns={"sell_trader_id": "seller", "buy_trader_id": "buyer"})
    )
    grouped["scenario_ids"] = grouped["scenario_ids"].apply(lambda values: ",".join(str(value) for value in values))
    return grouped.sort_values(["trade_count", "total_quantity"], ascending=False).reset_index(drop=True)
