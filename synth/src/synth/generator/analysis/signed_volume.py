from __future__ import annotations

from itertools import product
from typing import Iterable, Optional, Sequence

import pandas as pd

from synth.generator.analysis.dataset_loader import LoadedDataset


def build_signed_volume_matrix(
    dataset: LoadedDataset,
    source: str = "orders",
    bucket_minutes: int = 1,
    instrument_id: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    group_by: Sequence[str] = ("trader_id",),
) -> pd.DataFrame:
    events = _event_frame(dataset, source)
    if events.empty:
        return pd.DataFrame()
    events = events.copy()
    if instrument_id:
        events = events.loc[events["instrument_id"] == instrument_id]
    if start_time:
        events = events.loc[events["timestamp"] >= pd.Timestamp(start_time)]
    if end_time:
        events = events.loc[events["timestamp"] <= pd.Timestamp(end_time)]
    if events.empty:
        return pd.DataFrame()

    bucket_frequency = "{0}min".format(bucket_minutes)
    events["time_bucket"] = events["timestamp"].dt.floor(bucket_frequency)
    grouped = (
        events.groupby(["time_bucket", *group_by], dropna=False)["signed_volume"]
        .sum()
        .reset_index()
    )
    matrix = grouped.pivot_table(
        index="time_bucket",
        columns=list(group_by),
        values="signed_volume",
        aggfunc="sum",
        fill_value=0.0,
    )
    matrix = _complete_time_index(matrix, bucket_minutes)
    matrix = _complete_columns(matrix, dataset, group_by, instrument_id)
    return matrix.sort_index().fillna(0.0)


def _event_frame(dataset: LoadedDataset, source: str) -> pd.DataFrame:
    if source == "orders":
        frame = dataset.orders.copy()
        frame["signed_volume"] = frame["quantity"].astype(float).where(frame["side"] == "buy", -frame["quantity"].astype(float))
        return frame[["timestamp", "trader_id", "instrument_id", "signed_volume"]]
    if source == "trades":
        trades = dataset.trades.copy()
        buy_leg = trades[["timestamp", "buy_trader_id", "instrument_id", "quantity"]].rename(columns={"buy_trader_id": "trader_id"})
        buy_leg["signed_volume"] = buy_leg["quantity"].astype(float)
        sell_leg = trades[["timestamp", "sell_trader_id", "instrument_id", "quantity"]].rename(columns={"sell_trader_id": "trader_id"})
        sell_leg["signed_volume"] = -sell_leg["quantity"].astype(float)
        frame = pd.concat([buy_leg, sell_leg], ignore_index=True)
        return frame[["timestamp", "trader_id", "instrument_id", "signed_volume"]]
    raise ValueError("Unsupported source: {0}".format(source))


def _complete_time_index(matrix: pd.DataFrame, bucket_minutes: int) -> pd.DataFrame:
    if matrix.empty:
        return matrix
    full_index = pd.date_range(matrix.index.min(), matrix.index.max(), freq="{0}min".format(bucket_minutes))
    return matrix.reindex(full_index, fill_value=0.0)


def _complete_columns(
    matrix: pd.DataFrame,
    dataset: LoadedDataset,
    group_by: Sequence[str],
    instrument_id: Optional[str],
) -> pd.DataFrame:
    if matrix.empty:
        return matrix
    if tuple(group_by) == ("trader_id",):
        trader_ids = sorted(dataset.traders["trader_id"].astype(str).tolist())
        return matrix.reindex(columns=trader_ids, fill_value=0.0)
    if tuple(group_by) == ("instrument_id", "trader_id"):
        instrument_ids = [instrument_id] if instrument_id else sorted(dataset.instruments["instrument_id"].astype(str).tolist())
        trader_ids = sorted(dataset.traders["trader_id"].astype(str).tolist())
        full_columns = pd.MultiIndex.from_tuples(list(product(instrument_ids, trader_ids)), names=["instrument_id", "trader_id"])
        return matrix.reindex(columns=full_columns, fill_value=0.0)
    return matrix
