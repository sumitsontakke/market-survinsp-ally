from __future__ import annotations

import pandas as pd

from synth.generator.analysis.correlation import top_correlated_pairs
from synth.generator.analysis.dataset_loader import LoadedDataset
from synth.generator.analysis.signed_volume import build_signed_volume_matrix


def signed_volume_by_bucket(orders: list[dict], participant_ids: list[str]) -> dict[str, dict[str, float]]:
    orders_frame = pd.DataFrame(orders)
    if not orders_frame.empty and "timestamp" in orders_frame:
        orders_frame["timestamp"] = pd.to_datetime(orders_frame["timestamp"])
    dataset = LoadedDataset(
        root=None,
        traders=pd.DataFrame({"trader_id": participant_ids}),
        orders=orders_frame,
        trades=pd.DataFrame(),
        scenarios=pd.DataFrame(),
        instruments=pd.DataFrame({"instrument_id": sorted({order["instrument_id"] for order in orders})}),
        manifest={},
    )
    matrix = build_signed_volume_matrix(dataset=dataset, source="orders", bucket_minutes=1)
    selected = matrix.reindex(columns=participant_ids, fill_value=0.0)
    return {
        trader_id: {bucket.isoformat(): float(value) for bucket, value in series.items() if value != 0}
        for trader_id, series in selected.items()
    }


def pairwise_correlations(bucketed_volume: dict[str, dict[str, float]]) -> list[tuple[str, str, float]]:
    if not bucketed_volume:
        return []
    all_buckets = sorted({bucket for values in bucketed_volume.values() for bucket in values.keys()})
    matrix = pd.DataFrame(
        {
            trader_id: [values.get(bucket, 0.0) for bucket in all_buckets]
            for trader_id, values in bucketed_volume.items()
        }
    )
    return top_correlated_pairs(matrix.corr(method="pearson").fillna(0.0), top_k=10_000)
