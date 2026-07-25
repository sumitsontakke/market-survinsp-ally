from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def save_signed_volume_timeseries(
    matrix: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 8,
    title: str = "Signed Volume Time Series",
) -> None:
    output = Path(output_path)
    fig, axis = plt.subplots(figsize=(12, 6))
    if matrix.empty:
        axis.set_title("Empty Signed Volume Matrix")
        axis.axis("off")
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
        return

    trader_activity = matrix.abs().sum(axis=0).sort_values(ascending=False)
    selected_traders = trader_activity.head(top_n).index.tolist()
    for trader_id in selected_traders:
        axis.plot(matrix.index, matrix[trader_id], label=str(trader_id), linewidth=1.7)
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.7)
    axis.set_title(title)
    axis.set_xlabel("Time Bucket")
    axis.set_ylabel("Signed Volume")
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_group_timeseries(
    matrix: pd.DataFrame,
    suspicious_groups: pd.DataFrame,
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    if suspicious_groups.empty:
        save_signed_volume_timeseries(pd.DataFrame(), output, title="No Suspicious Group Available")
        return
    top_group = suspicious_groups.iloc[0]
    participants = json.loads(top_group["participant_ids"])
    group_matrix = matrix.reindex(columns=participants, fill_value=0.0)
    title = "Top Suspicious Group Signed Volume: {0}".format(top_group["group_id"])
    save_signed_volume_timeseries(
        matrix=group_matrix,
        output_path=output,
        top_n=len(participants),
        title=title,
    )
