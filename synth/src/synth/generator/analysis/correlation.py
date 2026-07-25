from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def filter_active_traders(matrix: pd.DataFrame, min_active_buckets: int) -> pd.DataFrame:
    if matrix.empty:
        return matrix
    active_counts = (matrix != 0).sum(axis=0)
    kept_columns = active_counts[active_counts >= min_active_buckets].index
    return matrix.loc[:, kept_columns]


def compute_correlation_matrix(matrix: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    if matrix.empty:
        return pd.DataFrame()
    return matrix.corr(method=method).fillna(0.0)


def top_correlated_pairs(correlation_matrix: pd.DataFrame, top_k: int = 10) -> list[tuple[str, str, float]]:
    if correlation_matrix.empty:
        return []
    pairs = []
    columns = list(correlation_matrix.columns)
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1 :]:
            pairs.append((str(left), str(right), float(correlation_matrix.loc[left, right])))
    pairs.sort(key=lambda item: item[2], reverse=True)
    return pairs[:top_k]


def correlation_edges(correlation_matrix: pd.DataFrame, threshold: float) -> pd.DataFrame:
    edges = []
    for left_index, left in enumerate(correlation_matrix.columns):
        for right in correlation_matrix.columns[left_index + 1 :]:
            weight = float(correlation_matrix.loc[left, right])
            if weight >= threshold:
                edges.append({"source": str(left), "target": str(right), "weight": weight})
    return pd.DataFrame(edges, columns=["source", "target", "weight"])


def save_heatmap(correlation_matrix: pd.DataFrame, output_path: str | Path) -> None:
    output = Path(output_path)
    if correlation_matrix.empty:
        plt.figure(figsize=(4, 3))
        plt.title("Empty Correlation Matrix")
        plt.savefig(output, bbox_inches="tight")
        plt.close()
        return
    size = max(6, min(14, int(len(correlation_matrix.columns) * 0.6)))
    fig, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(correlation_matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(correlation_matrix.columns)))
    axis.set_xticklabels(correlation_matrix.columns, rotation=90, fontsize=8)
    axis.set_yticks(range(len(correlation_matrix.index)))
    axis.set_yticklabels(correlation_matrix.index, fontsize=8)
    axis.set_title("Trader Correlation Heatmap")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
