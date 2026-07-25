from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from synth.generator.analysis.correlation import (
    compute_correlation_matrix,
    correlation_edges,
    filter_active_traders,
    save_heatmap,
    top_correlated_pairs,
)
from synth.generator.analysis.dataset_loader import load_dataset
from synth.generator.analysis.detection_evaluation import evaluate_detection_quality, write_detection_evaluation_artifacts
from synth.generator.analysis.network import (
    build_correlation_graph,
    export_edge_list,
    save_graph_visualization,
    save_group_context_graph,
    save_highlighted_suspicious_graph,
)
from synth.generator.analysis.reporting import write_analysis_summary
from synth.generator.analysis.signed_volume import build_signed_volume_matrix
from synth.generator.analysis.subgraph_detection import detect_suspicious_groups
from synth.generator.analysis.trade_flow import build_directed_trade_edges
from synth.generator.analysis.timeseries import save_group_timeseries, save_signed_volume_timeseries


def run_analysis(
    input_dir: str,
    output_dir: str,
    source: str = "orders",
    bucket_minutes: int = 1,
    corr_method: str = "pearson",
    corr_threshold: float = 0.7,
    min_active_buckets: int = 2,
    instrument_id: Optional[str] = None,
) -> dict[str, object]:
    dataset = load_dataset(input_dir)
    scoped_dataset = dataset.scoped_to_instrument(instrument_id)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    matrix = build_signed_volume_matrix(
        dataset=scoped_dataset,
        source=source,
        bucket_minutes=bucket_minutes,
        instrument_id=instrument_id,
    )
    filtered_matrix = filter_active_traders(matrix, min_active_buckets=min_active_buckets)
    corr_matrix = compute_correlation_matrix(filtered_matrix, method=corr_method)
    edges = correlation_edges(corr_matrix, threshold=corr_threshold)
    directed_trade_edges = build_directed_trade_edges(scoped_dataset)
    graph = build_correlation_graph(corr_matrix, threshold=corr_threshold)
    suspicious_groups = detect_suspicious_groups(graph, scoped_dataset.scenario_attribution_index())
    top_pairs = top_correlated_pairs(corr_matrix, top_k=10)

    matrix.to_csv(output_root / "signed_volume_matrix.csv")
    corr_matrix.to_csv(output_root / "correlation_matrix.csv")
    edges.to_csv(output_root / "correlation_edges.csv", index=False)
    directed_trade_edges.to_csv(output_root / "directed_trade_edges.csv", index=False)
    suspicious_groups.to_csv(output_root / "suspicious_groups.csv", index=False)
    detection_evaluation = evaluate_detection_quality(scoped_dataset, suspicious_groups)
    detection_evaluation["scope_type"] = "instrument_scoped" if instrument_id else "full_dataset"
    detection_evaluation["instrument_id"] = instrument_id or ""
    detection_artifacts = write_detection_evaluation_artifacts(output_root, detection_evaluation)
    save_heatmap(corr_matrix, output_root / "correlation_heatmap.png")
    save_graph_visualization(graph, output_root / "network_graph.png")
    save_highlighted_suspicious_graph(graph, suspicious_groups, detection_evaluation, output_root / "network_graph_highlighted.png")
    save_group_context_graph(graph, suspicious_groups, detection_evaluation, output_root / "network_graph_clique_context.png")
    save_signed_volume_timeseries(matrix, output_root / "signed_volume_timeseries.png")
    save_group_timeseries(matrix, suspicious_groups, output_root / "top_suspicious_group_timeseries.png")
    summary_path = write_analysis_summary(
        output_dir=output_root,
        input_dir=input_dir,
        source=source,
        bucket_minutes=bucket_minutes,
        corr_method=corr_method,
        corr_threshold=corr_threshold,
        matrix=filtered_matrix,
        top_pairs=top_pairs,
        directed_trade_edges=directed_trade_edges,
        suspicious_groups=suspicious_groups,
        detection_evaluation=detection_evaluation,
    )

    return {
        "scope_type": "instrument_scoped" if instrument_id else "full_dataset",
        "instrument_id": instrument_id or "",
        "analysis_source": source,
        "input_dataset": str(input_dir),
        "bucket_minutes": int(bucket_minutes),
        "correlation_method": corr_method,
        "correlation_threshold": float(corr_threshold),
        "retained_correlation_edges": int(len(edges)),
        "traders_analyzed": int(filtered_matrix.shape[1]) if not filtered_matrix.empty else 0,
        "matrix_shape": list(filtered_matrix.shape),
        "orders_in_dataset": int(len(scoped_dataset.orders)),
        "trades_in_dataset": int(len(scoped_dataset.trades)),
        "scenario_count": int(len(scoped_dataset.scenario_attribution_index().scenario_details)),
        "top_correlated_pairs": top_pairs[:5],
        "highest_pair_correlation": float(top_pairs[0][2]) if top_pairs else 0.0,
        "top_directed_trade_edges": directed_trade_edges.head(5).to_dict("records"),
        "suspicious_group_count": int(len(suspicious_groups)),
        "largest_group_size": int(suspicious_groups["trader_count"].max()) if not suspicious_groups.empty else 0,
        "detection_evaluation": detection_evaluation,
        "detection_artifacts": detection_artifacts,
        "summary_report": summary_path,
    }
