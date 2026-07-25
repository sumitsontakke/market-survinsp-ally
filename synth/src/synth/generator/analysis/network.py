from __future__ import annotations

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import networkx as nx
import pandas as pd


def build_correlation_graph(correlation_matrix: pd.DataFrame, threshold: float) -> nx.Graph:
    graph = nx.Graph()
    for node in correlation_matrix.columns:
        graph.add_node(str(node))
    for left_index, left in enumerate(correlation_matrix.columns):
        for right in correlation_matrix.columns[left_index + 1 :]:
            weight = float(correlation_matrix.loc[left, right])
            if weight >= threshold:
                graph.add_edge(str(left), str(right), weight=weight)
    return graph


def export_edge_list(graph: nx.Graph) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"source": source, "target": target, "weight": data["weight"]}
            for source, target, data in graph.edges(data=True)
        ]
    )


def save_graph_visualization(graph: nx.Graph, output_path: str | Path) -> None:
    output = Path(output_path)
    fig, axis = plt.subplots(figsize=(8, 6))
    if graph.number_of_nodes() == 0:
        axis.set_title("Empty Correlation Graph")
        axis.axis("off")
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
        return
    position = nx.spring_layout(graph, seed=7, weight="weight")
    edge_weights = [data["weight"] for _, _, data in graph.edges(data=True)]
    nx.draw_networkx_nodes(graph, position, ax=axis, node_color="#1f77b4", node_size=600, alpha=0.9)
    nx.draw_networkx_labels(graph, position, ax=axis, font_size=8)
    if edge_weights:
        nx.draw_networkx_edges(
            graph,
            position,
            ax=axis,
            width=[1.5 + weight * 2 for weight in edge_weights],
            edge_color=edge_weights,
            edge_cmap=plt.cm.Reds,
            alpha=0.75,
        )
    axis.set_title("Thresholded Trader Correlation Graph")
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_highlighted_suspicious_graph(
    graph: nx.Graph,
    suspicious_groups: pd.DataFrame,
    detection_evaluation: dict[str, object],
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    fig, axis = plt.subplots(figsize=(12, 8))
    if graph.number_of_nodes() == 0:
        axis.set_title("No Highlighted Suspicious Network Available")
        axis.axis("off")
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
        return

    position = nx.spring_layout(graph, seed=17, weight="weight")
    group_records = suspicious_groups.head(6).to_dict("records") if not suspicious_groups.empty else []
    group_palette = ["#d1495b", "#00798c", "#edae49", "#30638e", "#8f2d56", "#4f772d"]
    group_members = {
        record["group_id"]: set(json.loads(record["participant_ids"]))
        for record in group_records
    }
    node_to_group_color: dict[str, str] = {}
    for index, record in enumerate(group_records):
        color = group_palette[index % len(group_palette)]
        for node in group_members[record["group_id"]]:
            node_to_group_color.setdefault(node, color)

    injected = set(detection_evaluation.get("true_positive_traders", [])) | set(detection_evaluation.get("false_negative_traders", []))
    flagged = set(detection_evaluation.get("true_positive_traders", [])) | set(detection_evaluation.get("false_positive_traders", []))

    node_colors = [node_to_group_color.get(node, "#d9dde5") for node in graph.nodes()]
    edge_colors = []
    edge_widths = []
    for left, right in graph.edges():
        if any(left in members and right in members for members in group_members.values()):
            edge_colors.append("#202938")
            edge_widths.append(2.8)
        else:
            edge_colors.append("#c8ced8")
            edge_widths.append(0.8)

    nx.draw_networkx_edges(
        graph,
        position,
        ax=axis,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.82,
    )
    nx.draw_networkx_nodes(
        graph,
        position,
        ax=axis,
        node_color=node_colors,
        node_size=760,
        linewidths=[_node_border_width(node, injected, flagged) for node in graph.nodes()],
        edgecolors=[_node_border_color(node, injected, flagged) for node in graph.nodes()],
        alpha=0.95,
    )
    nx.draw_networkx_labels(graph, position, ax=axis, font_size=7)
    axis.set_title("Highlighted Suspicious Network")
    axis.axis("off")
    axis.legend(
        handles=_highlight_legend_handles(group_records, group_palette),
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8,
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def save_group_context_graph(
    graph: nx.Graph,
    suspicious_groups: pd.DataFrame,
    detection_evaluation: dict[str, object],
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    fig, axis = plt.subplots(figsize=(12, 8))
    if graph.number_of_nodes() == 0 or suspicious_groups.empty:
        axis.set_title("No Suspicious Group Context Network Available")
        axis.axis("off")
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
        return

    top_group = suspicious_groups.iloc[0].to_dict()
    selected_members = set(json.loads(top_group["participant_ids"]))
    injected = set(detection_evaluation.get("true_positive_traders", [])) | set(detection_evaluation.get("false_negative_traders", []))
    selected_neighbors = set()
    for member in selected_members:
        if member in graph:
            selected_neighbors.update(graph.neighbors(member))
    dominant_members = set()
    if top_group.get("dominant_scenario"):
        dominant_members = set(
            detection_evaluation.get("true_positive_traders", [])
        ) | (set(detection_evaluation.get("false_negative_traders", [])) & injected)
    context_nodes = selected_members | selected_neighbors | dominant_members
    subgraph = graph.subgraph(sorted(context_nodes)).copy() if context_nodes else graph.__class__()
    if subgraph.number_of_nodes() == 0:
        axis.set_title("No Suspicious Group Context Network Available")
        axis.axis("off")
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
        return

    flagged = set(detection_evaluation.get("true_positive_traders", [])) | set(detection_evaluation.get("false_positive_traders", []))
    position = nx.spring_layout(subgraph, seed=17, weight="weight")
    edge_colors = []
    edge_widths = []
    for left, right in subgraph.edges():
        if left in selected_members and right in selected_members:
            edge_colors.append("#111827")
            edge_widths.append(3.2)
        else:
            edge_colors.append("#c7d0da")
            edge_widths.append(1.0)

    node_colors = []
    for node in subgraph.nodes():
        if node in selected_members:
            node_colors.append("#d1495b")
        elif node in injected:
            node_colors.append("#8ecae6")
        else:
            node_colors.append("#d9dde5")

    nx.draw_networkx_edges(subgraph, position, ax=axis, edge_color=edge_colors, width=edge_widths, alpha=0.84)
    nx.draw_networkx_nodes(
        subgraph,
        position,
        ax=axis,
        node_color=node_colors,
        node_size=820,
        linewidths=[_node_border_width(node, injected, flagged) for node in subgraph.nodes()],
        edgecolors=[_node_border_color(node, injected, flagged) for node in subgraph.nodes()],
        alpha=0.96,
    )
    nx.draw_networkx_labels(subgraph, position, ax=axis, font_size=7)
    axis.set_title("Manipulative Context Network: {0}".format(top_group["group_id"]))
    axis.axis("off")
    axis.legend(
        handles=[
            Patch(facecolor="#d1495b", edgecolor="#1f2937", label="Selected suspicious group"),
            Patch(facecolor="#8ecae6", edgecolor="#6f42c1", label="Injected context"),
            Patch(facecolor="#d9dde5", edgecolor="#6b7280", label="Background context"),
            Line2D([0], [0], color="#18794e", linewidth=2.5, label="Injected + detected border"),
            Line2D([0], [0], color="#c76b00", linewidth=2.5, label="Detected-only border"),
            Line2D([0], [0], color="#6f42c1", linewidth=2.5, label="Injected-only border"),
        ],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8,
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _node_border_color(node: str, injected: set[str], flagged: set[str]) -> str:
    if node in injected and node in flagged:
        return "#18794e"
    if node in flagged:
        return "#c76b00"
    if node in injected:
        return "#6f42c1"
    return "#9aa5b1"


def _node_border_width(node: str, injected: set[str], flagged: set[str]) -> float:
    if node in injected or node in flagged:
        return 2.8
    return 1.1


def _highlight_legend_handles(group_records: list[dict[str, object]], group_palette: list[str]) -> list[object]:
    handles: list[object] = [
        Patch(facecolor="#d9dde5", edgecolor="#9aa5b1", label="Background trader"),
        Line2D([0], [0], color="#18794e", linewidth=2.5, label="Injected + detected border"),
        Line2D([0], [0], color="#c76b00", linewidth=2.5, label="Detected-only border"),
        Line2D([0], [0], color="#6f42c1", linewidth=2.5, label="Injected-only border"),
        Line2D([0], [0], color="#202938", linewidth=2.8, label="Bold suspicious-group edge"),
        Line2D([0], [0], color="#c8ced8", linewidth=1.0, label="Faint background edge"),
    ]
    for index, record in enumerate(group_records):
        handles.append(
            Patch(
                facecolor=group_palette[index % len(group_palette)],
                edgecolor="#1f2937",
                label="{0} | size {1} | {2} | purity {3:.2f}".format(
                    record["group_id"],
                    record.get("trader_count", 0),
                    record.get("dominant_scenario") or "no scenario",
                    float(record.get("purity", 0.0)),
                ),
            )
        )
    return handles
