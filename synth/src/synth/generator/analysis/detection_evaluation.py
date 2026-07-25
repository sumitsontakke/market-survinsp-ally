from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from synth.generator.analysis.dataset_loader import LoadedDataset


def evaluate_detection_quality(
    dataset: LoadedDataset,
    suspicious_groups: pd.DataFrame,
) -> dict[str, Any]:
    all_traders = _all_traders(dataset)
    injected_traders = sorted({str(participant) for participants in dataset.scenario_participants().values() for participant in participants})
    flagged_traders = sorted(_flagged_trader_ids(suspicious_groups))

    true_positive_traders = sorted(set(injected_traders) & set(flagged_traders))
    false_positive_traders = sorted(set(flagged_traders) - set(injected_traders))
    false_negative_traders = sorted(set(injected_traders) - set(flagged_traders))
    true_negative_count = len(set(all_traders) - set(injected_traders) - set(flagged_traders))

    total_scenarios = len(dataset.scenario_participants())
    scenario_evaluations = _scenario_evaluations(dataset, suspicious_groups)
    detected_scenario_count = sum(1 for item in scenario_evaluations if item["is_detected"])
    exact_match_count = sum(1 for item in scenario_evaluations if item["exact_match"])

    has_ground_truth = bool(injected_traders)
    true_positive_count = len(true_positive_traders)
    false_positive_count = len(false_positive_traders)
    false_negative_count = len(false_negative_traders)

    precision = _safe_ratio(true_positive_count, true_positive_count + false_positive_count)
    recall = _safe_ratio(true_positive_count, true_positive_count + false_negative_count)
    f1_score = _safe_ratio(2 * precision * recall, precision + recall) if precision and recall else 0.0
    false_positive_rate = _safe_ratio(false_positive_count, false_positive_count + true_negative_count)
    false_negative_rate = _safe_ratio(false_negative_count, false_negative_count + true_positive_count)

    if not has_ground_truth:
        verdict = "no_ground_truth"
    elif detected_scenario_count == 0:
        verdict = "missed"
    elif exact_match_count == total_scenarios and false_positive_count == 0 and false_negative_count == 0:
        verdict = "exact_match"
    elif false_positive_count > 0 or false_negative_count > 0:
        verdict = "partial_detection"
    else:
        verdict = "detected"

    return {
        "has_ground_truth": has_ground_truth,
        "total_traders": len(all_traders),
        "injected_trader_count": len(injected_traders),
        "flagged_trader_count": len(flagged_traders),
        "true_positive_count": true_positive_count,
        "false_positive_count": false_positive_count,
        "false_negative_count": false_negative_count,
        "true_negative_count": true_negative_count,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "true_positive_traders": true_positive_traders,
        "false_positive_traders": false_positive_traders,
        "false_negative_traders": false_negative_traders,
        "scenario_evaluations": scenario_evaluations,
        "total_scenarios": total_scenarios,
        "detected_scenario_count": detected_scenario_count,
        "exact_match_count": exact_match_count,
        "is_detected": detected_scenario_count > 0,
        "exact_match": exact_match_count == total_scenarios and total_scenarios > 0,
        "verdict": verdict,
    }


def write_detection_evaluation_artifacts(
    output_dir: str | Path,
    evaluation: dict[str, Any],
) -> dict[str, str]:
    output_root = Path(output_dir)
    detection_evaluation_path = output_root / "detection_evaluation.json"
    confusion_matrix_json_path = output_root / "confusion_matrix.json"
    confusion_matrix_csv_path = output_root / "confusion_matrix.csv"
    confusion_matrix_png_path = output_root / "confusion_matrix.png"

    confusion_matrix_payload = {
        "TP": evaluation["true_positive_count"],
        "FP": evaluation["false_positive_count"],
        "FN": evaluation["false_negative_count"],
        "TN": evaluation["true_negative_count"],
        "precision": evaluation["precision"],
        "recall": evaluation["recall"],
        "f1_score": evaluation["f1_score"],
        "false_positive_rate": evaluation["false_positive_rate"],
        "false_negative_rate": evaluation["false_negative_rate"],
        "has_ground_truth": evaluation["has_ground_truth"],
    }

    detection_evaluation_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    confusion_matrix_json_path.write_text(json.dumps(confusion_matrix_payload, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {"metric": "TP", "value": evaluation["true_positive_count"]},
            {"metric": "FP", "value": evaluation["false_positive_count"]},
            {"metric": "FN", "value": evaluation["false_negative_count"]},
            {"metric": "TN", "value": evaluation["true_negative_count"]},
            {"metric": "precision", "value": evaluation["precision"]},
            {"metric": "recall", "value": evaluation["recall"]},
            {"metric": "f1_score", "value": evaluation["f1_score"]},
            {"metric": "false_positive_rate", "value": evaluation["false_positive_rate"]},
            {"metric": "false_negative_rate", "value": evaluation["false_negative_rate"]},
        ]
    ).to_csv(confusion_matrix_csv_path, index=False)
    save_confusion_matrix_visualization(evaluation, confusion_matrix_png_path)

    return {
        "detection_evaluation": str(detection_evaluation_path),
        "confusion_matrix_json": str(confusion_matrix_json_path),
        "confusion_matrix_csv": str(confusion_matrix_csv_path),
        "confusion_matrix_png": str(confusion_matrix_png_path),
    }


def save_confusion_matrix_visualization(
    evaluation: dict[str, Any],
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    fig, axis = plt.subplots(figsize=(4.5, 4))
    if not evaluation.get("has_ground_truth"):
        axis.set_title("No Ground Truth Available")
        axis.axis("off")
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
        return

    matrix = pd.DataFrame(
        [
            [evaluation["true_positive_count"], evaluation["false_negative_count"]],
            [evaluation["false_positive_count"], evaluation["true_negative_count"]],
        ],
        index=["Injected +", "Injected -"],
        columns=["Flagged +", "Flagged -"],
    )
    image = axis.imshow(matrix.values, cmap="Blues")
    axis.set_xticks(range(len(matrix.columns)), matrix.columns)
    axis.set_yticks(range(len(matrix.index)), matrix.index)
    axis.set_title("Trader-Level Confusion Matrix")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(column_index, row_index, str(matrix.iat[row_index, column_index]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _scenario_evaluations(dataset: LoadedDataset, suspicious_groups: pd.DataFrame) -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    manipulative = dataset.manipulative_scenarios()
    if manipulative.empty:
        return evaluations
    for row in manipulative.to_dict("records"):
        scenario_id = str(row["scenario_id"])
        scenario_type = str(row.get("scenario_type", "unknown"))
        participants = {str(participant) for participant in row.get("participant_ids", [])}
        best_match = _best_group_match(participants, suspicious_groups)
        overlap_count = best_match["overlap_count"] if best_match else 0
        coverage = _safe_ratio(overlap_count, len(participants))
        purity = best_match["purity"] if best_match else 0.0
        evaluations.append(
            {
                "scenario_id": scenario_id,
                "scenario_type": scenario_type,
                "participant_count": len(participants),
                "best_group_id": best_match["group_id"] if best_match else "",
                "best_group_size": best_match["group_size"] if best_match else 0,
                "overlap_count": overlap_count,
                "coverage": coverage,
                "purity": purity,
                "is_detected": overlap_count > 0,
                "exact_match": best_match["group_members"] == participants if best_match else False,
            }
        )
    return evaluations


def _best_group_match(participants: set[str], suspicious_groups: pd.DataFrame) -> dict[str, Any] | None:
    best_match: dict[str, Any] | None = None
    if suspicious_groups.empty:
        return None
    for record in suspicious_groups.to_dict("records"):
        members = {str(member) for member in json.loads(record["participant_ids"])}
        overlap = len(participants & members)
        if overlap == 0:
            continue
        purity = _safe_ratio(overlap, len(members))
        candidate = {
            "group_id": record["group_id"],
            "group_size": len(members),
            "group_members": members,
            "overlap_count": overlap,
            "coverage": _safe_ratio(overlap, len(participants)),
            "purity": purity,
        }
        if best_match is None or (candidate["coverage"], candidate["purity"], candidate["overlap_count"]) > (
            best_match["coverage"],
            best_match["purity"],
            best_match["overlap_count"],
        ):
            best_match = candidate
    return best_match


def _flagged_trader_ids(suspicious_groups: pd.DataFrame) -> set[str]:
    flagged: set[str] = set()
    if suspicious_groups.empty or "participant_ids" not in suspicious_groups.columns:
        return flagged
    for value in suspicious_groups["participant_ids"].tolist():
        if pd.isna(value):
            continue
        flagged.update(str(member) for member in json.loads(value))
    return flagged


def _all_traders(dataset: LoadedDataset) -> list[str]:
    trader_ids: set[str] = set()
    if not dataset.traders.empty and "trader_id" in dataset.traders.columns:
        trader_ids.update(dataset.traders["trader_id"].dropna().astype(str).tolist())
    if not dataset.orders.empty and "trader_id" in dataset.orders.columns:
        trader_ids.update(dataset.orders["trader_id"].dropna().astype(str).tolist())
    for column in ("buy_trader_id", "sell_trader_id"):
        if not dataset.trades.empty and column in dataset.trades.columns:
            trader_ids.update(dataset.trades[column].dropna().astype(str).tolist())
    return sorted(trader_ids)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)
