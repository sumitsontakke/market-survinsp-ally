from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_analysis_summary(
    output_dir: str | Path,
    input_dir: str | Path,
    source: str,
    bucket_minutes: int,
    corr_method: str,
    corr_threshold: float,
    matrix: pd.DataFrame,
    top_pairs: list[tuple[str, str, float]],
    directed_trade_edges: pd.DataFrame,
    suspicious_groups: pd.DataFrame,
    detection_evaluation: dict[str, object] | None = None,
) -> str:
    output = Path(output_dir) / "analysis_summary.md"
    lines = [
        "# Analysis Summary",
        "",
        "## Configuration",
        "",
        "- Input dataset: `{0}`".format(input_dir),
        "- Source: `{0}`".format(source),
        "- Scope: `{0}`".format(
            "instrument_scoped:{0}".format(detection_evaluation.get("instrument_id", "selected"))
            if detection_evaluation and detection_evaluation.get("scope_type") == "instrument_scoped"
            else "full_dataset"
        ),
        "- Bucket size: `{0}` minute(s)".format(bucket_minutes),
        "- Correlation method: `{0}`".format(corr_method),
        "- Correlation threshold: `{0}`".format(corr_threshold),
        "- Matrix shape: `{0}`".format(matrix.shape),
        "- Time-series chart: `signed_volume_timeseries.png`",
        "- Suspicious-group chart: `top_suspicious_group_timeseries.png`",
        "",
        "## Top Correlated Pairs",
        "",
    ]
    if top_pairs:
        for left, right, value in top_pairs[:10]:
            lines.append("- `{0}` / `{1}`: `{2:.4f}`".format(left, right, value))
    else:
        lines.append("- None")
    lines.extend(["", "## Suspicious Groups", ""])
    if suspicious_groups.empty:
        lines.append("- None")
    else:
        for record in suspicious_groups.head(10).to_dict("records"):
            lines.append(
                "- `{0}` `{1}` size=`{2}` density=`{3:.3f}` avg_weight=`{4:.3f}` dominant_scenario=`{5}` scenario_type=`{6}` members=`{7}/{8}` coverage=`{9:.3f}` purity=`{10:.3f}` mixed=`{11}` unmatched=`{12}`".format(
                    record["group_id"],
                    record["group_type"],
                    record["trader_count"],
                    record["density"],
                    record["average_edge_weight"],
                    record.get("dominant_scenario") or "none",
                    record.get("scenario_type") or "none",
                    record.get("group_scenario_member_count", 0),
                    record.get("total_scenario_participants", 0),
                    record.get("coverage", 0.0),
                    record.get("purity", 0.0),
                    record.get("is_mixed", False),
                    record.get("unmatched_flagged_count", 0),
                )
            )
    lines.extend(["", "## Top Group Attribution", ""])
    if suspicious_groups.empty:
        lines.append("- None")
    else:
        top_group = suspicious_groups.iloc[0].to_dict()
        lines.append(
            "- Top group `{0}` best matches `{1}` (`{2}`) with members=`{3}/{4}`, coverage=`{5:.3f}`, purity=`{6:.3f}`, mixed=`{7}`.".format(
                top_group["group_id"],
                top_group.get("dominant_scenario") or "none",
                top_group.get("scenario_type") or "none",
                top_group.get("group_scenario_member_count", 0),
                top_group.get("total_scenario_participants", 0),
                top_group.get("coverage", 0.0),
                top_group.get("purity", 0.0),
                top_group.get("is_mixed", False),
            )
        )
        for record in suspicious_groups.head(10).to_dict("records"):
            lines.append(
                "- Group `{0}` -> scenario `{1}` (`{2}`), coverage=`{3:.3f}`, purity=`{4:.3f}`, mixed=`{5}`.".format(
                    record["group_id"],
                    record.get("dominant_scenario") or "none",
                    record.get("scenario_type") or "none",
                    record.get("coverage", 0.0),
                    record.get("purity", 0.0),
                    record.get("is_mixed", False),
                )
            )
    lines.extend(["", "## Top Directed Trade Edges", ""])
    if directed_trade_edges.empty:
        lines.append("- None")
    else:
        for record in directed_trade_edges.head(10).to_dict("records"):
            lines.append(
                "- `{0}` -> `{1}` trades=`{2}` quantity=`{3}` scenarios=`{4}`".format(
                    record["seller"],
                    record["buyer"],
                    record["trade_count"],
                    record["total_quantity"],
                    record["scenario_ids"],
                )
            )
    lines.extend(["", "## Trader-Level Confusion Matrix", ""])
    if not detection_evaluation or not detection_evaluation.get("has_ground_truth"):
        if detection_evaluation and detection_evaluation.get("scope_type") == "instrument_scoped":
            lines.append("- No ground truth available for selected instrument scope.")
        else:
            lines.append("- No ground truth available.")
    else:
        lines.extend(
            [
                "- TP=`{0}` FP=`{1}` FN=`{2}` TN=`{3}`".format(
                    detection_evaluation.get("true_positive_count", 0),
                    detection_evaluation.get("false_positive_count", 0),
                    detection_evaluation.get("false_negative_count", 0),
                    detection_evaluation.get("true_negative_count", 0),
                ),
                "- Precision=`{0:.3f}` Recall=`{1:.3f}` F1=`{2:.3f}` FPR=`{3:.3f}` FNR=`{4:.3f}`".format(
                    detection_evaluation.get("precision", 0.0),
                    detection_evaluation.get("recall", 0.0),
                    detection_evaluation.get("f1_score", 0.0),
                    detection_evaluation.get("false_positive_rate", 0.0),
                    detection_evaluation.get("false_negative_rate", 0.0),
                ),
                "- Flagged traders=`{0}` Injected traders=`{1}` True positives=`{2}` False positives=`{3}` False negatives=`{4}`".format(
                    detection_evaluation.get("flagged_trader_count", 0),
                    detection_evaluation.get("injected_trader_count", 0),
                    detection_evaluation.get("true_positive_count", 0),
                    detection_evaluation.get("false_positive_count", 0),
                    detection_evaluation.get("false_negative_count", 0),
                ),
            ]
        )
    lines.extend(["", "## Interpretation", ""])
    if not detection_evaluation or not detection_evaluation.get("has_ground_truth"):
        lines.append("- No injected manipulative ground truth was available, so detector quality metrics are omitted.")
    elif suspicious_groups.empty:
        lines.append("- No suspicious groups were detected above the configured correlation threshold.")
    else:
        top_group = suspicious_groups.iloc[0].to_dict()
        dominant_scenario = top_group.get("dominant_scenario") or ""
        false_positive_count = int(detection_evaluation.get("false_positive_count", 0))
        false_negative_count = int(detection_evaluation.get("false_negative_count", 0))
        if not dominant_scenario:
            lines.append(
                "- The highest-ranked suspicious group does not align with any manipulative scenario participants, which suggests background correlation or an uncovered pattern."
            )
        elif top_group.get("is_mixed", False):
            lines.append(
                "- The highest-ranked suspicious group aligns most strongly with `{0}` (`{1}`), but it mixes participants from multiple scenarios. Review the overlapping run segments and trade edges for leakage across scenarios.".format(
                    dominant_scenario,
                    top_group.get("scenario_type") or "unknown",
                )
            )
        else:
            lines.append(
                "- The highest-ranked suspicious group aligns with `{0}` (`{1}`), covering `{2}` of `{3}` attributed scenario participants with purity `{4:.3f}`.".format(
                    dominant_scenario,
                    top_group.get("scenario_type") or "unknown",
                    top_group.get("group_scenario_member_count", 0),
                    top_group.get("total_scenario_participants", 0),
                    top_group.get("purity", 0.0),
                )
            )
        if false_positive_count > false_negative_count:
            lines.append("- The detector is over-flagging relative to misses, so alert quality is noisier than ideal.")
        elif false_negative_count > false_positive_count:
            lines.append("- The detector is missing more injected traders than it is over-flagging, so recall needs attention.")
        else:
            lines.append("- False positives and false negatives are balanced at the trader level.")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(output)
