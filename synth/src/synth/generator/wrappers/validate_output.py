from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Union


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_dataset(input_dir: Union[str, Path]) -> list[str]:
    root = Path(input_dir)
    issues: list[str] = []
    required_files = [
        "beneficial_owners.csv",
        "accounts.csv",
        "traders.csv",
        "instruments.csv",
        "orders.csv",
        "scenarios.csv",
        "manifest.json",
    ]
    for file_name in required_files:
        if not (root / file_name).exists():
            issues.append(f"Missing required file: {file_name}")
    if issues:
        return issues

    accounts = _load_csv(root / "accounts.csv")
    traders = _load_csv(root / "traders.csv")
    instruments = _load_csv(root / "instruments.csv")
    orders = _load_csv(root / "orders.csv")
    trades = _load_csv(root / "trades.csv") if (root / "trades.csv").exists() else []
    beneficial_owners = _load_csv(root / "beneficial_owners.csv")
    scenarios = _load_csv(root / "scenarios.csv")

    account_ids = {row["account_id"] for row in accounts}
    trader_ids = {row["trader_id"] for row in traders}
    instrument_ids = {row["instrument_id"] for row in instruments}
    owner_ids = {row["beneficial_owner_id"] for row in beneficial_owners}
    order_ids = {row["order_id"] for row in orders}
    scenario_ids = {row["scenario_id"] for row in scenarios}
    scenario_map = {row["scenario_id"]: row for row in scenarios}

    for account in accounts:
        if account["beneficial_owner_id"] not in owner_ids:
            issues.append(f"Account references unknown beneficial owner: {account['account_id']}")
    for trader in traders:
        if trader["account_id"] not in account_ids:
            issues.append(f"Trader references unknown account: {trader['trader_id']}")
    for order in orders:
        if order["trader_id"] not in trader_ids:
            issues.append(f"Order references unknown trader: {order['order_id']}")
        if order["account_id"] not in account_ids:
            issues.append(f"Order references unknown account: {order['order_id']}")
        if order["instrument_id"] not in instrument_ids:
            issues.append(f"Order references unknown instrument: {order['order_id']}")
        if order["scenario_id"] not in scenario_ids:
            issues.append(f"Order references unknown scenario: {order['order_id']}")
        if order.get("is_manipulative", "").lower() == "true" and order["scenario_id"] == "normal":
            issues.append(f"Manipulative order cannot use normal scenario: {order['order_id']}")
    for trade in trades:
        if trade["buy_order_id"] not in order_ids or trade["sell_order_id"] not in order_ids:
            issues.append(f"Trade references missing order: {trade['trade_id']}")
        if trade["scenario_id"] not in scenario_ids:
            issues.append(f"Trade references unknown scenario: {trade['trade_id']}")
        if trade.get("is_manipulative", "").lower() == "true" and trade["scenario_id"] == "normal":
            issues.append(f"Manipulative trade cannot use normal scenario: {trade['trade_id']}")

    for scenario in scenarios:
        if scenario["instrument_id"] != "ALL" and scenario["instrument_id"] not in instrument_ids:
            issues.append(f"Scenario references unknown instrument: {scenario['scenario_id']}")
        participant_ids = json.loads(scenario["participant_ids"])
        ring_order = json.loads(scenario.get("ring_order", "[]") or "[]")
        for participant_id in participant_ids:
            if participant_id not in trader_ids:
                issues.append(f"Scenario references unknown participant: {scenario['scenario_id']} -> {participant_id}")
        if scenario["scenario_type"] == "circular_trading_ring":
            if not ring_order:
                issues.append(f"Circular trading ring missing ring_order: {scenario['scenario_id']}")
            elif len(ring_order) != len(participant_ids):
                issues.append(f"Ring order length mismatch: {scenario['scenario_id']}")
            elif len(set(ring_order)) != len(ring_order):
                issues.append(f"Ring order contains duplicates: {scenario['scenario_id']}")
            elif set(ring_order) != set(participant_ids):
                issues.append(f"Ring order participants do not match participant_ids: {scenario['scenario_id']}")
            if int(scenario.get("cycles", "0") or 0) <= 0:
                issues.append(f"Circular trading ring must define positive cycles: {scenario['scenario_id']}")

    for order in orders:
        if order["scenario_id"] != "normal":
            scenario = scenario_map.get(order["scenario_id"])
            if scenario is None:
                issues.append(f"Scenario-linked order missing scenario metadata: {order['order_id']}")
            elif scenario["is_manipulative"].lower() != "true":
                issues.append(f"Scenario-linked order maps to non-manipulative scenario: {order['order_id']}")
            elif scenario["scenario_type"] == "circular_trading_ring" and order["trader_id"] not in json.loads(scenario["participant_ids"]):
                issues.append(f"Ring-linked order trader not in ring participants: {order['order_id']}")

    for trade in trades:
        if trade["scenario_id"] != "normal":
            scenario = scenario_map.get(trade["scenario_id"])
            if scenario is None:
                continue
            if scenario["scenario_type"] == "circular_trading_ring":
                participants = set(json.loads(scenario["participant_ids"]))
                if trade["buy_trader_id"] not in participants and trade["sell_trader_id"] not in participants:
                    issues.append(f"Ring-linked trade missing ring participant: {trade['trade_id']}")

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["counts"]["orders"] != len(orders):
        issues.append("Manifest order count does not match orders.csv")
    if manifest.get("manipulative_order_count") != sum(1 for row in orders if row.get("is_manipulative", "").lower() == "true"):
        issues.append("Manifest manipulative order count does not match orders.csv")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated simulation output.")
    parser.add_argument("--input-dir", required=True, help="Directory containing exported dataset files.")
    args = parser.parse_args()
    issues = validate_dataset(args.input_dir)
    if issues:
        for issue in issues:
            print(issue)
        raise SystemExit(1)
    print("Validation passed")


if __name__ == "__main__":
    main()
