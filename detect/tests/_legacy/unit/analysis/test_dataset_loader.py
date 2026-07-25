from __future__ import annotations

import json

from synthetic_market_sim.analysis.dataset_loader import load_dataset


def test_scenario_attribution_index_augments_participants_from_activity(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    (dataset_dir / "manifest.json").write_text(json.dumps({"schema_version": "0.1.0"}), encoding="utf-8")
    (dataset_dir / "traders.csv").write_text("trader_id\nrun_a__trader_001\nrun_a__trader_002\n", encoding="utf-8")
    (dataset_dir / "instruments.csv").write_text("instrument_id,symbol\nrun_a__instrument_001,INST_1\n", encoding="utf-8")
    (dataset_dir / "scenarios.csv").write_text(
        "\n".join(
            [
                "scenario_id,scenario_type,start_time,end_time,participant_ids,instrument_id,intensity,is_manipulative,concealment,cycles,ring_order",
                'run_a__scenario_001,collusive_clique,2026-03-21 09:30:00,2026-03-21 09:40:00,"[""run_a__trader_001""]",run_a__instrument_001,medium,True,low,0,[]',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "orders.csv").write_text(
        "\n".join(
            [
                "order_id,timestamp,trader_id,account_id,broker_id,instrument_id,side,order_type,price,quantity,time_in_force,scenario_id,scenario_label,scenario_type,is_manipulative,parent_order_id,remaining_quantity",
                "run_a__order_001,2026-03-21 09:31:00,run_a__trader_002,run_a__account_001,run_a__broker_001,run_a__instrument_001,buy,limit,100,10,day,run_a__scenario_001,scenario_001,collusive_clique,True,,10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset_dir / "trades.csv").write_text(
        "\n".join(
            [
                "trade_id,timestamp,buy_order_id,sell_order_id,buy_trader_id,sell_trader_id,instrument_id,price,quantity,scenario_id,scenario_label,scenario_type,is_manipulative",
                "run_a__trade_001,2026-03-21 09:31:00,run_a__order_001,run_a__order_002,run_a__trader_002,run_a__trader_001,run_a__instrument_001,100,10,run_a__scenario_001,scenario_001,collusive_clique,True",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = load_dataset(dataset_dir)
    index = dataset.scenario_attribution_index()

    assert index.scenario_to_traders["run_a__scenario_001"] == ["run_a__trader_001", "run_a__trader_002"]
    assert {detail.scenario_id for detail in index.trader_to_scenarios["run_a__trader_002"]} == {"run_a__scenario_001"}
