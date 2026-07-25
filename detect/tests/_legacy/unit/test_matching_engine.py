from __future__ import annotations

from datetime import datetime

from synthetic_market_sim.domain.entities import Order
from synthetic_market_sim.simulation.matching_engine import MatchingEngine


def test_matching_engine_matches_crossing_orders() -> None:
    engine = MatchingEngine()
    sell_order = Order(
        order_id="order_00001",
        timestamp=datetime(2026, 3, 14, 9, 30),
        trader_id="trader_sell",
        account_id="account_sell",
        broker_id="broker_sell",
        instrument_id="instrument_00001",
        side="sell",
        order_type="limit",
        price=100.0,
        quantity=50,
        time_in_force="day",
        scenario_id="normal",
        scenario_label="normal",
        scenario_type="generic_background",
        is_manipulative=False,
        remaining_quantity=50,
    )
    engine.process(sell_order, lambda: "trade_00000")
    buy_order = Order(
        order_id="order_00002",
        timestamp=datetime(2026, 3, 14, 9, 31),
        trader_id="trader_buy",
        account_id="account_buy",
        broker_id="broker_buy",
        instrument_id="instrument_00001",
        side="buy",
        order_type="limit",
        price=101.0,
        quantity=20,
        time_in_force="day",
        scenario_id="normal",
        scenario_label="normal",
        scenario_type="generic_background",
        is_manipulative=False,
        remaining_quantity=20,
    )
    final_order, trades = engine.process(buy_order, lambda: "trade_00001")
    assert final_order.remaining_quantity == 0
    assert len(trades) == 1
    assert trades[0].price == 100.0
    assert trades[0].quantity == 20
