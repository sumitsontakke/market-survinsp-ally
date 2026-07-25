from __future__ import annotations

from datetime import datetime

from synthetic_market_sim.behaviors.manipulative import CircularTradingRingBehavior, CircularTradingRingPlan
from synthetic_market_sim.domain.entities import Instrument, Trader
from synthetic_market_sim.domain.value_objects import MarketSnapshot


def test_circular_trading_ring_behavior_emits_ordered_seller_then_buyer_intents() -> None:
    plan = CircularTradingRingPlan(
        scenario_id="scenario_002",
        scenario_type="circular_trading_ring",
        instrument_id="instrument_00001",
        participant_ids=("trader_00001", "trader_00002", "trader_00003"),
        participant_count=3,
        start_step=0,
        duration_steps=6,
        intensity="medium",
        concealment="low",
        cycles=2,
        ring_order=("trader_00001", "trader_00002", "trader_00003"),
        seed=17,
    )
    behavior = CircularTradingRingBehavior(plan)
    instrument = Instrument(
        instrument_id="instrument_00001",
        symbol="RING",
        asset_class="equity",
        tick_size=0.01,
        lot_size=1,
        price_band=(99.0, 101.0),
        session_calendar_id="session_00001",
    )
    traders_by_id = {
        trader_id: Trader(
            trader_id=trader_id,
            account_id="account_" + trader_id[-5:],
            beneficial_owner_id="owner_" + trader_id[-5:],
            broker_id="broker_00001",
            trader_profile_id="retail_random",
            risk_tier="medium",
            region="US",
            created_at=datetime(2026, 3, 21, 9, 30),
            status="active",
        )
        for trader_id in plan.ring_order
    }
    snapshot = MarketSnapshot(
        timestamp=datetime(2026, 3, 21, 9, 31),
        instrument_id=instrument.instrument_id,
        reference_price=100.0,
        drift=0.0,
        volatility=0.05,
        participation_intensity=1.0,
    )
    intents = behavior.generate_step_intents(
        step_index=0,
        snapshot=snapshot,
        instrument=instrument,
        traders_by_id=traders_by_id,
        base_order_size=20,
        max_order_size=100,
        passive_order_probability=0.3,
        aggressive_order_probability=0.5,
    )
    assert len(intents) == 2
    assert intents[0].trader_id == "trader_00001"
    assert intents[0].side.value == "sell"
    assert intents[1].trader_id == "trader_00002"
    assert intents[1].side.value == "buy"
    assert intents[0].scenario_type == "circular_trading_ring"
    assert intents[1].scenario_id == "scenario_002"
