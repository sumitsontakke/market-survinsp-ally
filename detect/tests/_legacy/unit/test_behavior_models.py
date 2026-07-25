from __future__ import annotations

from datetime import datetime

from synthetic_market_sim.behaviors.base import BehaviorContext
from synthetic_market_sim.behaviors.normal import RetailRandomBehavior
from synthetic_market_sim.domain.entities import Instrument, Trader
from synthetic_market_sim.domain.value_objects import MarketSnapshot
from synthetic_market_sim.utils.seed import build_rng


def test_retail_random_behavior_emits_linked_intent() -> None:
    rng = build_rng(1)
    behavior = RetailRandomBehavior()
    trader = Trader(
        trader_id="trader_00001",
        account_id="account_00001",
        beneficial_owner_id="owner_00001",
        broker_id="broker_00001",
        trader_profile_id="retail_random",
        risk_tier="low",
        region="US",
        created_at=datetime(2026, 3, 14, 9, 30),
        status="active",
    )
    instrument = Instrument(
        instrument_id="instrument_00001",
        symbol="ALPHA",
        asset_class="equity",
        tick_size=0.01,
        lot_size=1,
        price_band=(95.0, 105.0),
        session_calendar_id="session_00001",
    )
    snapshot = MarketSnapshot(
        timestamp=datetime(2026, 3, 14, 9, 31),
        instrument_id=instrument.instrument_id,
        reference_price=100.0,
        drift=0.01,
        volatility=0.05,
        participation_intensity=10.0,
    )
    intent = behavior.generate_order_intent(
        BehaviorContext(
            trader=trader,
            instrument=instrument,
            snapshot=snapshot,
            rng=rng,
            base_order_size=20,
            max_order_size=100,
            passive_order_probability=0.4,
            aggressive_order_probability=0.5,
        )
    )
    assert intent is not None
    assert intent.trader_id == trader.trader_id
    assert intent.instrument_id == instrument.instrument_id
    assert intent.quantity > 0
