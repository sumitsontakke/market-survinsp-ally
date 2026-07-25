from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from synth.generator.domain.enums import OrderType, Side, TimeInForce


@dataclass(frozen=True)
class OrderIntent:
    timestamp: datetime
    trader_id: str
    account_id: str
    broker_id: str
    instrument_id: str
    side: Side
    order_type: OrderType
    price: float
    quantity: int
    time_in_force: TimeInForce
    scenario_id: str
    scenario_label: str
    scenario_type: str
    is_manipulative: bool
    parent_order_id: Optional[str] = None


@dataclass(frozen=True)
class MarketSnapshot:
    timestamp: datetime
    instrument_id: str
    reference_price: float
    drift: float
    volatility: float
    participation_intensity: float
