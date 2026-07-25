from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class Broker:
    broker_id: str
    broker_name: str
    venue_access: str
    latency_profile: str


@dataclass(frozen=True)
class BeneficialOwner:
    beneficial_owner_id: str
    owner_type: str
    group_label: str
    linked_account_count: int


@dataclass(frozen=True)
class Account:
    account_id: str
    beneficial_owner_id: str
    broker_id: str
    account_type: str
    opened_at: datetime
    status: str


@dataclass(frozen=True)
class Trader:
    trader_id: str
    account_id: str
    beneficial_owner_id: str
    broker_id: str
    trader_profile_id: str
    risk_tier: str
    region: str
    created_at: datetime
    status: str


@dataclass(frozen=True)
class Instrument:
    instrument_id: str
    symbol: str
    asset_class: str
    tick_size: float
    lot_size: int
    price_band: tuple[float, float]
    session_calendar_id: str


@dataclass(frozen=True)
class TradingSession:
    session_id: str
    trade_date: str
    open_time: str
    close_time: str
    auction_windows: tuple[str, ...]
    timezone: str


@dataclass(frozen=True)
class Order:
    order_id: str
    timestamp: datetime
    trader_id: str
    account_id: str
    broker_id: str
    instrument_id: str
    side: str
    order_type: str
    price: float
    quantity: int
    time_in_force: str
    scenario_id: str
    scenario_label: str
    scenario_type: str
    is_manipulative: bool
    parent_order_id: Optional[str] = None
    remaining_quantity: int = 0


@dataclass(frozen=True)
class Trade:
    trade_id: str
    timestamp: datetime
    buy_order_id: str
    sell_order_id: str
    buy_trader_id: str
    sell_trader_id: str
    instrument_id: str
    price: float
    quantity: int
    scenario_id: str
    scenario_label: str
    scenario_type: str
    is_manipulative: bool


@dataclass(frozen=True)
class ScenarioEvent:
    scenario_id: str
    scenario_type: str
    start_time: datetime
    end_time: datetime
    participant_ids: tuple[str, ...]
    instrument_id: str
    intensity: str
    is_manipulative: bool
    concealment: str = ""
    cycles: int = 0
    ring_order: tuple[str, ...] = ()


def as_record(entity: object) -> dict[str, object]:
    record = asdict(entity)
    for key, value in tuple(record.items()):
        if isinstance(value, datetime):
            record[key] = value.isoformat()
        elif isinstance(value, tuple):
            record[key] = list(value)
    return record
