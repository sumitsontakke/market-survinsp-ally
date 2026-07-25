from __future__ import annotations

from enum import Enum


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"


class TimeInForce(str, Enum):
    DAY = "day"
    IOC = "ioc"


class AccountStatus(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"


class TraderStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"


class ScenarioLabel(str, Enum):
    NORMAL = "normal"
    MANIPULATIVE = "manipulative"
