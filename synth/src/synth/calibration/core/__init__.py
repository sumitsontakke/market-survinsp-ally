"""Shared code mounted into both fetcher and calibrator containers.

Exposes the public dataclass and the ticker universe so callers can do::

    from core import CalibrationParams, NIFTY_LIQUID_20

without reaching into submodules.

Reference
---------
Cont, R. (2001). Empirical properties of asset returns: stylized facts
and statistical implications. Quantitative Finance, 1(2), 223-236.
"""
from __future__ import annotations

from synth.calibration.core.config import (
    BUCKETS_PER_DAY,
    DB_PATH,
    EMPIRICAL_RETURN_DF_RANGE,
    EMPIRICAL_VOLUME_ALPHA_RANGE,
    IST_OFFSET_MINUTES,
    MIN_ROWS_PER_TICKER_DAY,
    NIFTY_LIQUID_20,
    NSE_CLOSE_UTC_HOUR,
    NSE_CLOSE_UTC_MINUTE,
    NSE_OPEN_UTC_HOUR,
    NSE_OPEN_UTC_MINUTE,
    PLOT_DIR,
    SYNTHETIC_BASELINES,
    TRADING_DAYS_PER_YEAR,
)
from synth.calibration.core.models import CalibrationParams

__all__ = [
    "BUCKETS_PER_DAY",
    "CalibrationParams",
    "DB_PATH",
    "EMPIRICAL_RETURN_DF_RANGE",
    "EMPIRICAL_VOLUME_ALPHA_RANGE",
    "IST_OFFSET_MINUTES",
    "MIN_ROWS_PER_TICKER_DAY",
    "NIFTY_LIQUID_20",
    "NSE_CLOSE_UTC_HOUR",
    "NSE_CLOSE_UTC_MINUTE",
    "NSE_OPEN_UTC_HOUR",
    "NSE_OPEN_UTC_MINUTE",
    "PLOT_DIR",
    "SYNTHETIC_BASELINES",
    "TRADING_DAYS_PER_YEAR",
]
