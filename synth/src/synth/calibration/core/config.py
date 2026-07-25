"""Calibration-service constants.

This module is mounted into both the fetcher and calibrator containers.
It is intentionally dependency-free so it can be imported during early
container start-up.

Reference
---------
Cont, R. (2001). Empirical properties of asset returns: stylized facts
and statistical implications. Quantitative Finance, 1(2), 223-236.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Ticker universe
# ---------------------------------------------------------------------------
# Fixed set of 20 liquid Nifty 50 names spread across sectors. Holding the
# universe constant means that calibration is reproducible and comparable
# across runs - the universe is not a tuning knob.
NIFTY_LIQUID_20: list[str] = [
    # Banking and financials
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "SBIN.NS",
    # IT services
    "TCS.NS",
    "INFY.NS",
    "WIPRO.NS",
    "HCLTECH.NS",
    # Energy
    "RELIANCE.NS",
    "ONGC.NS",
    # Consumer
    "HINDUNILVR.NS",
    "ITC.NS",
    # Pharma
    "SUNPHARMA.NS",
    "DRREDDY.NS",
    # Auto
    "TATAMOTORS.NS",
    "MARUTI.NS",
    # Metals
    "TATASTEEL.NS",
    "JSWSTEEL.NS",
    # Telecom
    "BHARTIARTL.NS",
]

# ---------------------------------------------------------------------------
# Trading session
# ---------------------------------------------------------------------------
# NSE regular session is 09:15-15:30 IST. Internally we keep everything in
# UTC; conversions to IST happen only at the display / plot layer.
#
# 03:45 UTC = 09:15 IST   (open)
# 10:00 UTC = 15:30 IST   (close)
#
# Total minutes per session = 6 * 60 + 15 = 375.
NSE_OPEN_UTC_HOUR: int = 3
NSE_OPEN_UTC_MINUTE: int = 45
NSE_CLOSE_UTC_HOUR: int = 10
NSE_CLOSE_UTC_MINUTE: int = 0
BUCKETS_PER_DAY: int = 375
IST_OFFSET_MINUTES: int = 5 * 60 + 30  # +05:30

# Number of trading minutes used in the realized-volatility annualization.
# Annualization factor = sqrt(TRADING_DAYS_PER_YEAR * BUCKETS_PER_DAY).
TRADING_DAYS_PER_YEAR: int = 252

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
# DB_PATH is supplied via the environment so the same code runs locally and
# inside Docker. Inside Docker it is "/data/market.db" (mounted volume).
DB_PATH: Path = Path(os.environ.get("DB_PATH", "/data/market.db"))

# Plot output root - inside the container this is /data/plots so the host
# can read the PNGs through the same volume mount used for the database.
PLOT_DIR: Path = Path(os.environ.get("PLOT_DIR", "/data/plots"))

# ---------------------------------------------------------------------------
# Quality thresholds
# ---------------------------------------------------------------------------
# Minimum rows we expect from a clean trading day after filtering. A typical
# NSE session yields ~375; anything below this triggers a logged warning.
MIN_ROWS_PER_TICKER_DAY: int = 200

# Empirical reference range for the Student-t degree-of-freedom parameter
# fitted on equity log-returns - see Cont (2001), Section 3.
EMPIRICAL_RETURN_DF_RANGE: tuple[float, float] = (3.0, 5.0)

# Empirical reference range for the Hill tail-index alpha on per-minute
# volume distributions across liquid equities.
EMPIRICAL_VOLUME_ALPHA_RANGE: tuple[float, float] = (1.5, 2.5)

# ---------------------------------------------------------------------------
# Synthetic-baseline anchors (Phase 1 measurements, hard-coded)
# ---------------------------------------------------------------------------
# These are the Phase 1 stylized-facts results that R3 calibration is
# designed to close. They live here so the calibrator's summary() output
# can show the gap on every run without re-running Phase 1.
#
#   - return_df = 2.00         → measured in data_quality/stylized_facts_*
#   - volume_alpha = "~inf"    → p99/median ratio = 1.11 (near-uniform);
#                                no Pareto tail, so Hill alpha diverges
#   - realized_volatility      → not directly measured in Phase 1
SYNTHETIC_BASELINES: dict[str, object] = {
    "return_df": 2.00,
    "volume_alpha": "~inf (p99/median=1.11, near-uniform)",
    "realized_volatility": "not measured in Phase 1",
}
