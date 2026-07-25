"""Daily-resolution NSE calibrator (bhavcopy path).

Companion to ``nse_calibrator.py`` which operates on 1-minute data.
This module reads from ``daily_market_data`` (the bhavcopy store) and
produces :class:`CalibrationParams` at end-of-day resolution.

Parameter mapping vs the intraday calibrator
--------------------------------------------

* ``realized_volatility``        -- annualized std of daily log-returns,
                                    volume-weighted across the focus universe
* ``intraday_volume_profile``    -- empty list (no intraday signal at this
                                    resolution; consumer must fall back
                                    to a default profile or merge with the
                                    intraday calibrator's output)
* ``return_df / loc / scale``    -- Student-t fit on pooled daily log-returns
* ``volume_alpha``               -- Hill estimator on the **cross-sectional**
                                    daily-volume distribution across the
                                    entire NSE EQ universe (not per-minute
                                    within one ticker). This is the direct
                                    target for R3's Pareto trader-activity
                                    multiplier.

Reference
---------
Cont, R. (2001). Empirical properties of asset returns: stylized facts
and statistical implications. Quantitative Finance, 1(2), 223-236.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sci_stats

from synth.calibration.core.config import (
    DB_PATH,
    EMPIRICAL_RETURN_DF_RANGE,
    EMPIRICAL_VOLUME_ALPHA_RANGE,
    NIFTY_LIQUID_20,
    TRADING_DAYS_PER_YEAR,
)
from synth.calibration.core.database import MarketDataDB
from synth.calibration.core.models import CalibrationParams


class NSEDailyCalibrator:
    """Compute :class:`CalibrationParams` from end-of-day bhavcopy data.

    Parameters
    ----------
    db_path:
        SQLite path. Defaults to ``DB_PATH``.
    focus_universe:
        Tickers used for per-ticker realized-vol / Student-t fit. Defaults
        to ``NIFTY_LIQUID_20``. Symbols may be passed with or without the
        yfinance ``.NS`` suffix - the database read strips them.
    pool_window_days:
        Trailing window of trading days used to pool returns for the
        Student-t fit and realized-vol estimate. Defaults to 30.
    """

    def __init__(
        self,
        db_path: "Path | str" = DB_PATH,
        focus_universe: Sequence[str] = tuple(NIFTY_LIQUID_20),
        pool_window_days: int = 30,
    ) -> None:
        self.db = MarketDataDB(db_path)
        self.focus_universe = tuple(focus_universe)
        self.pool_window_days = max(2, int(pool_window_days))

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def calibrate(self, date_str: Optional[str] = None) -> CalibrationParams:
        target_date = self._resolve_target_date(date_str)
        cached = self.db.get_calibration_for_date(target_date)
        if cached is not None and cached.intraday_volume_profile == []:
            # Existing daily-mode calibration cached for this date.
            return cached

        df = self._load_focus_window(target_date)
        if df.empty:
            raise ValueError(
                f"No daily_market_data rows for the focus universe on {target_date} "
                "or its trailing window. Run the bhavcopy fetcher first."
            )

        warnings: list[str] = []
        rv = self._realized_volatility(df, warnings)
        ret_dist = self._return_distribution(df, warnings)
        alpha = self._cross_sectional_volume_alpha(target_date, warnings)

        params = CalibrationParams(
            realized_volatility=float(rv),
            intraday_volume_profile=[],  # not derivable from EOD
            return_df=float(ret_dist["df"]),
            return_loc=float(ret_dist["loc"]),
            return_scale=float(ret_dist["scale"]),
            volume_alpha=float(alpha),
            calibration_date=target_date,
            tickers_used=sorted(df["ticker"].unique().tolist()),
            n_observations=int(len(df)),
            warnings=list(warnings) + ["resolution=daily (bhavcopy)"],
        )
        self.db.save_calibration(params)
        return params

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _resolve_target_date(self, date_str: Optional[str]) -> str:
        if date_str in (None, "latest"):
            latest = self.db.get_latest_daily_trade_date()
            if latest is None:
                raise ValueError(
                    "daily_market_data is empty. Run the bhavcopy fetcher first."
                )
            return latest
        pd.Timestamp(date_str)  # validate iso
        return date_str

    def _load_focus_window(self, target_date: str) -> pd.DataFrame:
        """Pull the trailing ``pool_window_days`` window for the focus
        universe, ending on ``target_date``."""
        end = date.fromisoformat(target_date)
        # Calendar window slightly larger than trading window to absorb
        # weekends and holidays.
        start = end - timedelta(days=int(self.pool_window_days * 1.6) + 7)
        df = self.db.get_daily_market_data(
            tickers=self.focus_universe,
            start_date=start.isoformat(),
            end_date=target_date,
        )
        if df.empty:
            return df
        # Keep only the most recent ``pool_window_days`` per ticker.
        df = df.sort_values(["ticker", "trade_date"])
        df = (
            df.groupby("ticker", group_keys=False)
            .apply(lambda g: g.tail(self.pool_window_days))
            .reset_index(drop=True)
        )
        return df

    @staticmethod
    def _per_ticker_log_returns(df: pd.DataFrame) -> pd.DataFrame:
        """Add a ``log_ret`` column per (ticker) using prev_close where
        available, falling back to the previous row's close."""
        out = df.copy().sort_values(["ticker", "trade_date"])
        # Prefer prev_close from the bhavcopy itself - it survives splits
        # better than a naive close.shift(1) but in practice both work for
        # liquid large caps over a small window.
        out["log_ret_pc"] = np.log(
            out["close"].astype(float) / out["prev_close"].astype(float).replace(0, np.nan)
        )
        out["log_ret_shift"] = (
            np.log(out["close"].astype(float))
            .groupby(out["ticker"])
            .diff()
        )
        out["log_ret"] = out["log_ret_pc"].fillna(out["log_ret_shift"])
        return out

    def _realized_volatility(
        self, df: pd.DataFrame, warnings: list[str]
    ) -> float:
        with_ret = self._per_ticker_log_returns(df)
        per_ticker_std = (
            with_ret.dropna(subset=["log_ret"])
            .groupby("ticker")["log_ret"]
            .std(ddof=1)
            .dropna()
        )
        if per_ticker_std.empty:
            warnings.append("realized_volatility: insufficient daily returns to compute std.")
            return 0.0
        annualization = float(np.sqrt(TRADING_DAYS_PER_YEAR))
        per_ticker_vol = per_ticker_std * annualization
        weights = (
            df.groupby("ticker")["volume"]
            .sum()
            .reindex(per_ticker_vol.index)
            .fillna(0.0)
        )
        total_w = float(weights.sum())
        if total_w <= 0:
            return float(per_ticker_vol.mean())
        return float((per_ticker_vol * weights).sum() / total_w)

    def _return_distribution(
        self, df: pd.DataFrame, warnings: list[str]
    ) -> dict[str, float]:
        with_ret = self._per_ticker_log_returns(df).dropna(subset=["log_ret"])
        x = with_ret["log_ret"].to_numpy()
        if x.size < 30:
            warnings.append(
                f"return_distribution: only {x.size} pooled daily returns; "
                "Student-t fit may be unstable."
            )
            return {"df": float("nan"), "loc": 0.0, "scale": float("nan")}
        df_fit, loc_fit, scale_fit = sci_stats.t.fit(x, floc=0.0)
        lo, hi = EMPIRICAL_RETURN_DF_RANGE
        if df_fit < lo or df_fit > hi:
            warnings.append(
                f"return_df={df_fit:.2f} outside empirical band [{lo}-{hi}]."
            )
        return {"df": float(df_fit), "loc": float(loc_fit), "scale": float(scale_fit)}

    def _cross_sectional_volume_alpha(
        self, target_date: str, warnings: list[str]
    ) -> float:
        """Hill estimator on the per-ticker daily volume distribution
        across the **entire NSE EQ universe** on ``target_date``.

        This is the population-level Pareto signal that R3 needs: how
        concentrated is volume across participants? Median ~ 145k,
        p99 ~ 28M for typical NSE EQ days (200x ratio).
        """
        vols = self.db.get_daily_universe_volumes(target_date)
        if vols.empty:
            warnings.append(
                "volume_alpha: no rows in daily_market_data for the date."
            )
            return float("nan")
        x = vols.sort_values().to_numpy()
        x = x[x > 0]
        if x.size < 30:
            warnings.append("volume_alpha: <30 tickers in cross-section.")
            return float("nan")
        x_min = float(np.percentile(x, 90))
        if x_min <= 0:
            return float("nan")
        tail = x[x > x_min]
        if tail.size < 10:
            return float("nan")
        denom = float(np.sum(np.log(tail / x_min)))
        if denom <= 0:
            return float("nan")
        alpha = tail.size / denom
        lo, hi = EMPIRICAL_VOLUME_ALPHA_RANGE
        if alpha < lo:
            warnings.append(
                f"volume_alpha={alpha:.2f} below empirical band [{lo}-{hi}] - "
                "tail heavier than typical literature reference."
            )
        elif alpha > hi:
            warnings.append(
                f"volume_alpha={alpha:.2f} above empirical band [{lo}-{hi}]."
            )
        return float(alpha)


# -- R3 INTEGRATION INTERFACE (DAILY MODE) ------------------------------------
# The daily calibrator's CalibrationParams feeds into synthetic_market_sim
# the same way as the intraday version, with two adjustments:
#
#   volatility_scale      = params.realized_volatility   # annualized, daily-derived
#   time_bucket_weights   = (use Phase 1 default profile - daily mode does
#                            not produce a 375-slot vector)
#   return_shock_df       = params.return_df             # ~ 4 on real NSE daily
#   return_shock_scale    = params.return_scale          # ~ 0.011 daily
#   trader_activity_alpha = params.volume_alpha          # ~ 0.95 cross-section EQ
#
# Pareto activity multipliers (revised to match the real cross-sectional signal):
#   For N traders in the synthetic population:
#     multipliers = scipy.stats.pareto(b=params.volume_alpha).rvs(N)
#     multipliers = clip(multipliers, 0, percentile(multipliers, 99))
#     multipliers = multipliers / mean(multipliers)
#   Effect: synthetic p99/median rises from 1.11 toward the ~200x ratio
#   observed across the live NSE EQ universe.
# -----------------------------------------------------------------------------
