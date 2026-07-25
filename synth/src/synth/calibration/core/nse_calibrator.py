"""Calibration parameter extraction from stored NSE 1-minute OHLCV.

The class :class:`NSECalibrator` reads market_data from the SQLite store,
computes four parameters that feed into ``synthetic_market_sim`` before
each synthetic session, and persists the result so a second invocation
on the same date is a free database read.

The four parameters
-------------------

1. **Realized volatility**  -  volume-weighted mean of per-ticker
   annualized log-return standard deviations. Annualization factor is
   ``sqrt(TRADING_DAYS_PER_YEAR * BUCKETS_PER_DAY)`` (252 * 375).
2. **Intraday volume profile**  -  length-375 normalized weight vector
   describing the U-shape of NSE volume by minute-of-session.
3. **Return distribution**  -  Student-t fit on pooled per-minute
   log-returns (``floc=0``); fields ``df``, ``loc``, ``scale``.
4. **Volume alpha**  -  Hill tail-index estimator on per-minute volume,
   median across tickers. The 90th-percentile threshold is the standard
   choice in equity microstructure work.

Reference
---------
Cont, R. (2001). Empirical properties of asset returns: stylized facts
and statistical implications. Quantitative Finance, 1(2), 223-236.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sci_stats

from synth.calibration.core.config import (
    BUCKETS_PER_DAY,
    DB_PATH,
    EMPIRICAL_VOLUME_ALPHA_RANGE,
    EMPIRICAL_RETURN_DF_RANGE,
    NSE_OPEN_UTC_HOUR,
    NSE_OPEN_UTC_MINUTE,
    TRADING_DAYS_PER_YEAR,
)
from synth.calibration.core.database import MarketDataDB
from synth.calibration.core.models import CalibrationParams

_OPEN_TOTAL_MIN = NSE_OPEN_UTC_HOUR * 60 + NSE_OPEN_UTC_MINUTE


class NSECalibrator:
    """Compute and persist :class:`CalibrationParams` for a target date."""

    def __init__(self, db_path: Path | str = DB_PATH) -> None:
        self.db = MarketDataDB(db_path)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def calibrate(self, date_str: Optional[str] = None) -> CalibrationParams:
        """Return CalibrationParams for ``date_str`` (or latest available).

        If a calibration row already exists for the target date it is
        loaded and returned without recomputation.
        """
        target_date = self._resolve_target_date(date_str)
        cached = self.db.get_calibration_for_date(target_date)
        if cached is not None:
            return cached

        df = self.db.get_market_data(
            tickers=self._tickers_with_data(target_date),
            start_date=target_date,
            end_date=target_date,
        )
        if df.empty:
            raise ValueError(
                f"No market_data rows for {target_date}. Run the fetcher first."
            )

        warnings: list[str] = []
        rv = self._compute_realized_volatility(df)
        profile = self._compute_volume_profile(df)
        ret_dist = self._compute_return_distribution(df, warnings)
        alpha = self._compute_volume_alpha(df, warnings)

        params = CalibrationParams(
            realized_volatility=float(rv),
            intraday_volume_profile=[float(x) for x in profile.tolist()],
            return_df=float(ret_dist["df"]),
            return_loc=float(ret_dist["loc"]),
            return_scale=float(ret_dist["scale"]),
            volume_alpha=float(alpha),
            calibration_date=target_date,
            tickers_used=sorted(df["ticker"].unique().tolist()),
            n_observations=int(len(df)),
            warnings=list(warnings),
        )
        self.db.save_calibration(params)
        return params

    # ------------------------------------------------------------------
    # date resolution
    # ------------------------------------------------------------------
    def _resolve_target_date(self, date_str: Optional[str]) -> str:
        if date_str in (None, "latest"):
            latest = self.db.get_latest_trade_date()
            if latest is None:
                raise ValueError(
                    "Database has no market_data rows yet. Run the fetcher first."
                )
            return latest
        # Validate ISO format up-front to surface bad CLI input early.
        pd.Timestamp(date_str)  # raises if malformed
        return date_str

    def _tickers_with_data(self, target_date: str) -> list[str]:
        """Tickers present in market_data on ``target_date``.

        We don't constrain to NIFTY_LIQUID_20 here so the calibrator works
        on whatever was actually fetched - this keeps the data layer the
        single source of truth.
        """
        df = self.db.get_market_data(
            tickers=self._all_known_tickers(),
            start_date=target_date,
            end_date=target_date,
        )
        return sorted(df["ticker"].unique().tolist()) if not df.empty else []

    def _all_known_tickers(self) -> list[str]:
        info = self.db.stats()
        # stats() doesn't return tickers directly; use a fresh query.
        with self.db._connect() as conn:  # noqa: SLF001 - internal helper
            rows = conn.execute(
                "SELECT DISTINCT ticker FROM market_data ORDER BY ticker"
            ).fetchall()
        return [r[0] for r in rows] or []

    # ------------------------------------------------------------------
    # parameter extractors
    # ------------------------------------------------------------------
    @staticmethod
    def _per_ticker_log_returns(df: pd.DataFrame) -> pd.Series:
        """Pooled log-returns computed per (ticker, day) so we never take
        a return across a ticker boundary or an overnight gap.
        """
        if df.empty:
            return pd.Series(dtype=float)
        df = df.copy()
        df["trade_date"] = df["datetime_utc"].dt.strftime("%Y-%m-%d")
        df = df.sort_values(["ticker", "datetime_utc"])
        # log(close / close.shift(1)) per (ticker, trade_date)
        df["log_ret"] = (
            np.log(df["close"].astype(float))
            .groupby([df["ticker"], df["trade_date"]])
            .diff()
        )
        return df["log_ret"].dropna()

    def _compute_realized_volatility(self, df: pd.DataFrame) -> float:
        """Volume-weighted mean of per-ticker annualized log-return std.

        ``vol_per_ticker = std(log_returns) * sqrt(252 * 375)``; weight is
        each ticker's total traded volume on the calibration date.
        """
        if df.empty:
            return 0.0
        returns = self._per_ticker_log_returns(df)
        if returns.empty:
            return 0.0
        # Re-attach ticker labels for groupby aggregation.
        returns = returns.rename("log_ret").to_frame()
        returns["ticker"] = df.sort_values(["ticker", "datetime_utc"]).loc[
            returns.index, "ticker"
        ].values
        per_ticker_std = returns.groupby("ticker")["log_ret"].std(ddof=1).dropna()
        if per_ticker_std.empty:
            return 0.0
        annualization = float(np.sqrt(TRADING_DAYS_PER_YEAR * BUCKETS_PER_DAY))
        per_ticker_vol = per_ticker_std * annualization
        weights = df.groupby("ticker")["volume"].sum().reindex(per_ticker_vol.index).fillna(0.0)
        total_w = float(weights.sum())
        if total_w <= 0:
            return float(per_ticker_vol.mean())
        return float((per_ticker_vol * weights).sum() / total_w)

    def _compute_volume_profile(self, df: pd.DataFrame) -> np.ndarray:
        """Length-375 normalized weight vector by minute-of-session.

        ``slot = (hour * 60 + minute) - (NSE open hour*60 + minute)``;
        only slots 0..374 are kept (the 10:00 close bar, slot 375, is
        intentionally dropped).
        """
        if df.empty:
            return np.zeros(BUCKETS_PER_DAY, dtype=float)
        ts = df["datetime_utc"]
        slot = ts.dt.hour * 60 + ts.dt.minute - _OPEN_TOTAL_MIN
        in_window = (slot >= 0) & (slot < BUCKETS_PER_DAY)
        slots = slot.loc[in_window].to_numpy()
        volumes = df["volume"].astype(float).loc[in_window].to_numpy()
        # Mean volume per slot across (ticker, day): use a sum-of-volume
        # divided by count-of-(ticker, day) observations contributing.
        sums = np.zeros(BUCKETS_PER_DAY, dtype=float)
        counts = np.zeros(BUCKETS_PER_DAY, dtype=float)
        np.add.at(sums, slots, volumes)
        np.add.at(counts, slots, 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_per_slot = np.where(counts > 0, sums / counts, 0.0)
        total = float(mean_per_slot.sum())
        if total <= 0:
            return np.full(BUCKETS_PER_DAY, 1.0 / BUCKETS_PER_DAY, dtype=float)
        return mean_per_slot / total

    def _compute_return_distribution(
        self, df: pd.DataFrame, warnings: list[str]
    ) -> dict[str, float]:
        """Student-t fit on pooled per-minute log-returns with ``floc=0``."""
        returns = self._per_ticker_log_returns(df).to_numpy()
        if returns.size < 50:
            warnings.append(
                f"return_distribution: only {returns.size} observations; "
                "Student-t fit may be unstable."
            )
            return {"df": float("nan"), "loc": 0.0, "scale": float("nan")}
        df_fit, loc_fit, scale_fit = sci_stats.t.fit(returns, floc=0.0)
        if df_fit < 2.5:
            warnings.append(
                f"Heavy tails df={df_fit:.2f} below empirical range "
                f"{EMPIRICAL_RETURN_DF_RANGE[0]}-{EMPIRICAL_RETURN_DF_RANGE[1]}. "
                "Benchmark harder than real market."
            )
        return {"df": float(df_fit), "loc": float(loc_fit), "scale": float(scale_fit)}

    def _compute_volume_alpha(
        self, df: pd.DataFrame, warnings: list[str]
    ) -> float:
        """Hill tail-index estimator, median across tickers.

        Per ticker:
          x        = sorted per-minute volumes (ascending)
          x_min    = 90th percentile of x
          tail     = x[x > x_min]
          alpha    = n / sum(log(tail / x_min))
        """
        if df.empty:
            return float("nan")
        per_ticker_alpha: list[float] = []
        for ticker, sub in df.groupby("ticker"):
            x = sub["volume"].astype(float).to_numpy()
            x = x[x > 0]
            if x.size < 30:
                continue
            x.sort()
            x_min = float(np.percentile(x, 90))
            if x_min <= 0:
                continue
            tail = x[x > x_min]
            n = tail.size
            if n < 5:
                continue
            denom = float(np.sum(np.log(tail / x_min)))
            if denom <= 0:
                continue
            per_ticker_alpha.append(n / denom)
        if not per_ticker_alpha:
            warnings.append(
                "volume_alpha: no ticker had enough data for the Hill estimator."
            )
            return float("nan")
        alpha = float(np.median(per_ticker_alpha))
        if alpha > 3.0:
            warnings.append(
                f"alpha={alpha:.2f} suggests weaker concentration than typical "
                "equity markets."
            )
        elif not (
            EMPIRICAL_VOLUME_ALPHA_RANGE[0]
            <= alpha
            <= EMPIRICAL_VOLUME_ALPHA_RANGE[1]
        ):
            warnings.append(
                f"alpha={alpha:.2f} outside empirical band "
                f"{EMPIRICAL_VOLUME_ALPHA_RANGE[0]}-"
                f"{EMPIRICAL_VOLUME_ALPHA_RANGE[1]}."
            )
        return alpha


# -- R3 INTEGRATION INTERFACE -------------------------------------------------
# CalibrationParams feeds into synthetic_market_sim as follows:
#
#   volatility_scale      = params.realized_volatility
#   time_bucket_weights   = params.intraday_volume_profile
#   return_shock_df       = params.return_df
#   return_shock_scale    = params.return_scale
#   trader_activity_alpha = params.volume_alpha
#
# Pareto activity multipliers (Dr. Milan, Review 2):
# For N traders in the synthetic population:
#   multipliers = scipy.stats.pareto(b=params.volume_alpha).rvs(N)
#   multipliers = clip(multipliers, 0, percentile(multipliers, 99))
#   multipliers = multipliers / mean(multipliers)        # normalize
# Each trader's order rate is scaled by their multiplier.
# Effect: p99/median ratio rises from 1.11 to ~10-30x, matching real NSE
# concentration (per Cont, 2001).
# -----------------------------------------------------------------------------
