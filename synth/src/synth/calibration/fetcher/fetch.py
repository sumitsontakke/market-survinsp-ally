"""NSE 1-minute OHLCV fetcher.

CLI usage::

    python fetch.py [--days N] [--t-minus N] [--tickers TICKER ...]

Defaults: ``--days 5 --t-minus 2`` over the full ``NIFTY_LIQUID_20``
universe. The range fetched is::

    end   = today - t_minus
    start = today - t_minus - days + 1

so a default invocation fetches the 5 trading days ending two days ago
(skips today and yesterday because intraday bars may still be settling).

The fetch is idempotent: ``fetch_log`` is consulted before any API call,
and ``INSERT OR IGNORE`` on ``market_data`` means re-running over the
same window costs at most a fetch_log lookup per (ticker, date).

Reference
---------
Cont, R. (2001). Empirical properties of asset returns: stylized facts
and statistical implications. Quantitative Finance, 1(2), 223-236.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import yfinance as yf

from synth.calibration.core.config import (
    BUCKETS_PER_DAY,
    DB_PATH,
    MIN_ROWS_PER_TICKER_DAY,
    NIFTY_LIQUID_20,
    NSE_CLOSE_UTC_HOUR,
    NSE_CLOSE_UTC_MINUTE,
    NSE_OPEN_UTC_HOUR,
    NSE_OPEN_UTC_MINUTE,
)
from synth.calibration.core.database import MarketDataDB

# NSE session boundaries as time-of-day for vectorized comparison.
_OPEN_TOTAL_MIN = NSE_OPEN_UTC_HOUR * 60 + NSE_OPEN_UTC_MINUTE
_CLOSE_TOTAL_MIN = NSE_CLOSE_UTC_HOUR * 60 + NSE_CLOSE_UTC_MINUTE


@dataclass
class FetchSummary:
    """Aggregated statistics from a fetch run, printed at the end."""

    tickers_processed: int = 0
    tickers_with_data: int = 0
    rows_inserted: int = 0
    rows_skipped_cached: int = 0
    dates_fetched: set[str] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_date_window(days: int, t_minus: int) -> list[str]:
    """Return ISO ``YYYY-MM-DD`` strings for the window (UTC dates).

    Skips Saturdays and Sundays - NSE is closed on weekends. Public
    holidays are not enumerated; ``fetch_log`` simply records zero rows
    for those days so we don't retry forever.
    """
    today = datetime.now(timezone.utc).date()
    end = today - timedelta(days=t_minus)
    start = end - timedelta(days=days - 1)
    out: list[str] = []
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5:  # Mon-Fri
            out.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return out


def _normalize_ticker_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize the raw yfinance frame into our flat schema.

    yfinance returns:
      - tz-aware DatetimeIndex (tz varies by version)
      - columns: Open, High, Low, Close, Volume, Adj Close (or a
        MultiIndex when called with a list of tickers)

    We collapse to a single-level frame with lower-case OHLCV columns
    and the canonical ``datetime_utc`` ISO string.
    """
    if raw is None or raw.empty:
        return pd.DataFrame(columns=[
            "ticker", "datetime_utc", "open", "high", "low", "close", "volume"
        ])

    df = raw.copy()
    # Drop the outer ticker level if yfinance returned a MultiIndex.
    if isinstance(df.columns, pd.MultiIndex):
        # Common shapes: ('Open', 'TICKER'), ('TICKER', 'Open').
        # Keep the OHLCV level whichever side it sits on.
        levels = df.columns.levels
        ohlcv_set = {"Open", "High", "Low", "Close", "Volume", "Adj Close"}
        if levels[0].isin(ohlcv_set).any():
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(-1)

    # Standardize index timezone -> UTC.
    if df.index.tz is None:
        # yfinance occasionally returns naive timestamps; assume IST for
        # NSE tickers and convert.
        df.index = df.index.tz_localize("Asia/Kolkata").tz_convert("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # Pull lower-case OHLCV columns.
    rename_map = {
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    }
    df = df.rename(columns=rename_map)
    needed = ["open", "high", "low", "close", "volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return pd.DataFrame(columns=[
            "ticker", "datetime_utc", *needed
        ])

    # Filter to NSE hours: 03:45-10:00 UTC inclusive on the bar's
    # timestamp (start of minute).
    minutes = df.index.hour * 60 + df.index.minute
    mask = (minutes >= _OPEN_TOTAL_MIN) & (minutes <= _CLOSE_TOTAL_MIN)
    df = df.loc[mask]
    if df.empty:
        return pd.DataFrame(columns=[
            "ticker", "datetime_utc", *needed
        ])

    # Drop zero-volume rows.
    df = df.loc[df["volume"].fillna(0) > 0]
    if df.empty:
        return pd.DataFrame(columns=[
            "ticker", "datetime_utc", *needed
        ])

    # Materialize the flat schema the DB expects.
    out = df[needed].copy()
    out.insert(0, "datetime_utc", df.index.strftime("%Y-%m-%dT%H:%M:%S+00:00"))
    out.insert(0, "ticker", ticker)
    return out.reset_index(drop=True)


def _trade_dates_present(df: pd.DataFrame) -> set[str]:
    """Distinct ``YYYY-MM-DD`` strings present in the normalized frame."""
    if df.empty:
        return set()
    return {ts[:10] for ts in df["datetime_utc"]}


def _yfinance_pull(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Single yfinance call with consistent parameters.

    yfinance's ``end`` is exclusive, so we add a one-day pad.
    """
    end_dt = (date.fromisoformat(end) + timedelta(days=1)).isoformat()
    return yf.download(
        tickers=[ticker],
        start=start,
        end=end_dt,
        interval="1m",
        auto_adjust=True,
        progress=False,
        threads=False,
    )


# ---------------------------------------------------------------------------
# Main fetch loop
# ---------------------------------------------------------------------------

def fetch(
    db: MarketDataDB,
    tickers: Iterable[str],
    target_dates: list[str],
) -> FetchSummary:
    """Pull each (ticker, date) pair that isn't already in fetch_log.

    The yfinance call covers the full date window per ticker - splitting
    it into one-call-per-day would multiply latency 5x for no benefit.
    After the pull we group by date and stamp fetch_log per date.
    """
    summary = FetchSummary()
    if not target_dates:
        summary.warnings.append("No target dates (window contained no weekdays).")
        return summary

    start_str, end_str = target_dates[0], target_dates[-1]

    for ticker in tickers:
        summary.tickers_processed += 1

        # Identify dates we still need for this ticker.
        wanted = [d for d in target_dates if not db.is_date_fetched(ticker, d)]
        cached = [d for d in target_dates if d not in wanted]
        for d in cached:
            print(f"  cached: {ticker} {d}")
        summary.rows_skipped_cached += len(cached)

        if not wanted:
            continue

        # One pull covers all uncached dates for this ticker.
        try:
            raw = _yfinance_pull(ticker, start_str, end_str)
        except Exception as exc:  # noqa: BLE001 - fetch must never crash
            err = f"{ticker}: yfinance error {exc!r}"
            print(f"  ERROR  {err}", file=sys.stderr)
            summary.errors.append(err)
            continue

        normalized = _normalize_ticker_frame(raw, ticker)
        if normalized.empty:
            warn = f"{ticker}: empty frame after filter for {start_str}..{end_str}"
            print(f"  warn   {warn}")
            summary.warnings.append(warn)
            # Stamp every wanted date with row_count=0 so we don't retry
            # forever (e.g. holiday or stale ticker).
            for d in wanted:
                db.log_fetch(ticker, d, row_count=0)
            continue

        summary.tickers_with_data += 1

        # Insert and per-date logging.
        n_inserted = db.insert_market_data(normalized)
        summary.rows_inserted += n_inserted

        per_date_counts = (
            normalized.assign(d=normalized["datetime_utc"].str.slice(0, 10))
            .groupby("d")
            .size()
            .to_dict()
        )
        for d in wanted:
            row_count = int(per_date_counts.get(d, 0))
            db.log_fetch(ticker, d, row_count=row_count)
            if row_count > 0:
                summary.dates_fetched.add(d)
            if 0 < row_count < MIN_ROWS_PER_TICKER_DAY:
                w = (
                    f"{ticker} {d}: only {row_count} rows after filter "
                    f"(min expected {MIN_ROWS_PER_TICKER_DAY})"
                )
                print(f"  warn   {w}")
                summary.warnings.append(w)

        kept_dates = sorted(per_date_counts)
        kept_total = int(sum(per_date_counts.values()))
        print(
            f"  fetched {ticker:15s} dates={','.join(kept_dates) or '-'} "
            f"rows={kept_total} inserted={n_inserted}"
        )

    return summary


def print_summary(summary: FetchSummary) -> None:
    print()
    print("=" * 72)
    print("FETCH SUMMARY")
    print("=" * 72)
    print(f"Tickers processed       : {summary.tickers_processed}")
    print(f"Tickers with data       : {summary.tickers_with_data}")
    if summary.dates_fetched:
        d_min = min(summary.dates_fetched)
        d_max = max(summary.dates_fetched)
        print(f"Dates covered           : {d_min} to {d_max} "
              f"({len(summary.dates_fetched)} distinct)")
    else:
        print("Dates covered           : (none)")
    print(f"Rows inserted (new)     : {summary.rows_inserted:,}")
    print(f"Rows skipped (cached)   : {summary.rows_skipped_cached}")
    if summary.warnings:
        print(f"Warnings ({len(summary.warnings)}):")
        for w in summary.warnings:
            print(f"  - {w}")
    if summary.errors:
        print(f"Errors ({len(summary.errors)}):")
        for e in summary.errors:
            print(f"  - {e}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch NSE 1-minute OHLCV via yfinance into SQLite."
    )
    parser.add_argument("--days", type=int, default=5,
                        help="Window length in calendar days (default 5).")
    parser.add_argument("--t-minus", type=int, default=2, dest="t_minus",
                        help="Days to step back from today for the window end "
                             "(default 2; skip the most recent 2 days).")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Override ticker universe (defaults to "
                             "NIFTY_LIQUID_20).")
    parser.add_argument("--db-path", type=Path, default=None,
                        help="Override DB path (defaults to env DB_PATH).")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    db_path = args.db_path if args.db_path is not None else DB_PATH
    db = MarketDataDB(db_path)

    tickers = args.tickers if args.tickers else list(NIFTY_LIQUID_20)
    dates = _build_date_window(args.days, args.t_minus)

    print(f"DB path             : {db_path}")
    print(f"Tickers in run      : {len(tickers)}")
    print(f"Date window (UTC)   : {dates[0] if dates else '-'} to "
          f"{dates[-1] if dates else '-'}  ({len(dates)} weekdays)")
    print(f"Bars per session    : {BUCKETS_PER_DAY} expected per ticker per day")
    print()

    summary = fetch(db, tickers, dates)
    print_summary(summary)
    return 0 if not summary.errors else 1


if __name__ == "__main__":
    sys.exit(main())
