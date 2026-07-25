"""NSE bhavcopy fetcher.

The bhavcopy is NSE's official end-of-day publication: one CSV per
trading day, every listed equity / ETF / G-Sec / SLB, with full
OHLCV + delivery columns. Files live under ``nsearchives.nseindia.com``
and are public (no auth, just a browser-like User-Agent).

This module pulls the CSV for a date range, parses, filters to the
configured ticker universe, and writes daily rows into
``daily_market_data`` via :class:`MarketDataDB`.

URL pattern (verified 2026-05-08)::

    https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_DDMMYYYY.csv

The endpoint returns:

* HTTP 200 + CSV body on a trading day
* HTTP 404 on weekends and dates with no archive

Holidays and weekends are recorded in ``daily_fetch_log`` with
``status='holiday'`` so we don't re-probe them.

Reference
---------
Cont, R. (2001). Empirical properties of asset returns: stylized facts
and statistical implications. Quantitative Finance, 1(2), 223-236.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from synth.calibration.core.config import DB_PATH
from synth.calibration.core.database import MarketDataDB

_BHAVCOPY_URL = (
    "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv"
)
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
_HTTP_TIMEOUT_SEC = 30

# The bhavcopy header has leading spaces in column names; we normalise
# them to lower_snake on parse.
_RENAME_MAP: dict[str, str] = {
    "SYMBOL": "ticker",
    "SERIES": "series",
    "DATE1": "trade_date_str",
    "PREV_CLOSE": "prev_close",
    "OPEN_PRICE": "open",
    "HIGH_PRICE": "high",
    "LOW_PRICE": "low",
    "LAST_PRICE": "last_price",
    "CLOSE_PRICE": "close",
    "AVG_PRICE": "avg_price",
    "TTL_TRD_QNTY": "volume",
    "TURNOVER_LACS": "turnover_lacs",
    "NO_OF_TRADES": "n_trades",
    "DELIV_QTY": "deliv_qty",
    "DELIV_PER": "deliv_pct",
}

_log = logging.getLogger(__name__)


@dataclass
class BhavcopySummary:
    """Aggregated outcome of a bhavcopy fetch run."""

    dates_processed: int = 0
    dates_with_data: int = 0
    dates_holiday: int = 0
    dates_missing: int = 0
    dates_cached: int = 0
    rows_inserted: int = 0
    universe_filter: tuple[str, ...] = ()
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class BhavcopyFetcher:
    """Pulls daily bhavcopy CSVs from NSE archives into ``daily_market_data``.

    Parameters
    ----------
    db_path:
        SQLite path. Defaults to ``core.config.DB_PATH``.
    universe:
        Optional set of tickers to keep (without exchange suffix - bhavcopy
        uses raw symbols like ``RELIANCE``, not ``RELIANCE.NS``). If None,
        all equity-series rows are stored.
    series:
        Series codes to keep. Default ``("EQ",)`` retains common equity only,
        excluding G-Sec (``GS``), warrants, etc.
    """

    def __init__(
        self,
        db_path: Path | str = DB_PATH,
        universe: Optional[Sequence[str]] = None,
        series: Sequence[str] = ("EQ",),
    ) -> None:
        self.db = MarketDataDB(db_path)
        self.universe = (
            None
            if universe is None
            else {self._strip_suffix(s).upper() for s in universe}
        )
        self.series = tuple(series)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def fetch_window(
        self,
        start_date: str,
        end_date: str,
        force: bool = False,
    ) -> BhavcopySummary:
        """Fetch every trading-day bhavcopy in ``[start_date, end_date]``.

        Weekends are skipped a priori. Dates already in ``daily_fetch_log``
        are skipped unless ``force=True``. Real holidays show up as a 404
        from the archive and get logged with ``status='holiday'``.
        """
        summary = BhavcopySummary(
            universe_filter=tuple(sorted(self.universe)) if self.universe else ()
        )
        dates = self._weekday_range(start_date, end_date)
        for trade_date in dates:
            summary.dates_processed += 1
            if not force and self.db.is_daily_fetched(trade_date):
                summary.dates_cached += 1
                _log.info("cached: bhavcopy %s", trade_date)
                continue
            try:
                csv_text = self._download_csv(trade_date)
            except HTTPError as exc:
                if exc.code == 404:
                    summary.dates_holiday += 1
                    self.db.log_daily_fetch(trade_date, row_count=0, status="holiday")
                    _log.info("holiday: bhavcopy %s (404)", trade_date)
                    continue
                err = f"{trade_date}: HTTP {exc.code} {exc.reason}"
                summary.errors.append(err)
                self.db.log_daily_fetch(trade_date, row_count=0, status="missing")
                _log.warning(err)
                continue
            except URLError as exc:
                err = f"{trade_date}: network error {exc.reason!r}"
                summary.errors.append(err)
                _log.warning(err)
                continue
            except Exception as exc:  # noqa: BLE001 - never crash the loop
                err = f"{trade_date}: unexpected {exc!r}"
                summary.errors.append(err)
                _log.exception(err)
                continue

            df = self._parse_csv(csv_text, trade_date)
            if df.empty:
                summary.dates_missing += 1
                self.db.log_daily_fetch(trade_date, row_count=0, status="missing")
                summary.warnings.append(
                    f"{trade_date}: parsed CSV had 0 rows after universe/series filter"
                )
                continue

            inserted = self.db.insert_daily_market_data(df)
            self.db.log_daily_fetch(trade_date, row_count=int(len(df)), status="ok")
            summary.dates_with_data += 1
            summary.rows_inserted += inserted
            _log.info(
                "fetched: bhavcopy %s rows=%d inserted=%d",
                trade_date, len(df), inserted,
            )
        return summary

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    @staticmethod
    def _strip_suffix(symbol: str) -> str:
        """Drop any exchange suffix - bhavcopy uses bare ticker codes."""
        return symbol.split(".", 1)[0]

    @staticmethod
    def _weekday_range(start_date: str, end_date: str) -> list[str]:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
        out: list[str] = []
        cursor = start
        while cursor <= end:
            if cursor.weekday() < 5:
                out.append(cursor.isoformat())
            cursor += timedelta(days=1)
        return out

    @staticmethod
    def _date_token(trade_date: str) -> str:
        """ISO ``YYYY-MM-DD`` -> ``DDMMYYYY`` token NSE expects in the URL."""
        d = date.fromisoformat(trade_date)
        return f"{d.day:02d}{d.month:02d}{d.year:04d}"

    def _download_csv(self, trade_date: str) -> str:
        """Fetch the bhavcopy CSV body for one date. Raises ``HTTPError`` on
        non-2xx so the caller can distinguish 404 (holiday) from other
        failures."""
        url = _BHAVCOPY_URL.format(ddmmyyyy=self._date_token(trade_date))
        req = Request(
            url,
            headers={
                "User-Agent": _USER_AGENT,
                "Accept": "text/csv,*/*;q=0.9",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.nseindia.com/",
            },
        )
        with urlopen(req, timeout=_HTTP_TIMEOUT_SEC) as resp:
            return resp.read().decode("utf-8", errors="replace")

    def _parse_csv(self, csv_text: str, trade_date: str) -> pd.DataFrame:
        """Parse the bhavcopy text into the schema ``daily_market_data`` expects.

        The column header has leading whitespace (``" SERIES"``, ``" DATE1"``
        etc.) so we strip after read. Column ``DATE1`` is ``08-May-2026``;
        we coerce to ISO ``YYYY-MM-DD`` and validate it matches the
        requested date.
        """
        try:
            raw = pd.read_csv(io.StringIO(csv_text), dtype=str)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"bhavcopy parse error for {trade_date}: {exc!r}") from exc
        if raw is None or raw.empty:
            return pd.DataFrame()

        raw.columns = [c.strip() for c in raw.columns]
        present = [c for c in _RENAME_MAP if c in raw.columns]
        if len(present) < 5:
            raise ValueError(
                f"bhavcopy schema unexpected for {trade_date}: cols={list(raw.columns)[:6]}"
            )
        df = raw[present].rename(columns=_RENAME_MAP)

        # Strip any per-cell whitespace introduced by NSE's CSV formatting.
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].str.strip()

        # Coerce numerics.
        for col in (
            "prev_close", "open", "high", "low", "last_price", "close",
            "avg_price", "volume", "turnover_lacs", "deliv_qty", "deliv_pct",
        ):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "n_trades" in df.columns:
            df["n_trades"] = pd.to_numeric(df["n_trades"], errors="coerce").astype("Int64")

        # Date column -> ISO and check.
        if "trade_date_str" in df.columns:
            parsed = pd.to_datetime(df["trade_date_str"], format="%d-%b-%Y", errors="coerce")
            df["trade_date"] = parsed.dt.strftime("%Y-%m-%d")
            file_dates = df["trade_date"].dropna().unique()
            if len(file_dates) == 1 and file_dates[0] != trade_date:
                # Sometimes Akamai serves a stale file under a holiday URL
                _log.warning(
                    "bhavcopy date mismatch: requested %s but file says %s",
                    trade_date, file_dates[0],
                )
                return pd.DataFrame()
            df = df.drop(columns=["trade_date_str"])
        else:
            df["trade_date"] = trade_date

        # Keep only target series.
        if self.series:
            df = df.loc[df["series"].isin(self.series)].copy()

        # Universe filter.
        if self.universe is not None:
            df = df.loc[df["ticker"].str.upper().isin(self.universe)].copy()

        # Drop rows with no usable price (mainly defensive).
        df = df.loc[df["close"].notna()].copy()

        # Make sure all expected columns exist for insert_daily_market_data.
        for col in (
            "series", "prev_close", "open", "high", "low", "last_price",
            "close", "avg_price", "volume", "turnover_lacs", "n_trades",
            "deliv_qty", "deliv_pct",
        ):
            if col not in df.columns:
                df[col] = pd.NA
        return df.reset_index(drop=True)


def print_summary(summary: BhavcopySummary) -> None:
    print()
    print("=" * 72)
    print("BHAVCOPY FETCH SUMMARY")
    print("=" * 72)
    print(f"Dates processed     : {summary.dates_processed}")
    print(f"Dates with data     : {summary.dates_with_data}")
    print(f"Dates flagged holiday (404) : {summary.dates_holiday}")
    print(f"Dates missing       : {summary.dates_missing}")
    print(f"Dates cached        : {summary.dates_cached}")
    print(f"Rows inserted (new) : {summary.rows_inserted:,}")
    if summary.universe_filter:
        print(f"Universe filter     : {len(summary.universe_filter)} symbols")
    if summary.warnings:
        print(f"Warnings ({len(summary.warnings)}):")
        for w in summary.warnings:
            print(f"  - {w}")
    if summary.errors:
        print(f"Errors ({len(summary.errors)}):")
        for e in summary.errors:
            print(f"  - {e}")
    print("=" * 72)
