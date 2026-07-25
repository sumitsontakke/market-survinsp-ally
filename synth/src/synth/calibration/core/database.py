"""SQLite layer for the NSE calibration service.

Five tables:

* ``market_data``         - 1-minute OHLCV bars (yfinance path)
* ``fetch_log``           - per-ticker / per-date 1-min fetch bookkeeping
* ``daily_market_data``   - daily OHLCV from NSE bhavcopy
* ``daily_fetch_log``     - per-date bhavcopy bookkeeping (one CSV per day)
* ``calibration_runs``    - persisted CalibrationParams keyed by date

All datetimes are stored as ISO-8601 strings in UTC. IST conversions happen
only at the display / plot layer.

Reference
---------
Cont, R. (2001). Empirical properties of asset returns: stylized facts
and statistical implications. Quantitative Finance, 1(2), 223-236.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional, Sequence

import pandas as pd

from synth.calibration.core.config import (
    DB_PATH,
    NSE_CLOSE_UTC_HOUR,
    NSE_CLOSE_UTC_MINUTE,
    NSE_OPEN_UTC_HOUR,
    NSE_OPEN_UTC_MINUTE,
)
from synth.calibration.core.models import CalibrationParams

_NSE_OPEN_HHMM = "{0:02d}:{1:02d}".format(NSE_OPEN_UTC_HOUR, NSE_OPEN_UTC_MINUTE)
_NSE_CLOSE_HHMM = "{0:02d}:{1:02d}".format(NSE_CLOSE_UTC_HOUR, NSE_CLOSE_UTC_MINUTE)

_MARKET_DATA_COLS: tuple[str, ...] = (
    "ticker", "datetime_utc", "open", "high", "low", "close", "volume",
)

_DAILY_COLS: tuple[str, ...] = (
    "ticker", "trade_date", "series", "prev_close", "open", "high",
    "low", "last_price", "close", "avg_price", "volume",
    "turnover_lacs", "n_trades", "deliv_qty", "deliv_pct",
)


def _opt_float(v) -> "float | None":
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class MarketDataDB:
    """SQLite wrapper for NSE market data and persisted calibration runs."""

    def __init__(self, db_path: "Path | str" = DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    # connection helpers
    # ------------------------------------------------------------------
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), isolation_level=None)
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA foreign_keys = ON")
            conn.row_factory = sqlite3.Row
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS market_data (
                    ticker        TEXT NOT NULL,
                    datetime_utc  TEXT NOT NULL,
                    open          REAL,
                    high          REAL,
                    low           REAL,
                    close         REAL,
                    volume        REAL,
                    PRIMARY KEY (ticker, datetime_utc)
                );
                CREATE INDEX IF NOT EXISTS idx_market_data_dt
                    ON market_data (datetime_utc);

                CREATE TABLE IF NOT EXISTS fetch_log (
                    ticker     TEXT NOT NULL,
                    trade_date TEXT NOT NULL,
                    fetched_at TEXT,
                    row_count  INTEGER,
                    PRIMARY KEY (ticker, trade_date)
                );

                CREATE TABLE IF NOT EXISTS daily_market_data (
                    ticker        TEXT NOT NULL,
                    trade_date    TEXT NOT NULL,
                    series        TEXT,
                    prev_close    REAL,
                    open          REAL,
                    high          REAL,
                    low           REAL,
                    last_price    REAL,
                    close         REAL,
                    avg_price     REAL,
                    volume        REAL,
                    turnover_lacs REAL,
                    n_trades      INTEGER,
                    deliv_qty     REAL,
                    deliv_pct     REAL,
                    PRIMARY KEY (ticker, trade_date)
                );
                CREATE INDEX IF NOT EXISTS idx_daily_dt
                    ON daily_market_data (trade_date);

                CREATE TABLE IF NOT EXISTS daily_fetch_log (
                    trade_date TEXT PRIMARY KEY,
                    fetched_at TEXT,
                    row_count  INTEGER,
                    status     TEXT
                );

                CREATE TABLE IF NOT EXISTS calibration_runs (
                    run_id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    calibration_date  TEXT NOT NULL,
                    tickers_used      TEXT NOT NULL,
                    n_observations    INTEGER NOT NULL,
                    realized_vol      REAL,
                    return_df         REAL,
                    return_loc        REAL,
                    return_scale      REAL,
                    volume_alpha      REAL,
                    volume_profile    TEXT,
                    warnings          TEXT,
                    created_at        TEXT,
                    UNIQUE(calibration_date)
                );
                """
            )

    # ------------------------------------------------------------------
    # 1-minute market_data + fetch_log (yfinance path)
    # ------------------------------------------------------------------
    def is_date_fetched(self, ticker: str, trade_date: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM fetch_log WHERE ticker = ? AND trade_date = ?",
                (ticker, trade_date),
            ).fetchone()
        return row is not None

    def insert_market_data(self, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        missing = [c for c in _MARKET_DATA_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"insert_market_data: missing columns {missing}")
        records = [
            (
                str(r["ticker"]), str(r["datetime_utc"]),
                _opt_float(r["open"]), _opt_float(r["high"]),
                _opt_float(r["low"]), _opt_float(r["close"]),
                _opt_float(r["volume"]),
            )
            for _, r in df[list(_MARKET_DATA_COLS)].iterrows()
        ]
        with self._connect() as conn:
            cur = conn.executemany(
                """
                INSERT OR IGNORE INTO market_data
                  (ticker, datetime_utc, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )
            return int(cur.rowcount)

    def log_fetch(self, ticker: str, trade_date: str, row_count: int) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO fetch_log
                  (ticker, trade_date, fetched_at, row_count)
                VALUES (?, ?, ?, ?)
                """,
                (ticker, trade_date, datetime.now(timezone.utc).isoformat(), int(row_count)),
            )

    def get_market_data(
        self,
        tickers: Sequence[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame(columns=list(_MARKET_DATA_COLS))
        placeholders = ",".join("?" for _ in tickers)
        query = (
            "SELECT ticker, datetime_utc, open, high, low, close, volume "
            "FROM market_data "
            f"WHERE ticker IN ({placeholders}) "
            "  AND substr(datetime_utc, 1, 10) >= ? "
            "  AND substr(datetime_utc, 1, 10) <= ? "
            "  AND substr(datetime_utc, 12, 5) >= ? "
            "  AND substr(datetime_utc, 12, 5) <= ? "
            "  AND volume > 0 "
            "ORDER BY ticker, datetime_utc"
        )
        params = list(tickers) + [start_date, end_date, _NSE_OPEN_HHMM, _NSE_CLOSE_HHMM]
        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        if not df.empty:
            df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
        return df

    def get_available_dates(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT substr(datetime_utc, 1, 10) AS d
                FROM market_data
                WHERE volume > 0
                  AND substr(datetime_utc, 12, 5) >= ?
                  AND substr(datetime_utc, 12, 5) <= ?
                ORDER BY d
                """,
                (_NSE_OPEN_HHMM, _NSE_CLOSE_HHMM),
            ).fetchall()
        return [row["d"] for row in rows]

    def get_latest_trade_date(self) -> Optional[str]:
        dates = self.get_available_dates()
        return dates[-1] if dates else None

    # ------------------------------------------------------------------
    # daily_market_data + daily_fetch_log (bhavcopy path)
    # ------------------------------------------------------------------
    def is_daily_fetched(self, trade_date: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM daily_fetch_log WHERE trade_date = ?",
                (trade_date,),
            ).fetchone()
        return row is not None

    def insert_daily_market_data(self, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        missing = [c for c in _DAILY_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"insert_daily_market_data: missing columns {missing}")
        records = [
            (
                str(r["ticker"]),
                str(r["trade_date"]),
                str(r["series"]) if pd.notna(r["series"]) else None,
                _opt_float(r["prev_close"]),
                _opt_float(r["open"]),
                _opt_float(r["high"]),
                _opt_float(r["low"]),
                _opt_float(r["last_price"]),
                _opt_float(r["close"]),
                _opt_float(r["avg_price"]),
                _opt_float(r["volume"]),
                _opt_float(r["turnover_lacs"]),
                int(r["n_trades"]) if pd.notna(r["n_trades"]) else None,
                _opt_float(r["deliv_qty"]),
                _opt_float(r["deliv_pct"]),
            )
            for _, r in df[list(_DAILY_COLS)].iterrows()
        ]
        with self._connect() as conn:
            cur = conn.executemany(
                """
                INSERT OR IGNORE INTO daily_market_data
                  (ticker, trade_date, series, prev_close, open, high,
                   low, last_price, close, avg_price, volume,
                   turnover_lacs, n_trades, deliv_qty, deliv_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )
            return int(cur.rowcount)

    def log_daily_fetch(self, trade_date: str, row_count: int, status: str = "ok") -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO daily_fetch_log
                  (trade_date, fetched_at, row_count, status)
                VALUES (?, ?, ?, ?)
                """,
                (trade_date, datetime.now(timezone.utc).isoformat(), int(row_count), str(status)),
            )

    def get_daily_market_data(
        self,
        tickers: Sequence[str],
        start_date: str,
        end_date: str,
        series: Sequence[str] = ("EQ",),
    ) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame()
        # Bhavcopy stores bare symbols ("RELIANCE"), but callers often pass
        # yfinance-suffixed forms ("RELIANCE.NS"). Strip the suffix on read
        # so the same NIFTY_LIQUID_20 list works against either source.
        tickers = [str(t).split(".", 1)[0].upper() for t in tickers]
        t_placeholders = ",".join("?" for _ in tickers)
        s_placeholders = ",".join("?" for _ in series)
        query = (
            "SELECT ticker, trade_date, series, prev_close, open, high, low, "
            "       last_price, close, avg_price, volume, turnover_lacs, "
            "       n_trades, deliv_qty, deliv_pct "
            "FROM daily_market_data "
            f"WHERE ticker IN ({t_placeholders}) "
            f"  AND series IN ({s_placeholders}) "
            "  AND trade_date >= ? AND trade_date <= ? "
            "ORDER BY ticker, trade_date"
        )
        params = list(tickers) + list(series) + [start_date, end_date]
        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        return df

    def get_daily_universe_volumes(
        self,
        trade_date: str,
        series: Sequence[str] = ("EQ",),
    ) -> pd.Series:
        s_placeholders = ",".join("?" for _ in series)
        query = (
            "SELECT ticker, volume FROM daily_market_data "
            f"WHERE trade_date = ? AND series IN ({s_placeholders}) "
            "  AND volume > 0"
        )
        with self._connect() as conn:
            df = pd.read_sql_query(query, conn, params=[trade_date, *series])
        if df.empty:
            return pd.Series(dtype=float)
        return df.set_index("ticker")["volume"].astype(float)

    def get_latest_daily_trade_date(self) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(trade_date) AS d FROM daily_market_data"
            ).fetchone()
        return row["d"] if row and row["d"] else None

    def get_available_daily_dates(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT trade_date FROM daily_market_data ORDER BY trade_date"
            ).fetchall()
        return [r["trade_date"] for r in rows]

    # ------------------------------------------------------------------
    # calibration_runs
    # ------------------------------------------------------------------
    def save_calibration(self, params: CalibrationParams) -> int:
        payload = (
            params.calibration_date,
            json.dumps(list(params.tickers_used)),
            int(params.n_observations),
            float(params.realized_volatility),
            float(params.return_df),
            float(params.return_loc),
            float(params.return_scale),
            float(params.volume_alpha),
            json.dumps(list(params.intraday_volume_profile)),
            json.dumps(list(params.warnings)),
            datetime.now(timezone.utc).isoformat(),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO calibration_runs (
                    calibration_date, tickers_used, n_observations,
                    realized_vol, return_df, return_loc, return_scale,
                    volume_alpha, volume_profile, warnings, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
            row = conn.execute(
                "SELECT run_id FROM calibration_runs WHERE calibration_date = ?",
                (params.calibration_date,),
            ).fetchone()
        return int(row["run_id"])

    def get_calibration_for_date(self, date_str: str) -> Optional[CalibrationParams]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM calibration_runs WHERE calibration_date = ?",
                (date_str,),
            ).fetchone()
        return self._row_to_params(row) if row else None

    def get_latest_calibration(self) -> Optional[CalibrationParams]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM calibration_runs
                ORDER BY calibration_date DESC, run_id DESC
                LIMIT 1
                """
            ).fetchone()
        return self._row_to_params(row) if row else None

    @staticmethod
    def _row_to_params(row) -> CalibrationParams:
        return CalibrationParams(
            realized_volatility=float(row["realized_vol"]),
            intraday_volume_profile=[
                float(x) for x in json.loads(row["volume_profile"] or "[]")
            ],
            return_df=float(row["return_df"]),
            return_loc=float(row["return_loc"]),
            return_scale=float(row["return_scale"]),
            volume_alpha=float(row["volume_alpha"]),
            calibration_date=str(row["calibration_date"]),
            tickers_used=list(json.loads(row["tickers_used"] or "[]")),
            n_observations=int(row["n_observations"]),
            warnings=list(json.loads(row["warnings"] or "[]")),
        )

    def stats(self) -> dict:
        with self._connect() as conn:
            n_market = conn.execute("SELECT COUNT(*) AS c FROM market_data").fetchone()["c"]
            n_log = conn.execute("SELECT COUNT(*) AS c FROM fetch_log").fetchone()["c"]
            n_daily = conn.execute("SELECT COUNT(*) AS c FROM daily_market_data").fetchone()["c"]
            n_dlog = conn.execute("SELECT COUNT(*) AS c FROM daily_fetch_log").fetchone()["c"]
            n_cal = conn.execute("SELECT COUNT(*) AS c FROM calibration_runs").fetchone()["c"]
        dates = self.get_available_dates()
        d_dates = self.get_available_daily_dates()
        return {
            "market_data_rows": int(n_market),
                    "fetch_log_rows": int(n_log),
            "daily_market_data_rows": int(n_daily),
            "daily_fetch_log_rows": int(n_dlog),
            "calibration_runs_rows": int(n_cal),
            "intraday_date_range": (dates[0], dates[-1]) if dates else (None, None),
            "daily_date_range": (d_dates[0], d_dates[-1]) if d_dates else (None, None),
        }
