"""CLI wrapper around BhavcopyFetcher.

Defaults to a 30-trading-day trailing window so a single ``run --rm
fetcher_bhav`` populates everything the calibrator's daily mode needs.

Usage::

    python fetch_bhavcopy.py [--days N] [--from YYYY-MM-DD] [--to YYYY-MM-DD]
                             [--universe nifty20|all]

Reference
---------
Cont, R. (2001). Empirical properties of asset returns: stylized facts
and statistical implications. Quantitative Finance, 1(2), 223-236.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

from synth.calibration.core.config import DB_PATH, NIFTY_LIQUID_20
from core.nse_bhavcopy import BhavcopyFetcher, print_summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch NSE bhavcopy CSVs.")
    p.add_argument("--days", type=int, default=30,
                   help="Trailing calendar days back from today (default 30).")
    p.add_argument("--from", dest="from_date", default=None,
                   help="Override start date YYYY-MM-DD.")
    p.add_argument("--to", dest="to_date", default=None,
                   help="Override end date YYYY-MM-DD.")
    p.add_argument("--universe", choices=["nifty20", "all"], default="all",
                   help="all (default) keeps the full NSE EQ universe so the "
                        "cross-sectional Pareto can be computed; nifty20 "
                        "keeps only NIFTY_LIQUID_20.")
    p.add_argument("--db-path", type=Path, default=None,
                   help="Override DB path (defaults to env DB_PATH).")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    db_path = args.db_path if args.db_path is not None else DB_PATH

    today = date.today()
    end = date.fromisoformat(args.to_date) if args.to_date else today
    start = (
        date.fromisoformat(args.from_date)
        if args.from_date
        else end - timedelta(days=args.days)
    )

    universe = NIFTY_LIQUID_20 if args.universe == "nifty20" else None
    fetcher = BhavcopyFetcher(db_path=db_path, universe=universe)
    print(f"DB path        : {db_path}")
    print(f"Window         : {start.isoformat()} to {end.isoformat()}")
    print(f"Universe       : {args.universe}")
    print(f"Series filter  : EQ")
    print()

    summary = fetcher.fetch_window(start.isoformat(), end.isoformat())
    print_summary(summary)
    return 0 if not summary.errors else 1


if __name__ == "__main__":
    sys.exit(main())
