"""NSE calibration CLI.

CLI usage::

    python calibrate.py [--date YYYY-MM-DD | latest] [--db-path PATH]
                        [--plot-dir PATH] [--no-plots]

Reads market_data from the shared SQLite store, computes (or loads from
the cache) :class:`CalibrationParams` for the target date, prints the
human-readable summary, and writes four validation plots:

    * ``volume_profile.png``       (Fact D - intraday U-shape)
    * ``return_distribution.png``  (Fact A - heavy tails vs Student-t)
    * ``volume_concentration.png`` (Fact E - Pareto / Zipf check)
    * ``calibration_gap_table.png`` (NSE vs Phase 1 synthetic baseline)

Reference
---------
Cont, R. (2001). Empirical properties of asset returns: stylized facts
and statistical implications. Quantitative Finance, 1(2), 223-236.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sci_stats

from synth.calibration.core.config import (
    BUCKETS_PER_DAY,
    DB_PATH,
    EMPIRICAL_RETURN_DF_RANGE,
    EMPIRICAL_VOLUME_ALPHA_RANGE,
    IST_OFFSET_MINUTES,
    NSE_OPEN_UTC_HOUR,
    NSE_OPEN_UTC_MINUTE,
    PLOT_DIR,
    SYNTHETIC_BASELINES,
)
from synth.calibration.core.database import MarketDataDB
from synth.calibration.core.models import CalibrationParams
from core.nse_bhavcopy import BhavcopyFetcher, print_summary as bhav_print_summary
from core.nse_calibrator import NSECalibrator
from core.nse_calibrator_daily import NSEDailyCalibrator

_OPEN_TOTAL_MIN = NSE_OPEN_UTC_HOUR * 60 + NSE_OPEN_UTC_MINUTE


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _ist_label_for_slot(slot: int) -> str:
    """IST clock-time label (HH:MM) for a 0-indexed minute-of-session."""
    total = _OPEN_TOTAL_MIN + slot + IST_OFFSET_MINUTES
    return "{0:02d}:{1:02d}".format((total // 60) % 24, total % 60)


def plot_volume_profile(params: CalibrationParams, out: Path) -> None:
    """Line chart of the 375-slot intraday weight curve (IST x-axis)."""
    profile = np.asarray(params.intraday_volume_profile, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(profile.size)
    ax.plot(x, profile, color="#1E2761", linewidth=1.5, label="NSE volume share")
    ax.fill_between(x, profile, color="#1E2761", alpha=0.18)

    # X axis ticks at IST hour boundaries: 09:15, 10:00, 11:00, ..., 15:30
    # Sessions: 09:15-15:30 = 375 mins. Mark 09:15, 10:00, 11:00, 12:00, 13:00, 14:00, 15:00, 15:30.
    tick_slots = [0]
    cursor_min = _OPEN_TOTAL_MIN + IST_OFFSET_MINUTES  # IST start
    next_hour = ((cursor_min // 60) + 1) * 60
    while next_hour < cursor_min + BUCKETS_PER_DAY:
        tick_slots.append(next_hour - cursor_min)
        next_hour += 60
    tick_slots.append(BUCKETS_PER_DAY - 1)
    tick_slots = sorted(set(s for s in tick_slots if 0 <= s < BUCKETS_PER_DAY))
    ax.set_xticks(tick_slots)
    ax.set_xticklabels([_ist_label_for_slot(s) for s in tick_slots], rotation=0, fontsize=9)

    # Annotate open / midday / close bands.
    open_band = profile[: max(1, BUCKETS_PER_DAY // 10)].mean()
    close_band = profile[-max(1, BUCKETS_PER_DAY // 10):].mean()
    mid_band = profile[BUCKETS_PER_DAY // 3 : 2 * BUCKETS_PER_DAY // 3].mean()
    ax.text(0.02, 0.95,
            f"open decile: {open_band:.4f}\nmidday third: {mid_band:.4f}\nclose decile: {close_band:.4f}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F4F6FA", edgecolor="#5C6480"))

    ax.set_xlabel("IST time")
    ax.set_ylabel("Normalized volume share (sum = 1.0)")
    ax.set_title(f"NSE Intraday Volume Profile - {params.calibration_date}")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_return_distribution(
    params: CalibrationParams,
    pooled_returns: np.ndarray,
    out: Path,
) -> None:
    """Histogram of pooled log-returns with Student-t and Normal overlays."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if pooled_returns.size > 0:
        # Clip absurd outliers (>10 sigma) for plot readability without
        # touching the fit numbers (those used the full sample).
        sigma = float(pooled_returns.std(ddof=1))
        if sigma > 0:
            mask = np.abs(pooled_returns) <= 10 * sigma
            x = pooled_returns[mask]
        else:
            x = pooled_returns
        ax.hist(x, bins=100, density=True, alpha=0.5, color="#1E2761",
                label="Pooled log-returns")
        grid = np.linspace(x.min(), x.max(), 400)
        ax.plot(grid,
                sci_stats.t.pdf(grid, params.return_df, loc=params.return_loc, scale=params.return_scale),
                color="#C8102E", linewidth=2,
                label=f"Student-t fit (df={params.return_df:.2f})")
        mu, sd = float(np.mean(x)), float(np.std(x, ddof=1))
        ax.plot(grid, sci_stats.norm.pdf(grid, mu, sd),
                color="#0F7A4D", linewidth=2, linestyle="--",
                label="Gaussian (for comparison)")
    df_lo, df_hi = EMPIRICAL_RETURN_DF_RANGE
    in_band = df_lo <= params.return_df <= df_hi
    band_note = (
        "OK in [{0}-{1}]".format(df_lo, df_hi)
        if in_band
        else "GAP outside [{0}-{1}]".format(df_lo, df_hi)
    )
    ax.set_yscale("log")
    ax.set_xlabel("1-min log-return")
    ax.set_ylabel("Density (log scale)")
    ax.set_title(
        f"NSE Log-Return Distribution - df={params.return_df:.2f}  ({band_note})"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_volume_concentration(
    params: CalibrationParams,
    pooled_volumes: np.ndarray,
    out: Path,
) -> None:
    """Log-log Zipf plot: rank vs volume + Pareto fit line."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if pooled_volumes.size > 0:
        v = np.asarray(pooled_volumes, dtype=float)
        v = v[v > 0]
        v = np.sort(v)[::-1]  # descending
        rank = np.arange(1, v.size + 1)
        ax.loglog(rank, v, marker=".", linestyle="none", color="#1E2761",
                  alpha=0.5, markersize=3, label="rank vs volume")
        # Pareto fit line using params.volume_alpha:
        # In a Pareto with shape alpha, log(rank) ≈ const - alpha * log(volume),
        # so log(volume) ≈ const' - (1/alpha) * log(rank).
        if not np.isnan(params.volume_alpha) and params.volume_alpha > 0:
            x_fit = rank
            y_fit = v[0] * (rank / rank[0]) ** (-1.0 / max(params.volume_alpha, 1e-3))
            ax.loglog(x_fit, y_fit, color="#C8102E", linewidth=2,
                      label=f"Pareto slope (alpha={params.volume_alpha:.2f})")
    a_lo, a_hi = EMPIRICAL_VOLUME_ALPHA_RANGE
    in_band = a_lo <= params.volume_alpha <= a_hi
    band_note = (
        "OK in [{0}-{1}]".format(a_lo, a_hi)
        if in_band
        else "GAP outside [{0}-{1}]".format(a_lo, a_hi)
    )
    ax.set_xlabel("Rank (log)")
    ax.set_ylabel("Per-minute volume (log)")
    ax.set_title(
        f"NSE Volume Concentration - alpha={params.volume_alpha:.2f}  ({band_note})"
    )
    ax.legend(fontsize=9)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_gap_table(params: CalibrationParams, out: Path) -> None:
    """Side-by-side table: NSE values vs Phase 1 synthetic baseline."""
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_axis_off()

    df_lo, df_hi = EMPIRICAL_RETURN_DF_RANGE
    a_lo, a_hi = EMPIRICAL_VOLUME_ALPHA_RANGE
    df_in = df_lo <= params.return_df <= df_hi
    a_in = a_lo <= params.volume_alpha <= a_hi

    # Direction commentary uses the recorded synthetic baselines.
    synth_df = float(SYNTHETIC_BASELINES["return_df"])
    if params.return_df > synth_df:
        df_dir = f"NSE thinner ({params.return_df:.2f} > {synth_df:.2f})"
    else:
        df_dir = f"NSE fatter ({params.return_df:.2f} <= {synth_df:.2f})"
    a_dir = "NSE more concentrated (real Pareto vs uniform synthetic)"

    rows = [
        ["Parameter", "NSE value", "Synthetic baseline", "Empirical band", "Direction"],
        [
            "realized_vol",
            f"{params.realized_volatility:.4f}",
            str(SYNTHETIC_BASELINES["realized_volatility"]),
            "n/a",
            "n/a",
        ],
        [
            "return_df",
            f"{params.return_df:.2f}",
            f"{synth_df:.2f}",
            f"[{df_lo}-{df_hi}]",
            df_dir,
        ],
        [
            "volume_alpha",
            f"{params.volume_alpha:.2f}",
            str(SYNTHETIC_BASELINES["volume_alpha"]),
            f"[{a_lo}-{a_hi}]",
            a_dir,
        ],
    ]
    table = ax.table(
        cellText=rows,
        loc="center",
        colWidths=[0.16, 0.14, 0.26, 0.14, 0.30],
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    # Header style.
    for col in range(5):
        cell = table[(0, col)]
        cell.set_facecolor("#1E2761")
        cell.set_text_props(color="white", weight="bold")

    # Color the band column based on in/out.
    band_color_df = "#DCFCE7" if df_in else "#FEE2E2"
    band_color_a = "#DCFCE7" if a_in else "#FEE2E2"
    table[(2, 3)].set_facecolor(band_color_df)
    table[(3, 3)].set_facecolor(band_color_a)

    ax.set_title(
        f"NSE vs Phase 1 Synthetic Baseline - calibration {params.calibration_date}",
        fontsize=12, fontweight="bold", color="#1E2761", pad=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot orchestrator
# ---------------------------------------------------------------------------

def emit_plots(
    params: CalibrationParams,
    db: MarketDataDB,
    plot_dir: Path,
) -> list[Path]:
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Pull the source frame for the calibration date so we can plot the
    # raw return + volume samples behind the four parameters.
    df = db.get_market_data(
        tickers=params.tickers_used,
        start_date=params.calibration_date,
        end_date=params.calibration_date,
    )
    pooled_returns = NSECalibrator._per_ticker_log_returns(df).to_numpy() if not df.empty else np.array([])
    pooled_volumes = df["volume"].astype(float).to_numpy() if not df.empty else np.array([])

    paths: list[Path] = []
    p1 = plot_dir / "volume_profile.png"
    plot_volume_profile(params, p1)
    paths.append(p1)

    p2 = plot_dir / "return_distribution.png"
    plot_return_distribution(params, pooled_returns, p2)
    paths.append(p2)

    p3 = plot_dir / "volume_concentration.png"
    plot_volume_concentration(params, pooled_volumes, p3)
    paths.append(p3)

    p4 = plot_dir / "calibration_gap_table.png"
    plot_gap_table(params, p4)
    paths.append(p4)
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute NSE calibration parameters from stored data."
    )
    parser.add_argument("--date", default="latest",
                        help="YYYY-MM-DD or 'latest' (default 'latest').")
    parser.add_argument("--mode", choices=["daily", "intraday"], default="daily",
                        help="daily: bhavcopy EOD path (default). "
                             "intraday: 1-minute path (yfinance fetcher).")
    parser.add_argument("--bhavcopy-window-days", type=int, default=30,
                        dest="bhavcopy_window_days",
                        help="In daily mode, trailing window of trading days "
                             "to pool returns over (default 30).")
    parser.add_argument("--auto-fetch-bhavcopy", action="store_true",
                        help="In daily mode, run BhavcopyFetcher first to "
                             "populate any missing dates in the trailing window.")
    parser.add_argument("--db-path", type=Path, default=None,
                        help="Override DB path (defaults to env DB_PATH).")
    parser.add_argument("--plot-dir", type=Path, default=None,
                        help="Override plot output dir "
                             "(defaults to env PLOT_DIR or /data/plots).")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip writing PNGs.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    db_path = args.db_path if args.db_path is not None else DB_PATH
    plot_dir = args.plot_dir if args.plot_dir is not None else PLOT_DIR

    if args.mode == "daily":
        if args.auto_fetch_bhavcopy:
            from datetime import date, timedelta
            today = date.today()
            start = today - timedelta(days=int(args.bhavcopy_window_days * 1.6) + 7)
            fetcher = BhavcopyFetcher(db_path=db_path, universe=None)
            print(f"auto-fetching bhavcopy {start} -> {today}")
            s = fetcher.fetch_window(start.isoformat(), today.isoformat())
            bhav_print_summary(s)

        daily = NSEDailyCalibrator(
            db_path=db_path,
            pool_window_days=args.bhavcopy_window_days,
        )
        try:
            params = daily.calibrate(date_str=args.date)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(params.summary())
        if not args.no_plots:
            paths = emit_plots(params, daily.db, plot_dir)
            print("Plots written:")
            for p in paths:
                print(f"  {p}")
        return 0

    # intraday mode (1-minute, yfinance path)
    calibrator = NSECalibrator(db_path)
    try:
        params = calibrator.calibrate(date_str=args.date)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(params.summary())
    print()
    if not args.no_plots:
        paths = emit_plots(params, calibrator.db, plot_dir)
        print("Plots written:")
        for p in paths:
            print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
