"""Run loader - reads orders / trades / scenarios from a run directory.

Lightweight wrapper that gives downstream code (features, GNN, eval) a
typed view over a run's artifacts without copy-pasting glob+parse logic.

Cohorts are resolved by name, not enumerated paths, so the same config
works whether ``outputs/runs/`` lives at the host root, inside a Docker
volume, or in a future S3 mount.

Reference
---------
Cont, R. (2001). Empirical properties of asset returns: stylized facts
and statistical implications. Quantitative Finance, 1(2), 223-236.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from detect.dataset.universe import infer_run_family

# Where to find runs. Can be overridden by the OUTPUTS_PATH env var.
_DEFAULT_OUTPUTS_ROOT = Path(os.environ.get("OUTPUTS_PATH", "/app/outputs"))


@dataclass(frozen=True)
class Run:
    """All artifacts of a single synthetic run, lazily loaded once."""

    run_id: str
    run_name: str
    family: str
    output_path: Path
    orders: pd.DataFrame
    trades: pd.DataFrame
    scenarios: pd.DataFrame
    manifest: dict = field(default_factory=dict)

    @property
    def n_orders(self) -> int:
        return int(len(self.orders))

    @property
    def n_trades(self) -> int:
        return int(len(self.trades))

    @property
    def n_traders(self) -> int:
        if self.trades.empty:
            return 0
        return int(
            pd.unique(
                pd.concat(
                    [self.trades["buy_trader_id"], self.trades["sell_trader_id"]],
                    ignore_index=True,
                ).astype(str)
            ).shape[0]
        )

    @property
    def n_manipulative_trades(self) -> int:
        if self.trades.empty or "is_manipulative" not in self.trades.columns:
            return 0
        return int(self.trades["is_manipulative"].astype(bool).sum())


# ---------------------------------------------------------------------------
# Cohort resolution
# ---------------------------------------------------------------------------

# Named cohorts referenced from YAML configs. Add new ones here as the
# project grows; keep names ALL_CAPS for visibility in configs.
KNOWN_COHORTS: dict[str, callable] = {}


def cohort(name: str):
    """Decorator: register a cohort resolver under ``name``."""
    def _decorator(func):
        KNOWN_COHORTS[name] = func
        return func
    return _decorator


@cohort("PHASE1_R01_R24")
def _resolve_phase1_r01_r24(outputs_root: Path) -> list[Path]:
    base = outputs_root / "runs"
    if not base.exists():
        return []
    return sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("R"))


@cohort("R3_CALIBRATED_24")
def _resolve_r3_calibrated_24(outputs_root: Path) -> list[Path]:
    """Calibrated regeneration of the R01-R24 fixture set.

    Created by ``training/synthetic/calibrated_runner.py`` in M3. Until
    that exists this returns an empty list and the harness will surface a
    clear "no runs" error.
    """
    base = outputs_root / "calibrated_runs"
    if not base.exists():
        return []
    return sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("R"))


def list_runs_in_cohort(
    cohort_name: str,
    outputs_root: Optional[Path] = None,
) -> list[Path]:
    if cohort_name not in KNOWN_COHORTS:
        raise ValueError(
            f"Unknown cohort '{cohort_name}'. Known: {sorted(KNOWN_COHORTS)}"
        )
    root = outputs_root if outputs_root is not None else _DEFAULT_OUTPUTS_ROOT
    return KNOWN_COHORTS[cohort_name](root)


# ---------------------------------------------------------------------------
# Single-run loader
# ---------------------------------------------------------------------------

def _read_csv_or_parquet(path: Path, stem: str) -> pd.DataFrame:
    p_path = path / f"{stem}.parquet"
    c_path = path / f"{stem}.csv"
    if p_path.exists():
        df = pd.read_parquet(p_path)
    elif c_path.exists():
        df = pd.read_csv(c_path)
    else:
        return pd.DataFrame()
    df.columns = [c.lower().strip() for c in df.columns]
    return df


def load_run(run_path: Path | str) -> Run:
    """Load all four artifacts for one run. Empty frames if missing."""
    p = Path(run_path)
    orders = _read_csv_or_parquet(p, "orders")
    trades = _read_csv_or_parquet(p, "trades")
    scenarios = _read_csv_or_parquet(p, "scenarios")

    # Manifest is optional; keep going if missing.
    manifest_path = p / "manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            manifest = {}

    # Light-touch normalization the rest of the package relies on.
    if not trades.empty:
        if "timestamp" in trades.columns:
            trades = trades.copy()
            trades["timestamp"] = pd.to_datetime(trades["timestamp"], errors="coerce")
        for col in ("buy_trader_id", "sell_trader_id", "instrument_id", "scenario_id", "scenario_type"):
            if col in trades.columns:
                trades[col] = trades[col].astype(str)
        if "quantity" not in trades.columns and "volume" in trades.columns:
            trades = trades.rename(columns={"volume": "quantity"})
        if "is_manipulative" in trades.columns:
            trades["is_manipulative"] = (
                trades["is_manipulative"].astype(str).str.lower().eq("true")
            )

    if not orders.empty and "timestamp" in orders.columns:
        orders = orders.copy()
        orders["timestamp"] = pd.to_datetime(orders["timestamp"], errors="coerce")

    return Run(
        run_id=p.name,
        run_name=p.name,
        family=infer_run_family(p.name, manifest.get("scenario_summary", "")),
        output_path=p,
        orders=orders,
        trades=trades,
        scenarios=scenarios,
        manifest=manifest,
    )


def load_cohort(cohort_name: str, outputs_root: Optional[Path] = None) -> Iterable[Run]:
    """Generator over loaded runs in a cohort. Skips empty runs with a warning."""
    paths = list_runs_in_cohort(cohort_name, outputs_root=outputs_root)
    for p in paths:
        run = load_run(p)
        if run.n_trades == 0:
            print(f"  warn: skipping {run.run_id} (no trades)")
            continue
        yield run
