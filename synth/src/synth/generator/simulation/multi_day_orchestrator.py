"""multi_day_orchestrator.py — Multi-day simulation with persistent entity universe.

Runs N trading days sequentially, carrying a shared entity pool (traders,
brokers, instruments) across days.  New traders are onboarded and existing
ones occasionally retire.  Manipulation scenarios can be injected across
specific day windows.

Output layout::

    outputs/multi_day/{run_id}/
        day_001/trades.csv, orders.csv, …
        day_002/…
        ground_truth_ledger.csv
        entity_timeline.csv
        run_manifest.json
"""
from __future__ import annotations

import copy
import csv
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain helpers (kept local to avoid coupling to inner simulation classes)
# ---------------------------------------------------------------------------

_DEFAULT_TRADER_PROFILES = {
    "retail_random": 0.40,
    "momentum": 0.15,
    "mean_reversion": 0.15,
    "institutional_slicer": 0.10,
    "liquidity_provider": 0.20,
}


def _trader_id(index: int) -> str:
    return f"trader_{index:05d}"


def _broker_id(index: int) -> str:
    return f"broker_{index:03d}"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class TraderState:
    trader_id: str
    broker_id: str
    profile: str
    onboard_day: int
    retire_day: int | None = None
    is_manipulator: bool = False
    ramp_up_days: int = 3  # 50 % activity for first ramp_up_days

    @property
    def is_active(self) -> bool:
        return self.retire_day is None

    def activity_scale(self, day: int) -> float:
        """Return activity scale in [0.5, 1.0] based on ramp-up status."""
        days_since_onboard = day - self.onboard_day
        if days_since_onboard < self.ramp_up_days:
            return 0.5
        return 1.0


@dataclass
class InstrumentState:
    instrument_id: str
    symbol: str
    asset_class: str
    tick_size: float
    lot_size: int
    price_band: tuple[float, float]
    join_day: int = 0  # day the instrument entered the market


@dataclass
class ManipulationWindow:
    scenario_id: str
    scenario_type: str
    participant_ids: list[str]
    day_start: int
    day_end: int
    intensity: str = "medium"
    concealment: str = "medium"

    def is_active(self, day: int) -> bool:
        return self.day_start <= day <= self.day_end


@dataclass
class DayResult:
    day: int
    trade_count: int
    order_count: int
    active_traders: int
    manipulative_traders: set[str] = field(default_factory=set)
    output_dir: Path = field(default_factory=Path)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


class MultiDayOrchestrator:
    """Orchestrate a multi-day synthetic market simulation.

    Parameters
    ----------
    n_days:
        Number of trading days to simulate.
    seed:
        Master random seed.
    output_dir:
        Root output directory; a subdirectory named ``run_id`` is created.
    run_id:
        Optional explicit run identifier; auto-generated UUID-based ID if omitted.
    new_trader_lambda:
        Poisson rate for new daily trader onboarding (default 2).
    retire_prob:
        Per-day per-trader retirement probability (default 0.005 = 0.5 %).
    base_trader_count:
        Number of traders at simulation start.
    base_broker_count:
        Number of brokers.
    base_instrument_count:
        Number of instruments from day 0.
    manipulation_windows:
        List of :class:`ManipulationWindow` objects to inject.
    new_instrument_days:
        Days on which a new instrument joins (default [10, 20]).
    session_minutes:
        Duration of each trading day in minutes (default 75).
    steps_per_minute:
        Simulation steps per minute (default 1).
    """

    def __init__(
        self,
        n_days: int = 30,
        seed: int = 42,
        output_dir: str | Path = "outputs/multi_day",
        run_id: str | None = None,
        *,
        new_trader_lambda: float = 2.0,
        retire_prob: float = 0.005,
        base_trader_count: int = 20,
        base_broker_count: int = 4,
        base_instrument_count: int = 2,
        manipulation_windows: list[ManipulationWindow] | None = None,
        new_instrument_days: list[int] | None = None,
        session_minutes: int = 75,
        steps_per_minute: int = 1,
    ) -> None:
        self.n_days = n_days
        self.seed = seed
        self.rng = Random(seed)
        self.run_id = run_id or f"multi_{uuid.uuid4().hex[:10]}"
        self.output_dir = Path(output_dir) / self.run_id
        self.new_trader_lambda = new_trader_lambda
        self.retire_prob = retire_prob
        self.base_trader_count = base_trader_count
        self.base_broker_count = base_broker_count
        self.base_instrument_count = base_instrument_count
        self.manipulation_windows: list[ManipulationWindow] = manipulation_windows or []
        self.new_instrument_days: set[int] = set(new_instrument_days if new_instrument_days is not None else [10, 20])
        self.session_minutes = session_minutes
        self.steps_per_minute = steps_per_minute

        # Entity universe (built lazily)
        self._traders: dict[str, TraderState] = {}
        self._instruments: list[InstrumentState] = []
        self._brokers: list[str] = []
        self._next_trader_idx: int = 0
        self._next_instrument_idx: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Execute the full multi-day simulation.

        Returns
        -------
        dict with keys: ``run_id``, ``n_days``, ``output_dir``, ``summary``.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("MultiDayOrchestrator starting run=%s (%d days)", self.run_id, self.n_days)

        self._initialise_entity_universe()

        ground_truth_rows: list[dict[str, Any]] = []
        entity_timeline_rows: list[dict[str, Any]] = []
        day_results: list[DayResult] = []

        for day in range(1, self.n_days + 1):
            logger.info("  Simulating day %d/%d …", day, self.n_days)

            # --- Check for new instruments
            if day in self.new_instrument_days:
                new_inst = self._create_instrument(day)
                self._instruments.append(new_inst)
                logger.info("    New instrument joined: %s", new_inst.symbol)

            # --- Trader lifecycle (retire + onboard)
            self._apply_retirements(day)
            self._onboard_new_traders(day)

            # --- Determine which manipulation windows are active today
            active_windows = [w for w in self.manipulation_windows if w.is_active(day)]
            manipulator_ids: set[str] = set()
            for w in active_windows:
                manipulator_ids.update(w.participant_ids)
            for tid, ts in self._traders.items():
                if tid in manipulator_ids:
                    ts.is_manipulator = True

            # --- Run single-day simulation
            active_traders = {tid: ts for tid, ts in self._traders.items() if ts.is_active}
            day_result = self._simulate_day(day, active_traders, active_windows)
            day_results.append(day_result)

            # --- Entity timeline snapshot
            for tid, ts in active_traders.items():
                days_since_onboard = day - ts.onboard_day
                if days_since_onboard < ts.ramp_up_days:
                    status = "onboarding"
                else:
                    status = "active"
                entity_timeline_rows.append({
                    "day": day,
                    "trader_id": tid,
                    "status": status,
                    "is_manipulator": ts.is_manipulator,
                    "broker_id": ts.broker_id,
                    "profile": ts.profile,
                })
            for tid, ts in self._traders.items():
                if ts.retire_day == day:
                    entity_timeline_rows.append({
                        "day": day,
                        "trader_id": tid,
                        "status": "retired",
                        "is_manipulator": ts.is_manipulator,
                        "broker_id": ts.broker_id,
                        "profile": ts.profile,
                    })

            # --- Ground truth rows for this day
            for w in self.manipulation_windows:
                for pid in w.participant_ids:
                    ground_truth_rows.append({
                        "day": day,
                        "trader_id": pid,
                        "scenario_id": w.scenario_id,
                        "manipulation_type": w.scenario_type,
                        "intensity": w.intensity,
                        "is_manipulative": w.is_active(day),
                    })

        # --- Save consolidated outputs
        self._save_ground_truth_ledger(ground_truth_rows)
        self._save_entity_timeline(entity_timeline_rows)
        run_manifest = self._save_run_manifest(day_results)

        logger.info("MultiDayOrchestrator complete.  Output: %s", self.output_dir)
        return run_manifest

    # ------------------------------------------------------------------
    # Entity universe management
    # ------------------------------------------------------------------

    def _initialise_entity_universe(self) -> None:
        """Create the starting set of traders, brokers, and instruments."""
        # Brokers
        self._brokers = [_broker_id(i) for i in range(1, self.base_broker_count + 1)]

        # Instruments
        self._instruments = []
        for i in range(1, self.base_instrument_count + 1):
            self._instruments.append(self._create_instrument(day=0, idx=i))

        # Traders
        for _ in range(self.base_trader_count):
            self._create_and_register_trader(onboard_day=0)

    def _create_instrument(self, day: int, idx: int | None = None) -> InstrumentState:
        self._next_instrument_idx += 1
        i = idx if idx is not None else self._next_instrument_idx
        mid = self.rng.uniform(90.0, 120.0)
        band_half = mid * 0.05
        return InstrumentState(
            instrument_id=f"instrument_{i:05d}",
            symbol=f"INST_{i}",
            asset_class="equity",
            tick_size=0.01,
            lot_size=1,
            price_band=(round(mid - band_half, 2), round(mid + band_half, 2)),
            join_day=day,
        )

    def _create_and_register_trader(self, onboard_day: int) -> TraderState:
        self._next_trader_idx += 1
        tid = _trader_id(self._next_trader_idx)
        broker = self.rng.choice(self._brokers)
        profile = self.rng.choices(
            list(_DEFAULT_TRADER_PROFILES.keys()),
            weights=list(_DEFAULT_TRADER_PROFILES.values()),
        )[0]
        ts = TraderState(
            trader_id=tid,
            broker_id=broker,
            profile=profile,
            onboard_day=onboard_day,
        )
        self._traders[tid] = ts
        return ts

    def _apply_retirements(self, day: int) -> None:
        """Randomly retire active traders with probability ``retire_prob``."""
        for tid, ts in self._traders.items():
            if ts.is_active and not ts.is_manipulator:
                if self.rng.random() < self.retire_prob:
                    ts.retire_day = day
                    logger.debug("    Trader %s retired on day %d", tid, day)

    def _onboard_new_traders(self, day: int) -> None:
        """Onboard a Poisson-distributed number of new traders."""
        import math

        # Poisson draw via sum-of-uniforms method for small lambda
        lam = self.new_trader_lambda
        L = math.exp(-lam)
        k, p = 0, 1.0
        while p > L:
            p *= self.rng.random()
            k += 1
        n_new = max(0, k - 1)
        for _ in range(n_new):
            ts = self._create_and_register_trader(onboard_day=day)
            logger.debug("    New trader onboarded: %s", ts.trader_id)

    # ------------------------------------------------------------------
    # Single-day simulation
    # ------------------------------------------------------------------

    def _simulate_day(
        self,
        day: int,
        active_traders: dict[str, TraderState],
        active_windows: list[ManipulationWindow],
    ) -> DayResult:
        """Simulate one trading day and write outputs to ``day_{N:03d}/``."""
        day_dir = self.output_dir / f"day_{day:03d}"
        day_dir.mkdir(parents=True, exist_ok=True)

        day_rng = Random(self.seed ^ (day * 997))
        manipulator_ids: set[str] = set()
        for w in active_windows:
            manipulator_ids.update(w.participant_ids)

        steps = self.session_minutes * self.steps_per_minute
        trades: list[dict[str, Any]] = []
        orders: list[dict[str, Any]] = []

        instruments = [inst for inst in self._instruments if inst.join_day <= day]

        for step in range(steps):
            for inst in instruments:
                ref_price = (inst.price_band[0] + inst.price_band[1]) / 2
                ref_price += day_rng.gauss(0, inst.tick_size * 0.5)

                for tid, ts in active_traders.items():
                    activity = ts.activity_scale(day)
                    is_manip = tid in manipulator_ids

                    # Background order generation
                    if day_rng.random() < 0.03 * activity:
                        side = "BUY" if day_rng.random() < 0.5 else "SELL"
                        order_type = "LIMIT" if day_rng.random() < 0.7 else "MARKET"
                        price = round(ref_price + day_rng.uniform(-3, 3) * inst.tick_size, 4)
                        qty = day_rng.randint(1, 20)

                        # If manipulator in layering window, add extra large orders
                        if is_manip:
                            qty = int(qty * day_rng.uniform(2, 5))
                            order_type = "LIMIT"

                        order: dict[str, Any] = {
                            "day": day,
                            "step": step,
                            "trader_id": tid,
                            "broker_id": ts.broker_id,
                            "instrument_id": inst.instrument_id,
                            "side": side,
                            "order_type": order_type,
                            "price": round(price, 4),
                            "quantity": qty,
                            "is_manipulative": is_manip,
                            "scenario_id": next(
                                (w.scenario_id for w in active_windows if tid in w.participant_ids), ""
                            ),
                        }
                        orders.append(order)

                        # Simplified trade generation (matching)
                        if day_rng.random() < 0.4:
                            counter_trader_id = day_rng.choice(
                                [t for t in active_traders if t != tid] or [tid]
                            )
                            counter_ts = active_traders[counter_trader_id]
                            trade: dict[str, Any] = {
                                "day": day,
                                "step": step,
                                "buy_trader_id": tid if side == "BUY" else counter_trader_id,
                                "sell_trader_id": counter_trader_id if side == "BUY" else tid,
                                "buy_broker_id": ts.broker_id if side == "BUY" else counter_ts.broker_id,
                                "sell_broker_id": counter_ts.broker_id if side == "BUY" else ts.broker_id,
                                "instrument_id": inst.instrument_id,
                                "price": round(ref_price + day_rng.gauss(0, inst.tick_size), 4),
                                "quantity": min(qty, day_rng.randint(1, qty)),
                                "is_manipulative": is_manip or counter_trader_id in manipulator_ids,
                            }
                            trades.append(trade)

        # Write day output files
        trades_df = pd.DataFrame(trades)
        orders_df = pd.DataFrame(orders)
        trades_df.to_csv(day_dir / "trades.csv", index=False)
        orders_df.to_csv(day_dir / "orders.csv", index=False)

        # Day manifest
        day_manifest = {
            "day": day,
            "active_traders": len(active_traders),
            "instruments": [i.symbol for i in instruments],
            "active_manipulation_windows": [w.scenario_id for w in active_windows],
            "trade_count": len(trades),
            "order_count": len(orders),
        }
        (day_dir / "day_manifest.json").write_text(json.dumps(day_manifest, indent=2), encoding="utf-8")

        return DayResult(
            day=day,
            trade_count=len(trades),
            order_count=len(orders),
            active_traders=len(active_traders),
            manipulative_traders=manipulator_ids,
            output_dir=day_dir,
        )

    # ------------------------------------------------------------------
    # Consolidated output writers
    # ------------------------------------------------------------------

    def _save_ground_truth_ledger(self, rows: list[dict[str, Any]]) -> None:
        """Write ``ground_truth_ledger.csv``."""
        path = self.output_dir / "ground_truth_ledger.csv"
        if not rows:
            pd.DataFrame(columns=["day", "trader_id", "scenario_id",
                                   "manipulation_type", "intensity", "is_manipulative"]).to_csv(path, index=False)
            return
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        logger.info("Saved ground_truth_ledger → %s (%d rows)", path, len(df))

    def _save_entity_timeline(self, rows: list[dict[str, Any]]) -> None:
        """Write ``entity_timeline.csv``."""
        path = self.output_dir / "entity_timeline.csv"
        if not rows:
            pd.DataFrame(columns=["day", "trader_id", "status", "is_manipulator",
                                   "broker_id", "profile"]).to_csv(path, index=False)
            return
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        logger.info("Saved entity_timeline → %s (%d rows)", path, len(df))

    def _save_run_manifest(self, day_results: list[DayResult]) -> dict[str, Any]:
        """Write ``run_manifest.json`` and return its contents."""
        manifest: dict[str, Any] = {
            "run_id": self.run_id,
            "n_days": self.n_days,
            "seed": self.seed,
            "output_dir": str(self.output_dir),
            "base_trader_count": self.base_trader_count,
            "base_broker_count": self.base_broker_count,
            "base_instrument_count": self.base_instrument_count,
            "new_trader_lambda": self.new_trader_lambda,
            "retire_prob": self.retire_prob,
            "new_instrument_days": sorted(self.new_instrument_days),
            "manipulation_windows": [
                {
                    "scenario_id": w.scenario_id,
                    "scenario_type": w.scenario_type,
                    "day_start": w.day_start,
                    "day_end": w.day_end,
                    "intensity": w.intensity,
                    "participant_count": len(w.participant_ids),
                }
                for w in self.manipulation_windows
            ],
            "summary": {
                "total_trades": sum(d.trade_count for d in day_results),
                "total_orders": sum(d.order_count for d in day_results),
                "days_with_manipulation": sum(
                    1 for d in day_results if d.manipulative_traders
                ),
                "final_trader_count": len(self._traders),
                "final_active_traders": sum(1 for ts in self._traders.values() if ts.is_active),
                "final_instrument_count": len(self._instruments),
            },
            "artifacts": {
                "ground_truth_ledger": str(self.output_dir / "ground_truth_ledger.csv"),
                "entity_timeline": str(self.output_dir / "entity_timeline.csv"),
            },
        }
        path = self.output_dir / "run_manifest.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("Saved run_manifest → %s", path)
        return manifest
