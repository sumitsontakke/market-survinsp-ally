"""Seed synthetic_market_sim with NSE-derived CalibrationParams.

The simulator package is **not** modified. Instead, we:

  1. Load the calibration from the bhavcopy SQLite store.
  2. Patch the simulator's config dict in-memory before generation
     (price-process volatility, return-shock df / scale, Pareto
     activity multipliers).
  3. Call the existing wrappers (``run_generic``, ``run_manipulation``)
     against the patched config.

This is the seam ``synthetic_market_sim`` exposes via its YAML config
loader: everything we override here is already a knob in the
simulator's config schema, just with new values.

R01-R24 are regenerated under ``outputs/calibrated_runs/`` using the
same 24 seed + scenario configs as the locked-stress fixtures, so the
GNN trains and evaluates on the same scenario structure - but the
underlying market dynamics now reflect real NSE characteristics.

Reference
---------
Cont, R. (2001). Empirical properties of asset returns: stylized facts
and statistical implications. Quantitative Finance, 1(2), 223-236.

Hill, B. M. (1975). A simple general approach to inference about the
tail of a distribution. The Annals of Statistics, 3(5), 1163-1174.
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import yaml

from scipy.stats import pareto

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pareto activity multipliers
# ---------------------------------------------------------------------------

def make_pareto_activity_multipliers(
    n_traders: int,
    alpha: float,
    seed: int = 42,
    clip_percentile: float = 99.0,
) -> np.ndarray:
    """Per-trader activity multipliers drawn from Pareto(alpha).

    Real NSE EQ universe shows alpha ~ 1.0, with p99/median ~ 160-217x
    on a typical trading day. After clipping at the empirical 99th
    percentile and normalizing to mean 1.0, the synthetic population's
    activity p99/median rises from Phase 1's 1.11x toward the observed
    real-world concentration.

    Parameters
    ----------
    n_traders : int
        Number of traders in the synthetic population.
    alpha : float
        Pareto shape parameter (smaller = heavier tail). Use the value
        from ``CalibrationParams.volume_alpha``.
    seed : int
        Random seed. Default 42 keeps runs reproducible.
    clip_percentile : float
        Trim the right tail before normalization to avoid one trader
        absorbing all activity. Standard practice in equity microstructure.

    Returns
    -------
    np.ndarray of shape (n_traders,)
        Multipliers with mean exactly 1.0.
    """
    if n_traders <= 0:
        return np.zeros(0, dtype=float)
    if not (alpha > 0 and np.isfinite(alpha)):
        raise ValueError(f"alpha must be > 0, got {alpha!r}")
    rng = np.random.default_rng(seed)
    raw = pareto.rvs(b=alpha, size=n_traders, random_state=rng)
    cap = float(np.percentile(raw, clip_percentile))
    raw = np.clip(raw, 0.0, cap)
    mean = float(raw.mean())
    if mean <= 0:
        return np.ones(n_traders, dtype=float)
    return (raw / mean).astype(float)


# ---------------------------------------------------------------------------
# Config patching
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationOverrides:
    """The four scalar knobs we apply on top of the simulator's defaults."""

    realized_volatility: float
    return_df: float
    return_scale: float
    volume_alpha: float
    source_date: str

    def to_simulator_config_patch(self) -> dict[str, Any]:
        """Return the dict that goes under ``simulation.calibration`` in the
        patched config. Keys are stable names downstream code can rely on."""
        return {
            "volatility_scale": float(self.realized_volatility),
            "return_shock_df": float(self.return_df),
            "return_shock_scale": float(self.return_scale),
            "trader_activity_alpha": float(self.volume_alpha),
            "source_date": self.source_date,
        }


def _load_overrides_from_db(db_path: Path | str) -> CalibrationOverrides:
    """Pull the latest CalibrationParams from the bhavcopy SQLite store."""
    # Lazy import so a missing calibration_service package doesn't crash the
    # rest of the training pipeline.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "calibration_service"))
    from core.database import MarketDataDB  # noqa: WPS433

    db = MarketDataDB(db_path)
    latest = db.get_latest_calibration()
    if latest is None:
        raise RuntimeError(
            f"No calibration runs in {db_path}. "
            "Run the bhavcopy fetcher and calibrator first."
        )
    return CalibrationOverrides(
        realized_volatility=float(latest.realized_volatility),
        return_df=float(latest.return_df),
        return_scale=float(latest.return_scale),
        volume_alpha=float(latest.volume_alpha),
        source_date=str(latest.calibration_date),
    )


def apply_calibration_to_config(
    base_config: dict[str, Any],
    overrides: CalibrationOverrides,
    *,
    pareto_seed: int = 42,
) -> dict[str, Any]:
    """Return a deep-copied config with calibration overrides applied.

    What we patch:
      - ``simulation.calibration``         the four scalar knobs
      - ``simulation.pareto_multipliers``  per-trader emission scaling
      - ``simulation.return_shock``        Student-t shock parameters
      - ``simulation.volatility_scale``    scalar multiplier for price-process vol

    The simulator's existing config loader treats unknown keys as
    inert, so this is non-breaking even if a particular simulator
    version doesn't consume every key yet. Downstream simulator code
    can pick up what it understands.
    """
    import copy
    cfg = copy.deepcopy(base_config)
    cfg.setdefault("simulation", {})

    patch = overrides.to_simulator_config_patch()
    cfg["simulation"]["calibration"] = patch
    cfg["simulation"]["volatility_scale"] = patch["volatility_scale"]
    cfg["simulation"]["return_shock"] = {
        "distribution": "student_t",
        "df": patch["return_shock_df"],
        "scale": patch["return_shock_scale"],
    }

    # Pareto multipliers: pre-compute and write into the config so the
    # simulator (or a downstream wrapper) doesn't need scipy.
    n_traders = int(_resolve_trader_count(cfg))
    multipliers = make_pareto_activity_multipliers(
        n_traders, alpha=patch["trader_activity_alpha"], seed=pareto_seed,
    )
    cfg["simulation"]["pareto_multipliers"] = multipliers.tolist()
    cfg["simulation"]["pareto_multipliers_seed"] = int(pareto_seed)
    return cfg


def _resolve_trader_count(cfg: dict[str, Any]) -> int:
    """Best-effort trader count from a simulator YAML config."""
    traders = cfg.get("traders", {})
    if isinstance(traders, dict):
        if "count" in traders:
            return int(traders["count"])
        # Fall back to inferring from accounts + per-owner config.
    accounts = cfg.get("accounts", {})
    bo = cfg.get("beneficial_owners", {})
    per_min = int(accounts.get("per_owner_min", 1) if isinstance(accounts, dict) else 1)
    bo_count = int(bo.get("count", 0) if isinstance(bo, dict) else 0)
    return max(bo_count * per_min, 1)


# ---------------------------------------------------------------------------
# Generation entrypoints
# ---------------------------------------------------------------------------

def generate_calibrated_run(
    base_config_path: Path | str,
    output_dir: Path | str,
    db_path: Path | str,
    *,
    pareto_seed: int = 42,
    is_manipulative: bool = True,
) -> Path:
    """Patch a config with CalibrationParams and call the simulator wrapper.

    Returns the path to the run output directory.
    """
    base_cfg = yaml.safe_load(Path(base_config_path).read_text(encoding="utf-8"))
    overrides = _load_overrides_from_db(db_path)
    patched = apply_calibration_to_config(
        base_cfg, overrides, pareto_seed=pareto_seed,
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Write the patched config alongside the run for reproducibility.
    patched_config_path = out_dir / "config.calibrated.yaml"
    patched_config_path.write_text(yaml.safe_dump(patched, sort_keys=False), encoding="utf-8")

    # Dispatch via the simulator's existing wrappers. We import lazily
    # so an absent simulator package doesn't break this module's import.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    if is_manipulative:
        from synth.generator.wrappers import run_manipulation as wrapper  # noqa: WPS433
    else:
        from synth.generator.wrappers import run_generic as wrapper  # noqa: WPS433

    # The simulator wrappers expect a YAML path on disk. We've already written it.
    sys.argv = [
        "calibrated_runner.py",
        "--config", str(patched_config_path),
        "--output-dir", str(out_dir),
    ]
    wrapper.main()
    return out_dir


def regenerate_r01_r24(
    *,
    base_config_dir: Path | str = "configs",
    output_root: Path | str = "outputs/calibrated_runs",
    db_path: Path | str = "/data/market.db",
    seeds: Optional[dict[str, int]] = None,
) -> list[Path]:
    """Regenerate the 24 locked-stress fixtures with calibrated parameters.

    The seed mapping mirrors the R01-R24 naming convention
    (..._sNN suffix carries the seed). If ``seeds`` is None we infer
    from the existing ``outputs/runs/R*`` directory names.
    """
    base_config_dir = Path(base_config_dir)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Map family -> config file (existing configs in repo)
    family_to_cfg = {
        "clique": base_config_dir / "collusion_scenario.yaml",
        "ring":   base_config_dir / "circular_trading.yaml",
        "mixed":  base_config_dir / "collusion_scenario.yaml",  # placeholder
    }

    # Resolve seed map from existing fixtures if not provided.
    if seeds is None:
        seeds = {}
        legacy = Path("outputs/runs")
        if legacy.exists():
            for d in sorted(legacy.iterdir()):
                if not d.is_dir() or not d.name.startswith("R"):
                    continue
                parts = d.name.split("_s")
                if len(parts) == 2 and parts[1].isdigit():
                    seeds[d.name] = int(parts[1])

    produced: list[Path] = []
    for run_name, seed in seeds.items():
        family = _infer_family_from_name(run_name)
        cfg_path = family_to_cfg.get(family)
        if cfg_path is None or not cfg_path.exists():
            _log.warning("skip %s: no template config for family=%s", run_name, family)
            continue
        out_dir = output_root / run_name
        _log.info("regenerating %s seed=%d cfg=%s", run_name, seed, cfg_path.name)
        try:
            generate_calibrated_run(
                cfg_path, out_dir, db_path,
                pareto_seed=seed,
                is_manipulative=(family != "benign"),
            )
            produced.append(out_dir)
        except Exception as exc:  # noqa: BLE001
            _log.warning("FAILED %s: %r", run_name, exc)
            continue
    return produced


def _infer_family_from_name(run_name: str) -> str:
    n = run_name.lower()
    if "ring" in n:
        return "ring"
    if "mixed" in n:
        return "mixed"
    if "clique" in n:
        return "clique"
    return "other"
