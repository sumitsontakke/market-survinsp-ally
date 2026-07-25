"""Calibration-parameter dataclasses.

The single public type is :class:`CalibrationParams` - the contract
between the calibrator and synthetic_market_sim. All four NSE-derived
parameters plus provenance metadata live here.

Reference
---------
Cont, R. (2001). Empirical properties of asset returns: stylized facts
and statistical implications. Quantitative Finance, 1(2), 223-236.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from synth.calibration.core.config import (
    EMPIRICAL_RETURN_DF_RANGE,
    EMPIRICAL_VOLUME_ALPHA_RANGE,
    SYNTHETIC_BASELINES,
)


@dataclass
class CalibrationParams:
    """Calibration parameters extracted from real NSE 1-minute OHLCV.

    These values feed into ``synthetic_market_sim`` before each synthetic
    session - see the integration block at the bottom of
    ``core/nse_calibrator.py`` for the exact mapping.

    Attributes
    ----------
    realized_volatility:
        Volume-weighted average of per-ticker annualized log-return
        standard deviation, computed on the calibration date's pooled
        1-minute returns.
    intraday_volume_profile:
        Length-375 list of normalized weights (sum to 1.0) describing
        the U-shape of NSE intraday volume.
    return_df, return_loc, return_scale:
        Student-t fit parameters on pooled log-returns. Empirical
        reference range for ``return_df`` is ~3-5 (Cont, 2001).
    volume_alpha:
        Hill tail-index estimator on per-minute volume distributions,
        median across tickers. Reference range ~1.5-2.5 for liquid
        equities.
    calibration_date:
        ISO date (YYYY-MM-DD) the calibration was computed for.
    tickers_used:
        Tickers that contributed data on this date (subset of
        NIFTY_LIQUID_20 if any failed to fetch).
    n_observations:
        Total number of 1-minute bars across tickers and date(s) used.
    warnings:
        Free-text findings about the calibration: tail-index outside
        empirical band, sparse data, etc. Empty list = clean run.
    """

    realized_volatility: float
    intraday_volume_profile: list[float]  # length 375
    return_df: float
    return_loc: float
    return_scale: float
    volume_alpha: float
    calibration_date: str  # YYYY-MM-DD
    tickers_used: list[str]
    n_observations: int
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # (de)serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form suitable for JSON / SQLite storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CalibrationParams":
        """Inverse of :meth:`to_dict`. Tolerates JSON-encoded list fields."""
        payload = dict(d)
        for key in ("intraday_volume_profile", "tickers_used", "warnings"):
            value = payload.get(key)
            if isinstance(value, str):
                payload[key] = json.loads(value)
        # Defensive: ensure list types and float casts on numerics.
        payload["intraday_volume_profile"] = [
            float(x) for x in payload.get("intraday_volume_profile", [])
        ]
        payload["tickers_used"] = list(payload.get("tickers_used", []))
        payload["warnings"] = list(payload.get("warnings", []))
        payload["realized_volatility"] = float(payload["realized_volatility"])
        payload["return_df"] = float(payload["return_df"])
        payload["return_loc"] = float(payload["return_loc"])
        payload["return_scale"] = float(payload["return_scale"])
        payload["volume_alpha"] = float(payload["volume_alpha"])
        payload["n_observations"] = int(payload["n_observations"])
        return cls(**payload)

    # ------------------------------------------------------------------
    # Human-readable report
    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Multi-line report comparing NSE values to Phase 1 baselines.

        The report is what the calibrator CLI prints after every run.
        Each line carries context the reviewer needs - the bare number is
        not useful without its empirical band.
        """
        empirical_df_lo, empirical_df_hi = EMPIRICAL_RETURN_DF_RANGE
        empirical_alpha_lo, empirical_alpha_hi = EMPIRICAL_VOLUME_ALPHA_RANGE

        df_in_band = empirical_df_lo <= self.return_df <= empirical_df_hi
        alpha_in_band = empirical_alpha_lo <= self.volume_alpha <= empirical_alpha_hi

        synth_df = SYNTHETIC_BASELINES["return_df"]
        synth_alpha = SYNTHETIC_BASELINES["volume_alpha"]
        synth_vol = SYNTHETIC_BASELINES["realized_volatility"]

        lines: list[str] = []
        lines.append("=" * 72)
        lines.append("NSE CALIBRATION  ·  {date}".format(date=self.calibration_date))
        lines.append("=" * 72)
        lines.append("Tickers used      : {n} / 20  ({tickers})".format(
            n=len(self.tickers_used),
            tickers=", ".join(self.tickers_used) if len(self.tickers_used) <= 6
            else ", ".join(self.tickers_used[:6]) + ", ...",
        ))
        lines.append("Observations      : {n:,}".format(n=self.n_observations))
        lines.append("")
        lines.append("PARAMETER              NSE VALUE        SYNTH BASELINE        STATUS")
        lines.append("-" * 72)
        lines.append("realized_vol          {nse:>10.4f}     {syn}".format(
            nse=self.realized_volatility,
            syn=str(synth_vol),
        ))
        lines.append("return_df             {nse:>10.4f}     {syn:<22}{stat}".format(
            nse=self.return_df,
            syn="{0:.2f}".format(synth_df),
            stat="OK in [{0}-{1}]".format(empirical_df_lo, empirical_df_hi)
            if df_in_band
            else "GAP outside [{0}-{1}]".format(empirical_df_lo, empirical_df_hi),
        ))
        lines.append("return_loc            {nse:>10.6f}".format(nse=self.return_loc))
        lines.append("return_scale          {nse:>10.6f}".format(nse=self.return_scale))
        lines.append("volume_alpha          {nse:>10.4f}     {syn:<22}{stat}".format(
            nse=self.volume_alpha,
            syn=str(synth_alpha)[:22],
            stat="OK in [{0}-{1}]".format(empirical_alpha_lo, empirical_alpha_hi)
            if alpha_in_band
            else "GAP outside [{0}-{1}]".format(empirical_alpha_lo, empirical_alpha_hi),
        ))
        lines.append("intraday_profile      length={n}, sum={s:.4f}, peak slot={p}".format(
            n=len(self.intraday_volume_profile),
            s=sum(self.intraday_volume_profile) if self.intraday_volume_profile else 0.0,
            p=max(
                range(len(self.intraday_volume_profile)),
                key=lambda i: self.intraday_volume_profile[i],
            ) if self.intraday_volume_profile else -1,
        ))
        lines.append("")
        if self.warnings:
            lines.append("WARNINGS  ({n}):".format(n=len(self.warnings)))
            for w in self.warnings:
                lines.append("  - {0}".format(w))
        else:
            lines.append("WARNINGS  : none")
        lines.append("")
        lines.append("Reference: Cont, R. (2001). Empirical properties of asset returns.")
        lines.append("=" * 72)
        return "\n".join(lines)
