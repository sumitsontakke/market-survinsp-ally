from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegimeState:
    regime_type: str
    drift: float
    volatility: float
    participation_intensity: float


class MarketRegimeModel:
    def state_at(self, progress: float, intensity_multiplier: float) -> RegimeState:
        if progress < 0.1:
            return RegimeState("opening_burst", drift=0.02, volatility=0.18, participation_intensity=1.3 * intensity_multiplier)
        if progress > 0.9:
            return RegimeState("closing_ramp", drift=-0.01, volatility=0.14, participation_intensity=1.15 * intensity_multiplier)
        return RegimeState("calm", drift=0.0, volatility=0.08, participation_intensity=0.9 * intensity_multiplier)
