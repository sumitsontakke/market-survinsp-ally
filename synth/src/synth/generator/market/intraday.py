from __future__ import annotations


class IntradaySeasonalityModel:
    def participation_multiplier(self, progress: float) -> float:
        distance_from_mid = abs(progress - 0.5) * 2
        return 1.25 - 0.4 * (1 - distance_from_mid)
