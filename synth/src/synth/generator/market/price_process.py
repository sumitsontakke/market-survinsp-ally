from __future__ import annotations

from random import Random


class PriceProcessModel:
    def next_price(self, current_price: float, drift: float, volatility: float, rng: Random, tick_size: float) -> float:
        raw_move = drift + rng.gauss(0.0, volatility)
        candidate = max(tick_size, current_price + raw_move)
        ticks = round(candidate / tick_size)
        return round(ticks * tick_size, 10)
