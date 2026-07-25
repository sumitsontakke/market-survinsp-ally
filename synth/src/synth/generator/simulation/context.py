from __future__ import annotations

from dataclasses import dataclass
from random import Random

from synth.generator.behaviors.selectors import BehaviorRegistry
from synth.generator.market.intraday import IntradaySeasonalityModel
from synth.generator.market.price_process import PriceProcessModel
from synth.generator.market.regime import MarketRegimeModel
from synth.generator.market.session_clock import SessionClock
from synth.generator.registry.entity_registry import EntityRegistry
from synth.generator.registry.id_factory import IdFactory


@dataclass
class SimulationContext:
    config: dict
    rng: Random
    id_factory: IdFactory
    registry: EntityRegistry
    behavior_registry: BehaviorRegistry
    session_clock: SessionClock
    seasonality_model: IntradaySeasonalityModel
    regime_model: MarketRegimeModel
    price_process_model: PriceProcessModel
