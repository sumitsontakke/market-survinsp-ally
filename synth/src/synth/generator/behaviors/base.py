from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from random import Random
from typing import Optional

from synth.generator.domain.entities import Instrument, Trader
from synth.generator.domain.value_objects import MarketSnapshot, OrderIntent


@dataclass(frozen=True)
class BehaviorContext:
    trader: Trader
    instrument: Instrument
    snapshot: MarketSnapshot
    rng: Random
    base_order_size: int
    max_order_size: int
    passive_order_probability: float
    aggressive_order_probability: float
    step_index: int = 0
    total_steps: int = 0


class BehaviorModel(ABC):
    label = "normal"
    scenario_id = "normal"
    scenario_type = "generic_background"
    is_manipulative = False

    @abstractmethod
    def generate_order_intent(self, context: BehaviorContext) -> Optional[OrderIntent]:
        raise NotImplementedError

    def update_state(self, feedback: dict) -> None:
        return None

    def get_label(self) -> str:
        return self.label
