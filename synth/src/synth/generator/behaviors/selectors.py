from __future__ import annotations

from synth.generator.behaviors.base import BehaviorModel
from synth.generator.behaviors.normal import (
    InstitutionalSlicerBehavior,
    LiquidityProviderBehavior,
    MeanReversionBehavior,
    MomentumBehavior,
    RetailRandomBehavior,
)


class BehaviorRegistry:
    def __init__(self) -> None:
        self._models: dict[str, type[BehaviorModel]] = {
            "retail_random": RetailRandomBehavior,
            "momentum": MomentumBehavior,
            "mean_reversion": MeanReversionBehavior,
            "institutional_slicer": InstitutionalSlicerBehavior,
            "liquidity_provider": LiquidityProviderBehavior,
        }

    def build(self, profile_id: str) -> BehaviorModel:
        model_cls = self._models.get(profile_id, RetailRandomBehavior)
        return model_cls()

    def register(self, profile_id: str, model_cls: type[BehaviorModel]) -> None:
        self._models[profile_id] = model_cls
