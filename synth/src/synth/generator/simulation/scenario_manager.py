from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from synth.generator.behaviors.manipulative import (
    CircularTradingRingBehavior,
    CircularTradingRingPlan,
    CollusiveCliqueBehavior,
    CollusiveCliquePlan,
    ManipulativeBehaviorModel,
)
from synth.generator.domain.entities import Instrument, ScenarioEvent, Trader
from synth.generator.domain.value_objects import MarketSnapshot, OrderIntent
from synth.generator.market.session_clock import SessionClock
from synth.generator.registry.entity_registry import EntityRegistry


@dataclass(frozen=True)
class ConfiguredScenario:
    event: ScenarioEvent
    behavior: ManipulativeBehaviorModel


class ScenarioManager:
    def __init__(self, config: dict, registry: EntityRegistry, session_clock: SessionClock, rng) -> None:
        self.config = config
        self.registry = registry
        self.session_clock = session_clock
        self.rng = rng
        self._traders_by_id = {trader.trader_id: trader for trader in registry.traders}
        self._configured_scenarios = self._build_scenarios()

    def scenarios(self) -> list[ScenarioEvent]:
        return [configured.event for configured in self._configured_scenarios]

    def generate_intents(
        self,
        instrument: Instrument,
        step_index: int,
        snapshot: MarketSnapshot,
        base_order_size: int,
        max_order_size: int,
        passive_order_probability: float,
        aggressive_order_probability: float,
    ) -> list[OrderIntent]:
        intents: list[OrderIntent] = []
        for configured in self._configured_scenarios:
            if instrument.instrument_id != configured.event.instrument_id:
                continue
            intents.extend(
                configured.behavior.generate_step_intents(
                    step_index=step_index,
                    snapshot=snapshot,
                    instrument=instrument,
                    traders_by_id=self._traders_by_id,
                    base_order_size=base_order_size,
                    max_order_size=max_order_size,
                    passive_order_probability=passive_order_probability,
                    aggressive_order_probability=aggressive_order_probability,
                )
            )
        return intents

    def _build_scenarios(self) -> list[ConfiguredScenario]:
        configured: list[ConfiguredScenario] = []
        for index, scenario_cfg in enumerate(self.config.get("scenarios", []), start=1):
            scenario_type = scenario_cfg["scenario_type"]
            instrument = self._resolve_instrument(scenario_cfg)
            participant_count = int(scenario_cfg["participant_count"])
            traders = self.rng.sample(self.registry.traders, k=min(participant_count, len(self.registry.traders)))
            participant_ids = tuple(sorted(trader.trader_id for trader in traders))
            start_step = self._minute_to_step(int(scenario_cfg["start_minute"]))
            duration_steps = max(1, self._minute_to_step(int(scenario_cfg["duration_minutes"]), clamp_to_total=False))
            duration_steps = min(duration_steps, max(self.session_clock.steps - start_step, 1))
            if scenario_type == "collusive_clique":
                configured.append(
                    self._build_collusive_clique(
                        index=index,
                        scenario_cfg=scenario_cfg,
                        instrument=instrument,
                        participant_ids=participant_ids,
                        participant_count=participant_count,
                        start_step=start_step,
                        duration_steps=duration_steps,
                    )
                )
            elif scenario_type == "circular_trading_ring":
                configured.append(
                    self._build_circular_trading_ring(
                        index=index,
                        scenario_cfg=scenario_cfg,
                        instrument=instrument,
                        participant_ids=participant_ids,
                        participant_count=participant_count,
                        start_step=start_step,
                        duration_steps=duration_steps,
                    )
                )
            else:
                raise ValueError("Unsupported scenario_type: {0}".format(scenario_type))
        return configured

    def _build_collusive_clique(
        self,
        index: int,
        scenario_cfg: dict,
        instrument: Instrument,
        participant_ids: tuple[str, ...],
        participant_count: int,
        start_step: int,
        duration_steps: int,
    ) -> ConfiguredScenario:
        plan = CollusiveCliquePlan(
            scenario_id=scenario_cfg.get("scenario_id", "scenario_{0:03d}".format(index)),
            scenario_type="collusive_clique",
            instrument_id=instrument.instrument_id,
            participant_ids=participant_ids,
            participant_count=participant_count,
            start_step=start_step,
            duration_steps=duration_steps,
            intensity=scenario_cfg.get("intensity", "medium"),
            concealment=scenario_cfg.get("concealment", "low"),
            side_pattern=scenario_cfg.get("side_pattern", "synchronized_buy_sell"),
            seed=int(self.config["seed"]),
        )
        event = self._build_event(
            scenario_id=plan.scenario_id,
            scenario_type=plan.scenario_type,
            participant_ids=participant_ids,
            instrument_id=instrument.instrument_id,
            start_step=start_step,
            duration_steps=duration_steps,
            intensity=plan.intensity,
            concealment=plan.concealment,
            cycles=0,
            ring_order=(),
        )
        return ConfiguredScenario(event=event, behavior=CollusiveCliqueBehavior(plan))

    def _build_circular_trading_ring(
        self,
        index: int,
        scenario_cfg: dict,
        instrument: Instrument,
        participant_ids: tuple[str, ...],
        participant_count: int,
        start_step: int,
        duration_steps: int,
    ) -> ConfiguredScenario:
        ring_order = tuple(participant_ids)
        plan = CircularTradingRingPlan(
            scenario_id=scenario_cfg.get("scenario_id", "scenario_{0:03d}".format(index)),
            scenario_type="circular_trading_ring",
            instrument_id=instrument.instrument_id,
            participant_ids=participant_ids,
            participant_count=participant_count,
            start_step=start_step,
            duration_steps=duration_steps,
            intensity=scenario_cfg.get("intensity", "medium"),
            concealment=scenario_cfg.get("concealment", "low"),
            cycles=int(scenario_cfg.get("cycles", max(1, duration_steps // max(len(ring_order), 1)))),
            ring_order=ring_order,
            seed=int(self.config["seed"]),
        )
        event = self._build_event(
            scenario_id=plan.scenario_id,
            scenario_type=plan.scenario_type,
            participant_ids=participant_ids,
            instrument_id=instrument.instrument_id,
            start_step=start_step,
            duration_steps=duration_steps,
            intensity=plan.intensity,
            concealment=plan.concealment,
            cycles=plan.cycles,
            ring_order=ring_order,
        )
        return ConfiguredScenario(event=event, behavior=CircularTradingRingBehavior(plan))

    def _build_event(
        self,
        scenario_id: str,
        scenario_type: str,
        participant_ids: tuple[str, ...],
        instrument_id: str,
        start_step: int,
        duration_steps: int,
        intensity: str,
        concealment: str,
        cycles: int,
        ring_order: tuple[str, ...],
    ) -> ScenarioEvent:
        start_time = self.session_clock.start + self.session_clock.step_size * start_step
        end_time = self.session_clock.start + self.session_clock.step_size * min(start_step + duration_steps, self.session_clock.steps)
        return ScenarioEvent(
            scenario_id=scenario_id,
            scenario_type=scenario_type,
            start_time=start_time,
            end_time=end_time,
            participant_ids=participant_ids,
            instrument_id=instrument_id,
            intensity=intensity,
            is_manipulative=True,
            concealment=concealment,
            cycles=cycles,
            ring_order=ring_order,
        )

    def _resolve_instrument(self, scenario_cfg: dict) -> Instrument:
        instrument_id = scenario_cfg.get("instrument_id")
        instrument_symbol = scenario_cfg.get("instrument_symbol")
        for instrument in self.registry.instruments:
            if instrument_id and instrument.instrument_id == instrument_id:
                return instrument
            if instrument_symbol and instrument.symbol == instrument_symbol:
                return instrument
        if len(self.registry.instruments) == 1:
            return self.registry.instruments[0]
        raise ValueError("Scenario must reference a valid instrument_id or instrument_symbol.")

    def _minute_to_step(self, minute_offset: int, clamp_to_total: bool = True) -> int:
        total_minutes = max((self.session_clock.end - self.session_clock.start) / timedelta(minutes=1), 1)
        ratio = minute_offset / total_minutes
        step = int(round(ratio * self.session_clock.steps))
        if clamp_to_total:
            return max(0, min(step, max(self.session_clock.steps - 1, 0)))
        return max(1, step)
