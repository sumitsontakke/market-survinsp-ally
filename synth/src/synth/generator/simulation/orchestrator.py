from __future__ import annotations

from synth.generator.behaviors.selectors import BehaviorRegistry
from synth.generator.behaviors.base import BehaviorContext
from synth.generator.domain.entities import ScenarioEvent, as_record
from synth.generator.domain.value_objects import MarketSnapshot
from synth.generator.exporters.csv_exporter import CsvExporter
from synth.generator.exporters.manifest_exporter import ManifestExporter
from synth.generator.exporters.parquet_exporter import ParquetExporter
from synth.generator.market.intraday import IntradaySeasonalityModel
from synth.generator.market.price_process import PriceProcessModel
from synth.generator.market.regime import MarketRegimeModel
from synth.generator.market.session_clock import SessionClock
from synth.generator.registry.entity_registry import EntityRegistry
from synth.generator.registry.id_factory import IdFactory
from synth.generator.simulation.context import SimulationContext
from synth.generator.simulation.matching_engine import MatchingEngine
from synth.generator.simulation.order_emitter import OrderEmitter
from synth.generator.simulation.scenario_manager import ScenarioManager


class SimulationOrchestrator:
    def __init__(self, config: dict, rng) -> None:
        self.config = config
        self.rng = rng
        self.id_factory = IdFactory()
        self.registry = EntityRegistry.from_config(config, rng, self.id_factory)
        self.context = SimulationContext(
            config=config,
            rng=rng,
            id_factory=self.id_factory,
            registry=self.registry,
            behavior_registry=BehaviorRegistry(),
            session_clock=SessionClock(self.registry.session, steps=config["simulation"]["steps"]),
            seasonality_model=IntradaySeasonalityModel(),
            regime_model=MarketRegimeModel(),
            price_process_model=PriceProcessModel(),
        )
        self.order_emitter = OrderEmitter(self.id_factory)
        self.matching_engine = MatchingEngine()
        self.scenario_manager = ScenarioManager(config, self.registry, self.context.session_clock, rng)

    def run(self) -> dict[str, list[dict]]:
        prices = {
            instrument.instrument_id: sum(instrument.price_band) / 2
            for instrument in self.registry.instruments
        }
        orders = []
        trades = []
        scenarios = [
            ScenarioEvent(
                scenario_id="normal",
                scenario_type="generic_background",
                start_time=self.context.session_clock.start,
                end_time=self.context.session_clock.end,
                participant_ids=tuple(trader.trader_id for trader in self.registry.traders),
                instrument_id="ALL",
                intensity="baseline",
                is_manipulative=False,
            )
        ]
        scenarios.extend(self.scenario_manager.scenarios())
        timestamps = self.context.session_clock.iter_timestamps()
        trade_id = 0

        def next_trade_id() -> str:
            nonlocal trade_id
            trade_id += 1
            return f"trade_{trade_id:05d}"

        for index, timestamp in enumerate(timestamps):
            progress = index / max(len(timestamps) - 1, 1)
            intensity_multiplier = self.context.seasonality_model.participation_multiplier(progress)
            regime = self.context.regime_model.state_at(progress, intensity_multiplier)
            for instrument in self.registry.instruments:
                next_price = self.context.price_process_model.next_price(
                    current_price=prices[instrument.instrument_id],
                    drift=regime.drift,
                    volatility=regime.volatility,
                    rng=self.rng,
                    tick_size=instrument.tick_size,
                )
                low, high = instrument.price_band
                prices[instrument.instrument_id] = min(max(next_price, low), high)
                snapshot = MarketSnapshot(
                    timestamp=timestamp,
                    instrument_id=instrument.instrument_id,
                    reference_price=prices[instrument.instrument_id],
                    drift=regime.drift,
                    volatility=regime.volatility,
                    participation_intensity=regime.participation_intensity,
                )
                scenario_intents = self.scenario_manager.generate_intents(
                    instrument=instrument,
                    step_index=index,
                    snapshot=snapshot,
                    base_order_size=self.config["simulation"]["base_order_size"],
                    max_order_size=self.config["simulation"]["max_order_size"],
                    passive_order_probability=self.config["simulation"]["passive_order_probability"],
                    aggressive_order_probability=self.config["simulation"]["aggressive_order_probability"],
                )
                scenario_consumed_traders = set()
                for intent in scenario_intents:
                    scenario_consumed_traders.add(intent.trader_id)
                    order = self.order_emitter.materialize(intent)
                    final_order, generated_trades = self.matching_engine.process(order, next_trade_id)
                    orders.append(as_record(final_order))
                    trades.extend(as_record(trade) for trade in generated_trades)

                for trader in self.registry.traders:
                    if trader.trader_id in scenario_consumed_traders:
                        continue
                    behavior_context = BehaviorContext(
                        trader=trader,
                        instrument=instrument,
                        snapshot=snapshot,
                        rng=self.rng,
                        base_order_size=self.config["simulation"]["base_order_size"],
                        max_order_size=self.config["simulation"]["max_order_size"],
                        passive_order_probability=self.config["simulation"]["passive_order_probability"],
                        aggressive_order_probability=self.config["simulation"]["aggressive_order_probability"],
                        step_index=index,
                        total_steps=len(timestamps),
                    )
                    behavior = self.context.behavior_registry.build(trader.trader_profile_id)
                    intent = behavior.generate_order_intent(behavior_context)
                    if intent is None:
                        continue
                    order = self.order_emitter.materialize(intent)
                    final_order, generated_trades = self.matching_engine.process(order, next_trade_id)
                    orders.append(as_record(final_order))
                    trades.extend(as_record(trade) for trade in generated_trades)

        return {
            "brokers": [as_record(entity) for entity in self.registry.brokers],
            "beneficial_owners": [as_record(entity) for entity in self.registry.beneficial_owners],
            "accounts": [as_record(entity) for entity in self.registry.accounts],
            "traders": [as_record(entity) for entity in self.registry.traders],
            "instruments": [as_record(entity) for entity in self.registry.instruments],
            "sessions": [as_record(self.registry.session)],
            "orders": orders,
            "trades": trades,
            "scenarios": [as_record(scenario) for scenario in scenarios],
        }

    def export(self, output_dir: str, dataset: dict[str, list[dict]] = None) -> dict[str, str]:
        dataset = dataset or self.run()
        CsvExporter(output_dir).export(dataset)
        ParquetExporter(output_dir).export(dataset)
        manifest_path = ManifestExporter(output_dir).export(dataset, self.config)
        return {"output_dir": output_dir, "manifest": manifest_path}
