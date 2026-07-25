from __future__ import annotations

import hashlib
from dataclasses import dataclass
from random import Random
from typing import Optional

from synth.generator.behaviors.base import BehaviorContext, BehaviorModel
from synth.generator.domain.entities import Instrument, Trader
from synth.generator.domain.enums import OrderType, Side, TimeInForce
from synth.generator.domain.value_objects import MarketSnapshot, OrderIntent


@dataclass(frozen=True)
class ManipulativeScenarioPlan:
    scenario_id: str
    scenario_type: str
    instrument_id: str
    participant_ids: tuple[str, ...]
    participant_count: int
    start_step: int
    duration_steps: int
    intensity: str
    concealment: str
    seed: int

    @property
    def end_step(self) -> int:
        return self.start_step + self.duration_steps


@dataclass(frozen=True)
class CollusiveCliquePlan(ManipulativeScenarioPlan):
    side_pattern: str


@dataclass(frozen=True)
class CircularTradingRingPlan(ManipulativeScenarioPlan):
    cycles: int
    ring_order: tuple[str, ...]


@dataclass(frozen=True)
class LayeringPlan(ManipulativeScenarioPlan):
    """Plan for a layering / spoofing manipulation scenario.

    The manipulator places a staircase of large LIMIT orders on one side
    of the book to create artificial price pressure, then cancels them by
    issuing aggressive orders in the *opposite* direction before they fill.

    Attributes
    ----------
    layer_count:
        Number of price levels (layers) stacked on the book side.
    target_side:
        The *deceptive* side ("BUY" creates apparent buying pressure,
        then the aggressive orders are SELL).
    cancel_window_steps:
        How many steps after layer placement before the cancel/aggress phase.
    """

    layer_count: int
    target_side: str  # "BUY" or "SELL"
    cancel_window_steps: int


class ManipulativeBehaviorModel(BehaviorModel):
    label = "manipulative"
    is_manipulative = True

    def generate_step_intents(
        self,
        step_index: int,
        snapshot: MarketSnapshot,
        instrument: Instrument,
        traders_by_id: dict[str, Trader],
        base_order_size: int,
        max_order_size: int,
        passive_order_probability: float,
        aggressive_order_probability: float,
    ) -> list[OrderIntent]:
        raise NotImplementedError

    def _local_rng(self, seed: int, scenario_id: str, step_index: int, token: str) -> Random:
        payload = f"{seed}:{scenario_id}:{step_index}:{token}".encode("utf-8")
        derived_seed = int(hashlib.sha256(payload).hexdigest()[:16], 16)
        return Random(derived_seed)

    def _build_intent(
        self,
        trader: Trader,
        instrument_id: str,
        timestamp,
        side: Side,
        order_type: OrderType,
        price: float,
        quantity: int,
        scenario_id: str,
        scenario_type: str,
    ) -> OrderIntent:
        return OrderIntent(
            timestamp=timestamp,
            trader_id=trader.trader_id,
            account_id=trader.account_id,
            broker_id=trader.broker_id,
            instrument_id=instrument_id,
            side=side,
            order_type=order_type,
            price=price,
            quantity=quantity,
            time_in_force=TimeInForce.IOC if order_type == OrderType.MARKET else TimeInForce.DAY,
            scenario_id=scenario_id,
            scenario_label=self.get_label(),
            scenario_type=scenario_type,
            is_manipulative=True,
        )


class CollusiveCliqueBehavior(ManipulativeBehaviorModel):
    def __init__(self, plan: CollusiveCliquePlan) -> None:
        self.plan = plan
        self.scenario_id = plan.scenario_id
        self.scenario_type = plan.scenario_type

    def generate_order_intent(self, context: BehaviorContext) -> Optional[OrderIntent]:
        if context.instrument.instrument_id != self.plan.instrument_id:
            return None
        if context.trader.trader_id not in self.plan.participant_ids:
            return None
        if context.step_index < self.plan.start_step or context.step_index >= self.plan.end_step:
            return None
        relative_step = context.step_index - self.plan.start_step
        if not self._is_active_step(relative_step):
            return None

        local_rng = self._local_rng(self.plan.seed, self.scenario_id, context.step_index, context.trader.trader_id)
        if local_rng.random() > self._participation_probability():
            return None
        side = self._side_for_step(relative_step)
        order_type = self._order_type_for_concealment(local_rng)
        quantity = self._quantity_for_step(relative_step, context.base_order_size, context.max_order_size, local_rng)
        price = self._price_for_order(context.snapshot.reference_price, context.instrument.tick_size, side, order_type, local_rng)
        return self._build_intent(
            trader=context.trader,
            instrument_id=context.instrument.instrument_id,
            timestamp=context.snapshot.timestamp,
            side=side,
            order_type=order_type,
            price=price,
            quantity=quantity,
            scenario_id=self.scenario_id,
            scenario_type=self.scenario_type,
        )

    def generate_step_intents(
        self,
        step_index: int,
        snapshot: MarketSnapshot,
        instrument: Instrument,
        traders_by_id: dict[str, Trader],
        base_order_size: int,
        max_order_size: int,
        passive_order_probability: float,
        aggressive_order_probability: float,
    ) -> list[OrderIntent]:
        intents: list[OrderIntent] = []
        for trader_id in self.plan.participant_ids:
            trader = traders_by_id[trader_id]
            context = BehaviorContext(
                trader=trader,
                instrument=instrument,
                snapshot=snapshot,
                rng=Random(self.plan.seed),
                base_order_size=base_order_size,
                max_order_size=max_order_size,
                passive_order_probability=passive_order_probability,
                aggressive_order_probability=aggressive_order_probability,
                step_index=step_index,
                total_steps=0,
            )
            intent = self.generate_order_intent(context)
            if intent is not None:
                intents.append(intent)
        return intents

    def _is_active_step(self, relative_step: int) -> bool:
        stride_by_intensity = {"low": 3, "medium": 2, "high": 1}
        return relative_step % stride_by_intensity.get(self.plan.intensity, 1) == 0

    def _participation_probability(self) -> float:
        return {"low": 1.0, "medium": 0.9, "high": 0.78}.get(self.plan.concealment, 1.0)

    def _side_for_step(self, relative_step: int) -> Side:
        midpoint = max(self.plan.duration_steps // 2, 1)
        if self.plan.side_pattern == "synchronized_buy_sell":
            return Side.BUY if relative_step < midpoint else Side.SELL
        return Side.BUY

    def _order_type_for_concealment(self, rng: Random) -> OrderType:
        if self.plan.concealment == "low":
            return OrderType.MARKET
        if self.plan.concealment == "medium":
            return OrderType.MARKET if rng.random() < 0.8 else OrderType.LIMIT
        return OrderType.MARKET if rng.random() < 0.65 else OrderType.LIMIT

    def _quantity_for_step(self, relative_step: int, base_order_size: int, max_order_size: int, rng: Random) -> int:
        intensity_scale = {"low": 1.0, "medium": 1.35, "high": 1.8}.get(self.plan.intensity, 1.0)
        wave = 1.0 + (relative_step % 4) * 0.12
        anchor = base_order_size * intensity_scale * wave
        concealment_noise = {"low": 0.04, "medium": 0.12, "high": 0.22}.get(self.plan.concealment, 0.04)
        quantity = int(anchor * (1 + rng.uniform(-concealment_noise, concealment_noise)))
        return max(1, min(quantity, max_order_size))

    def _price_for_order(
        self,
        reference_price: float,
        tick_size: float,
        side: Side,
        order_type: OrderType,
        rng: Random,
    ) -> float:
        if order_type == OrderType.MARKET:
            return reference_price
        concealment_ticks = {"low": 0, "medium": 1, "high": 2}.get(self.plan.concealment, 0)
        jitter_ticks = rng.randint(0, concealment_ticks)
        sign = 1 if side == Side.BUY else -1
        return round(reference_price + sign * tick_size * jitter_ticks, 10)


class CircularTradingRingBehavior(ManipulativeBehaviorModel):
    def __init__(self, plan: CircularTradingRingPlan) -> None:
        self.plan = plan
        self.scenario_id = plan.scenario_id
        self.scenario_type = plan.scenario_type
        self._event_schedule = self._build_schedule()

    def generate_order_intent(self, context: BehaviorContext) -> Optional[OrderIntent]:
        step_intents = self.generate_step_intents(
            step_index=context.step_index,
            snapshot=context.snapshot,
            instrument=context.instrument,
            traders_by_id={context.trader.trader_id: context.trader},
            base_order_size=context.base_order_size,
            max_order_size=context.max_order_size,
            passive_order_probability=context.passive_order_probability,
            aggressive_order_probability=context.aggressive_order_probability,
        )
        for intent in step_intents:
            if intent.trader_id == context.trader.trader_id:
                return intent
        return None

    def generate_step_intents(
        self,
        step_index: int,
        snapshot: MarketSnapshot,
        instrument: Instrument,
        traders_by_id: dict[str, Trader],
        base_order_size: int,
        max_order_size: int,
        passive_order_probability: float,
        aggressive_order_probability: float,
    ) -> list[OrderIntent]:
        scheduled_events = self._event_schedule.get(step_index, [])
        if not scheduled_events:
            return []
        intents: list[OrderIntent] = []
        for event_index, seller_id, buyer_id in scheduled_events:
            seller = traders_by_id.get(seller_id)
            buyer = traders_by_id.get(buyer_id)
            if seller is None or buyer is None:
                continue
            local_rng = self._local_rng(self.plan.seed, self.scenario_id, step_index, f"{seller_id}:{buyer_id}:{event_index}")
            if local_rng.random() > self._participation_probability():
                continue
            quantity = self._quantity_for_event(event_index, base_order_size, max_order_size, local_rng)
            seller_price = self._seller_limit_price(snapshot.reference_price, instrument, local_rng)
            buyer_order_type = OrderType.MARKET if self.plan.concealment != "high" else OrderType.LIMIT
            buyer_price = snapshot.reference_price + instrument.tick_size * 4 if buyer_order_type == OrderType.LIMIT else snapshot.reference_price
            intents.append(
                self._build_intent(
                    trader=seller,
                    instrument_id=instrument.instrument_id,
                    timestamp=snapshot.timestamp,
                    side=Side.SELL,
                    order_type=OrderType.LIMIT,
                    price=seller_price,
                    quantity=quantity,
                    scenario_id=self.scenario_id,
                    scenario_type=self.scenario_type,
                )
            )
            intents.append(
                self._build_intent(
                    trader=buyer,
                    instrument_id=instrument.instrument_id,
                    timestamp=snapshot.timestamp,
                    side=Side.BUY,
                    order_type=buyer_order_type,
                    price=buyer_price,
                    quantity=quantity,
                    scenario_id=self.scenario_id,
                    scenario_type=self.scenario_type,
                )
            )
        return intents

    def _build_schedule(self) -> dict[int, list[tuple[int, str, str]]]:
        schedule: dict[int, list[tuple[int, str, str]]] = {}
        ring_size = max(len(self.plan.ring_order), 1)
        total_events = max(1, self.plan.cycles * ring_size)
        divisor = max(total_events - 1, 1)
        for event_index in range(total_events):
            base_relative_step = int(round(event_index * max(self.plan.duration_steps - 1, 0) / divisor))
            jitter = self._schedule_jitter(event_index)
            relative_step = max(0, min(base_relative_step + jitter, max(self.plan.duration_steps - 1, 0)))
            step_index = self.plan.start_step + relative_step
            seller_id = self.plan.ring_order[event_index % ring_size]
            buyer_id = self.plan.ring_order[(event_index + 1) % ring_size]
            schedule.setdefault(step_index, []).append((event_index, seller_id, buyer_id))
        return schedule

    def _schedule_jitter(self, event_index: int) -> int:
        if self.plan.concealment == "low":
            return 0
        rng = self._local_rng(self.plan.seed, self.scenario_id, self.plan.start_step + event_index, "schedule")
        if self.plan.concealment == "medium":
            return rng.choice([-1, 0, 0, 1])
        return rng.choice([-2, -1, 0, 1, 2])

    def _participation_probability(self) -> float:
        return {"low": 1.0, "medium": 0.92, "high": 0.82}.get(self.plan.concealment, 1.0)

    def _quantity_for_event(self, event_index: int, base_order_size: int, max_order_size: int, rng: Random) -> int:
        intensity_scale = {"low": 1.0, "medium": 1.25, "high": 1.6}.get(self.plan.intensity, 1.0)
        cycle_wave = 1.0 + (event_index % max(len(self.plan.ring_order), 1)) * 0.08
        anchor = base_order_size * intensity_scale * cycle_wave
        concealment_noise = {"low": 0.02, "medium": 0.08, "high": 0.16}.get(self.plan.concealment, 0.02)
        quantity = int(anchor * (1 + rng.uniform(-concealment_noise, concealment_noise)))
        return max(1, min(quantity, max_order_size))

    def _seller_limit_price(self, reference_price: float, instrument: Instrument, rng: Random) -> float:
        ask_offset_ticks = {"low": 1, "medium": 2, "high": 3}.get(self.plan.concealment, 1)
        jitter_ticks = 0 if self.plan.concealment == "low" else rng.randint(0, ask_offset_ticks)
        candidate = reference_price + instrument.tick_size * (ask_offset_ticks + jitter_ticks)
        low, high = instrument.price_band
        return round(min(max(candidate, low), high), 10)


class LayeringBehavior(ManipulativeBehaviorModel):
    """Layering / spoofing manipulation behaviour.

    The manipulator (a single dominant trader or a small coordinated group)
    builds a staircase of large visible LIMIT orders on the *deceptive side*
    of the order book across ``layer_count`` price levels to create apparent
    directional pressure.  After ``cancel_window_steps`` steps the deceptive
    orders are abandoned (simulated by placing aggressively-priced LIMIT orders
    in the *opposite* direction that will match quickly) and replaced with
    smaller aggressive MARKET / IOC orders that profit from the price movement
    induced by the spoofing.

    Empirical signature:
    - High order-to-trade ratio for the primary participant(s).
    - Large imbalance between placed vs. filled volume on one side.
    - Clustered cancellation events shortly before aggressive fills.
    """

    label = "layering"

    def __init__(self, plan: LayeringPlan) -> None:
        self.plan = plan
        self.scenario_id = plan.scenario_id
        self.scenario_type = plan.scenario_type
        self._deceptive_side: Side = Side.BUY if plan.target_side.upper() == "BUY" else Side.SELL
        self._aggressive_side: Side = Side.SELL if self._deceptive_side == Side.BUY else Side.BUY
        self._phase_boundary: int = plan.start_step + max(plan.cancel_window_steps, 1)

    # ------------------------------------------------------------------
    # BehaviorModel interface
    # ------------------------------------------------------------------

    def generate_order_intent(self, context: BehaviorContext) -> Optional[OrderIntent]:
        intents = self.generate_step_intents(
            step_index=context.step_index,
            snapshot=context.snapshot,
            instrument=context.instrument,
            traders_by_id={context.trader.trader_id: context.trader},
            base_order_size=context.base_order_size,
            max_order_size=context.max_order_size,
            passive_order_probability=context.passive_order_probability,
            aggressive_order_probability=context.aggressive_order_probability,
        )
        for intent in intents:
            if intent.trader_id == context.trader.trader_id:
                return intent
        return None

    def generate_step_intents(
        self,
        step_index: int,
        snapshot: MarketSnapshot,
        instrument: Instrument,
        traders_by_id: dict[str, Trader],
        base_order_size: int,
        max_order_size: int,
        passive_order_probability: float,
        aggressive_order_probability: float,
    ) -> list[OrderIntent]:
        if step_index < self.plan.start_step or step_index >= self.plan.end_step:
            return []
        if context_trader := self._primary_trader(traders_by_id):
            pass
        else:
            return []

        local_rng = self._local_rng(self.plan.seed, self.scenario_id, step_index, "layering_main")
        if local_rng.random() > self._participation_probability():
            return []

        intents: list[OrderIntent] = []
        in_layering_phase = step_index < self._phase_boundary

        if in_layering_phase:
            # Place staircase of large LIMIT orders on the deceptive side
            intents.extend(
                self._build_layering_orders(
                    context_trader, snapshot, instrument, base_order_size, max_order_size, step_index, local_rng
                )
            )
        else:
            # Aggress in the opposite direction (profit phase)
            intents.extend(
                self._build_aggress_orders(
                    context_trader, snapshot, instrument, base_order_size, max_order_size, step_index, local_rng
                )
            )
        return intents

    # ------------------------------------------------------------------
    # Layering phase helpers
    # ------------------------------------------------------------------

    def _build_layering_orders(
        self,
        trader: Trader,
        snapshot: MarketSnapshot,
        instrument: Instrument,
        base_order_size: int,
        max_order_size: int,
        step_index: int,
        rng: Random,
    ) -> list[OrderIntent]:
        """Build one LIMIT order per price layer on the deceptive side."""
        intents: list[OrderIntent] = []
        layer_count = max(self.plan.layer_count, 1)
        intensity_multiplier = {"low": 2.0, "medium": 3.5, "high": 5.0}.get(self.plan.intensity, 3.0)
        tick = instrument.tick_size
        low_band, high_band = instrument.price_band
        ref = snapshot.reference_price

        for layer_idx in range(layer_count):
            # Each layer is further from mid by an extra tick — creates visible wall
            offset_ticks = layer_idx + 1
            if self._deceptive_side == Side.BUY:
                # Buy-side layers below mid (less likely to fill)
                price = round(ref - tick * offset_ticks, 10)
                price = max(price, low_band)
            else:
                price = round(ref + tick * offset_ticks, 10)
                price = min(price, high_band)

            size_noise = rng.uniform(0.8, 1.2)
            quantity = int(base_order_size * intensity_multiplier * size_noise * (1 + layer_idx * 0.15))
            quantity = max(1, min(quantity, max_order_size * 3))  # layers are oversized

            intents.append(
                self._build_intent(
                    trader=trader,
                    instrument_id=instrument.instrument_id,
                    timestamp=snapshot.timestamp,
                    side=self._deceptive_side,
                    order_type=OrderType.LIMIT,
                    price=price,
                    quantity=quantity,
                    scenario_id=self.scenario_id,
                    scenario_type=self.scenario_type,
                )
            )
        return intents

    def _build_aggress_orders(
        self,
        trader: Trader,
        snapshot: MarketSnapshot,
        instrument: Instrument,
        base_order_size: int,
        max_order_size: int,
        step_index: int,
        rng: Random,
    ) -> list[OrderIntent]:
        """Place aggressive MARKET orders on the side opposite to the layers."""
        intensity_multiplier = {"low": 1.0, "medium": 1.3, "high": 1.7}.get(self.plan.intensity, 1.2)
        quantity = int(base_order_size * intensity_multiplier * rng.uniform(0.9, 1.1))
        quantity = max(1, min(quantity, max_order_size))
        return [
            self._build_intent(
                trader=trader,
                instrument_id=instrument.instrument_id,
                timestamp=snapshot.timestamp,
                side=self._aggressive_side,
                order_type=OrderType.MARKET,
                price=snapshot.reference_price,
                quantity=quantity,
                scenario_id=self.scenario_id,
                scenario_type=self.scenario_type,
            )
        ]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _primary_trader(self, traders_by_id: dict[str, Trader]) -> Optional[Trader]:
        """Return the first participant that exists in traders_by_id."""
        for pid in self.plan.participant_ids:
            if pid in traders_by_id:
                return traders_by_id[pid]
        return None

    def _participation_probability(self) -> float:
        return {"low": 1.0, "medium": 0.90, "high": 0.80}.get(self.plan.concealment, 1.0)
