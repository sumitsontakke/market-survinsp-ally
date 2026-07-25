from __future__ import annotations

from random import Random
from typing import Optional

from synth.generator.behaviors.base import BehaviorContext, BehaviorModel
from synth.generator.domain.enums import OrderType, Side, TimeInForce
from synth.generator.domain.value_objects import OrderIntent


def _bounded_quantity(rng: Random, base_order_size: int, max_order_size: int, scale: float = 1.0) -> int:
    quantity = int(base_order_size * scale + abs(rng.gauss(0, base_order_size / 3)))
    return max(1, min(quantity, max_order_size))


def _limit_price(reference_price: float, tick_size: float, side: Side, ticks_from_mid: int) -> float:
    sign = -1 if side == Side.BUY else 1
    return round(reference_price + sign * tick_size * ticks_from_mid, 10)


class RetailRandomBehavior(BehaviorModel):
    def generate_order_intent(self, context: BehaviorContext) -> Optional[OrderIntent]:
        activation = 0.18 * context.snapshot.participation_intensity
        if context.rng.random() > activation:
            return None
        side = context.rng.choice([Side.BUY, Side.SELL])
        aggressive = context.rng.random() < context.aggressive_order_probability
        order_type = OrderType.MARKET if aggressive else OrderType.LIMIT
        price = context.snapshot.reference_price
        if not aggressive:
            price = _limit_price(price, context.instrument.tick_size, side, ticks_from_mid=context.rng.randint(1, 4))
        return OrderIntent(
            timestamp=context.snapshot.timestamp,
            trader_id=context.trader.trader_id,
            account_id=context.trader.account_id,
            broker_id=context.trader.broker_id,
            instrument_id=context.instrument.instrument_id,
            side=side,
            order_type=order_type,
            price=price,
            quantity=_bounded_quantity(context.rng, context.base_order_size, context.max_order_size),
            time_in_force=TimeInForce.DAY if not aggressive else TimeInForce.IOC,
            scenario_id=self.scenario_id,
            scenario_label=self.get_label(),
            scenario_type=self.scenario_type,
            is_manipulative=self.is_manipulative,
        )


class MomentumBehavior(BehaviorModel):
    def generate_order_intent(self, context: BehaviorContext) -> Optional[OrderIntent]:
        activation = 0.15 * context.snapshot.participation_intensity
        if context.rng.random() > activation:
            return None
        side = Side.BUY if context.snapshot.drift >= 0 else Side.SELL
        aggressive = True
        return OrderIntent(
            timestamp=context.snapshot.timestamp,
            trader_id=context.trader.trader_id,
            account_id=context.trader.account_id,
            broker_id=context.trader.broker_id,
            instrument_id=context.instrument.instrument_id,
            side=side,
            order_type=OrderType.MARKET if aggressive else OrderType.LIMIT,
            price=context.snapshot.reference_price,
            quantity=_bounded_quantity(context.rng, context.base_order_size, context.max_order_size, scale=1.2),
            time_in_force=TimeInForce.IOC,
            scenario_id=self.scenario_id,
            scenario_label=self.get_label(),
            scenario_type=self.scenario_type,
            is_manipulative=self.is_manipulative,
        )


class MeanReversionBehavior(BehaviorModel):
    def generate_order_intent(self, context: BehaviorContext) -> Optional[OrderIntent]:
        activation = 0.12 * context.snapshot.participation_intensity
        if context.rng.random() > activation:
            return None
        side = Side.SELL if context.snapshot.drift >= 0 else Side.BUY
        price = _limit_price(context.snapshot.reference_price, context.instrument.tick_size, side, ticks_from_mid=1)
        return OrderIntent(
            timestamp=context.snapshot.timestamp,
            trader_id=context.trader.trader_id,
            account_id=context.trader.account_id,
            broker_id=context.trader.broker_id,
            instrument_id=context.instrument.instrument_id,
            side=side,
            order_type=OrderType.LIMIT,
            price=price,
            quantity=_bounded_quantity(context.rng, context.base_order_size, context.max_order_size, scale=0.8),
            time_in_force=TimeInForce.DAY,
            scenario_id=self.scenario_id,
            scenario_label=self.get_label(),
            scenario_type=self.scenario_type,
            is_manipulative=self.is_manipulative,
        )


class InstitutionalSlicerBehavior(BehaviorModel):
    def generate_order_intent(self, context: BehaviorContext) -> Optional[OrderIntent]:
        activation = 0.1 * context.snapshot.participation_intensity
        if context.rng.random() > activation:
            return None
        side = context.rng.choice([Side.BUY, Side.SELL])
        aggressive = context.rng.random() < 0.35
        order_type = OrderType.MARKET if aggressive else OrderType.LIMIT
        price = context.snapshot.reference_price
        if not aggressive:
            price = _limit_price(price, context.instrument.tick_size, side, ticks_from_mid=1)
        return OrderIntent(
            timestamp=context.snapshot.timestamp,
            trader_id=context.trader.trader_id,
            account_id=context.trader.account_id,
            broker_id=context.trader.broker_id,
            instrument_id=context.instrument.instrument_id,
            side=side,
            order_type=order_type,
            price=price,
            quantity=_bounded_quantity(context.rng, context.base_order_size, context.max_order_size, scale=1.6),
            time_in_force=TimeInForce.DAY if not aggressive else TimeInForce.IOC,
            scenario_id=self.scenario_id,
            scenario_label=self.get_label(),
            scenario_type=self.scenario_type,
            is_manipulative=self.is_manipulative,
        )


class LiquidityProviderBehavior(BehaviorModel):
    def generate_order_intent(self, context: BehaviorContext) -> Optional[OrderIntent]:
        activation = 0.2 * context.snapshot.participation_intensity
        if context.rng.random() > activation:
            return None
        side = context.rng.choice([Side.BUY, Side.SELL])
        price = _limit_price(context.snapshot.reference_price, context.instrument.tick_size, side, ticks_from_mid=context.rng.randint(1, 2))
        return OrderIntent(
            timestamp=context.snapshot.timestamp,
            trader_id=context.trader.trader_id,
            account_id=context.trader.account_id,
            broker_id=context.trader.broker_id,
            instrument_id=context.instrument.instrument_id,
            side=side,
            order_type=OrderType.LIMIT,
            price=price,
            quantity=_bounded_quantity(context.rng, context.base_order_size, context.max_order_size, scale=0.9),
            time_in_force=TimeInForce.DAY,
            scenario_id=self.scenario_id,
            scenario_label=self.get_label(),
            scenario_type=self.scenario_type,
            is_manipulative=self.is_manipulative,
        )
