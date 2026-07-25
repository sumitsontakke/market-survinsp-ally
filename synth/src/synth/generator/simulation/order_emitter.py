from __future__ import annotations

from synth.generator.domain.entities import Order
from synth.generator.domain.value_objects import OrderIntent
from synth.generator.registry.id_factory import IdFactory


class OrderEmitter:
    def __init__(self, id_factory: IdFactory) -> None:
        self.id_factory = id_factory

    def materialize(self, intent: OrderIntent) -> Order:
        return Order(
            order_id=self.id_factory.next("order"),
            timestamp=intent.timestamp,
            trader_id=intent.trader_id,
            account_id=intent.account_id,
            broker_id=intent.broker_id,
            instrument_id=intent.instrument_id,
            side=intent.side.value,
            order_type=intent.order_type.value,
            price=float(intent.price),
            quantity=int(intent.quantity),
            time_in_force=intent.time_in_force.value,
            scenario_id=intent.scenario_id,
            scenario_label=intent.scenario_label,
            scenario_type=intent.scenario_type,
            is_manipulative=bool(intent.is_manipulative),
            parent_order_id=intent.parent_order_id,
            remaining_quantity=int(intent.quantity),
        )
