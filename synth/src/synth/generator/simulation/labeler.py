from __future__ import annotations

from synth.generator.domain.entities import Order


def resolve_trade_label(buy_order: Order, sell_order: Order) -> tuple[str, str, str, bool]:
    if buy_order.is_manipulative:
        return (
            buy_order.scenario_id,
            buy_order.scenario_label,
            buy_order.scenario_type,
            buy_order.is_manipulative,
        )
    if sell_order.is_manipulative:
        return (
            sell_order.scenario_id,
            sell_order.scenario_label,
            sell_order.scenario_type,
            sell_order.is_manipulative,
        )
    return "normal", "normal", "generic_background", False
