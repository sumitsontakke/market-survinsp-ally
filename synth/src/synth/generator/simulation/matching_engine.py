from __future__ import annotations

from dataclasses import replace

from synth.generator.domain.entities import Order, Trade
from synth.generator.simulation.labeler import resolve_trade_label


class MatchingEngine:
    def __init__(self) -> None:
        self.buy_books: dict[str, list[Order]] = {}
        self.sell_books: dict[str, list[Order]] = {}

    def process(self, order: Order, next_trade_id) -> tuple[Order, list[Trade]]:
        instrument_id = order.instrument_id
        trades: list[Trade] = []
        if order.side == "buy":
            order, trades = self._match_against_book(order, self.sell_books.setdefault(instrument_id, []), next_trade_id, aggressive_is_buy=True)
            if order.remaining_quantity > 0 and order.order_type == "limit":
                self.buy_books.setdefault(instrument_id, []).append(order)
                self.buy_books[instrument_id].sort(key=lambda entry: (-entry.price, entry.timestamp, entry.order_id))
        else:
            order, trades = self._match_against_book(order, self.buy_books.setdefault(instrument_id, []), next_trade_id, aggressive_is_buy=False)
            if order.remaining_quantity > 0 and order.order_type == "limit":
                self.sell_books.setdefault(instrument_id, []).append(order)
                self.sell_books[instrument_id].sort(key=lambda entry: (entry.price, entry.timestamp, entry.order_id))
        return order, trades

    def _match_against_book(self, incoming: Order, opposite_book: list[Order], next_trade_id, aggressive_is_buy: bool) -> tuple[Order, list[Trade]]:
        trades: list[Trade] = []
        remaining = incoming.remaining_quantity
        cursor = 0
        updated_book: list[Order] = []
        while cursor < len(opposite_book) and remaining > 0:
            resting = opposite_book[cursor]
            if not self._crosses(incoming, resting):
                updated_book.extend(opposite_book[cursor:])
                break
            trade_qty = min(remaining, resting.remaining_quantity)
            remaining -= trade_qty
            resting_remaining = resting.remaining_quantity - trade_qty
            trade_price = resting.price
            buy_order = incoming if aggressive_is_buy else resting
            sell_order = resting if aggressive_is_buy else incoming
            scenario_id, scenario_label, scenario_type, is_manipulative = resolve_trade_label(buy_order, sell_order)
            trades.append(
                Trade(
                    trade_id=next_trade_id(),
                    timestamp=incoming.timestamp,
                    buy_order_id=buy_order.order_id,
                    sell_order_id=sell_order.order_id,
                    buy_trader_id=buy_order.trader_id,
                    sell_trader_id=sell_order.trader_id,
                    instrument_id=incoming.instrument_id,
                    price=trade_price,
                    quantity=trade_qty,
                    scenario_id=scenario_id,
                    scenario_label=scenario_label,
                    scenario_type=scenario_type,
                    is_manipulative=is_manipulative,
                )
            )
            if resting_remaining > 0:
                updated_book.append(replace(resting, remaining_quantity=resting_remaining))
            cursor += 1
        else:
            updated_book.extend(opposite_book[cursor:])
        opposite_book[:] = updated_book
        return replace(incoming, remaining_quantity=remaining), trades

    def _crosses(self, incoming: Order, resting: Order) -> bool:
        if incoming.order_type == "market":
            return True
        if incoming.side == "buy":
            return incoming.price >= resting.price
        return incoming.price <= resting.price
