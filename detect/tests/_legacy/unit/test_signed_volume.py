from __future__ import annotations


def signed_quantity(side: str, quantity: int) -> int:
    return quantity if side == "buy" else -quantity


def test_signed_quantity() -> None:
    assert signed_quantity("buy", 10) == 10
    assert signed_quantity("sell", 10) == -10
