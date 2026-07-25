from __future__ import annotations

from synthetic_market_sim.registry.entity_registry import EntityRegistry
from synthetic_market_sim.registry.id_factory import IdFactory
from synthetic_market_sim.utils.config import load_config
from synthetic_market_sim.utils.seed import build_rng


def test_id_factory_yields_unique_ids() -> None:
    factory = IdFactory()
    ids = {factory.next("order") for _ in range(10)}
    assert len(ids) == 10


def test_registry_referential_integrity() -> None:
    config = load_config("tests/fixtures/small_config.yaml")
    registry = EntityRegistry.from_config(config, build_rng(config["seed"]), IdFactory())
    owner_ids = {owner.beneficial_owner_id for owner in registry.beneficial_owners}
    account_ids = {account.account_id for account in registry.accounts}
    broker_ids = {broker.broker_id for broker in registry.brokers}
    assert registry.traders
    assert all(account.beneficial_owner_id in owner_ids for account in registry.accounts)
    assert all(account.broker_id in broker_ids for account in registry.accounts)
    assert all(trader.account_id in account_ids for trader in registry.traders)
