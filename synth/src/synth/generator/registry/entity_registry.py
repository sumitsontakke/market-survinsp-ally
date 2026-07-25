from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from random import Random

from synth.generator.domain.entities import (
    Account,
    BeneficialOwner,
    Broker,
    Instrument,
    Trader,
    TradingSession,
)
from synth.generator.domain.enums import AccountStatus, TraderStatus
from synth.generator.registry.id_factory import IdFactory


@dataclass
class EntityRegistry:
    session: TradingSession
    brokers: list[Broker] = field(default_factory=list)
    beneficial_owners: list[BeneficialOwner] = field(default_factory=list)
    accounts: list[Account] = field(default_factory=list)
    traders: list[Trader] = field(default_factory=list)
    instruments: list[Instrument] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: dict, rng: Random, id_factory: IdFactory) -> "EntityRegistry":
        session_cfg = config["session"]
        session = TradingSession(
            session_id=id_factory.next("session"),
            trade_date=session_cfg["trade_date"],
            open_time=session_cfg["open_time"],
            close_time=session_cfg["close_time"],
            auction_windows=tuple(session_cfg.get("auction_windows", [])),
            timezone=session_cfg["timezone"],
        )
        registry = cls(session=session)
        registry._build_brokers(config, id_factory)
        registry._build_beneficial_owners(config, id_factory)
        registry._build_accounts(config, rng, id_factory)
        registry._build_traders(config, rng, id_factory)
        registry._build_instruments(config, id_factory)
        return registry

    def _build_brokers(self, config: dict, id_factory: IdFactory) -> None:
        count = int(config["brokers"]["count"])
        for index in range(count):
            self.brokers.append(
                Broker(
                    broker_id=id_factory.next("broker"),
                    broker_name=f"Broker {index + 1}",
                    venue_access="lit_equity",
                    latency_profile="standard",
                )
            )

    def _build_beneficial_owners(self, config: dict, id_factory: IdFactory) -> None:
        count = int(config["beneficial_owners"]["count"])
        for _ in range(count):
            self.beneficial_owners.append(
                BeneficialOwner(
                    beneficial_owner_id=id_factory.next("owner"),
                    owner_type="individual",
                    group_label="normal",
                    linked_account_count=0,
                )
            )

    def _build_accounts(self, config: dict, rng: Random, id_factory: IdFactory) -> None:
        account_cfg = config["accounts"]
        opened_at = datetime.fromisoformat(f"{self.session.trade_date}T{self.session.open_time}")
        built_owners: list[BeneficialOwner] = []
        for owner in self.beneficial_owners:
            account_count = rng.randint(account_cfg["per_owner_min"], account_cfg["per_owner_max"])
            built_owners.append(
                BeneficialOwner(
                    beneficial_owner_id=owner.beneficial_owner_id,
                    owner_type=owner.owner_type,
                    group_label=owner.group_label,
                    linked_account_count=account_count,
                )
            )
            for _ in range(account_count):
                broker = rng.choice(self.brokers)
                self.accounts.append(
                    Account(
                        account_id=id_factory.next("account"),
                        beneficial_owner_id=owner.beneficial_owner_id,
                        broker_id=broker.broker_id,
                        account_type="cash",
                        opened_at=opened_at,
                        status=AccountStatus.ACTIVE.value,
                    )
                )
        self.beneficial_owners = built_owners

    def _build_traders(self, config: dict, rng: Random, id_factory: IdFactory) -> None:
        created_at = datetime.fromisoformat(f"{self.session.trade_date}T{self.session.open_time}")
        profile_weights = config["traders"]["profiles"]
        profile_names = list(profile_weights.keys())
        weight_values = list(profile_weights.values())
        for account in self.accounts:
            profile = rng.choices(profile_names, weights=weight_values, k=1)[0]
            self.traders.append(
                Trader(
                    trader_id=id_factory.next("trader"),
                    account_id=account.account_id,
                    beneficial_owner_id=account.beneficial_owner_id,
                    broker_id=account.broker_id,
                    trader_profile_id=profile,
                    risk_tier=rng.choice(["low", "medium", "high"]),
                    region=rng.choice(["US", "EU", "APAC"]),
                    created_at=created_at,
                    status=TraderStatus.ACTIVE.value,
                )
            )

    def _build_instruments(self, config: dict, id_factory: IdFactory) -> None:
        for instrument_cfg in config["instruments"]:
            self.instruments.append(
                Instrument(
                    instrument_id=id_factory.next("instrument"),
                    symbol=instrument_cfg["symbol"],
                    asset_class=instrument_cfg["asset_class"],
                    tick_size=float(instrument_cfg["tick_size"]),
                    lot_size=int(instrument_cfg["lot_size"]),
                    price_band=tuple(float(value) for value in instrument_cfg["price_band"]),
                    session_calendar_id=self.session.session_id,
                )
            )

    def trader_by_id(self) -> dict[str, Trader]:
        return {trader.trader_id: trader for trader in self.traders}

    def account_by_id(self) -> dict[str, Account]:
        return {account.account_id: account for account in self.accounts}

    def instrument_by_id(self) -> dict[str, Instrument]:
        return {instrument.instrument_id: instrument for instrument in self.instruments}
