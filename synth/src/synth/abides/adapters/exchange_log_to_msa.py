"""
ABIDES exchange-log → Market Surveillance Ally (MSA) per-run schema adapter.

Contract: given an ABIDES simulation run's outputs (trade log + agent registry +
optional manipulator-label sidecar), produce a directory matching the MSA
synthesizer schema observed at outputs/runs/R*/. Specifically:

    <out_dir>/
      trades.csv          trade_id, timestamp, buy_order_id, sell_order_id,
                          buy_trader_id, sell_trader_id, instrument_id, price,
                          quantity, scenario_id, scenario_label, scenario_type,
                          is_manipulative
      orders.csv          order_id, timestamp, trader_id, account_id, broker_id,
                          instrument_id, side, order_type, price, quantity,
                          time_in_force, scenario_id, scenario_label,
                          scenario_type, is_manipulative, parent_order_id,
                          remaining_quantity
      traders.csv         trader_id, account_id, beneficial_owner_id, broker_id,
                          trader_profile_id, risk_tier, region, created_at, status
      accounts.csv        account_id, beneficial_owner_id, opened_at, status
      beneficial_owners.csv  owner_id, name, kyc_status, region, created_at
      brokers.csv         broker_id, name, region, registered_at, status
      instruments.csv     instrument_id, symbol, asset_class, listing_venue, currency
      sessions.csv        session_id, instrument_id, session_date, open_ts, close_ts
      scenarios.csv       scenario_id, scenario_label, scenario_type,
                          start_ts, end_ts, manipulator_count
      manifest.json       schema_version, generator_version, counts, ...

Notes:
- ABIDES does not natively distinguish brokers / beneficial_owners / accounts.
  We synthesize stable IDs by hashing trader_id with a configured fanout so the
  surveillance pipeline's edge features (broker-overlap, owner-overlap) remain
  well-defined.
- Manipulator labels come from a sidecar (manipulator_labels.csv) emitted by
  our custom agents in Phase C. Until those land, every trade is labelled
  benign — which is exactly what the Phase 3 false-alarm-rate cohort needs.

This module imports nothing from abides itself, so it is fully testable
without the vendored upstream installed.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

SCHEMA_VERSION = "0.1.0"
ADAPTER_VERSION = "0.1.0-phase-a"


# --- Column manifests (must match MSA synthesizer exactly) -------------------

TRADES_COLUMNS = [
    "trade_id", "timestamp", "buy_order_id", "sell_order_id",
    "buy_trader_id", "sell_trader_id", "instrument_id", "price", "quantity",
    "scenario_id", "scenario_label", "scenario_type", "is_manipulative",
]
ORDERS_COLUMNS = [
    "order_id", "timestamp", "trader_id", "account_id", "broker_id",
    "instrument_id", "side", "order_type", "price", "quantity",
    "time_in_force", "scenario_id", "scenario_label", "scenario_type",
    "is_manipulative", "parent_order_id", "remaining_quantity",
]
TRADERS_COLUMNS = [
    "trader_id", "account_id", "beneficial_owner_id", "broker_id",
    "trader_profile_id", "risk_tier", "region", "created_at", "status",
]
ACCOUNTS_COLUMNS = ["account_id", "beneficial_owner_id", "opened_at", "status"]
OWNERS_COLUMNS = ["owner_id", "name", "kyc_status", "region", "created_at"]
BROKERS_COLUMNS = ["broker_id", "name", "region", "registered_at", "status"]
INSTRUMENTS_COLUMNS = ["instrument_id", "symbol", "asset_class", "listing_venue", "currency"]
SESSIONS_COLUMNS = ["session_id", "instrument_id", "session_date", "open_ts", "close_ts"]
SCENARIOS_COLUMNS = [
    "scenario_id", "scenario_label", "scenario_type",
    "start_ts", "end_ts", "manipulator_count",
]


@dataclass
class AdapterConfig:
    """Knobs controlling how synthetic identity hierarchies are synthesized."""

    num_brokers: int = 20
    accounts_per_owner: int = 1
    instrument_symbol: str = "ABIDES_SYN"
    instrument_id: str = "instrument_00001"
    listing_venue: str = "NSE_SYN"
    asset_class: str = "equity"
    currency: str = "INR"
    default_region: str = "IN"
    benign_scenario_id: str = "normal"
    benign_scenario_label: str = "normal"
    benign_scenario_type: str = "generic_background"


def _stable_int(key: str, modulo: int) -> int:
    """Deterministic non-cryptographic hash → integer in [0, modulo)."""
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def _broker_id_for_trader(trader_id: str, num_brokers: int) -> str:
    return f"broker_{_stable_int('broker:' + trader_id, num_brokers) + 1:05d}"


def _account_id_for_trader(trader_id: str) -> str:
    # 1:1 trader→account by default; can be relaxed via AdapterConfig later.
    return trader_id.replace("trader_", "account_")


def _owner_id_for_trader(trader_id: str) -> str:
    return trader_id.replace("trader_", "owner_")


# --- ABIDES reader -----------------------------------------------------------


@dataclass
class AbidesTrade:
    """One trade row as parsed from an ABIDES exchange log."""

    timestamp: str
    buy_order_id: str
    sell_order_id: str
    buy_trader_id: str
    sell_trader_id: str
    price: float
    quantity: int


@dataclass
class AbidesOrder:
    """One order row as parsed from an ABIDES exchange log."""

    order_id: str
    timestamp: str
    trader_id: str
    side: str          # "buy" | "sell"
    order_type: str    # "limit" | "market"
    price: float
    quantity: int
    time_in_force: str = "day"
    remaining_quantity: int = 0
    parent_order_id: str = ""


@dataclass
class AbidesRun:
    """In-memory representation of one ABIDES run, post-parsing."""

    session_date: str
    open_ts: str
    close_ts: str
    trader_ids: list[str] = field(default_factory=list)
    orders: list[AbidesOrder] = field(default_factory=list)
    trades: list[AbidesTrade] = field(default_factory=list)
    # trader_id -> manipulator label tuple (scenario_id, scenario_label, scenario_type)
    # If absent, trader is treated as benign.
    manipulator_labels: dict[str, tuple[str, str, str]] = field(default_factory=dict)


class AbidesExchangeLogReader:
    """Parse ABIDES exchange log + optional manipulator-label sidecar.

    ABIDES emits ExchangeAgent.processStream{trades,orders} into pickled
    pandas DataFrames in their canonical config. For Phase A we accept a
    flat CSV form (which our custom configs write directly via a small
    monkeypatch around the ExchangeAgent log_pretty path). This keeps the
    adapter independent of pickle format changes in the upstream.

    Expected input directory layout::

        <abides_run_dir>/
            trades.csv            (ts, buy_oid, sell_oid, buy_tid, sell_tid, price, qty)
            orders.csv            (oid, ts, tid, side, type, price, qty, [tif], [rem], [parent])
            traders.csv           (tid)                       # optional; derived if absent
            manipulator_labels.csv (tid, scenario_id, scenario_label, scenario_type)  # optional
            session.json          (date, open_ts, close_ts)   # optional
    """

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        if not self.run_dir.is_dir():
            raise FileNotFoundError(f"ABIDES run dir not found: {self.run_dir}")

    def read(self) -> AbidesRun:
        trades = list(self._read_trades())
        orders = list(self._read_orders())
        trader_ids = sorted(self._collect_traders(orders, trades))
        session = self._read_session(trades, orders)
        labels = self._read_manipulator_labels()
        return AbidesRun(
            session_date=session["session_date"],
            open_ts=session["open_ts"],
            close_ts=session["close_ts"],
            trader_ids=trader_ids,
            orders=orders,
            trades=trades,
            manipulator_labels=labels,
        )

    def _read_trades(self) -> Iterable[AbidesTrade]:
        path = self.run_dir / "trades.csv"
        if not path.is_file():
            return []
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                yield AbidesTrade(
                    timestamp=row["timestamp"],
                    buy_order_id=row.get("buy_order_id", ""),
                    sell_order_id=row.get("sell_order_id", ""),
                    buy_trader_id=row["buy_trader_id"],
                    sell_trader_id=row["sell_trader_id"],
                    price=float(row["price"]),
                    quantity=int(row["quantity"]),
                )

    def _read_orders(self) -> Iterable[AbidesOrder]:
        path = self.run_dir / "orders.csv"
        if not path.is_file():
            return []
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                yield AbidesOrder(
                    order_id=row["order_id"],
                    timestamp=row["timestamp"],
                    trader_id=row["trader_id"],
                    side=row["side"],
                    order_type=row.get("order_type", "limit"),
                    price=float(row["price"]),
                    quantity=int(row["quantity"]),
                    time_in_force=row.get("time_in_force", "day"),
                    remaining_quantity=int(row.get("remaining_quantity") or 0),
                    parent_order_id=row.get("parent_order_id", ""),
                )

    def _read_manipulator_labels(self) -> dict[str, tuple[str, str, str]]:
        path = self.run_dir / "manipulator_labels.csv"
        if not path.is_file():
            return {}
        labels: dict[str, tuple[str, str, str]] = {}
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                labels[row["trader_id"]] = (
                    row["scenario_id"],
                    row["scenario_label"],
                    row["scenario_type"],
                )
        return labels

    def _read_session(
        self, trades: Sequence[AbidesTrade], orders: Sequence[AbidesOrder]
    ) -> dict[str, str]:
        path = self.run_dir / "session.json"
        if path.is_file():
            return json.loads(path.read_text())
        # Derive from data: take earliest and latest timestamps.
        all_ts = [t.timestamp for t in trades] + [o.timestamp for o in orders]
        if not all_ts:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            return {
                "session_date": today,
                "open_ts": f"{today}T09:30:00",
                "close_ts": f"{today}T15:30:00",
            }
        open_ts = min(all_ts)
        close_ts = max(all_ts)
        session_date = open_ts.split("T", 1)[0]
        return {"session_date": session_date, "open_ts": open_ts, "close_ts": close_ts}

    @staticmethod
    def _collect_traders(
        orders: Sequence[AbidesOrder], trades: Sequence[AbidesTrade]
    ) -> set[str]:
        ids: set[str] = set()
        for o in orders:
            ids.add(o.trader_id)
        for t in trades:
            ids.add(t.buy_trader_id)
            ids.add(t.sell_trader_id)
        return ids


# --- MSA writer --------------------------------------------------------------


class MSARunWriter:
    """Write an AbidesRun to disk in MSA per-run schema."""

    def __init__(self, out_dir: str | Path, config: AdapterConfig | None = None) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or AdapterConfig()

    def write(self, run: AbidesRun, run_label: str = "abides_run") -> dict:
        cfg = self.config
        self._write_brokers()
        self._write_instruments()
        owners = self._write_owners(run.trader_ids)
        accounts = self._write_accounts(run.trader_ids, owners, run.open_ts)
        self._write_traders(run, accounts, owners, run.open_ts)
        self._write_sessions(run)
        scenarios = self._write_scenarios(run)
        self._write_orders(run, scenarios)
        self._write_trades(run, scenarios)
        manifest = self._write_manifest(run, run_label, scenarios)
        return manifest

    # -- individual writers --

    def _write_brokers(self) -> None:
        cfg = self.config
        rows = []
        for i in range(1, cfg.num_brokers + 1):
            rows.append({
                "broker_id": f"broker_{i:05d}",
                "name": f"AbidesBroker_{i:02d}",
                "region": cfg.default_region,
                "registered_at": "2026-01-01T00:00:00",
                "status": "active",
            })
        _write_csv(self.out_dir / "brokers.csv", BROKERS_COLUMNS, rows)

    def _write_instruments(self) -> None:
        cfg = self.config
        rows = [{
            "instrument_id": cfg.instrument_id,
            "symbol": cfg.instrument_symbol,
            "asset_class": cfg.asset_class,
            "listing_venue": cfg.listing_venue,
            "currency": cfg.currency,
        }]
        _write_csv(self.out_dir / "instruments.csv", INSTRUMENTS_COLUMNS, rows)

    def _write_owners(self, trader_ids: Sequence[str]) -> dict[str, str]:
        cfg = self.config
        owner_rows = []
        owner_map: dict[str, str] = {}
        for tid in trader_ids:
            oid = _owner_id_for_trader(tid)
            owner_map[tid] = oid
            owner_rows.append({
                "owner_id": oid,
                "name": f"Owner_{tid}",
                "kyc_status": "verified",
                "region": cfg.default_region,
                "created_at": "2026-01-01T00:00:00",
            })
        _write_csv(self.out_dir / "beneficial_owners.csv", OWNERS_COLUMNS, owner_rows)
        return owner_map

    def _write_accounts(
        self, trader_ids: Sequence[str], owner_map: Mapping[str, str], opened_at: str
    ) -> dict[str, str]:
        account_rows = []
        account_map: dict[str, str] = {}
        for tid in trader_ids:
            aid = _account_id_for_trader(tid)
            account_map[tid] = aid
            account_rows.append({
                "account_id": aid,
                "beneficial_owner_id": owner_map[tid],
                "opened_at": opened_at,
                "status": "active",
            })
        _write_csv(self.out_dir / "accounts.csv", ACCOUNTS_COLUMNS, account_rows)
        return account_map

    def _write_traders(
        self,
        run: AbidesRun,
        account_map: Mapping[str, str],
        owner_map: Mapping[str, str],
        opened_at: str,
    ) -> None:
        cfg = self.config
        rows = []
        for tid in run.trader_ids:
            # Profile inferred from manipulator label if present, else generic.
            label = run.manipulator_labels.get(tid)
            profile = label[2] if label else "abides_background"
            rows.append({
                "trader_id": tid,
                "account_id": account_map[tid],
                "beneficial_owner_id": owner_map[tid],
                "broker_id": _broker_id_for_trader(tid, cfg.num_brokers),
                "trader_profile_id": profile,
                "risk_tier": "high" if label else "medium",
                "region": cfg.default_region,
                "created_at": opened_at,
                "status": "active",
            })
        _write_csv(self.out_dir / "traders.csv", TRADERS_COLUMNS, rows)

    def _write_sessions(self, run: AbidesRun) -> None:
        cfg = self.config
        rows = [{
            "session_id": "session_00001",
            "instrument_id": cfg.instrument_id,
            "session_date": run.session_date,
            "open_ts": run.open_ts,
            "close_ts": run.close_ts,
        }]
        _write_csv(self.out_dir / "sessions.csv", SESSIONS_COLUMNS, rows)

    def _write_scenarios(self, run: AbidesRun) -> dict[str, tuple[str, str, str]]:
        """Returns trader_id -> (scenario_id, scenario_label, scenario_type)."""
        cfg = self.config
        scenarios: dict[str, dict] = {
            cfg.benign_scenario_id: {
                "scenario_id": cfg.benign_scenario_id,
                "scenario_label": cfg.benign_scenario_label,
                "scenario_type": cfg.benign_scenario_type,
                "start_ts": run.open_ts,
                "end_ts": run.close_ts,
                "manipulator_count": 0,
            },
        }
        per_trader: dict[str, tuple[str, str, str]] = {}
        for tid in run.trader_ids:
            label = run.manipulator_labels.get(tid)
            if label is None:
                per_trader[tid] = (cfg.benign_scenario_id, cfg.benign_scenario_label, cfg.benign_scenario_type)
                continue
            sid, slabel, stype = label
            per_trader[tid] = label
            sc = scenarios.setdefault(sid, {
                "scenario_id": sid,
                "scenario_label": slabel,
                "scenario_type": stype,
                "start_ts": run.open_ts,
                "end_ts": run.close_ts,
                "manipulator_count": 0,
            })
            sc["manipulator_count"] += 1
        _write_csv(
            self.out_dir / "scenarios.csv",
            SCENARIOS_COLUMNS,
            list(scenarios.values()),
        )
        return per_trader

    def _write_orders(self, run: AbidesRun, scenarios: Mapping[str, tuple[str, str, str]]) -> None:
        cfg = self.config
        rows = []
        for o in run.orders:
            sid, slabel, stype = scenarios.get(
                o.trader_id,
                (cfg.benign_scenario_id, cfg.benign_scenario_label, cfg.benign_scenario_type),
            )
            is_manip = stype != cfg.benign_scenario_type
            rows.append({
                "order_id": o.order_id,
                "timestamp": o.timestamp,
                "trader_id": o.trader_id,
                "account_id": _account_id_for_trader(o.trader_id),
                "broker_id": _broker_id_for_trader(o.trader_id, cfg.num_brokers),
                "instrument_id": cfg.instrument_id,
                "side": o.side,
                "order_type": o.order_type,
                "price": o.price,
                "quantity": o.quantity,
                "time_in_force": o.time_in_force,
                "scenario_id": sid,
                "scenario_label": slabel,
                "scenario_type": stype,
                "is_manipulative": is_manip,
                "parent_order_id": o.parent_order_id,
                "remaining_quantity": o.remaining_quantity,
            })
        _write_csv(self.out_dir / "orders.csv", ORDERS_COLUMNS, rows)

    def _write_trades(self, run: AbidesRun, scenarios: Mapping[str, tuple[str, str, str]]) -> None:
        cfg = self.config
        rows = []
        for i, t in enumerate(run.trades, start=1):
            # A trade is manipulative if either side is manipulative.
            buy_sid, buy_slabel, buy_stype = scenarios.get(
                t.buy_trader_id,
                (cfg.benign_scenario_id, cfg.benign_scenario_label, cfg.benign_scenario_type),
            )
            sell_sid, sell_slabel, sell_stype = scenarios.get(
                t.sell_trader_id,
                (cfg.benign_scenario_id, cfg.benign_scenario_label, cfg.benign_scenario_type),
            )
            buy_manip = buy_stype != cfg.benign_scenario_type
            sell_manip = sell_stype != cfg.benign_scenario_type
            if buy_manip and not sell_manip:
                sid, slabel, stype = buy_sid, buy_slabel, buy_stype
            elif sell_manip and not buy_manip:
                sid, slabel, stype = sell_sid, sell_slabel, sell_stype
            elif buy_manip and sell_manip:
                sid, slabel, stype = buy_sid, buy_slabel, buy_stype  # break tie on buy side
            else:
                sid, slabel, stype = cfg.benign_scenario_id, cfg.benign_scenario_label, cfg.benign_scenario_type
            rows.append({
                "trade_id": f"trade_{i:05d}",
                "timestamp": t.timestamp,
                "buy_order_id": t.buy_order_id,
                "sell_order_id": t.sell_order_id,
                "buy_trader_id": t.buy_trader_id,
                "sell_trader_id": t.sell_trader_id,
                "instrument_id": cfg.instrument_id,
                "price": t.price,
                "quantity": t.quantity,
                "scenario_id": sid,
                "scenario_label": slabel,
                "scenario_type": stype,
                "is_manipulative": buy_manip or sell_manip,
            })
        _write_csv(self.out_dir / "trades.csv", TRADES_COLUMNS, rows)

    def _write_manifest(
        self, run: AbidesRun, run_label: str, scenarios: Mapping[str, tuple[str, str, str]]
    ) -> dict:
        cfg = self.config
        scenario_types = sorted({s[2] for s in scenarios.values()})
        scenario_ids = sorted({s[0] for s in scenarios.values()})
        manip_trade_count = sum(
            1 for t in run.trades
            if scenarios.get(t.buy_trader_id, ("", "", cfg.benign_scenario_type))[2] != cfg.benign_scenario_type
            or scenarios.get(t.sell_trader_id, ("", "", cfg.benign_scenario_type))[2] != cfg.benign_scenario_type
        )
        manip_order_count = sum(
            1 for o in run.orders
            if scenarios.get(o.trader_id, ("", "", cfg.benign_scenario_type))[2] != cfg.benign_scenario_type
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generator_version": f"abides-adapter-{ADAPTER_VERSION}",
            "package_version": ADAPTER_VERSION,
            "config_hash": _hash_config(cfg),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_label": run_label,
            "counts": {
                "brokers": cfg.num_brokers,
                "beneficial_owners": len(run.trader_ids),
                "accounts": len(run.trader_ids),
                "traders": len(run.trader_ids),
                "instruments": 1,
                "sessions": 1,
                "orders": len(run.orders),
                "trades": len(run.trades),
                "scenarios": len(scenario_ids),
            },
            "scenario_types": scenario_types,
            "scenario_ids": scenario_ids,
            "manipulative_order_count": manip_order_count,
            "manipulative_trade_count": manip_trade_count,
            "entity_relationships": {
                "beneficial_owner_to_account": "1..n",
                "account_to_trader": "1..n",
                "trader_to_order": "1..n",
                "order_to_trade": "0..n",
            },
            "label_definitions": {
                "normal": "Background non-manipulative activity",
                "manipulative": "Scenario-linked coordinated activity",
            },
            "data_source": "abides",
        }
        (self.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return manifest


def adapt_abides_run_to_msa(
    abides_run_dir: str | Path,
    out_dir: str | Path,
    run_label: str = "abides_run",
    config: AdapterConfig | None = None,
) -> dict:
    """One-call convenience: read an ABIDES run dir, write MSA per-run dir.

    Returns the manifest dict.
    """
    reader = AbidesExchangeLogReader(abides_run_dir)
    run = reader.read()
    writer = MSARunWriter(out_dir, config=config)
    return writer.write(run, run_label=run_label)


# --- helpers -----------------------------------------------------------------


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def _hash_config(cfg: AdapterConfig) -> str:
    blob = json.dumps(cfg.__dict__, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
