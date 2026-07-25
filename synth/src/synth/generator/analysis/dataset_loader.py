from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml


@dataclass(frozen=True)
class ScenarioAttribution:
    scenario_id: str
    scenario_type: str
    is_manipulative: bool


@dataclass(frozen=True)
class ScenarioAttributionIndex:
    trader_to_scenarios: dict[str, list[ScenarioAttribution]]
    scenario_to_traders: dict[str, list[str]]
    scenario_details: dict[str, ScenarioAttribution]


@dataclass(frozen=True)
class LoadedDataset:
    root: Optional[Path]
    traders: pd.DataFrame
    orders: pd.DataFrame
    trades: pd.DataFrame
    scenarios: pd.DataFrame
    instruments: pd.DataFrame
    manifest: dict
    instrument_scope: str | None = None

    def scoped_to_instrument(self, instrument_id: str | None) -> "LoadedDataset":
        if not instrument_id:
            return self

        scoped_orders = self.orders.copy()
        if not scoped_orders.empty and "instrument_id" in scoped_orders.columns:
            scoped_orders = scoped_orders.loc[scoped_orders["instrument_id"].astype(str) == str(instrument_id)].copy()

        scoped_trades = self.trades.copy()
        if not scoped_trades.empty and "instrument_id" in scoped_trades.columns:
            scoped_trades = scoped_trades.loc[scoped_trades["instrument_id"].astype(str) == str(instrument_id)].copy()

        scoped_scenarios = self.scenarios.copy()
        scoped_scenario_ids: set[str] = set()
        for frame in (scoped_orders, scoped_trades):
            if not frame.empty and "scenario_id" in frame.columns:
                scoped_scenario_ids.update(
                    str(value)
                    for value in frame["scenario_id"].dropna().astype(str).tolist()
                    if str(value) and str(value) != "normal"
                )
        if not scoped_scenarios.empty:
            scoped_scenarios = scoped_scenarios.loc[
                (scoped_scenarios.get("instrument_id", pd.Series(dtype=str)).astype(str) == str(instrument_id))
                | (scoped_scenarios["scenario_id"].astype(str).isin(scoped_scenario_ids))
            ].copy()
            scoped_scenario_ids.update(scoped_scenarios["scenario_id"].dropna().astype(str).tolist())

        scoped_instruments = self.instruments.copy()
        if not scoped_instruments.empty and "instrument_id" in scoped_instruments.columns:
            scoped_instruments = scoped_instruments.loc[scoped_instruments["instrument_id"].astype(str) == str(instrument_id)].copy()

        participant_ids: set[str] = set()
        if not scoped_scenarios.empty and "participant_ids" in scoped_scenarios.columns:
            for values in scoped_scenarios["participant_ids"].tolist():
                if isinstance(values, list):
                    participant_ids.update(str(value) for value in values)
        active_trader_ids: set[str] = set(participant_ids)
        if not scoped_orders.empty and "trader_id" in scoped_orders.columns:
            active_trader_ids.update(scoped_orders["trader_id"].dropna().astype(str).tolist())
        if not scoped_trades.empty:
            for column in ("buy_trader_id", "sell_trader_id"):
                if column in scoped_trades.columns:
                    active_trader_ids.update(scoped_trades[column].dropna().astype(str).tolist())

        scoped_traders = self.traders.copy()
        if not scoped_traders.empty and active_trader_ids and "trader_id" in scoped_traders.columns:
            scoped_traders = scoped_traders.loc[scoped_traders["trader_id"].astype(str).isin(active_trader_ids)].copy()

        scoped_manifest = dict(self.manifest)
        counts = dict(scoped_manifest.get("counts", {}))
        counts.update(
            {
                "traders": int(len(scoped_traders)),
                "orders": int(len(scoped_orders)),
                "trades": int(len(scoped_trades)),
                "scenarios": int(len(scoped_scenarios)),
                "instruments": int(len(scoped_instruments)),
            }
        )
        scoped_manifest["counts"] = counts
        scoped_manifest["instrument_scope"] = {
            "instrument_id": str(instrument_id),
            "scope_type": "instrument_scoped",
        }
        scoped_manifest["scenario_ids"] = sorted(
            scenario_id for scenario_id in scoped_scenario_ids if scenario_id != "normal"
        )
        scoped_manifest["scenario_types"] = (
            sorted(scoped_scenarios["scenario_type"].dropna().astype(str).unique().tolist())
            if not scoped_scenarios.empty and "scenario_type" in scoped_scenarios.columns
            else []
        )

        return LoadedDataset(
            root=self.root,
            traders=scoped_traders.reset_index(drop=True),
            orders=scoped_orders.reset_index(drop=True),
            trades=scoped_trades.reset_index(drop=True),
            scenarios=scoped_scenarios.reset_index(drop=True),
            instruments=scoped_instruments.reset_index(drop=True),
            manifest=scoped_manifest,
            instrument_scope=str(instrument_id),
        )

    def manipulative_scenarios(self) -> pd.DataFrame:
        if self.scenarios.empty:
            return self.scenarios.copy()
        mask = self.scenarios["is_manipulative"].astype(str).str.lower() == "true"
        return self.scenarios.loc[mask].copy()

    def scenario_participants(self) -> dict[str, list[str]]:
        return self.scenario_attribution_index().scenario_to_traders

    def scenario_attribution_index(self) -> ScenarioAttributionIndex:
        scenario_to_traders: dict[str, set[str]] = {}
        scenario_details: dict[str, ScenarioAttribution] = {}
        for row in self.manipulative_scenarios().to_dict("records"):
            scenario_id = str(row["scenario_id"])
            scenario_type = str(row.get("scenario_type", "unknown"))
            scenario_details[scenario_id] = ScenarioAttribution(
                scenario_id=scenario_id,
                scenario_type=scenario_type,
                is_manipulative=True,
            )
            scenario_to_traders[scenario_id] = {
                str(participant_id)
                for participant_id in row.get("participant_ids", [])
                if pd.notna(participant_id)
            }

        self._add_manifest_scenario_details(scenario_details, scenario_to_traders)
        self._add_config_snapshot_scenario_details(scenario_details, scenario_to_traders)
        self._add_activity_attributions(self.orders, scenario_to_traders, scenario_details, trader_columns=["trader_id"])
        self._add_activity_attributions(
            self.trades,
            scenario_to_traders,
            scenario_details,
            trader_columns=["buy_trader_id", "sell_trader_id"],
        )

        if not scenario_details:
            return ScenarioAttributionIndex(trader_to_scenarios={}, scenario_to_traders={}, scenario_details={})

        trader_to_scenarios: dict[str, list[ScenarioAttribution]] = {}
        for scenario_id, traders in scenario_to_traders.items():
            detail = scenario_details[scenario_id]
            for trader_id in traders:
                trader_to_scenarios.setdefault(trader_id, []).append(detail)

        ordered_trader_mapping = {
            trader_id: sorted(
                details,
                key=lambda detail: (detail.scenario_id, detail.scenario_type),
            )
            for trader_id, details in trader_to_scenarios.items()
        }
        ordered_scenario_mapping = {
            scenario_id: sorted(traders)
            for scenario_id, traders in scenario_to_traders.items()
        }
        return ScenarioAttributionIndex(
            trader_to_scenarios=ordered_trader_mapping,
            scenario_to_traders=ordered_scenario_mapping,
            scenario_details=scenario_details,
        )

    def _add_activity_attributions(
        self,
        frame: pd.DataFrame,
        scenario_to_traders: dict[str, set[str]],
        scenario_details: dict[str, ScenarioAttribution],
        *,
        trader_columns: list[str],
    ) -> None:
        if frame.empty or "scenario_id" not in frame.columns:
            return
        manipulative_rows = frame.loc[frame["scenario_id"].astype(str).isin(scenario_details)]
        for row in manipulative_rows.to_dict("records"):
            scenario_id = str(row["scenario_id"])
            for trader_column in trader_columns:
                trader_id = row.get(trader_column)
                if pd.isna(trader_id):
                    continue
                scenario_to_traders[scenario_id].add(str(trader_id))

    def _add_manifest_scenario_details(
        self,
        scenario_details: dict[str, ScenarioAttribution],
        scenario_to_traders: dict[str, set[str]],
    ) -> None:
        for scenario_id in self.manifest.get("scenario_ids", []):
            scenario_key = str(scenario_id)
            if not scenario_key or scenario_key == "normal":
                continue
            scenario_details.setdefault(
                scenario_key,
                ScenarioAttribution(
                    scenario_id=scenario_key,
                    scenario_type="unknown",
                    is_manipulative=True,
                ),
            )
            scenario_to_traders.setdefault(scenario_key, set())

    def _add_config_snapshot_scenario_details(
        self,
        scenario_details: dict[str, ScenarioAttribution],
        scenario_to_traders: dict[str, set[str]],
    ) -> None:
        if self.root is None:
            return
        config_path = self.root / "config_snapshot.yaml"
        if not config_path.exists():
            return
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        allowed_symbols = set()
        if self.instrument_scope and not self.instruments.empty and "symbol" in self.instruments.columns:
            allowed_symbols = set(self.instruments["symbol"].dropna().astype(str).tolist())
        for scenario in payload.get("scenarios", []):
            scenario_id = str(scenario.get("scenario_id", "")).strip()
            if not scenario_id:
                continue
            if self.instrument_scope:
                configured_instrument_id = str(scenario.get("instrument_id", "")).strip()
                configured_symbol = str(scenario.get("instrument_symbol", "")).strip()
                if configured_instrument_id and configured_instrument_id != self.instrument_scope:
                    continue
                if configured_symbol and allowed_symbols and configured_symbol not in allowed_symbols:
                    continue
            scenario_details.setdefault(
                scenario_id,
                ScenarioAttribution(
                    scenario_id=scenario_id,
                    scenario_type=str(scenario.get("scenario_type", "unknown")),
                    is_manipulative=True,
                ),
            )
            scenario_to_traders.setdefault(scenario_id, set())


def load_dataset(root: str | Path) -> LoadedDataset:
    root_path = Path(root)
    return LoadedDataset(
        root=root_path,
        traders=_load_csv(root_path / "traders.csv", parse_dates=None),
        orders=_load_csv(root_path / "orders.csv", parse_dates=["timestamp"]),
        trades=_load_csv(root_path / "trades.csv", parse_dates=["timestamp"]),
        scenarios=_load_scenarios(root_path / "scenarios.csv"),
        instruments=_load_csv(root_path / "instruments.csv", parse_dates=None),
        manifest=json.loads((root_path / "manifest.json").read_text(encoding="utf-8")),
        instrument_scope=None,
    )


def _load_csv(path: Path, parse_dates: Optional[list[str]]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if parse_dates:
        return pd.read_csv(path, parse_dates=parse_dates)
    return pd.read_csv(path)


def _load_scenarios(path: Path) -> pd.DataFrame:
    scenarios = _load_csv(path, parse_dates=["start_time", "end_time"])
    if scenarios.empty:
        return scenarios
    scenarios = scenarios.copy()
    scenarios["participant_ids"] = scenarios["participant_ids"].apply(json.loads)
    if "ring_order" in scenarios.columns:
        scenarios["ring_order"] = scenarios["ring_order"].apply(lambda value: json.loads(value) if isinstance(value, str) and value else [])
    return scenarios
