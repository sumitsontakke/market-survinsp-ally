from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from synth.generator import __version__
from synth.generator.utils.config import hash_config


class ManifestExporter:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, dataset: dict[str, list[dict]], config: dict) -> str:
        manifest = {
            "schema_version": config["schema_version"],
            "generator_version": config.get("generator_version", __version__),
            "package_version": __version__,
            "config_hash": hash_config(config),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "counts": {name: len(rows) for name, rows in dataset.items()},
            "scenario_types": sorted({row["scenario_type"] for row in dataset.get("scenarios", [])}),
            "scenario_ids": sorted({row["scenario_id"] for row in dataset.get("scenarios", [])}),
            "manipulative_order_count": sum(1 for row in dataset.get("orders", []) if row.get("is_manipulative")),
            "manipulative_trade_count": sum(1 for row in dataset.get("trades", []) if row.get("is_manipulative")),
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
        }
        file_path = self.output_dir / "manifest.json"
        with file_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        return str(file_path)
