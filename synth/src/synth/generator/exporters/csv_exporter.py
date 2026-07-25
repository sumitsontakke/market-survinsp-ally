from __future__ import annotations

import csv
import json
from pathlib import Path


class CsvExporter:
    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, dataset: dict[str, list[dict]]) -> None:
        for name, rows in dataset.items():
            if not rows:
                continue
            file_path = self.output_dir / f"{name}.csv"
            normalized_rows = [self._normalize_row(row) for row in rows]
            with file_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(normalized_rows[0].keys()))
                writer.writeheader()
                writer.writerows(normalized_rows)

    def _normalize_row(self, row: dict) -> dict:
        normalized = {}
        for key, value in row.items():
            if isinstance(value, list):
                normalized[key] = json.dumps(value)
            else:
                normalized[key] = value
        return normalized
