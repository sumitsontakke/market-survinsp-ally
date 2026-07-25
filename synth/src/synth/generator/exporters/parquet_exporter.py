from __future__ import annotations

import logging
from pathlib import Path

_log = logging.getLogger(__name__)


class ParquetExporter:
    """Optional parquet sidecar — never fatal.

    Skips silently (with a warning) if pandas has no usable parquet engine
    (pyarrow or fastparquet). CSV remains the canonical artifact downstream
    so a missing parquet engine should not crash an otherwise-successful
    run.
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, dataset):
        try:
            import pandas as pd
        except ImportError:
            return
        try:
            from pandas.io.parquet import get_engine
            get_engine("auto")
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "ParquetExporter: no parquet engine available (%s). "
                "Skipping parquet output. CSVs were still written.",
                exc,
            )
            return
        for name, rows in dataset.items():
            if not rows:
                continue
            try:
                pd.DataFrame(rows).to_parquet(
                    self.output_dir / f"{name}.parquet", index=False,
                )
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "ParquetExporter: failed to write %s.parquet (%s) "
                    "— continuing; CSV is the canonical artifact.",
                    name, exc,
                )
                return
