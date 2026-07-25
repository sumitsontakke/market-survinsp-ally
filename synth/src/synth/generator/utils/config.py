from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union

import yaml


def load_config(path: Union[str, Path]) -> dict:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def hash_config(config: dict) -> str:
    payload = yaml.safe_dump(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
