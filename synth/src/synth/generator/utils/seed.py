from __future__ import annotations

import random


def build_rng(seed: int) -> random.Random:
    return random.Random(seed)
