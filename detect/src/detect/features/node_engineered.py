"""Engineered per-trader node features for the feature-augmented GraphSAGE.

Path A2 (Phase H): the six manipulation-signature features from the
tier-2 study, injected as GraphSAGE *node* features so message-passing
can learn feature-graph interactions, not just the trader-marginal
feature combinations a bolt-on classifier sees.

Thin adapter over ``engineered_core.compute_features_frame`` in node mode
(exactly one row per graph node, in node-id order). The feature math is
shared with ``scripts/compute_engineered_features.py`` so the GNN inputs
and the tier-2 analysis features are provably identical.
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd

from detect.features.engineered_core import (
    FEATURE_NAMES,
    compute_features_frame,
)

# The six engineered features, in the canonical order used everywhere.
ENGINEERED_NODE_FEATURE_NAMES: tuple[str, ...] = FEATURE_NAMES


def engineered_node_feature_frame(orders: pd.DataFrame,
                                  trades: pd.DataFrame,
                                  node_ids: Sequence[str]) -> pd.DataFrame:
    """DataFrame indexed by node_id with the six engineered features.

    Rows are aligned to ``node_ids`` order; any id without usable order
    data gets 0.0 across all six features. NaNs are scrubbed to 0.0 so
    the result is trainer-ready.
    """
    ids = [str(n) for n in node_ids]
    if not ids:
        return pd.DataFrame(columns=list(ENGINEERED_NODE_FEATURE_NAMES),
                            dtype=float)
    feats = compute_features_frame(orders, trades, trader_ids=ids)
    feats = feats.set_index("trader_id")
    feats = feats.reindex(ids)
    out = feats[list(ENGINEERED_NODE_FEATURE_NAMES)].astype(float)
    out = out.fillna(0.0)
    out.index = ids
    return out
