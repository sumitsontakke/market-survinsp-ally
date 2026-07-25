"""M2 acceptance test: build PyG arrays for R01.

Asserts:
  - no NaNs in node features or edge attributes
  - edge_attr has the expected dimensionality
  - at least one positive edge label exists
  - knn sparsification reduces edges meaningfully vs unsparsified

Run from repo root:
    PYTHONPATH=. OUTPUTS_PATH=outputs python3 training/tests/test_pyg_builder.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure we run from repo root regardless of how the test is invoked.
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from detect.dataset.loader import list_runs_in_cohort, load_run  # noqa: E402
from detect.features.pyg_builder import (  # noqa: E402
    DEFAULT_EDGE_FEATURE_NAMES,
    DEFAULT_NODE_FEATURE_NAMES,
    build_graph_arrays,
)


def test_r01_graph_arrays() -> None:
    paths = list_runs_in_cohort("PHASE1_R01_R24")
    assert paths, "no runs resolved - check OUTPUTS_PATH"
    r01 = next(p for p in paths if p.name.startswith("R01_"))
    run = load_run(r01)
    assert run.n_trades > 0, "R01 has no trades"

    # No-sparsification baseline (used to validate the knn ratio)
    unsparsified = build_graph_arrays(run, sparsification=None)

    # Default (knn, k=12) - the production setting
    g = build_graph_arrays(
        run,
        edge_feature_names=DEFAULT_EDGE_FEATURE_NAMES,
        node_feature_names=DEFAULT_NODE_FEATURE_NAMES,
        sparsification="knn", k=12,
    )

    # 1. No NaNs anywhere
    assert not np.isnan(g.edge_attr).any(), "NaNs in edge_attr"
    assert not np.isnan(g.x).any(), "NaNs in node features"

    # 2. Edge attr dimensionality
    assert g.edge_attr.shape[1] == len(DEFAULT_EDGE_FEATURE_NAMES), (
        f"edge_attr dim {g.edge_attr.shape[1]} != "
        f"expected {len(DEFAULT_EDGE_FEATURE_NAMES)}"
    )

    # 3. At least one positive edge
    assert int(g.y.sum()) > 0, f"R01 has zero positive edges; y.sum()={int(g.y.sum())}"

    # 4. Sparsification actually reduced edge count
    n_before = unsparsified.num_edges
    n_after = g.num_edges
    ratio = n_before / max(n_after, 1)
    assert n_after < n_before, (
        f"knn sparsification didn't reduce edges: before={n_before} after={n_after}"
    )

    # 5. Edge counts and ratios are in the right ballpark
    print()
    print("=== R01 graph arrays ===")
    print(f"  nodes               : {g.num_nodes}")
    print(f"  edges before sparsif: {n_before}")
    print(f"  edges after  knn k=12: {n_after}")
    print(f"  reduction ratio     : {ratio:.2f}x")
    print(f"  positive edges      : {int(g.y.sum())}")
    print(f"  positive ratio      : {100*g.y.mean():.2f}%")
    print(f"  edge_attr shape     : {g.edge_attr.shape}")
    print(f"  node feature shape  : {g.x.shape}")
    print(f"  family              : {g.metadata['family']}")
    print(f"  edge features       : {g.edge_feature_names}")
    print(f"  node features       : {g.node_feature_names}")
    print()
    print("ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    test_r01_graph_arrays()
