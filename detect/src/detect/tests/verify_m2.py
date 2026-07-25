"""M2 acceptance check: per-run stats table across the cohort.

Builds graph arrays for every (or a sample of) run in the cohort with both
unsparsified and k-NN sparsified settings, then prints a markdown stats
table for review.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from detect.dataset.loader import list_runs_in_cohort, load_run
from detect.features.pyg_builder import build_graph_arrays


def _process(run_path: Path) -> dict:
    t0 = time.perf_counter()
    run = load_run(run_path)
    raw = build_graph_arrays(run, sparsification=None)
    knn = build_graph_arrays(run, sparsification="knn", k=12)
    return {
        "run_id": run.run_id,
        "family": run.family,
        "trades": int(run.n_trades),
        "nodes": int(raw.num_nodes),
        "edges_raw": int(raw.num_edges),
        "edges_knn12": int(knn.num_edges),
        "reduction_ratio": float(raw.num_edges / max(knn.num_edges, 1)),
        "pos_edges_raw": int(raw.num_positives),
        "pos_edges_knn12": int(knn.num_positives),
        "neg_edges_knn12": int(knn.num_edges - knn.num_positives),
        "pos_ratio_knn12_pct": float(100 * knn.num_positives / max(knn.num_edges, 1)),
        "has_nan_x": bool(np.isnan(knn.x).any()),
        "has_nan_edge_attr": bool(np.isnan(knn.edge_attr).any()),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=None,
                   help="Run only N representative runs (one+ per family).")
    p.add_argument("--cohort", default="PHASE1_R01_R24")
    p.add_argument("--out", type=Path, default=None,
                   help="Optional JSON output path.")
    args = p.parse_args()

    paths = list_runs_in_cohort(args.cohort)
    if args.sample:
        from detect.dataset.universe import infer_run_family
        seen: dict[str, list[Path]] = {}
        for q in paths:
            seen.setdefault(infer_run_family(q.name), []).append(q)
        per_family = max(1, args.sample // max(len(seen), 1))
        chosen: list[Path] = []
        for fam, items in seen.items():
            chosen.extend(items[:per_family])
        chosen = chosen[: args.sample]
        paths = chosen

    print(f"verifying {len(paths)} runs from {args.cohort}")
    rows: list[dict] = []
    for q in paths:
        try:
            row = _process(q)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {q.name}: {exc!r}")
            continue
        rows.append(row)
        print(
            f"  {row['run_id']:<42s} fam={row['family']:<7s} "
            f"nodes={row['nodes']:>4d} raw_edges={row['edges_raw']:>5d} "
            f"knn12={row['edges_knn12']:>5d} ({row['reduction_ratio']:.2f}x)  "
            f"pos={row['pos_edges_knn12']:>3d} ({row['pos_ratio_knn12_pct']:>4.1f}%)  "
            f"NaN={row['has_nan_x'] or row['has_nan_edge_attr']}  "
            f"{row['elapsed_sec']:>4.1f}s"
        )

    print()
    print("=== Aggregate ===")
    if rows:
        raw_total = sum(r["edges_raw"] for r in rows)
        knn_total = sum(r["edges_knn12"] for r in rows)
        pos_total = sum(r["pos_edges_knn12"] for r in rows)
        print(f"  runs processed       : {len(rows)}")
        print(f"  total raw edges      : {raw_total:,}")
        print(f"  total knn12 edges    : {knn_total:,}")
        print(f"  aggregate reduction  : {raw_total/max(knn_total,1):.2f}x")
        print(f"  total positive edges : {pos_total:,}")
        print(f"  any NaN encountered  : {any(r['has_nan_x'] or r['has_nan_edge_attr'] for r in rows)}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"  report written to    : {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
