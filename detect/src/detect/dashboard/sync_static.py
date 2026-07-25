"""Mirror the small set of artifacts the Demo Review page wants to expose.

Streamlit's static file serving refuses to serve directories >1 GB, and the
top-level outputs/ folder is much bigger because of bhavcopy CSVs and per-run
artifacts. This script copies only the curated, small artifacts (PNGs, JSONs,
the comparison CSV, the observation .md docs) into /app/static/ at container
start time (see docker-compose webapp command).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Source roots inside the container.
SRC_OUTPUTS = Path("/outputs")
SRC_TRAINING = Path("/training")  # only present when training/ is bind-mounted
DST = Path("/app/static")

# (src_root, filename) tuples. None src_root means search both.
INCLUDE_FROM_OUTPUTS = {
    # Comparison + metrics
    "r3_locked_stress.csv",
    "_m3_full_metrics.json",
    "_m3_full_loss_curve.json",
    "_m3_boosted_metrics.json",
    "_m3_boosted_loss_curve.json",
    "_m3_boosted_run.log",
    # M4 charts
    "m4_four_rung_bars.png",
    "m4_loss_curves.png",
    "m4_gpu_speedup.png",
    "m4_per_run_recall.png",
    # Demo Review diagrams
    "demo_gnn_architecture.png",
    "demo_synth_flow.png",
    "demo_stylized_facts.png",
    # Per-family trader-network visualizations + confusion matrices
    "demo_network_R01_clique.png",
    "demo_network_R09_ring.png",
    "demo_network_R17_mixed.png",
    "demo_confusion_R01_clique.png",
    "demo_confusion_R09_ring.png",
    "demo_confusion_R17_mixed.png",
}

INCLUDE_FROM_TRAINING = {
    # Observations docs — surfaced in the Demo Review page expander + link.
    "M3_FULL_OBSERVATIONS.md",
    "M3_BOOSTED_OBSERVATIONS.md",
    # Chapter 6 (limitations) — opens in new tab from the Demo Review footer.
    "LIMITATIONS.md",
}


def _copy_set(src_root: Path, names: set[str]) -> list[str]:
    copied: list[str] = []
    if not src_root.exists():
        print(f"  src missing: {src_root}", file=sys.stderr)
        return copied
    for name in sorted(names):
        src_p = src_root / name
        if not src_p.exists():
            print(f"  miss   {src_root}/{name}", file=sys.stderr)
            continue
        shutil.copyfile(src_p, DST / name)
        copied.append(name)
    return copied


def main() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    copied = _copy_set(SRC_OUTPUTS, INCLUDE_FROM_OUTPUTS)
    copied += _copy_set(SRC_TRAINING, INCLUDE_FROM_TRAINING)
    print(f"sync_static: copied {len(copied)} artifacts to {DST}")
    for n in copied:
        print(f"  {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
