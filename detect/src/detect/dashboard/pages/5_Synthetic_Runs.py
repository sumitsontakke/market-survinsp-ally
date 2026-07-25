"""Synthetic Runs viewer.

Surfaces the existing ``synthetic_market_sim`` outputs without modifying
anything in that package. Lists run directories under ``/outputs/runs/``
(read-only mount) and renders the artifacts each run produced.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Synthetic Runs", page_icon="🔬", layout="wide")
st.title("Synthetic Runs")
st.caption(
    "Read-only viewer for `synthetic_market_sim` runs. Lists every run "
    "directory and renders its artifacts. Nothing in `synthetic_market_sim/` "
    "is modified."
)

# Inside the container the host's outputs/ is mounted at /outputs.
# Outside the container we fall back to the repo path for local dev.
DEFAULT_PATHS = [
    Path("/outputs"),
    Path(os.environ.get("SYNTHETIC_OUTPUTS_DIR", "")),
    Path(__file__).resolve().parent.parent.parent.parent / "outputs",
]


def _find_outputs_root() -> Path | None:
    for p in DEFAULT_PATHS:
        if p and p.exists():
            return p
    return None


def _list_run_dirs(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for sub in ("runs", "ui_runs", ""):
        d = root / sub if sub else root
        if d.exists():
            for child in sorted(d.iterdir()):
                if child.is_dir() and (child / "manifest.json").exists():
                    candidates.append(child)
    # Dedupe while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def render() -> None:
    root = _find_outputs_root()
    if root is None:
        st.info(
            "No synthetic outputs directory found. Mount `./outputs:/outputs` "
            "into this container, or set the `SYNTHETIC_OUTPUTS_DIR` env var."
        )
        return

    runs = _list_run_dirs(root)
    if not runs:
        st.info(
            f"No synthetic runs found under `{root}`. "
            "Generate one via `synthetic_market_sim.wrappers.run_generic` or `run_manipulation`."
        )
        return

    st.markdown(f"**Found {len(runs)} runs under** `{root}`")
    selected = st.selectbox(
        "Pick a run",
        options=[str(p.relative_to(root)) for p in runs],
        index=0,
    )
    run_dir = root / selected

    # Manifest
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        import json
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        st.markdown("### Manifest")
        with st.expander("Full manifest JSON", expanded=False):
            st.json(manifest)

    # Counts
    cols = st.columns(4)
    for col, fname, label in zip(
        cols,
        ["orders.csv", "trades.csv", "scenarios.csv", "traders.csv"],
        ["Orders", "Trades", "Scenarios", "Traders"],
    ):
        f = run_dir / fname
        if f.exists():
            try:
                n = sum(1 for _ in f.open()) - 1
                col.metric(label, f"{n:,}")
            except Exception:
                col.metric(label, "?")
        else:
            col.metric(label, "—")

    # Common artifacts
    st.markdown("### Artifacts")
    artifacts = sorted(run_dir.glob("**/*"))
    rendered: set[str] = set()

    # Render the analysis_summary.md if present
    summary = run_dir / "analysis_summary.md"
    if summary.exists():
        st.markdown("#### `analysis_summary.md`")
        st.markdown(summary.read_text(encoding="utf-8"))
        rendered.add(str(summary))

    # Display PNG plots inline
    pngs = sorted(run_dir.glob("**/*.png"))
    if pngs:
        st.markdown("#### Plots")
        for png in pngs:
            with png.open("rb") as fh:
                st.image(fh.read(), caption=str(png.relative_to(run_dir)))
            rendered.add(str(png))

    # Show small CSVs
    st.markdown("#### Tabular outputs")
    for csv_path in run_dir.glob("**/*.csv"):
        if str(csv_path) in rendered:
            continue
        size = csv_path.stat().st_size
        if size > 1_000_000:
            st.caption(f"`{csv_path.relative_to(run_dir)}` ({size/1024:.0f} KB) - too large to preview inline")
            continue
        try:
            df = pd.read_csv(csv_path, nrows=200)
            with st.expander(f"`{csv_path.relative_to(run_dir)}` ({size/1024:.1f} KB, head 200)"):
                st.dataframe(df, hide_index=True, use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            st.caption(f"Could not read `{csv_path.relative_to(run_dir)}`: {exc!r}")


render()
