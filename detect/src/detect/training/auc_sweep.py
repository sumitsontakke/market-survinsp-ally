"""GNN architecture sweep — is CV AUC 0.536 a robust ceiling or task-bound?

The M3+ baseline reports cv_auc ≈ 0.5363 (barely above random) while trader-level
recall is 0.896-0.956. The two-layer-task explanation says edge-level AUC is
limited by the projection layer's `0.7*max + 0.3*top3` aggregation — the model
only needs the *peak* edge per trader to be informative, not the average. This
sweep tests that explanation against reasonable architecture variations.

Each variant trains end-to-end against the locked R01-R24 cohort + the M3+
balanced holdout, uses the cached graph pickles (no rebuild), reports:
  * cv_auc        — edge-level ROC AUC on the inner val split
  * cv_f1         — edge-level F1 at threshold 0.5
  * locked_*_recall — trader-level recall on the holdout (the production metric)
  * fit_seconds   — wall-clock for the model fit
  * epochs_used   — how many epochs trained before early-stop

The honest interpretation:
  * If cv_auc changes meaningfully across variants → architecture-bound,
    headroom for improvement.
  * If cv_auc stays near 0.54 across variants → task-bound, confirms the
    LIMITATIONS chapter's framing (the trader-projection is doing the work).

Run inside the trainer-gpu container:
    docker-compose run --rm trainer-gpu python -u /app/training/auc_sweep.py
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, "/app")

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from detect.dataset.loader import list_runs_in_cohort, load_run  # noqa: E402
from detect.dataset.splitter import iter_run_holdout_split  # noqa: E402
from detect.evaluate.locked_stress import evaluate_locked_stress  # noqa: E402
from detect.features.pyg_builder import build_graph_arrays  # noqa: E402
from detect.models.gnn_graphsage import (  # noqa: E402
    EdgeGraphSAGEModel, GraphSAGEConfig,
)
import pickle  # noqa: E402

COHORT_NAME = "PHASE1_R01_R24"
GRAPH_CACHE_DIR = Path("/app/outputs/_m3_graph_cache_v2")
OUT_JSON = "/app/outputs/_auc_sweep_results.json"
SEED = 42

HOLDOUT_RUNS = [
    "R03_clique_medium_medium_single_s17",
    "R07_clique_low_high_single_s31",
    "R09_ring_high_low_single_s41",
    "R11_ring_medium_medium_single_s47",
    "R17_mixed_high_low_single_s73",
    "R19_mixed_medium_medium_single_s83",
]


@dataclass(frozen=True)
class SweepVariant:
    """One row in the sweep table."""

    name: str
    hidden_dims: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.3
    aggregator: str = "mean"
    head_hidden: int = 64
    head_dropout: float = 0.2
    epochs: int = 200
    patience: int = 20
    lr: float = 1e-3
    rationale: str = ""


# Each variant tests one architectural dimension. Keeping holdout, optimizer,
# LR, and graph features fixed so changes are attributable.
VARIANTS: list[SweepVariant] = [
    SweepVariant(
        name="V0_baseline_m3p",
        hidden_dims=(256, 128, 64), dropout=0.3, aggregator="mean",
        rationale="The current M3+ champion. Reproducibility baseline.",
    ),
    SweepVariant(
        name="V1_wider",
        hidden_dims=(512, 256, 128), dropout=0.3, aggregator="mean",
        rationale="2x capacity — tests whether the model is under-fitting.",
    ),
    SweepVariant(
        name="V2_aggr_max",
        hidden_dims=(256, 128, 64), dropout=0.3, aggregator="max",
        rationale="Max aggregator — keeps peak-neighbor signal instead of "
                  "smoothing over all neighbors. Often helps with sparse "
                  "rare-positive tasks.",
    ),
    SweepVariant(
        name="V3_higher_dropout",
        hidden_dims=(256, 128, 64), dropout=0.5, aggregator="mean",
        rationale="Stronger regularization — tests whether the 0.3 baseline "
                  "is overfitting the inner train.",
    ),
    SweepVariant(
        name="V4_deeper",
        hidden_dims=(256, 256, 128, 64), dropout=0.3, aggregator="mean",
        rationale="4 SAGE layers instead of 3 — tests whether deeper "
                  "neighbourhood propagation helps catch layered manipulators.",
    ),
]


def _log(msg: str) -> None:
    print(msg, flush=True)


def _load_or_build_graph(run_path: Path, k: int = 12):
    GRAPH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pkl = GRAPH_CACHE_DIR / f"{run_path.name}.pkl"
    if pkl.exists():
        with pkl.open("rb") as fh:
            return pickle.load(fh)
    run = load_run(run_path)
    g = build_graph_arrays(run, sparsification="knn", k=k)
    with pkl.open("wb") as fh:
        pickle.dump(g, fh)
    return g


def _resolve_run_path(run_id: str) -> Path | None:
    for root in ("/app/outputs/calibrated_runs", "/app/outputs/runs"):
        cand = Path(root) / run_id
        if cand.exists():
            return cand
    return None


def main() -> int:
    _log("=" * 72)
    _log("GNN ARCHITECTURE SWEEP — does cv_auc move with reasonable tweaks?")
    _log("=" * 72)

    # Device probe
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            cap = torch.cuda.get_device_capability(0)
            _log(f"  device      : cuda (capability sm_{cap[0]}{cap[1]})")
        else:
            _log("  device      : cpu")
    except ImportError:
        _log("  torch NOT installed")
        return 1

    np.random.seed(SEED)
    import torch
    torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)

    # Build the dataset once — variants share train/val/holdout splits.
    cohort_paths = list_runs_in_cohort(COHORT_NAME)
    all_run_ids = [p.name for p in cohort_paths]
    train_ids, eval_ids = iter_run_holdout_split(all_run_ids, HOLDOUT_RUNS)
    _log(f"  cohort      : {COHORT_NAME}  "
         f"(train={len(train_ids)}, eval={len(eval_ids)})")

    _log("loading graph cache...")
    train_graphs = []
    eval_graphs = []
    for run_id in train_ids:
        p = _resolve_run_path(run_id)
        if p is not None:
            train_graphs.append(_load_or_build_graph(p))
    for run_id in eval_ids:
        p = _resolve_run_path(run_id)
        if p is not None:
            eval_graphs.append(_load_or_build_graph(p))
    _log(f"  graphs      : train={len(train_graphs)}  eval={len(eval_graphs)}")

    # Inner train/val split — fixed across variants so cv_auc is comparable.
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(train_graphs))
    val_cut = max(1, int(0.2 * len(train_graphs)))
    val_idx = set(perm[:val_cut].tolist())
    inner_train = [g for i, g in enumerate(train_graphs) if i not in val_idx]
    inner_val   = [g for i, g in enumerate(train_graphs) if i in val_idx]
    _log(f"  inner train : {len(inner_train)}   inner val : {len(inner_val)}")
    _log("")

    f_node = train_graphs[0].x.shape[1]
    f_edge = train_graphs[0].edge_attr.shape[1]

    results: list[dict] = []
    for variant in VARIANTS:
        _log("-" * 72)
        _log(f"=== {variant.name} ===")
        _log(f"  rationale   : {variant.rationale}")
        _log(f"  hidden_dims : {variant.hidden_dims}")
        _log(f"  aggregator  : {variant.aggregator}   dropout: {variant.dropout}")

        cfg = GraphSAGEConfig(
            node_in_dim=f_node, edge_in_dim=f_edge,
            hidden_dims=variant.hidden_dims,
            dropout=variant.dropout,
            aggregator=variant.aggregator,
            head_hidden=variant.head_hidden,
            head_dropout=variant.head_dropout,
        )
        model = EdgeGraphSAGEModel(
            cfg, lr=variant.lr, epochs=variant.epochs,
            early_stopping_patience=variant.patience,
            seed=SEED, device=device,
        )
        t_fit = time.perf_counter()
        try:
            model.fit(inner_train, val_graphs=inner_val)
        except Exception as exc:  # noqa: BLE001
            _log(f"  FIT FAILED  : {exc!r}")
            results.append({
                "name": variant.name, "error": repr(exc),
                "hidden_dims": list(variant.hidden_dims),
                "aggregator": variant.aggregator, "dropout": variant.dropout,
            })
            continue
        fit_secs = time.perf_counter() - t_fit
        epochs_used = len(model.epoch_losses)

        # CV AUC
        cv_probs = np.concatenate(
            [model.predict_proba(g) for g in inner_val]
            or [np.zeros(0, dtype=np.float32)]
        )
        cv_labels = np.concatenate(
            [g.y.astype(int) for g in inner_val]
            or [np.zeros(0, dtype=int)]
        )
        cv_auc = -1.0
        if cv_labels.size > 1 and len(np.unique(cv_labels)) > 1:
            cv_auc = float(roc_auc_score(cv_labels, cv_probs))

        # Holdout metrics
        eval_probs = [model.predict_proba(g) for g in eval_graphs]
        metrics = evaluate_locked_stress(
            train_graphs=train_graphs, eval_graphs=eval_graphs,
            cv_probs=cv_probs, cv_labels=cv_labels,
            eval_probs_per_graph=eval_probs,
            cv_auc=cv_auc,
            n_train_runs=len(train_graphs), n_eval_runs=len(eval_graphs),
            seed=SEED,
        )
        _log(f"  fit elapsed : {fit_secs:.2f}s   epochs used: {epochs_used}")
        _log(f"  cv_auc      : {cv_auc:.4f}")
        _log(f"  cv_f1       : {metrics.cv_f1:.4f}")
        _log(f"  locked      : clique={metrics.locked_clique_recall:.3f} "
             f"ring={metrics.locked_ring_recall:.3f} "
             f"mixed={metrics.locked_mixed_recall:.3f}")

        results.append({
            "name":         variant.name,
            "hidden_dims":  list(variant.hidden_dims),
            "aggregator":   variant.aggregator,
            "dropout":      variant.dropout,
            "rationale":    variant.rationale,
            "epochs_used":  epochs_used,
            "fit_seconds":  fit_secs,
            "cv_auc":       cv_auc,
            "cv_f1":        metrics.cv_f1,
            "locked_clique_recall": metrics.locked_clique_recall,
            "locked_ring_recall":   metrics.locked_ring_recall,
            "locked_mixed_recall":  metrics.locked_mixed_recall,
        })

    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_JSON).write_text(json.dumps({
        "schema_version": "1",
        "cohort": COHORT_NAME,
        "seed":   SEED,
        "holdout": HOLDOUT_RUNS,
        "variants": results,
    }, indent=2), encoding="utf-8")
    _log("")
    _log("=" * 72)
    _log("SWEEP COMPLETE")
    _log("=" * 72)
    _log("name                  cv_auc    cv_f1   clique    ring   mixed   fit_s")
    for r in results:
        if "error" in r:
            _log(f"  {r['name']:<22s}  ERROR: {r['error']}")
            continue
        _log(f"  {r['name']:<22s}  {r['cv_auc']:.4f}   {r['cv_f1']:.4f}  "
             f"{r['locked_clique_recall']:.3f}   {r['locked_ring_recall']:.3f}  "
             f"{r['locked_mixed_recall']:.3f}   {r['fit_seconds']:.2f}")
    _log(f"\n  results : {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
