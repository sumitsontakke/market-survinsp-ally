"""Seeded GNN architecture sweep — for the Phase-F write-up.

The single-seed `auc_sweep.py` answered "is cv_auc architecture-bound?" — it
isn't, and AUC and trader recall are inversely correlated across the variants.
But a single seed is a single seed. For the dissertation chapter we need mean ±
std across seeds so the trends are defensible against the "small-N" critique
the LIMITATIONS chapter raises about R3 itself.

Same five variants, same locked cohort and holdout, same cached graphs. The
seed is varied across {42, 11, 7, 23, 99} — five seeds × five variants = 25
fits. With the cached graphs and the early-stop on validation loss each fit
runs in 1-5 s on GPU, so total wall is ~2 minutes.

What changes per seed:
    * numpy / torch / torch.cuda RNG
    * inner train/val split (so the val set is different each seed)
    * GraphSAGE Glorot init + dropout mask order
    * `EdgeGraphSAGEModel(seed=...)` passed through to the model

What stays fixed across seeds:
    * cohort selection PHASE1_R01_R24
    * M3+ balanced holdout (the 6 named runs)
    * cached graphs (no rebuild between seeds — features deterministic)
    * variant architecture knobs

Reports per (variant, seed): cv_auc, cv_f1, per-family recall.
Reports per variant:        mean ± std on the above.

Run inside the trainer-gpu container:
    docker-compose run --rm trainer-gpu python -u /app/training/auc_sweep_seeds.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

# Allow override for non-docker runs (e.g. inside the sandbox where /app
# isn't mounted). Default to /app for container runs.
_APP_ROOT = os.environ.get("MSA_APP_ROOT", "/app")
sys.path.insert(0, _APP_ROOT)

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
GRAPH_CACHE_DIR = Path(_APP_ROOT) / "outputs/_m3_graph_cache_v2"
OUT_JSON = str(Path(_APP_ROOT) / "outputs/_auc_sweep_seeds_results.json")
OUT_JSONL = str(Path(_APP_ROOT) / "outputs/_auc_sweep_seeds_rows.jsonl")
_RUN_ROOTS = (
    str(Path(_APP_ROOT) / "outputs/calibrated_runs"),
    str(Path(_APP_ROOT) / "outputs/runs"),
)
SEEDS = [42, 11, 7, 23, 99]
# Wall budget per invocation. The sandbox kills detached processes when the
# shell exits, so the run is resumable: each fit is appended to OUT_JSONL,
# and the script skips rows already present. Re-launch to continue.
TIME_BUDGET_S = float(os.environ.get("MSA_SWEEP_BUDGET_S", "300"))

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
    name: str
    hidden_dims: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.3
    aggregator: str = "mean"
    head_hidden: int = 64
    head_dropout: float = 0.2
    epochs: int = 200
    patience: int = 20
    lr: float = 1e-3


VARIANTS: list[SweepVariant] = [
    SweepVariant(name="V0_baseline_m3p",
                 hidden_dims=(256, 128, 64), dropout=0.3, aggregator="mean"),
    SweepVariant(name="V1_wider",
                 hidden_dims=(512, 256, 128), dropout=0.3, aggregator="mean"),
    SweepVariant(name="V2_aggr_max",
                 hidden_dims=(256, 128, 64), dropout=0.3, aggregator="max"),
    SweepVariant(name="V3_higher_dropout",
                 hidden_dims=(256, 128, 64), dropout=0.5, aggregator="mean"),
    SweepVariant(name="V4_deeper",
                 hidden_dims=(256, 256, 128, 64), dropout=0.3, aggregator="mean"),
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
    for root in _RUN_ROOTS:
        cand = Path(root) / run_id
        if cand.exists():
            return cand
    return None


def _split_inner(train_graphs, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_graphs))
    val_cut = max(1, int(0.2 * len(train_graphs)))
    val_idx = set(perm[:val_cut].tolist())
    inner_train = [g for i, g in enumerate(train_graphs) if i not in val_idx]
    inner_val = [g for i, g in enumerate(train_graphs) if i in val_idx]
    return inner_train, inner_val


def _run_one(variant: SweepVariant, seed: int, train_graphs, eval_graphs,
             device: str, f_node: int, f_edge: int) -> dict:
    """One (variant × seed) fit. Returns a row dict including all metrics."""
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    inner_train, inner_val = _split_inner(train_graphs, seed=seed)

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
        seed=seed, device=device,
    )
    t_fit = time.perf_counter()
    try:
        model.fit(inner_train, val_graphs=inner_val)
    except Exception as exc:  # noqa: BLE001
        return {
            "variant": variant.name, "seed": seed,
            "error": repr(exc),
        }
    fit_secs = time.perf_counter() - t_fit
    epochs_used = len(model.epoch_losses)

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

    eval_probs = [model.predict_proba(g) for g in eval_graphs]
    metrics = evaluate_locked_stress(
        train_graphs=train_graphs, eval_graphs=eval_graphs,
        cv_probs=cv_probs, cv_labels=cv_labels,
        eval_probs_per_graph=eval_probs,
        cv_auc=cv_auc,
        n_train_runs=len(train_graphs), n_eval_runs=len(eval_graphs),
        seed=seed,
    )
    return {
        "variant":               variant.name,
        "seed":                  seed,
        "epochs_used":           epochs_used,
        "fit_seconds":           fit_secs,
        "cv_auc":                cv_auc,
        "cv_f1":                 metrics.cv_f1,
        "locked_clique_recall":  metrics.locked_clique_recall,
        "locked_ring_recall":    metrics.locked_ring_recall,
        "locked_mixed_recall":   metrics.locked_mixed_recall,
    }


def _agg(rows: list[dict], key: str) -> tuple[float, float]:
    vals = [r[key] for r in rows if "error" not in r and key in r]
    if not vals:
        return (float("nan"), float("nan"))
    if len(vals) == 1:
        return (float(vals[0]), 0.0)
    return (mean(vals), pstdev(vals))


def main() -> int:
    _log("=" * 78)
    _log("SEEDED GNN ARCHITECTURE SWEEP — 5 variants × 5 seeds")
    _log("=" * 78)

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

    cohort_paths = list_runs_in_cohort(COHORT_NAME)
    all_run_ids = [p.name for p in cohort_paths]
    train_ids, eval_ids = iter_run_holdout_split(all_run_ids, HOLDOUT_RUNS)
    _log(f"  cohort      : {COHORT_NAME}  "
         f"(train={len(train_ids)}, eval={len(eval_ids)})")
    _log(f"  seeds       : {SEEDS}")
    _log(f"  variants    : {[v.name for v in VARIANTS]}")

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
    f_node = train_graphs[0].x.shape[1]
    f_edge = train_graphs[0].edge_attr.shape[1]

    # Load already-completed (variant, seed) pairs from JSONL.
    done: set[tuple[str, int]] = set()
    all_rows: list[dict] = []
    if Path(OUT_JSONL).exists():
        with open(OUT_JSONL, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in r:
                    continue
                done.add((r.get("variant", ""), int(r.get("seed", -1))))
                all_rows.append(r)
        _log(f"  resume      : {len(done)} (variant, seed) already done in JSONL")

    Path(OUT_JSONL).parent.mkdir(parents=True, exist_ok=True)
    jsonl_fh = open(OUT_JSONL, "a", encoding="utf-8")

    t0 = time.perf_counter()
    budget_hit = False
    for variant in VARIANTS:
        if budget_hit:
            break
        _log("-" * 78)
        _log(f"=== {variant.name} ===  "
             f"hidden={variant.hidden_dims} aggr={variant.aggregator} "
             f"dropout={variant.dropout}")
        variant_rows: list[dict] = [r for r in all_rows
                                    if r["variant"] == variant.name]
        for seed in SEEDS:
            if (variant.name, seed) in done:
                _log(f"  seed={seed:3d}   (cached)")
                continue
            elapsed = time.perf_counter() - t0
            if elapsed > TIME_BUDGET_S:
                _log(f"  budget hit ({elapsed:.1f}s > {TIME_BUDGET_S}s) — "
                     f"stopping; resume by re-launching")
                budget_hit = True
                break
            row = _run_one(variant, seed, train_graphs, eval_graphs,
                           device, f_node, f_edge)
            all_rows.append(row)
            variant_rows.append(row)
            jsonl_fh.write(json.dumps(row) + "\n")
            jsonl_fh.flush()
            if "error" in row:
                _log(f"  seed={seed:3d}   ERROR: {row['error']}")
                continue
            _log(f"  seed={seed:3d}   "
                 f"auc={row['cv_auc']:.4f}  f1={row['cv_f1']:.4f}  "
                 f"clique={row['locked_clique_recall']:.3f}  "
                 f"ring={row['locked_ring_recall']:.3f}  "
                 f"mixed={row['locked_mixed_recall']:.3f}  "
                 f"({row['fit_seconds']:.1f}s, {row['epochs_used']}e)")
        auc_m, auc_s = _agg(variant_rows, "cv_auc")
        f1_m, f1_s = _agg(variant_rows, "cv_f1")
        cli_m, cli_s = _agg(variant_rows, "locked_clique_recall")
        rng_m, rng_s = _agg(variant_rows, "locked_ring_recall")
        mix_m, mix_s = _agg(variant_rows, "locked_mixed_recall")
        _log(f"  MEAN±STD : auc={auc_m:.4f}±{auc_s:.4f}  "
             f"f1={f1_m:.4f}±{f1_s:.4f}")
        _log(f"           : clique={cli_m:.3f}±{cli_s:.3f}  "
             f"ring={rng_m:.3f}±{rng_s:.3f}  "
             f"mixed={mix_m:.3f}±{mix_s:.3f}")

    total_secs = time.perf_counter() - t0
    jsonl_fh.close()

    # Aggregate per variant for the JSON
    summary: list[dict] = []
    for variant in VARIANTS:
        rows = [r for r in all_rows if r["variant"] == variant.name]
        auc_m, auc_s = _agg(rows, "cv_auc")
        f1_m, f1_s = _agg(rows, "cv_f1")
        cli_m, cli_s = _agg(rows, "locked_clique_recall")
        rng_m, rng_s = _agg(rows, "locked_ring_recall")
        mix_m, mix_s = _agg(rows, "locked_mixed_recall")
        summary.append({
            "variant":       variant.name,
            "hidden_dims":   list(variant.hidden_dims),
            "aggregator":    variant.aggregator,
            "dropout":       variant.dropout,
            "n_seeds":       sum(1 for r in rows if "error" not in r),
            "cv_auc_mean":   auc_m, "cv_auc_std":   auc_s,
            "cv_f1_mean":    f1_m,  "cv_f1_std":    f1_s,
            "locked_clique_recall_mean": cli_m, "locked_clique_recall_std": cli_s,
            "locked_ring_recall_mean":   rng_m, "locked_ring_recall_std":   rng_s,
            "locked_mixed_recall_mean":  mix_m, "locked_mixed_recall_std":  mix_s,
            "mean_recall_overall_mean":  (cli_m + rng_m + mix_m) / 3.0,
        })

    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_JSON).write_text(json.dumps({
        "schema_version": "1",
        "cohort":         COHORT_NAME,
        "seeds":          SEEDS,
        "holdout":        HOLDOUT_RUNS,
        "rows":           all_rows,
        "summary":        summary,
        "total_seconds":  total_secs,
    }, indent=2), encoding="utf-8")

    _log("")
    _log("=" * 78)
    _log("SWEEP COMPLETE — mean ± std table")
    _log("=" * 78)
    _log(f"{'variant':<22s} {'cv_auc':<14s} {'clique':<12s} {'ring':<12s} {'mixed':<12s} {'mean_R':<8s}")
    for s in summary:
        _log(f"  {s['variant']:<20s} "
             f"{s['cv_auc_mean']:.4f}±{s['cv_auc_std']:.4f}  "
             f"{s['locked_clique_recall_mean']:.3f}±{s['locked_clique_recall_std']:.3f}  "
             f"{s['locked_ring_recall_mean']:.3f}±{s['locked_ring_recall_std']:.3f}  "
             f"{s['locked_mixed_recall_mean']:.3f}±{s['locked_mixed_recall_std']:.3f}  "
             f"{s['mean_recall_overall_mean']:.3f}")
    _log(f"\n  total wall  : {total_secs:.1f}s   results: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
