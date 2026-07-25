"""Exhaustive Phase F sweep — K-fold CV on the 5000-trader cohort.

User asks (2026-05-17):
  * Skip CPU vs GPU comparison (GPU established as ~53x).
  * Exhaustive training + validation with multiple runs.
  * 10x data volume → 5000 traders per run.
  * Measure purity (precision-side) of detected positives.
  * Architecture sweep: V2 (max aggregator) + V4 (deeper).

Design:
  * Reads outputs/scaled_5000_cohort/ — 24 runs at 5000 traders each.
  * K=5 folds across the 24 runs (each fold ≈ 5 holdout / 19 train).
  * Per (variant, fold): train 3 seeds for within-fold variance.
  * Per fit: register via training.registry.save_experiment so the
    Metric Timeline page picks up every result automatically.
  * Reports mean ± std across folds & seeds for each variant on:
      cv_auc, locked_*_recall, locked_*_purity (Phase F-new).

Total work: 2 variants × 5 folds × 3 seeds = 30 fits.
On GPU (sm_120): ~5-8 min per fit at N=5000 → ~3-4 hours total.

Resumable: each completed fit is appended to outputs/_kfold_5000_rows.jsonl
and skipped on re-launch. Re-run as many times as needed.

Run inside the trainer-gpu container:
    docker compose run --rm trainer-gpu \\
        python -u /app/training/kfold_5000_sweep.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

_APP_ROOT = os.environ.get("MSA_APP_ROOT", "/app")
sys.path.insert(0, _APP_ROOT)

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from detect.dataset.loader import load_run  # noqa: E402
from detect.evaluate.locked_stress import evaluate_locked_stress  # noqa: E402
from detect.evaluate.metrics import NOT_COMPUTED  # noqa: E402
from detect.features.pyg_builder import build_graph_arrays  # noqa: E402
from detect.models.gnn_graphsage import (  # noqa: E402
    EdgeGraphSAGEModel, GraphSAGEConfig,
)
from detect.registry import ExperimentConfig, save_experiment  # noqa: E402

COHORT_DIR = Path(_APP_ROOT) / "outputs/scaled_5000_cohort"
GRAPH_CACHE_DIR = Path(_APP_ROOT) / "outputs/_kfold_5000_graph_cache"
OUT_JSONL = Path(_APP_ROOT) / "outputs/_kfold_5000_rows.jsonl"
OUT_SUMMARY = Path(_APP_ROOT) / "outputs/_kfold_5000_summary.json"
N_FOLDS = 5
SEEDS_PER_FOLD = (42, 11, 7)


@dataclass(frozen=True)
class SweepVariant:
    name: str
    hidden_dims: tuple[int, ...]
    dropout: float
    aggregator: str
    head_hidden: int = 64
    head_dropout: float = 0.2
    epochs: int = 200
    patience: int = 20
    lr: float = 1e-3
    rationale: str = ""


# Sweep variants the user picked (no V0 baseline, no V1 wider).
VARIANTS: list[SweepVariant] = [
    SweepVariant(
        name="V2_aggr_max",
        hidden_dims=(256, 128, 64), dropout=0.3, aggregator="max",
        rationale=("Max aggregator composes cleanly with the trader-projection's "
                   "0.7*max + 0.3*top3 inductive bias. Best recall in the "
                   "single-seed sweep at 500 traders; verify holds at 5000."),
    ),
    SweepVariant(
        name="V4_deeper",
        hidden_dims=(256, 256, 128, 64), dropout=0.3, aggregator="mean",
        rationale=("4 SAGE layers. Killed recall at 500 traders (graph diameter "
                   "≈ 3 → over-smoothing). 5000-trader graphs are denser and "
                   "may tolerate the extra depth — testing the hypothesis."),
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


def _family_of(name: str) -> str:
    n = name.lower()
    if "clique" in n:
        return "clique"
    if "ring" in n:
        return "ring"
    if "mixed" in n:
        return "mixed"
    return "unknown"


def _make_folds(run_names: list[str], k: int, seed: int = 42
                ) -> list[tuple[list[str], list[str]]]:
    """Family-stratified K-fold split. Each fold's eval set draws roughly
    proportionally from clique/ring/mixed so per-family recall is always
    measurable.
    """
    families: dict[str, list[str]] = {"clique": [], "ring": [], "mixed": []}
    for name in run_names:
        fam = _family_of(name)
        families.setdefault(fam, []).append(name)

    rng = np.random.default_rng(seed)
    for fam in families:
        order = list(families[fam])
        rng.shuffle(order)
        families[fam] = order

    folds: list[tuple[list[str], list[str]]] = []
    for fi in range(k):
        eval_runs: list[str] = []
        for fam, members in families.items():
            for j, name in enumerate(members):
                if j % k == fi:
                    eval_runs.append(name)
        train_runs = [r for r in run_names if r not in set(eval_runs)]
        folds.append((train_runs, eval_runs))
    return folds


def _split_inner(train_graphs, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_graphs))
    val_cut = max(1, int(0.2 * len(train_graphs)))
    val_idx = set(perm[:val_cut].tolist())
    inner_train = [g for i, g in enumerate(train_graphs) if i not in val_idx]
    inner_val = [g for i, g in enumerate(train_graphs) if i in val_idx]
    return inner_train, inner_val


def _experiment_id(variant: SweepVariant, fold: int, seed: int) -> str:
    return f"kfold_5000_{variant.name}__fold{fold}__seed{seed}"


def _build_config(variant: SweepVariant, fold: int, seed: int,
                  train_runs: list[str], eval_runs: list[str]
                  ) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id=_experiment_id(variant, fold, seed),
        seed=seed,
        model={
            "family":      "gnn",
            "kind":        "edge_graphsage",
            "hidden_dims": list(variant.hidden_dims),
            "dropout":     variant.dropout,
            "aggregator":  variant.aggregator,
            "head_hidden": variant.head_hidden,
            "head_dropout": variant.head_dropout,
        },
        features={"sparsification": "knn", "k": 12},
        data={
            "cohort": "SCALED_5000_R01_R24",
            "runs":   list(train_runs) + list(eval_runs),
            "n_traders_per_run": 5000,
            "source": str(COHORT_DIR),
        },
        calibration={"alpha_source": "calibrated_nse",
                     "target_manip_ratio": 0.0010},
        split={
            "policy":     "kfold_family_stratified",
            "k_folds":    N_FOLDS,
            "fold":       fold,
            "train_runs": train_runs,
            "eval_runs":  eval_runs,
        },
        loss={"type": "binary_cross_entropy"},
        evaluation={
            "edge_threshold_grid": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "projection":         "trader_score = 0.7*max + 0.3*top3_mean",
            "report":             ["cv_auc", "cv_f1", "locked_*_recall",
                                   "locked_*_purity", "locked_benign_alarm"],
        },
        training={
            "lr": variant.lr,
            "epochs": variant.epochs,
            "early_stopping_patience": variant.patience,
            "rationale": variant.rationale,
        },
    )


def _run_one(variant: SweepVariant, fold: int, seed: int,
             train_graphs, eval_graphs, train_runs, eval_runs,
             device: str, f_node: int, f_edge: int) -> dict:
    """One (variant, fold, seed) fit. Trains, evaluates, registers."""
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
            "variant": variant.name, "fold": fold, "seed": seed,
            "error":   repr(exc),
        }
    fit_secs = time.perf_counter() - t_fit

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

    # Register in the model registry so Metric Timeline picks this up.
    exp_cfg = _build_config(variant, fold, seed, train_runs, eval_runs)
    try:
        folder = save_experiment(
            exp_cfg, metrics,
            notes=(f"K-fold 5000-cohort sweep — variant={variant.name} "
                   f"fold={fold}/{N_FOLDS} seed={seed}\n"
                   f"Rationale: {variant.rationale}\n"
                   f"Fit elapsed: {fit_secs:.1f}s   epochs: "
                   f"{len(model.epoch_losses)}"),
        )
        registered_at = folder.name
    except Exception as exc:  # noqa: BLE001
        registered_at = f"register_failed: {exc!r}"

    return {
        "variant":             variant.name,
        "fold":                fold,
        "seed":                seed,
        "registered_at":       registered_at,
        "fit_seconds":         fit_secs,
        "epochs_used":         len(model.epoch_losses),
        "cv_auc":              cv_auc,
        "cv_f1":               metrics.cv_f1,
        "cv_precision":        metrics.cv_precision,
        "cv_recall":           metrics.cv_recall,
        "locked_clique_recall": metrics.locked_clique_recall,
        "locked_ring_recall":   metrics.locked_ring_recall,
        "locked_mixed_recall":  metrics.locked_mixed_recall,
        "locked_clique_purity": metrics.locked_clique_purity,
        "locked_ring_purity":   metrics.locked_ring_purity,
        "locked_mixed_purity":  metrics.locked_mixed_purity,
        "locked_benign_alarm":  metrics.locked_benign_alarm,
        "n_train_runs":         metrics.n_train_runs,
        "n_eval_runs":          metrics.n_eval_runs,
    }


def _agg(rows: list[dict], key: str) -> tuple[float, float, int]:
    vals = [r[key] for r in rows
            if "error" not in r and r.get(key, NOT_COMPUTED) > NOT_COMPUTED]
    if not vals:
        return (float("nan"), float("nan"), 0)
    if len(vals) == 1:
        return (float(vals[0]), 0.0, 1)
    return (mean(vals), pstdev(vals), len(vals))


def main() -> int:
    _log("=" * 78)
    _log("EXHAUSTIVE K-FOLD SWEEP — 5000-trader cohort, V2 + V4")
    _log("=" * 78)

    if not COHORT_DIR.is_dir():
        _log(f"FATAL: cohort dir not found: {COHORT_DIR}")
        _log(f"  generate it first via scripts/regen_5000_cohort.py")
        return 2

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        cap = torch.cuda.get_device_capability(0)
        _log(f"  device      : cuda sm_{cap[0]}{cap[1]}")
    else:
        _log("  device      : cpu (no GPU detected — this will be very slow at N=5000)")

    run_dirs = [p for p in sorted(COHORT_DIR.iterdir())
                if p.is_dir() and p.name.startswith("SCALED_5000_")]
    run_names = [p.name for p in run_dirs]
    _log(f"  cohort runs : {len(run_names)} at {COHORT_DIR}")
    if len(run_names) < N_FOLDS:
        _log(f"FATAL: need ≥ {N_FOLDS} runs, found {len(run_names)}")
        return 3

    folds = _make_folds(run_names, k=N_FOLDS, seed=42)
    for fi, (tr, ev) in enumerate(folds):
        _log(f"  fold {fi}     : train={len(tr)}  eval={len(ev)}  "
             f"(eval families: "
             f"{ {f: sum(1 for r in ev if _family_of(r)==f) for f in ('clique','ring','mixed')} })")

    _log("loading graph cache...")
    all_graphs: dict[str, object] = {}
    for d in run_dirs:
        all_graphs[d.name] = _load_or_build_graph(d)
    _log(f"  graphs      : {len(all_graphs)} cached/loaded")

    # Pick any one to read tensor shape.
    first = next(iter(all_graphs.values()))
    f_node = first.x.shape[1]
    f_edge = first.edge_attr.shape[1]
    _log(f"  features    : node_in={f_node}  edge_in={f_edge}")

    # Resume tracking.
    done: set[tuple[str, int, int]] = set()
    all_rows: list[dict] = []
    if OUT_JSONL.exists():
        with OUT_JSONL.open(encoding="utf-8") as fh:
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
                done.add((r.get("variant", ""), int(r.get("fold", -1)),
                          int(r.get("seed", -1))))
                all_rows.append(r)
        _log(f"  resume      : {len(done)} (variant, fold, seed) already done")
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    jsonl_fh = OUT_JSONL.open("a", encoding="utf-8")

    t0 = time.perf_counter()
    for variant in VARIANTS:
        _log("-" * 78)
        _log(f"=== {variant.name} ===  hidden={variant.hidden_dims} "
             f"aggr={variant.aggregator} dropout={variant.dropout}")
        for fi, (train_runs, eval_runs) in enumerate(folds):
            train_graphs = [all_graphs[r] for r in train_runs]
            eval_graphs = [all_graphs[r] for r in eval_runs]
            for seed in SEEDS_PER_FOLD:
                key = (variant.name, fi, seed)
                if key in done:
                    _log(f"  fold {fi} seed {seed:3d}: cached")
                    continue
                row = _run_one(variant, fi, seed, train_graphs, eval_graphs,
                               train_runs, eval_runs, device, f_node, f_edge)
                all_rows.append(row)
                jsonl_fh.write(json.dumps(row) + "\n")
                jsonl_fh.flush()
                if "error" in row:
                    _log(f"  fold {fi} seed {seed:3d}: ERROR {row['error']}")
                    continue
                _log(f"  fold {fi} seed {seed:3d}: "
                     f"auc={row['cv_auc']:.4f}  "
                     f"R clique={row['locked_clique_recall']:.3f} "
                     f"ring={row['locked_ring_recall']:.3f} "
                     f"mixed={row['locked_mixed_recall']:.3f}  | "
                     f"P clique={row['locked_clique_purity']:.3f} "
                     f"ring={row['locked_ring_purity']:.3f} "
                     f"mixed={row['locked_mixed_purity']:.3f}  "
                     f"({row['fit_seconds']:.1f}s)")
    jsonl_fh.close()
    total_secs = time.perf_counter() - t0

    # ---- Summary per variant (mean ± std across folds × seeds) -----
    summary: list[dict] = []
    for variant in VARIANTS:
        v_rows = [r for r in all_rows if r["variant"] == variant.name]
        auc_m, auc_s, n_auc = _agg(v_rows, "cv_auc")
        f1_m, f1_s, _ = _agg(v_rows, "cv_f1")
        cl_r_m, cl_r_s, _ = _agg(v_rows, "locked_clique_recall")
        rg_r_m, rg_r_s, _ = _agg(v_rows, "locked_ring_recall")
        mx_r_m, mx_r_s, _ = _agg(v_rows, "locked_mixed_recall")
        cl_p_m, cl_p_s, _ = _agg(v_rows, "locked_clique_purity")
        rg_p_m, rg_p_s, _ = _agg(v_rows, "locked_ring_purity")
        mx_p_m, mx_p_s, _ = _agg(v_rows, "locked_mixed_purity")
        ba_m, ba_s, _ = _agg(v_rows, "locked_benign_alarm")
        summary.append({
            "variant":      variant.name,
            "hidden_dims":  list(variant.hidden_dims),
            "aggregator":   variant.aggregator,
            "dropout":      variant.dropout,
            "n_fits":       n_auc,
            "cv_auc_mean":  auc_m, "cv_auc_std":  auc_s,
            "cv_f1_mean":   f1_m,  "cv_f1_std":   f1_s,
            "locked_clique_recall_mean": cl_r_m, "locked_clique_recall_std": cl_r_s,
            "locked_ring_recall_mean":   rg_r_m, "locked_ring_recall_std":   rg_r_s,
            "locked_mixed_recall_mean":  mx_r_m, "locked_mixed_recall_std":  mx_r_s,
            "locked_clique_purity_mean": cl_p_m, "locked_clique_purity_std": cl_p_s,
            "locked_ring_purity_mean":   rg_p_m, "locked_ring_purity_std":   rg_p_s,
            "locked_mixed_purity_mean":  mx_p_m, "locked_mixed_purity_std":  mx_p_s,
            "locked_benign_alarm_mean":  ba_m,   "locked_benign_alarm_std":  ba_s,
        })

    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.write_text(json.dumps({
        "schema_version": "1",
        "cohort":         "SCALED_5000_R01_R24",
        "n_runs_total":   len(run_names),
        "k_folds":        N_FOLDS,
        "seeds_per_fold": list(SEEDS_PER_FOLD),
        "summary":        summary,
        "total_seconds":  total_secs,
    }, indent=2), encoding="utf-8")

    _log("")
    _log("=" * 78)
    _log("SWEEP COMPLETE")
    _log("=" * 78)
    _log(f"{'variant':<18s} {'n':>3s}  {'cv_auc':<14s}  "
         f"{'R_clique':<14s} {'R_ring':<14s} {'R_mixed':<14s}  "
         f"{'P_clique':<14s} {'P_ring':<14s} {'P_mixed':<14s}")
    for s in summary:
        _log(f"  {s['variant']:<16s} {s['n_fits']:>3d}  "
             f"{s['cv_auc_mean']:.4f}+/-{s['cv_auc_std']:.4f}  "
             f"{s['locked_clique_recall_mean']:.3f}+/-{s['locked_clique_recall_std']:.3f} "
             f"{s['locked_ring_recall_mean']:.3f}+/-{s['locked_ring_recall_std']:.3f} "
             f"{s['locked_mixed_recall_mean']:.3f}+/-{s['locked_mixed_recall_std']:.3f}  "
             f"{s['locked_clique_purity_mean']:.3f}+/-{s['locked_clique_purity_std']:.3f} "
             f"{s['locked_ring_purity_mean']:.3f}+/-{s['locked_ring_purity_std']:.3f} "
             f"{s['locked_mixed_purity_mean']:.3f}+/-{s['locked_mixed_purity_std']:.3f}")
    _log(f"  total wall : {total_secs:.1f}s")
    _log(f"  rows JSONL : {OUT_JSONL}")
    _log(f"  summary    : {OUT_SUMMARY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
