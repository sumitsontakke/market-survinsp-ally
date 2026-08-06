# Reproducing the paper cohorts

> This document describes the full-cohort reproduction path. For a
> 5-minute smoke test on a small demo cohort, see the `Quick start`
> section in the top-level [`README.md`](../README.md).

The two conference papers depend on three cohort families:

1. **Within-generator OOD cohort** — 240 trading days, 50 runs, held-out
   seeds and manipulation-parameter ranges. Used for the AUC 0.638 →
   0.968 → 0.984 headline table (v1 baseline → tier-2 GBM → v4 GraphSAGE).
2. **Cross-generator ABIDES cohort** — 15 runs from the ABIDES agent-based
   simulator using calibrated NSE-like microstructure. Used for the
   cross-generator column (v1 chance → v4 0.842 AUC).
3. **Family-disjoint leave-one-out cohorts** — three retrains each
   holding out one of {clique, ring, front-account}. Used for the
   architectural-finding chart (tier-2 holds, v4 collapses).

## Prerequisites

- Python 3.11 (matched to CI). 3.10 and 3.12 also work.
- ~40 GB free disk for full cohort regeneration.
- Recommended: NVIDIA GPU with ≥8 GB VRAM for detector training. CPU
  training works but takes ~10× longer.
- Docker (only needed for GPU-in-container training, otherwise optional).

## Fresh clone → smoke run

```bash
git clone https://github.com/sumitsontakke/market-survinsp-ally.git
cd market-survinsp-ally

# Install both modules editable
pip install -e ./synth -e ./detect

# Small demo cohort (~5 MB, ~15 min on CPU, ~5 min on GPU)
make reproduce
```

The `make reproduce` target runs, in order:

```bash
python -m synth.generate  --config configs/synth/demo_cohort.yaml --out cohorts/demo
python -m detect.train    --cohort cohorts/demo --model tier2
python -m detect.evaluate --cohort cohorts/demo --model tier2 --out reports/demo.json
```

The final `reports/demo.json` should show a per-family AUC in the
0.90–0.99 range for the demo cohort. This is not the paper number
(which uses the much larger OOD cohort) but it demonstrates the
pipeline works end-to-end from a clean clone.

## Full within-generator OOD cohort (paper table row 1)

```bash
# Regenerate 240 days × 50 runs (~10 GB, ~4 hours on CPU)
python -m synth.generate --config configs/synth/ood_cohort.yaml --out cohorts/ood

# Train v1 baseline (2-feature GraphSAGE)
python -m detect.train --cohort cohorts/ood --model graphsage_v1 --gpu

# Train v4 (8-feature augmented GraphSAGE)
python -m detect.train --cohort cohorts/ood --model graphsage_v4 --gpu

# Train tier-2 GBM (uses v1 score + 6 engineered features)
python -m detect.train --cohort cohorts/ood --model tier2_gbm

# Evaluate all three
python -m detect.evaluate --cohort cohorts/ood --model graphsage_v1 --out reports/v1.json
python -m detect.evaluate --cohort cohorts/ood --model tier2_gbm --out reports/tier2.json
python -m detect.evaluate --cohort cohorts/ood --model graphsage_v4 --out reports/v4.json
```

Numbers should reproduce the paper's within-generator column to within
~0.01 AUC (differences are seed-dependent).

## Cross-generator ABIDES cohort (paper table row 2)

```bash
# Regenerate 15 ABIDES runs (~3 hours on CPU, ABIDES is single-threaded)
python -m synth.abides.run_cohort --config configs/synth/abides_cohort.yaml \
    --out cohorts/abides

# Evaluate the models you already trained on the ABIDES cohort
python -m detect.evaluate --cohort cohorts/abides --model graphsage_v1 --out reports/v1_abides.json
python -m detect.evaluate --cohort cohorts/abides --model graphsage_v4 --out reports/v4_abides.json
```

## Family-disjoint leave-one-out (paper table row 3)

Three retrains, one per held-out family:

```bash
for family in clique ring front_account; do
    MSA_PHASE_G_HOLDOUT_FAMILY=$family \
        python -m detect.train --cohort cohorts/ood --model graphsage_v4 --gpu \
        --out models/v4_no_${family}.pt

    python -m detect.evaluate --cohort cohorts/ood --model graphsage_v4 \
        --checkpoint models/v4_no_${family}.pt --held-out-family $family \
        --out reports/v4_no_${family}.json
done
```

Do the same with `--model tier2_gbm` (much faster, CPU-only).

## Getting the trained checkpoints without retraining

**Status as of v0.2.0: not available yet.** Trained model checkpoints are
planned for the v0.3.0 release, at which point they will be archived on
Zenodo with SHA-256-verified downloads. See
[`docs/CHECKPOINTS.md`](CHECKPOINTS.md) for the intended layout and the
`docs/v0.2.0_MILESTONE.md` § "Deferred to later" section for the roadmap.

Until then, retraining from the cohort takes ~2 hours on a single GPU for
v4 and ~30 minutes for the tier-2 GBM — see the commands above.

## Troubleshooting

- **CUDA out of memory during v4 training** — reduce batch size in the
  training config, or fall back to CPU with `--gpu` omitted.
- **ABIDES cohort generation hangs** — ABIDES uses a single-threaded
  event loop; expect ~12 minutes per run. Not a hang, just slow.
- **Numbers differ by >0.02 AUC from the paper** — seed sensitivity. Try
  averaging over 5 seeds via `--seeds 42,43,44,45,46`.
- **`synth validate` reports missing SCHEMA-required columns** — expected
  in v0.2.0; the validator's strict column set doesn't yet match the
  v0.1.0 generator's actual output. `make reproduce` still completes
  end-to-end. Reconciliation is a v0.3.0 task.
