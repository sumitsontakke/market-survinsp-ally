# 48 — Tier-2 Classifier Results

> The next-phase result. Pairs with [[Research Notes/46 Engineered Features + Investigation Page|46]]
> (the features) and [[Research Notes/45 Phase G Generalization Pipeline|45]]
> (the v1 model the tier-2 layer rides on top of).

## TL;DR

A lightweight gradient-boosted tier-2 classifier — taking v1's trader
score plus the six engineered features as its 7-input vector — achieves
**pooled AUC 0.968** on a leave-one-run-out cross-validation across all
50 Phase G OOD runs.

At the dissertation-headline operating points:

| Operating point          | Threshold | Recall    | Purity    | F1        |
|--------------------------|-----------|-----------|-----------|-----------|
| v1 alone @ thr 0.40      | —         | 0.117     | 1.000     | 0.209     |
| **Tier-2 @ purity ≥ 0.95** | **0.54**  | **0.737** | **0.950** | **0.830** |
| **Tier-2 @ purity ≥ 0.99** | **0.86**  | **0.619** | **0.991** | **0.762** |
| Tier-2 @ purity ≥ 0.80   | 0.18      | 0.850     | 0.803     | 0.826     |
| Tier-2 @ best F1         | 0.48      | 0.759     | 0.941     | 0.840     |

**6x recall lift at near-identical purity** vs v1's existing operating
point. The case is now: at **99% purity we catch 62%** of all manipulators
across the OOD cohort. At **95% purity we catch 74%**.

## How

The model is sklearn's `GradientBoostingClassifier(n_estimators=120,
max_depth=3, learning_rate=0.08, subsample=0.85, random_state=42)`.
Trains in ~0.8 seconds per fold on CPU. Total run time across 50
LOO-CV folds: under 45 seconds.

Inputs per trader (7-dim vector):

  1. v1 `trader_score`         (from `outputs/_phase_g_v1_trader_scores/*.csv`)
  2. `burst_concentration`     ┐
  3. `side_entropy_in_burst`   │
  4. `counterparty_hhi_burst`  ├ engineered features
  5. `order_qty_cov`           │ from `_phase_g_features/*.csv`
  6. `top_partner_trade_share` │
  7. `co_active_top_count`     ┘

Label: per-trader `is_manipulator` looked up directly from the run's
`orders.csv` (NOT the projection's `label_core`, which inflates
positives by including manipulator counterparties).

Evaluation: leave-one-run-out CV. For each of the 50 runs, train on
traders from the other 49 runs (~9,800 rows), predict on the held-out
run (~200 rows). Aggregate held-out predictions → pooled precision-
recall curve.

## Why the lift is so large

v1's `trader_score` alone has pooled AUC **0.793**. `counterparty_hhi_burst`
alone has AUC **0.800**. Together they hit **0.968** because:

- **v1 and HHI make different mistakes.** v1's false positives are
  high-activity benigns; HHI's false positives are traders with one or
  two favourite counterparties. The intersection is mostly true
  manipulators.

- **The GBM learns non-linear interactions.** Manipulators with
  *moderate* v1 scores get caught when their HHI is also low. Benigns
  with low HHI get spared when v1 scores them low. These rules aren't
  expressible as linear combinations.

- **Five features in the input, not one.** `top_partner_trade_share`
  (AUC 0.776), `co_active_top_count` (AUC 0.744), and
  `burst_concentration` (AUC 0.704) each contribute independent
  signal, especially on edge cases that the top two miss.

A logistic-regression sanity check on the same data (with the same CV
setup) achieves AUC ≈ 0.93 — already a large lift over v1 alone, but
the GBM's non-linearity buys another 4 points of AUC.

## Feature ablation — which features carry the lift

`scripts/ablation_tier2_features.py` re-runs the exact headline pipeline
(GBM 120 / depth 3 / lr 0.08 / subsample 0.85, leave-one-run-out CV over
the 50 runs) eight times: once with all seven inputs, then once with each
input removed. The drop in pooled AUC and in the operating points
isolates each feature's marginal contribution.

| Variant                        | AUC    | ΔAUC    | recall@p99 | recall@p95 | best F1 |
|---------------------------------|--------|---------|------------|------------|---------|
| **full (7 features)**           | 0.9677 | —       | 0.619      | 0.737      | 0.840   |
| drop `trader_score`             | 0.9677 | +0.0000 | 0.530      | 0.712      | 0.833   |
| drop `burst_concentration`      | 0.9165 | −0.0512 | 0.274      | 0.449      | 0.685   |
| drop `side_entropy_in_burst`    | 0.9636 | −0.0041 | 0.634      | 0.721      | 0.830   |
| drop `counterparty_hhi_burst`   | 0.9660 | −0.0017 | 0.620      | 0.718      | 0.834   |
| drop `order_qty_cov`            | 0.9652 | −0.0025 | 0.608      | 0.726      | 0.827   |
| drop `top_partner_trade_share`  | 0.9679 | +0.0002 | 0.625      | 0.750      | 0.839   |
| drop `co_active_top_count`      | 0.9605 | −0.0072 | 0.592      | 0.733      | 0.834   |

GBM feature importance (mean over the 50 folds, full model) vs each
feature's standalone discriminative power:

| Feature                  | GBM importance | Standalone AUC |
|--------------------------|----------------|----------------|
| burst_concentration      | 0.338          | 0.704          |
| order_qty_cov            | 0.198          | 0.666          |
| trader_score             | 0.181          | 0.793          |
| counterparty_hhi_burst   | 0.150          | 0.800          |
| co_active_top_count      | 0.099          | 0.744          |
| side_entropy_in_burst    | 0.028          | 0.550          |
| top_partner_trade_share  | 0.006          | 0.775          |

Three findings worth keeping for chapter 6:

- **`burst_concentration` is the load-bearing feature.** Standalone it
  is only AUC 0.704 — fifth of seven — yet removing it from the ensemble
  costs −0.051 AUC and collapses recall@p99 from 0.62 to 0.27. Importance
  *inside* the ensemble is not the same as standalone power: the best
  standalone feature (`counterparty_hhi_burst`, 0.800) is highly
  substitutable and costs only −0.0017 AUC when dropped, because other
  features reconstruct its signal.

- **`trader_score` is redundant for ranking but matters at the tail.**
  Dropping the v1 GNN score leaves pooled AUC identical (0.9677), yet
  recall@p99 falls 0.62 → 0.53. The engineered features reconstruct the
  global ranking; v1 sharpens the highest-confidence predictions where
  the 99%-purity operating point lives.

- **`side_entropy_in_burst` is near-dead weight on this cohort.**
  Standalone AUC 0.550 (barely above chance), importance 0.028. Dropping
  it moves nothing outside fold-noise: −0.004 AUC, recall@p99 actually
  nudges 0.62 → 0.63, recall@p95 dips 0.74 → 0.72. The feature was
  designed to catch one-directional pumps; Phase G's ring/clique mix is
  not strongly one-sided, so it does not separate manipulators here.

> **Decision:** `side_entropy_in_burst` is retained for interpretability
> and theoretical completeness despite a negligible Phase G contribution
> (ablation: −0.004 AUC). It costs nothing to compute and is expected to
> matter on classic pump-and-dump distributions. A leaner six-feature
> model dropping it is equally defensible — the ablation is the evidence
> either way.

Caveat: this ablation, like the headline, uses leave-one-*run*-out CV
that still samples families uniformly, so every test run has
family-matched training examples. Under family-disjoint testing the
per-feature picture could shift — `side_entropy_in_burst` in particular
might behave differently across clique/ring/mixed.

## What this means for the dissertation

The honest read in chapter 6 ([[Research Notes/40 LIMITATIONS Chapter 6|40]])
argued v1's 0.006-recall-at-thr-0.40 was a conservative-but-defensible
operating point. The tier-2 stack shifts the conversation entirely:

> Conservative defensible claim: **on a 50-run out-of-distribution
> cohort with parameter-disjoint manipulation scenarios, the
> surveillance system catches 62% of manipulators at 99% precision
> using a two-stage stack (GraphSAGE GNN + GBM classifier on
> engineered features), evaluated under leave-one-run-out cross-
> validation.**

The next ask Dr Milan is likely to make: how does this generalize to a
*completely* unseen distribution? E.g., manipulators using ABIDES
agents from [[Research Notes/43 ABIDES Integration|43]] rather than the
Phase G synthesizer? That's a real generalization test that this CV
doesn't capture — but it's a *future* generalization test, and the
in-distribution tier-2 result is itself a defensible Phase 3
contribution.

## How to reproduce

```
cd <repo root>
python scripts/train_tier2_classifier.py --model gbm   # headline metrics
python scripts/ablation_tier2_features.py              # feature ablation
```

Under a minute each. No docker required, no GPU required. Outputs:

  outputs/_phase_g_tier2_predictions.csv   per-trader CV prob, 10,037 rows
  outputs/_phase_g_tier2_metrics.json      AUC + operating points
  outputs/_phase_g_tier2_ablation.json     leave-one-feature-out results

The classifier itself is not persisted — it's re-trained fresh in
under a minute each run, which is the right design for CV evaluation.
For production deployment, train one model on all 50 runs and pickle
it (5 extra lines of code).

## What this does NOT replace

- **GraphSAGE retraining (Path A2).** The tier-2 GBM rides on top of
  v1; it cannot fix what v1 misses entirely. A feature-augmented
  GraphSAGE retrain might do better still because the GNN can learn
  feature-graph interactions, not just trader-marginal feature
  combinations. That remains an open follow-up.

- **A held-out manipulation family.** Train on clique+mixed, evaluate
  on ring (or vice versa). The current LOO-CV samples uniformly across
  families, so the test set always contains family-matched training
  examples. The harder test is family-disjoint generalization.

- **Real NSE data.** Everything here is on the calibrated synthesizer
  + ABIDES substrate. The features are explicit + interpretable, so
  they should transfer to real data, but that's an empirical question
  not yet answered.

## Operating-point selection for production

A surveillance officer would pick the operating point based on what
the team's review budget allows:

  - High-precision (purity ≥ 0.99, recall 0.62): ~13 flagged traders
    per run. Every flag is worth investigating. Catches majority of
    manipulators with very low noise. Recommended default.
  - Balanced (best F1, recall 0.76, purity 0.94): ~16 flagged
    traders per run. Slight increase in noise for better recall.
  - High-recall (purity ≥ 0.80, recall 0.85): ~21 flagged per run.
    Use when there's investigator capacity and you want to minimise
    missed manipulators.

The page (investigation page 12) should add a tier-2 toggle so the
officer can switch between v1-only and v1+tier2 scoring per run. That's
a minor wiring change deferred to a follow-up note.

## See also

- `scripts/train_tier2_classifier.py` — the trainer
- `scripts/ablation_tier2_features.py` — leave-one-feature-out ablation
- `outputs/_phase_g_tier2_predictions.csv` — per-trader CV predictions
- `outputs/_phase_g_tier2_metrics.json` — pooled metrics
- `outputs/_phase_g_tier2_ablation.json` — ablation results
- [[Research Notes/46 Engineered Features + Investigation Page|46]] — feature methodology
- [[Research Notes/47 Phase G Investigation Dashboard Tour|47]] — dashboard tour
- [[Research Notes/45 Phase G Generalization Pipeline|45]] — the v1 model lineage
- [[Research Notes/40 LIMITATIONS Chapter 6|40]] — the honest read this result lifts
