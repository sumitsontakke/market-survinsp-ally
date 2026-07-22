# Directory Layout & Migration Plan

> **Purpose:** map the current repository layout onto the target `synth/` + `detect/` structure. Read alongside [`SCHEMA.md`](SCHEMA.md) for the boundary contract. This document is intended for the first round of restructuring; delete after the migration lands.

## Target layout

```
market-survinsp-ally/                          (repo root)
│
├── SCHEMA.md                                  ← boundary contract
├── README.md                                  ← project arc + headline results
├── LICENSE                                    ← MIT
├── CITATION.cff                               ← "cite this repo" metadata
├── Makefile                                   ← make reproduce | test | lint
├── .gitignore                                 ← comprehensive
├── .github/                                   ← CI + issue templates
│
├── synth/                                     ← MODULE 1: data generator
│   ├── README.md                              ← module-scoped docs
│   ├── pyproject.toml
│   ├── src/synth/
│   │   ├── __init__.py
│   │   ├── calibration/                       ← from calibration_service/{fetcher,calibrator,core}
│   │   ├── generator/                         ← from src/synthetic_market_sim/
│   │   ├── abides/                            ← from abides_integration/ + src/…/abides
│   │   ├── validate/                          ← SCHEMA.md conformance checker
│   │   ├── cli.py                             ← python -m synth {generate,validate,fetch}
│   │   └── constants.py
│   └── tests/
│
├── detect/                                    ← MODULE 2: detectors + dashboard
│   ├── README.md
│   ├── pyproject.toml
│   ├── docker-compose.yml                     ← webapp + trainer-gpu only
│   ├── src/detect/
│   │   ├── __init__.py
│   │   ├── features/                          ← six engineered features
│   │   │   ├── engineered.py                  ← from src/…/node_engineered.py
│   │   │   └── projection.py                  ← 0.7*max + 0.3*top3 rule
│   │   ├── models/                            ← from src/surveillance_ml/
│   │   │   ├── graphsage.py                   ← v1 and v4 (dim=2 or 8)
│   │   │   └── tier2_gbm.py
│   │   ├── training/                          ← from training/
│   │   │   ├── driver.py                      ← run_m3_boosted rewritten
│   │   │   └── loops.py
│   │   ├── evaluation/                        ← from training/phase_g_eval.py
│   │   │   ├── phase_g.py                     ← within-generator OOD
│   │   │   ├── abides.py                      ← cross-generator
│   │   │   ├── family_disjoint.py             ← leave-one-family-out
│   │   │   └── metrics.py
│   │   ├── dashboard/                         ← from calibration_service/webapp/
│   │   │   ├── app.py
│   │   │   ├── pages/
│   │   │   └── static/
│   │   ├── registry/                          ← model registry local reader
│   │   └── cli.py                             ← python -m detect {train,evaluate,serve}
│   └── tests/
│
├── docs/
│   ├── REPRODUCE.md                           ← full-cohort reproduction guide
│   ├── DEVELOPING.md                          ← contributor guide
│   ├── CHECKPOINTS.md                         ← Zenodo links
│   ├── ARCHITECTURE.md                        ← how the modules interact
│   ├── papers/
│   │   ├── model_journey_paper.pdf            ← paper v1 (OOD framing)
│   │   ├── model_journey_paper.tex
│   │   ├── model_journey_paper_v2.pdf         ← paper v2 (pipeline framing)
│   │   ├── model_journey_paper_v2.tex
│   │   └── CITATIONS.md
│   └── img/
│       ├── pipeline_overview.png              ← README hero
│       ├── dashboard_metric_timeline.png      ← README screenshot
│       └── (existing fig_pipeline / fig_graphsage / fig_workflow / fig_results / fig_family)
│
├── configs/                                   ← example configs
│   ├── demo_cohort.yaml                       ← small cohort for make reproduce
│   ├── nse_calibrated_ood.yaml
│   ├── abides_cross_generator.yaml
│   └── detector/{v1,v4,tier2}.yaml
│
└── cohorts/                                   ← .gitignored except demo
    └── demo/                                  ← git-lfs, 3-run mini cohort
```

## Migration mapping — current → target

Legend: **[move]** unchanged code, new path · **[refactor]** move + rename or reshape · **[split]** one path becomes several · **[stay]** already in the right place · **[delete]** not part of the public repo · **[external]** hosted off-repo (Zenodo, HuggingFace Hub, etc.)

### synth module

| Current path | Action | Target path | Notes |
|---|---|---|---|
| `src/synthetic_market_sim/` | **[refactor]** | `synth/src/synth/generator/` | Rename package. Update all imports. |
| `src/synthetic_market_sim/behaviors/` | **[move]** | `synth/src/synth/generator/behaviors/` | |
| `src/synthetic_market_sim/exporters/` | **[refactor]** | `synth/src/synth/generator/exporters/` | Ensure output conforms to `SCHEMA.md`. |
| `src/synthetic_market_sim/market/` | **[move]** | `synth/src/synth/generator/market/` | |
| `src/synthetic_market_sim/simulation/` | **[move]** | `synth/src/synth/generator/simulation/` | |
| `src/synthetic_market_sim/domain/` | **[move]** | `synth/src/synth/generator/domain/` | |
| `abides_integration/` | **[move]** | `synth/src/synth/abides/` | |
| `calibration_service/fetcher/` | **[move]** | `synth/src/synth/calibration/fetcher/` | Bhavcopy fetch stays a synth concern. |
| `calibration_service/calibrator/` | **[move]** | `synth/src/synth/calibration/calibrator/` | |
| `calibration_service/core/` | **[split]** | `synth/src/synth/calibration/core/` + shared bits into `synth/src/synth/constants.py` | Some pieces (`config.py`) may end up in constants; `database.py` stays in synth. |
| `configs/generic_market.yaml` | **[move]** | `configs/synth/generic_market.yaml` | |
| `configs/collusion_scenario.yaml` | **[move]** | `configs/synth/collusion_scenario.yaml` | |
| `configs/circular_trading.yaml` | **[move]** | `configs/synth/circular_trading.yaml` | |
| `docs/scenario_design.md` | **[move]** | `synth/docs/scenario_design.md` | |
| `src/synthetic_market_sim.egg-info/` | **[delete]** | — | Regenerated by pip; never track. |

### detect module

| Current path | Action | Target path | Notes |
|---|---|---|---|
| `src/surveillance_ml/` | **[refactor]** | `detect/src/detect/` | Rename package. |
| `src/surveillance_ml/models.py` | **[split]** | `detect/src/detect/models/graphsage.py` + `detect/src/detect/models/tier2_gbm.py` | Split monolithic module. |
| `src/surveillance_ml/feature_store.py` | **[refactor]** | `detect/src/detect/features/engineered.py` | Reads orders per `SCHEMA.md`; drops any generator imports. |
| `src/surveillance_ml/trainer.py` | **[refactor]** | `detect/src/detect/training/driver.py` | |
| `src/surveillance_ml/evaluator.py` | **[refactor]** | `detect/src/detect/evaluation/metrics.py` | |
| `training/phase_g_eval.py` | **[split]** | `detect/src/detect/evaluation/{phase_g,abides,family_disjoint}.py` | Splits by regime. |
| `training/run_m3_boosted.py` | **[refactor]** | `detect/src/detect/training/driver.py` (CLI shim in `detect/cli.py`) | |
| `training/hyperparam_tuner.py` | **[move]** | `detect/src/detect/training/hyperparam.py` | |
| `training/node_engineered.py` | **[move]** | `detect/src/detect/features/engineered.py` | Consolidate with feature_store.py. |
| `src/surveillance_app/` | **[delete]** | — | Superseded by `_v2`. |
| `src/surveillance_app_v2/` | **[refactor]** | `detect/src/detect/dashboard/` | Rename package. |
| `calibration_service/webapp/` | **[delete]** | — | Superseded by v2. |
| `calibration_service/webapp_v2/` | **[move]** | `detect/src/detect/dashboard/` | Merge with v2 above. |
| `calibration_service/webapp_v2/pages/` | **[move]** | `detect/src/detect/dashboard/pages/` | Same page files. |
| `calibration_service/docker-compose.yml` | **[split]** | `detect/docker-compose.yml` (webapp + trainer-gpu) and drop synth-only services | Fetcher/calibrator/bhavcopy compose lives in `synth/docker-compose.yml`. |
| `calibration_service/trainer-gpu/` | **[move]** | `detect/docker/trainer-gpu/` | GPU trainer image is detect-side. |
| `configs/detector/*.yaml` | **[stay]** | `configs/detector/` | Already in the right structural place. |

### Documentation

| Current path | Action | Target path | Notes |
|---|---|---|---|
| `outputs/45 Phase G …md` through `outputs/51 Phase J …md` | **[refactor]** | `docs/research-notes/` | Rename to slugs (e.g., `45-phase-g-ood-eval.md`) for URL-friendliness. Keep the arc. |
| `outputs/model_journey_paper.pdf` + `.tex` | **[move]** | `docs/papers/` | |
| `outputs/model_journey_paper_v2.pdf` + `.tex` | **[move]** | `docs/papers/` | |
| `outputs/fig_pipeline.png` etc | **[move]** | `docs/img/` | Reused by README + papers. |
| `outputs/mtech_report_sumit.pdf` | **[external]** | Zenodo | The full dissertation is archival; not core to the code repo. Link from README. |
| `outputs/final_submission_drmilan_deck.pptx` | **[delete]** | — | Dissertation artefact, not code-repo material. |
| `outputs/final_review_drmilan_deck.pptx` | **[delete]** | — | |
| `outputs/submission_letter_drmilan.docx` | **[delete]** | — | Private university admin artefact. |
| `outputs/_qa/`, `outputs/_deck_*`, `outputs/_exercise_*` | **[delete]** | — | Session cruft. |
| `training/DETAILS.md` | **[move]** | `detect/docs/DETAILS.md` | |
| `training/LIMITATIONS.md` | **[move]** | `docs/LIMITATIONS.md` (project-wide) | |
| `training/M3_*.md` | **[refactor]** | `docs/research-notes/m3-*.md` | Older phase notes. |

### External hosting

The following are explicitly **not** in-repo. Store elsewhere and link from `docs/CHECKPOINTS.md` / `docs/DATA.md`:

- Model checkpoints for v1 / v3 / v4 / Tier-2 → **Zenodo** with DOI.
- Full 240-day training cohort → regenerated deterministically from `synth/`. Not stored.
- Full 15-run ABIDES cohort → same.
- Raw NSE bhavcopy CSVs → not distributed; each user fetches their own via the synth fetcher.
- Old model registry snapshots (`data/model_registry/`) → Zenodo companion release if useful, otherwise dropped.
- `Archive.zip` (early codebase snapshot) → dropped.

### What stays at repo root

Only files that are read by first-time visitors or by CI:

- `README.md`, `SCHEMA.md`, `LICENSE`, `CITATION.cff`, `Makefile`, `.gitignore`, `.github/`
- Config templates for the two example paths `configs/demo_cohort.yaml`

Everything else lives under one of the module directories or `docs/`.

## Migration steps (do in this order)

1. **Freeze the current tree.** Tag it: `git tag pre-monorepo-restructure`.
2. **Create the target skeleton.** New empty `synth/`, `detect/`, `docs/`, `configs/` directories with `README.md` placeholders.
3. **Move the synth code.** One package at a time. `git mv` preserves history. Fix imports in each module before moving the next.
4. **Move the detect code.** Same discipline. `detect/` must not import `synth.*` — if you find one, refactor to read via schema instead.
5. **Move the docs.** Papers, research notes, images. Slug-rename as you go.
6. **Kill the deletes.** All `._*`, session cruft, superseded webapp v1, old egg-info. Verify with `git status`.
7. **Add `pyproject.toml` to each module.** Test `pip install -e synth/` and `pip install -e detect/` from a fresh venv.
8. **Wire up `Makefile`.** `make reproduce` runs a full synth → detect → evaluate loop on the demo cohort. It must pass end-to-end from a clean clone.
9. **Wire up CI.** `.github/workflows/test.yml` runs pytest + lint + docker build sanity. Add branch protection.
10. **Push to GitHub, private first.** Let the initial layout settle for a few days before flipping public. If Dr Milan flags any venue-disclosure concerns, this window catches them.

## Rough time budget

- Steps 1–3 (synth move): ~2 hours focused. Boring, mostly `git mv` and import fixes.
- Steps 4–5 (detect move + docs): ~2 hours.
- Step 6 (delete cruft): 30 minutes.
- Steps 7–8 (pyproject + Makefile + reproduce): ~2 hours; may go long if the first `make reproduce` reveals hidden coupling.
- Step 9 (CI): 1 hour to get the yaml right.
- Step 10 (GitHub push + private review): 30 minutes.

Roughly a two-evening job. Not a weekend if you don't want it to be.

## What to check before flipping the repo public

- [ ] No `.env`, `secrets.yaml`, `credentials.json`, API tokens anywhere.
- [ ] No `._*` macOS metadata (should be caught by `.gitignore` but verify).
- [ ] No `Archive.zip` or old codebase snapshots.
- [ ] No PII in commit messages (search `git log` for email addresses, phone numbers, personal names beyond the authorship line).
- [ ] `make reproduce` passes from a fresh clone in a fresh venv.
- [ ] README screenshots render on the GitHub preview.
- [ ] LICENSE is present and MIT.
- [ ] CITATION.cff parses (GitHub renders a "Cite this repository" button if it does).
- [ ] `synth/README.md` and `detect/README.md` are non-empty.
- [ ] `SCHEMA.md` is at the root and unambiguous.

## Notes on the papers repository

Both paper PDFs and their LaTeX source live under `docs/papers/`. This is deliberate — the code repo is the discovery surface, and the papers should be one click away for anyone who lands on it. If either paper is later accepted at a venue with a strict PDF-hosting policy, the file can be replaced with a link to the venue's official version and a shorter preprint.

Do *not* remove the LaTeX source. Keeping it in the repo lets external contributors flag typos and reproduce the figures.
