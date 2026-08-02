# Model checkpoints

Trained model checkpoints are hosted on Zenodo rather than checked
into git, for two reasons:

1. **Size.** The v4 GraphSAGE `.pt` alone is ~120 MB; storing large
   binaries in git bloats every clone forever.
2. **Provenance.** Zenodo mints a DOI for each checkpoint release, so
   any paper or downstream user can cite the exact weights they used
   with a permanent identifier.

## v0.1.0 status

The v0.1.0 foundation release (the current release as of this writing)
**does not include trained checkpoints**. It ships the code and the
scaffolding only. To get numbers, you need to train from a cohort — see
[`REPRODUCE.md`](REPRODUCE.md) for the training commands.

Checkpoints will land in v0.2.0 alongside a `python -m detect.registry.download`
helper that pulls them from Zenodo automatically.

## What v0.2.0 will include

Four artefacts, each with SHA-256 + metrics captured in a manifest:

| File | Model | Size | Reported metric |
|---|---|---|---|
| `v1_graphsage.pt` | 2-feature GraphSAGE baseline | ~40 MB | AUC 0.638 within-generator OOD |
| `v3_graphsage.pt` | v1 + focal loss + val-recall early stop | ~40 MB | AUC 0.79 within-generator OOD |
| `v4_graphsage.pt` | 8-feature augmented GraphSAGE | ~120 MB | AUC 0.984 within-generator OOD, 0.842 cross-generator ABIDES |
| `tier2_gbm.pkl` | Gradient Boosting stacked on v1 score + 6 features | ~2 MB | AUC 0.968 within-generator OOD, 0.91–0.99 family-disjoint |

Plus `checkpoints_manifest.json` with SHA-256, training-cohort ID, and
paper-reported metric per file. The download helper verifies the SHA-256
before returning the local path, so a corrupted download can't silently
poison downstream evaluation.

## Manual download (v0.2.0 onwards)

Once the checkpoints ship, you'll be able to grab them from
[Zenodo record page](https://zenodo.org/records/21755230) manually
(there'll be a separate DOI for the checkpoints release) or
programmatically:

```bash
python -m detect.registry.download --version v0.2.0 --dest models/
```

This creates:

```
models/
├── v1_graphsage.pt
├── v3_graphsage.pt
├── v4_graphsage.pt
├── tier2_gbm.pkl
└── checkpoints_manifest.json
```

## Verifying a checkpoint by hand

```bash
sha256sum models/v4_graphsage.pt
# should match the value in checkpoints_manifest.json
```

Or in Python:

```python
import hashlib, pathlib, json
manifest = json.loads(pathlib.Path("models/checkpoints_manifest.json").read_text())
expected = manifest["v4_graphsage.pt"]["sha256"]
actual   = hashlib.sha256(pathlib.Path("models/v4_graphsage.pt").read_bytes()).hexdigest()
assert actual == expected, "checkpoint corrupted or wrong version"
```

## Optional: HuggingFace Hub mirror

For the v4 checkpoint specifically (the one most likely to be pulled
by downstream users), a HuggingFace Hub mirror is planned. This gives
`from_pretrained` compatibility for people who prefer the HF workflow.

Not required — Zenodo is the source of truth and stays authoritative.

## Which checkpoint to use

- **You want the paper's headline result:** use `v4_graphsage.pt`.
- **You want the family-disjoint story:** use `tier2_gbm.pkl` — it's
  the model that survives held-out manipulation families.
- **You want to compare against the baseline:** use `v1_graphsage.pt`.
- **You're debugging:** train fresh on the demo cohort instead — much
  faster iteration than downloading a full checkpoint.

## Reproducibility notes

Each checkpoint's manifest entry includes:

- `training_cohort_id` — the exact cohort the model was trained on
- `training_seed` — the numpy/torch seed used
- `training_epochs` — actual epochs before early stop
- `paper_metric` — the number cited in the papers for this checkpoint
- `sha256` — file integrity hash

You can regenerate any checkpoint from scratch given the cohort + seed,
though wall-clock training time on a single GPU is ~2 hours for v4 and
~30 minutes for tier-2.
