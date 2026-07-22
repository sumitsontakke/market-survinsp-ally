# Checkpoint Hosting — Zenodo Setup Guide

> Where the v1 / v3 / v4 GraphSAGE weights and the Tier-2 GBM live. Repo stays lean; users pull the checkpoints only when they need them. Ships with a citable DOI you can put on your CV.

**Time to set up end-to-end:** ~45 minutes for a first-time user, split across account creation, GitHub linkage, and a first release.

**Recurring effort per release:** ~5 minutes — the GitHub↔Zenodo integration does the archival automatically once wired up.

---

## Why Zenodo (versus alternatives)

| Option | Pro | Con | Verdict |
|---|---|---|---|
| **Zenodo** | Free, permanent, CERN-backed, DOI per release, GitHub integration is one-click. | 50 GB per record limit (not a problem for you). | **Recommended for the paper artefacts.** |
| **HuggingFace Hub** | Model-first UX, easy `from_pretrained` API, nice for demos. | No DOI (yet), storage is model-specific. | Second, complementary. Use for the *deployed* v4 checkpoint if you want a `pip install`-style download experience. |
| **Git LFS** | Lives in the repo. | 1 GB free quota, 50 GB paid. Wrong tool for large binary artefacts a research repo. | Only for the ~5 MB demo cohort. |
| **GitHub Releases (binary attach)** | Simple. | No DOI, 2 GB per file limit, not indexed by academic search. | Don't use for citable artefacts. |
| **Google Drive / Dropbox** | Familiar. | Not citable, links rot. | Never for research. |

**Recommendation:** Zenodo for the citable release, optionally mirror the deployable v4 checkpoint on HuggingFace Hub for the `from_pretrained` UX later. Skip the rest.

---

## Prerequisites

- Your GitHub account: `sumitsontakke` (confirmed).
- The public repo: [`sumitsontakke/market-survinsp-ally`](https://github.com/sumitsontakke/market-survinsp-ally) (confirmed).
- Your four (or so) model artefacts, staged locally:
  - `v1_graphsage.pt` — baseline GraphSAGE, 2-feature node input.
  - `v3_graphsage.pt` — v2 focal loss + val-recall early stopping.
  - `v4_graphsage.pt` — feature-augmented GraphSAGE, 8-feature node input.
  - `tier2_gbm.pkl` — sklearn `GradientBoostingClassifier` (fitted).
  - Plus (optional) per-checkpoint `config.yaml` and `metrics.json` from the model registry.
- Approximately 10 minutes now, then a chunk of time when you're ready to tag the release.

---

## Step 1 — Create a Zenodo account (5 min)

1. Go to <https://zenodo.org>.
2. Click **Sign up** in the top-right.
3. Choose **Sign up with GitHub** (recommended — it makes the integration in Step 2 automatic). Alternatively use email; you can add GitHub later.
4. Authorise the OAuth prompt when GitHub asks.
5. Complete your profile: full name, ORCID (create one at <https://orcid.org> if you don't already have one — takes 2 minutes and is worth it for academic citations).

You now have a Zenodo account associated with your GitHub identity.

---

## Step 2 — Link Zenodo to your GitHub repo (2 min)

1. In Zenodo, click your username (top-right) → **GitHub**.
2. You'll see a list of your GitHub repositories. Find `sumitsontakke/market-survinsp-ally`.
3. Toggle the switch next to it to **ON**. This tells Zenodo to watch for new releases and archive them automatically.
4. Zenodo installs a webhook on the repo. You can verify at GitHub → repo → **Settings → Webhooks**; you'll see a `https://zenodo.org/api/hooks/receivers/github/…` entry.

The link is now live. **Any GitHub release you tag from this point onward gets archived to Zenodo and assigned a DOI automatically.** No manual upload.

---

## Step 3 — Prepare the release archive (10 min)

Decide what goes into the tagged release. Zenodo archives the entire repo tree at the tag *plus* whatever binary assets you attach to the GitHub release. Recommended contents:

- **In-repo (source of truth):** everything already in `market-survinsp-ally` at the tag commit. Zenodo takes the tarball automatically.
- **Attached binaries (upload to the GitHub release page):**
  - `v1_graphsage.pt` (~10 MB expected)
  - `v3_graphsage.pt` (~10 MB)
  - `v4_graphsage.pt` (~12 MB)
  - `tier2_gbm.pkl` (~1 MB)
  - `checkpoints_manifest.json` — small file listing each `.pt` / `.pkl` with SHA256, git SHA, config hash, and headline metrics. Lets downstream users verify what they downloaded.
  - Optional: `sample_cohort.tar.gz` (10-run sample cohort, ~50 MB) — makes the "reproduce table 6.2 without generating cohorts" story work in one download.

**Total binary payload:** ~80 MB. Well under Zenodo's 50 GB per-record limit.

### The manifest file (write this first)

Create `checkpoints_manifest.json` at the repo root or under `docs/`. Example content:

```json
{
  "release": "v1.0.0",
  "git_sha": "abc1234…",
  "generated_at": "2026-07-15T10:00:00Z",
  "checkpoints": [
    {
      "name": "v1_graphsage",
      "file": "v1_graphsage.pt",
      "sha256": "…",
      "size_bytes": 10485760,
      "framework": "pytorch",
      "framework_version": "2.4.0",
      "model_type": "GraphSAGE",
      "input_dim": 2,
      "hidden_dim": 64,
      "config_hash": "…",
      "training_cohort": "diverse-variation-240d",
      "metrics": {
        "within_ood_edge_auc": 0.638,
        "within_ood_pooled_trader_auc": 0.793
      }
    },
    {
      "name": "v4_graphsage",
      "file": "v4_graphsage.pt",
      "sha256": "…",
      "size_bytes": 12582912,
      "framework": "pytorch",
      "framework_version": "2.4.0",
      "model_type": "GraphSAGE",
      "input_dim": 8,
      "hidden_dim": 64,
      "config_hash": "…",
      "training_cohort": "diverse-variation-240d",
      "metrics": {
        "within_ood_edge_auc": 0.984,
        "cross_generator_abides_auc": 0.842
      }
    },
    {
      "name": "tier2_gbm",
      "file": "tier2_gbm.pkl",
      "sha256": "…",
      "size_bytes": 1048576,
      "framework": "scikit-learn",
      "framework_version": "1.4.0",
      "model_type": "GradientBoostingClassifier",
      "n_estimators": 200,
      "learning_rate": 0.05,
      "max_depth": 3,
      "input_features": ["v1_score", "phi1_burst_conc", "phi2_side_ent", "phi3_cp_hhi", "phi4_qty_cov", "phi5_top_partner", "phi6_co_active"],
      "training_protocol": "leave-one-run-out CV over 50 OOD runs",
      "metrics": {
        "within_ood_pooled_trader_auc": 0.968,
        "family_disjoint_ring_auc": 0.994,
        "family_disjoint_clique_auc": 0.975,
        "family_disjoint_mixed_auc": 0.906
      }
    }
  ]
}
```

Compute the SHA256 of each artefact with:

```bash
sha256sum v1_graphsage.pt v3_graphsage.pt v4_graphsage.pt tier2_gbm.pkl
```

Paste the results into the manifest.

---

## Step 4 — Tag the release on GitHub (5 min)

Zenodo triggers **only on GitHub Releases**, not plain git tags. Do it via the GitHub web UI to keep it simple:

1. Go to <https://github.com/sumitsontakke/market-survinsp-ally/releases>.
2. Click **Draft a new release**.
3. **Choose a tag** → type `v1.0.0` → *"Create new tag: v1.0.0 on publish"*.
4. **Release title:** `v1.0.0 — Initial public release`.
5. **Description:** short summary of what the release contains. Suggested template:

   ```markdown
   ## market-survinsp-ally v1.0.0

   First public release. Companion code for the two conference-length papers.

   ### What's included
   - Full source of the `synth/` (calibrated synthesizer + ABIDES adapter) and `detect/` (GraphSAGE + Tier-2 GBM + dashboard) modules.
   - Reproducibility script `make reproduce` on a 3-run demo cohort.
   - Trained checkpoints attached as release assets:
     - `v1_graphsage.pt` — baseline (2-feature node vector).
     - `v3_graphsage.pt` — focal loss + val-recall early stopping.
     - `v4_graphsage.pt` — feature-augmented (8-feature node vector).
     - `tier2_gbm.pkl` — bolt-on GBM over v1 score + 6 engineered features.
   - `checkpoints_manifest.json` — SHA256 + metrics for each artefact.

   ### Papers
   - Feature-Augmented GNNs for OOD Detection — `docs/papers/model_journey_paper.pdf`
   - Calibrated-Synthesis-to-Deployment Surveillance Pipeline — `docs/papers/model_journey_paper_v2.pdf`

   ### Citation
   See `CITATION.cff` at the repo root, or use the "Cite this repository" button in the sidebar.
   ```

6. **Attach binaries.** Drag `v1_graphsage.pt`, `v3_graphsage.pt`, `v4_graphsage.pt`, `tier2_gbm.pkl`, `checkpoints_manifest.json` (and optionally `sample_cohort.tar.gz`) onto the release page.
7. **Publish release.**

At this moment: GitHub creates the tag, publishes the release page, sends the webhook to Zenodo.

---

## Step 5 — Verify Zenodo received it (2 min)

1. Wait ~30 seconds for Zenodo to process the webhook.
2. Go to Zenodo → your dashboard → **Uploads**.
3. You should see a new record titled *"market-survinsp-ally: Graph-neural-network market surveillance toolkit"* — the metadata is drawn from your `CITATION.cff`.
4. Click it. The record shows:
   - A permanent DOI in the format `10.5281/zenodo.<numeric>` (e.g., `10.5281/zenodo.12345678`).
   - The repo tarball archived.
   - The binary release assets archived.
   - A "Cite as" block ready to copy.

If Zenodo didn't pick it up:
- Confirm the GitHub↔Zenodo toggle from Step 2 is still ON.
- Confirm the release is *published*, not draft.
- GitHub → repo → Settings → Webhooks → click the Zenodo webhook → check **Recent Deliveries** for errors.

---

## Step 6 — Wire the DOI into CITATION.cff (3 min)

Once you have the numeric DOI:

1. Edit `CITATION.cff` at the repo root.
2. Replace `doi: 10.5281/zenodo.XXXXXXX` with the actual DOI (e.g., `doi: 10.5281/zenodo.12345678`).
3. Commit: `git commit -am "docs: pin Zenodo DOI in CITATION.cff"` and push to `main`.

Zenodo maintains one **concept DOI** (stable across all versions of the same repo) and one **version DOI** (unique per release). Use the *concept DOI* in `CITATION.cff` — it always resolves to the most recent version and is what you want on your CV.

Also update the BibTeX block in `README.md`:

```bibtex
@software{sontakke2026msa,
  author  = {Sontakke, Sumit and Joshi, Milan},
  title   = {market-survinsp-ally: Graph-neural-network market surveillance toolkit},
  year    = {2026},
  url     = {https://github.com/sumitsontakke/market-survinsp-ally},
  version = {1.0.0},
  doi     = {10.5281/zenodo.12345678}   ← replace with your actual DOI
}
```

---

## Step 7 — Add a download helper (5 min)

Give users one command to pull the checkpoints. Add this to `detect/src/detect/registry/download.py`:

```python
"""Download trained checkpoints from the Zenodo release."""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

ZENODO_RECORD = "12345678"   # replace with your actual record number
ZENODO_BASE = f"https://zenodo.org/record/{ZENODO_RECORD}/files"

MANIFEST_URL = f"{ZENODO_BASE}/checkpoints_manifest.json"


def download_checkpoints(dest: Path = Path("./data/model_registry")) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(urllib.request.urlopen(MANIFEST_URL).read())
    for ckpt in manifest["checkpoints"]:
        target = dest / ckpt["file"]
        if target.exists():
            print(f"skip {ckpt['file']} (exists)")
            continue
        url = f"{ZENODO_BASE}/{ckpt['file']}"
        print(f"download {ckpt['file']} ({ckpt['size_bytes'] // 1024**2} MB)…")
        urllib.request.urlretrieve(url, target)
        sha = hashlib.sha256(target.read_bytes()).hexdigest()
        assert sha == ckpt["sha256"], f"sha256 mismatch on {ckpt['file']}"
        print(f"  ok, sha256 verified")


if __name__ == "__main__":
    download_checkpoints()
```

Then in your `Makefile`:

```makefile
checkpoints:
	python -m detect.registry.download
```

And in the README quick-start:

```
make checkpoints                     # pull v1 / v3 / v4 / tier-2 from Zenodo
make reproduce                       # end-to-end reproduction
```

---

## Step 8 — Add a DOI badge to the README (2 min)

Once the DOI exists, add this line near the top of `README.md`:

```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12345678.svg)](https://doi.org/10.5281/zenodo.12345678)
```

Renders as a green DOI badge — universal signal that the repo has a proper archival home.

---

## Optional: HuggingFace Hub mirror for v4

If you want the `AutoModel.from_pretrained("sumitsontakke/msa-v4-graphsage")` UX later:

1. Create a HuggingFace account at <https://huggingface.co>.
2. Install the CLI: `pip install huggingface_hub`.
3. `huggingface-cli login`.
4. Create a model repo: `huggingface-cli repo create msa-v4-graphsage`.
5. Push:

   ```bash
   git clone https://huggingface.co/sumitsontakke/msa-v4-graphsage
   cd msa-v4-graphsage
   cp ../v4_graphsage.pt ./pytorch_model.bin
   cp ../v4_config.json ./config.json
   cp ../v4_model_card.md ./README.md
   git add . && git commit -m "Initial v4 GraphSAGE checkpoint" && git push
   ```

HF is nicer for a demo experience (the model card renders inline, users can inference in-browser via HF Inference API). Zenodo is where the citable version lives. Both is fine; if you're picking one, Zenodo first.

---

## Ongoing operations

**When you tag a new release** (`v1.1.0`, `v2.0.0`, etc.), Zenodo automatically archives the new version and mints a new version DOI. The concept DOI keeps pointing to the latest.

**Versioning rule of thumb:**

- **Patch** (`v1.0.1`) — bug fix, docs, small config tweak. Rebuild checkpoints only if they changed.
- **Minor** (`v1.1.0`) — new feature, new checkpoint variant, backward-compatible schema field. Ship new checkpoints if they exist.
- **Major** (`v2.0.0`) — breaking schema or API change. Ship new checkpoints and clearly deprecate old ones in the release notes.

**When you retrain a model** without changing anything else, that's a **minor** — bump to `v1.1.0`, ship the new checkpoint alongside a *"Retrained v4"* changelog entry. Old checkpoints stay downloadable from the earlier record.

---

## Cost

Zenodo: **free**, no strings.

HuggingFace: **free** for public models, paid for private (not relevant here).

ORCID: **free**.

Total ongoing cost: **zero**. This is unusual and worth appreciating.

---

## When to actually do this

Sensible order:

1. **Now (this session or next):** Finish the repo restructure. Push to GitHub. Do *not* tag a release yet.
2. **Before your papers are accepted:** Complete Steps 1–2 (Zenodo account + repo linkage). Zero cost, zero disclosure risk.
3. **On paper acceptance OR when you're ready to promote the repo:** Complete Steps 3–8. This is when the DOI matters — you can cite it in the camera-ready paper, on your CV, and in any conference presentations.
4. **After the first release:** revisit annually. If you retrain checkpoints for a follow-up paper, tag a new release; if not, the v1.0.0 record stands.

The link between GitHub and Zenodo doesn't do anything visible until you tag a release, so setting it up early (Steps 1–2) costs nothing and saves you fumbling on release day.
