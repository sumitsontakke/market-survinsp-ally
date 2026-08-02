# Developing on market-survinsp-ally

Welcome. This guide covers how to make changes to the codebase, run the
tests locally, and open a pull request that has a good chance of getting
merged.

## Repo philosophy

Two modules with a **strict on-disk schema boundary**:

- `synth/` produces cohorts of trading days on disk.
- `detect/` reads those cohorts from disk.

They **never** import from each other. If you find yourself writing
`from synth import ...` inside `detect/`, stop — that's a schema
change waiting to happen and it should be routed through
[`SCHEMA.md`](../SCHEMA.md) instead. The whole point of the two-module
split is that either module can be replaced independently as long as
the other honours the schema.

Two rules for schema-touching contributions:

1. **Never add a required field without a schema major bump.** Optional
   fields with sane defaults are always welcome as minor bumps.
2. **Every schema change ships with a validator update** in
   `synth/src/synth/validate/`.

## Environment setup

```bash
git clone https://github.com/sumitsontakke/market-survinsp-ally.git
cd market-survinsp-ally

# Editable installs
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e ./synth[dev,test]
pip install -e ./detect[dev,test]
```

Python 3.11 is what CI uses. 3.10 and 3.12 also work; older versions
are not supported.

If you're on Windows and pytest can't find `synth` or `detect` after
install, use `python -m pytest` instead of bare `pytest`. See
[the CI test workflow](../.github/workflows/test.yml) for the pattern.

## Running tests

```bash
# Everything (smoke tests currently — real tests migrate in over time)
python -m pytest -ra synth/tests detect/tests

# Just one module
python -m pytest -ra synth/tests
python -m pytest -ra detect/tests

# With coverage (matches CI)
python -m pytest -ra --cov=synth  synth/tests
python -m pytest -ra --cov=detect detect/tests
```

The `detect/tests/_legacy/` subtree is excluded from CI (see
`detect/tests/conftest.py`). Those are pre-monorepo tests whose fixture
paths still reference the old layout — they need a fixture-migration
pass before they can run. Contributions to that migration are welcome.

## Running the linter

```bash
ruff check synth/src synth/tests
ruff check detect/src detect/tests
```

CI runs ruff with `continue-on-error: true` right now — findings warn
but don't gate the build — because the migrated code has cosmetic
issues that need a dedicated cleanup pass. If you're doing that pass,
remove the `continue-on-error` line to make ruff blocking.

Auto-fix what's auto-fixable:

```bash
ruff check --fix synth/src detect/src
```

## Branching + commits

- `main` is protected; open PRs against it.
- Feature branches: `feat/short-description`.
- Bug-fix branches: `fix/short-description`.
- Docs-only branches: `docs/short-description`.

Commit messages follow a light form of conventional commits:

```
feat(detect): add family-disjoint eval mode
fix(synth): correct off-by-one in bhavcopy fetcher window
docs(readme): update DOI badge for v0.2.0
ci: switch to python -m pytest
```

Small, reviewable commits are preferred over large "wall of change"
commits.

## Opening a pull request

Include in the PR description:

1. **What changed and why.** One paragraph.
2. **How to test it.** A copy-pasteable command list.
3. **Any schema impact.** If your PR touches the fields in
   [`SCHEMA.md`](../SCHEMA.md), call it out and increment the
   version. Otherwise say "no schema impact".
4. **Screenshots** if you touched dashboard UI.

CI runs pytest + ruff on Python 3.10, 3.11, 3.12 automatically. Wait
for green before requesting review.

## Where things live

Fastest way to find something:

| Looking for | Where to look |
|---|---|
| Data boundary contract | [`SCHEMA.md`](../SCHEMA.md) |
| Generator agent implementations | `synth/src/synth/generator/`, `synth/src/synth/abides/` |
| Six engineered features (φ₁ … φ₆) | `detect/src/detect/features/` |
| GraphSAGE v1 / v4 models | `detect/src/detect/models/gnn_graphsage.py` |
| Tier-2 GBM | `detect/src/detect/models/tier2_gbm.py` |
| Evaluators (OOD, ABIDES, family-disjoint) | `detect/src/detect/evaluation/` |
| Streamlit dashboard | `detect/src/detect/dashboard/` |
| Repo-level docs | `docs/` |
| Preprint PDFs | `docs/papers/` |
| Research vault notes | `docs/research-notes/` |
| CI workflow | `.github/workflows/test.yml` |

## Getting help

- **Bugs / feature requests:** open an issue with a minimal repro.
- **Design discussions:** GitHub Discussions.
- **Security concerns:** email the maintainer directly — see
  the [Security policy](https://github.com/sumitsontakke/market-survinsp-ally/security/policy)
  (to be added).

## Code of conduct

Be kind. This is a research codebase and everyone is here to learn.
Assume good faith, review substance not style, and remember that other
contributors may be students working around class schedules.
