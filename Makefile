# market-survinsp-ally — top-level orchestrator
#
# Two-module monorepo: `synth/` produces cohorts, `detect/` consumes them.
# Every recipe here composes on top of the module-level pyproject.toml files.

.DEFAULT_GOAL := help
.PHONY: help install synth-install detect-install reproduce test lint format \
        docs clean checkpoints demo-cohort schema-check


# ── Environment ────────────────────────────────────────────────
install:  ## Install both modules in editable mode with test extras
	pip install -e "./synth[test]"
	pip install -e "./detect[test]"

synth-install:  ## Install synth only
	pip install -e "./synth"

detect-install:  ## Install detect only
	pip install -e "./detect"


# ── Reproduction ───────────────────────────────────────────────
demo-cohort:  ## Generate the 3-run demo cohort (~1 min CPU)
	python -m synth.cli generate \
		--config configs/synth/demo_cohort.yaml \
		--out cohorts/demo

schema-check:  ## Verify a cohort is SCHEMA.md-conformant
	python -m synth.validate cohorts/demo

reproduce: demo-cohort schema-check  ## Run the full demo pipeline end-to-end
	@echo "→ Extracting six engineered features"
	python -m detect.cli features --cohort cohorts/demo
	@echo "→ Training Tier-2 GBM (leave-one-run-out)"
	python -m detect.cli train --cohort cohorts/demo --model tier2
	@echo "→ Evaluating"
	python -m detect.cli evaluate --cohort cohorts/demo --model tier2
	@echo "✓ Reproduction complete. See cohorts/demo/*/detect_tier2/ for scores."


# ── Checkpoints ────────────────────────────────────────────────
checkpoints:  ## Pull v1/v3/v4/Tier-2 trained checkpoints from Zenodo
	python -m detect.registry.download


# ── Quality gates ──────────────────────────────────────────────
test:  ## Run all tests (pytest on both modules)
	pytest synth/tests/ detect/tests/

lint:  ## Lint both modules with ruff
	ruff check synth/src synth/tests
	ruff check detect/src detect/tests

format:  ## Format both modules with ruff
	ruff format synth/src synth/tests
	ruff format detect/src detect/tests


# ── Docs ────────────────────────────────────────────────────────
docs:  ## Build the local docs site (placeholder)
	@echo "Doc site build not yet wired. See docs/ for hand-written markdown."


# ── Housekeeping ───────────────────────────────────────────────
clean:  ## Remove build artefacts (not source, not cohorts)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name build -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name dist -exec rm -rf {} + 2>/dev/null || true


# ── Help ────────────────────────────────────────────────────────
help:  ## Show this help
	@echo "market-survinsp-ally — monorepo Make targets"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
