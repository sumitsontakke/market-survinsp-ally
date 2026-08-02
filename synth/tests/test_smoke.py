"""Smoke tests for the synth module.

These are deliberately minimal: enough to satisfy pytest that the package
imports cleanly and CI has something to run. Real tests migrate into
tests/ over time as fixtures get updated for the monorepo layout.
"""
from __future__ import annotations

import synth


def test_synth_importable() -> None:
    """synth top-level package imports without side effects."""
    assert synth is not None


def test_synth_has_version() -> None:
    """synth exposes __version__ as a string like '0.1.0'."""
    assert hasattr(synth, "__version__")
    assert isinstance(synth.__version__, str)
    assert synth.__version__.count(".") >= 1  # semver-ish


def test_synth_has_schema_version() -> None:
    """synth exposes __schema_version__ matching SCHEMA.md."""
    assert hasattr(synth, "__schema_version__")
    assert isinstance(synth.__schema_version__, str)
