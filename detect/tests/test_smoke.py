"""Smoke tests for the detect module.

These are deliberately minimal: enough to satisfy pytest that the package
imports cleanly and CI has something to run. Real tests migrate into
tests/ over time as the _legacy/ fixtures get updated for the monorepo
layout (see tests/conftest.py which excludes _legacy/ from collection).
"""
from __future__ import annotations

import detect


def test_detect_importable() -> None:
    """detect top-level package imports without side effects."""
    assert detect is not None


def test_detect_has_version() -> None:
    """detect exposes __version__ as a string like '0.1.0'."""
    assert hasattr(detect, "__version__")
    assert isinstance(detect.__version__, str)
    assert detect.__version__.count(".") >= 1


def test_detect_has_schema_version() -> None:
    """detect exposes __schema_version__ matching SCHEMA.md."""
    assert hasattr(detect, "__schema_version__")
    assert isinstance(detect.__schema_version__, str)
