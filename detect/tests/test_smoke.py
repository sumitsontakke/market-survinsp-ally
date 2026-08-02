"""Smoke tests for the detect module.

Minimal by design: verify the package imports cleanly. If import succeeds,
the editable install and sys.path are correctly wired. Metadata like
__version__ / __schema_version__ is optional and only warned on, not gated.

The _legacy/ subtree is excluded from collection in tests/conftest.py.
"""
from __future__ import annotations

import warnings


def test_detect_importable() -> None:
    """detect top-level package imports without side effects."""
    import detect  # noqa: F401  (import IS the assertion)


def test_detect_has_file_attribute() -> None:
    """detect resolves to an on-disk file, not a namespace package stub."""
    import detect

    assert getattr(detect, "__file__", None), (
        "detect has no __file__ - likely a broken editable install or "
        "namespace package collision"
    )


def test_detect_metadata_present_if_set() -> None:
    """Optional: warn (don't fail) if __version__ isn't wired yet."""
    import detect

    if not hasattr(detect, "__version__"):
        warnings.warn(
            f"detect.__version__ not set. Loaded from: {getattr(detect, '__file__', '?')}",
            stacklevel=2,
        )
