"""Smoke tests for the synth module.

Minimal by design: verify the package imports cleanly. If import succeeds,
the editable install and sys.path are correctly wired. Metadata like
__version__ / __schema_version__ is optional and only warned on, not gated.
"""
from __future__ import annotations

import warnings


def test_synth_importable() -> None:
    """synth top-level package imports without side effects."""
    import synth  # noqa: F401  (import IS the assertion)


def test_synth_has_file_attribute() -> None:
    """synth resolves to an on-disk file, not a namespace package stub."""
    import synth

    assert getattr(synth, "__file__", None), (
        "synth has no __file__ - likely a broken editable install or "
        "namespace package collision"
    )


def test_synth_metadata_present_if_set() -> None:
    """Optional: warn (don't fail) if __version__ isn't wired yet."""
    import synth

    if not hasattr(synth, "__version__"):
        warnings.warn(
            f"synth.__version__ not set. Loaded from: {getattr(synth, '__file__', '?')}",
            stacklevel=2,
        )
