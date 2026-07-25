from __future__ import annotations

from surveillance_app.utils.formatting import safe_text


def test_safe_text_strips_whitespace_and_handles_none() -> None:
    assert safe_text(None) == ""
    assert safe_text("  run name  ") == "run name"
    assert safe_text("value") == "value"

