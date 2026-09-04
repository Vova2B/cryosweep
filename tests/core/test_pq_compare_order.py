"""The checker must never draw the canvas before the image is captured.

A pre-capture draw() changes first-save tight-bbox bytes on 10 of the 41 renderable
gallery entries (spec 2026-08-10-visual-gate-hardening-design.md 4.3), so the call
ORDER inside _run_entry is a byte-identity contract, not an implementation detail.
"""
from __future__ import annotations

import pathlib
import sys

TOOLS = pathlib.Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def test_check_runs_after_image_capture(monkeypatch):
    """Pin the CALL ORDER, not the source text.

    A source-index assertion is fragile to the point of being wrong: a comment
    naming _check_fig above the _fig_to_img line flips the comparison. Record the
    real order by monkeypatching both functions.
    """
    import pq_compare

    calls: list[str] = []
    monkeypatch.setattr(pq_compare, "_render_v2", lambda e: ("FIG", "ok"))
    monkeypatch.setattr(pq_compare, "_fig_to_img", lambda f: calls.append("img"))
    monkeypatch.setattr(pq_compare, "_check_fig", lambda e, f, s: (calls.append("check"), [])[1])
    monkeypatch.setattr(pq_compare, "_tile", lambda e, i: None)
    monkeypatch.setattr(pq_compare.plt, "close", lambda f: None)

    pq_compare._run_entry({"id": "x", "v2_kind": "k"}, True)

    assert calls == ["img", "check"], (
        f"byte-identity contract: capture must precede checks, got {calls}"
    )
