"""The font sweep re-renders at each size; it must not disturb the pinned 9pt gallery.

The sweep exists because the hc_full_cp_t legend/inset occlusion is INVISIBLE at the
9pt baseline and severe at 16pt -- goodness at one font size says nothing about layout
robustness at another, and GlobalStyle.font_pt is a user-facing GUI knob.
"""
from __future__ import annotations

import pathlib
import sys

TOOLS = pathlib.Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def test_sweep_default_sizes():
    import pq_compare

    assert pq_compare.SWEEP_DEFAULT == (9.0, 12.0, 16.0)


def test_parse_font_pt_list():
    import pq_compare

    assert pq_compare._parse_font_pts("9,12,16") == (9.0, 12.0, 16.0)
    assert pq_compare._parse_font_pts("11") == (11.0,)
    assert pq_compare._parse_font_pts(" 9 , 16 ") == (9.0, 16.0)


def test_style_is_restored_after_sweep():
    """The module-level STYLE must not be left mutated -- later callers read it."""
    import pq_compare

    before = pq_compare.STYLE
    swept = pq_compare._with_font_pt(before, 16.0)
    assert swept.font_pt == 16.0
    assert before.font_pt == 9.0, "the base style must not be mutated in place"
    assert pq_compare.STYLE is before
