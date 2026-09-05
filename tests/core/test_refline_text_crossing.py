"""The gate: no reference LINE may run through a text artist — measured, on every shipped
example x every built-in plot kind.

Placement decided without measuring has produced this defect class four times now (legend,
low-T inset, reference-line labels, and the rho(T) stats box this file was written for: on
`resistivity_superconductor.dat` the vertical dashed T_c guide ran straight through
`RRR = 86.7 / T_c = 8.03 K (onset 8.80, zero 7.49)`). The text-over-DATA audit next door
cannot see this one — `_axes_points_in_host_frac` skips `gid == "refline"` by construction,
so text-over-refline was unmeasured until now.

THE SCAN MUST PROVE IT CAN SEE, twice over.

1. Vertex containment is blind here. An `axvline` has exactly TWO vertices, at the top and
   the bottom of the axes; against a stats box in the middle of the panel both of them are
   outside it while the segment between them crosses it. A point-in-box version of this
   scan reported a clean zero across all 101 figures with the defect on screen. Only
   segment-vs-rectangle intersection finds it, so the first test below pins that exact
   geometry on the shipped helper.

2. A scan whose renders all raise inside a bare `except: continue` examines nothing and
   reports a false all-clear (that scan was written, twice, in this repo's history). So the
   audit asserts the number of figures rendered, reference lines found and text artists
   examined against measured floors, and pins every non-`NothingToPlot` render failure BY
   NAME.

Two further traps, both of which produced a false all-clear while this was being written:
  - reference lines carry one coordinate in AXES fraction on a blended transform. Use
    `ln.get_transform()`; `ax.transData` yields nonsense (measured: y = -107).
  - `BUILTIN_PLOTKINDS` holds PlotKind OBJECTS. Passing one to `render_kind` in place of
    its `.key` renders zero artists — 101 empty figures, zero hits, all green.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pytest

from cryosweep_core.plotting.render import render_kind, _segment_hits_rect, NothingToPlot
from cryosweep_core.plotting.catalog import BUILTIN_PLOTKINDS
from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.registry import build_default_registry
from cryosweep_core.analyzers.dispatch import analyze_file

EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"

# The reproducer. Asserted to be IN the scan: if this example ever stops producing the box,
# the gate must say so rather than pass because it no longer looks.
REPRODUCER = ("resistivity_superconductor.dat", "resistivity_rho_t")

# Render failures other than NothingToPlot, pinned BY NAME. `NothingToPlot` is the ordinary
# "this file carries no series for that kind" and is counted, not listed (measured: 570 of
# 671 attempts). The one below is a PRE-EXISTING cross-probe defect — a TTO kind rendered
# against an ACMS result broadcasts a (220,) against a (0,) — out of scope for this gate but
# pinned so the set cannot grow silently.
EXPECTED_RENDER_ERRORS = {
    ("ac_susceptibility.dat", "tto_kappa_t"): "ValueError",
}

# Measured 2026-09-05: 101 figures, 35 reference lines, 74 text artists examined. Floors,
# not equalities — a new plot kind must not fail this test, but a scan that has stopped
# seeing must.
MIN_FIGURES, MIN_REFLINES, MIN_TEXTS = 90, 30, 65

# A reference line's own rotated label sits ON its line by design (measured: 4 of them).
# Identified by gid prefix, never by suppressing text wholesale — the box this gate exists
# for carries no gid at all, and a blanket skip would hide it.
OWN_LABEL_GID_PREFIX = "refline-label"

# Named exceptions: (file, kind, text prefix) -> the measured reason it is tolerated. Empty
# today. Add a case here, with its measurement, ONLY when a panel genuinely has nowhere
# clear to put the box — never by loosening the rule for every figure.
ALLOWED = {}


def test_segment_hits_rect_sees_what_vertex_containment_cannot():
    """The shipped defect's geometry, in the small: a two-vertex vertical line crossing a
    wide, short box whose four corners are all outside the line's endpoints."""
    box = (0.1, 0.4, 0.9, 0.6)                    # a wide stats box across the panel middle
    axvline = np.array([[0.5, 0.0], [0.5, 1.0]])  # exactly two vertices: bottom and top
    assert not any(box[0] <= x <= box[2] and box[1] <= y <= box[3] for x, y in axvline), (
        "premise of this test: BOTH vertices lie outside the box — which is precisely why a "
        "point-in-box scan reports a clean zero while the line crosses the text")
    assert _segment_hits_rect(axvline, box)


@pytest.mark.parametrize("pts, expect", [
    ([[0.5, 0.0], [0.5, 1.0]], True),    # axvline through the box (the defect)
    ([[0.05, 0.0], [0.05, 1.0]], False),  # axvline left of it
    ([[0.95, 0.0], [0.95, 1.0]], False),  # axvline right of it
    ([[0.0, 0.5], [1.0, 0.5]], True),    # axhline through it
    ([[0.0, 0.9], [1.0, 0.9]], False),   # axhline above it
    ([[0.0, 0.0], [1.0, 1.0]], True),    # diagonal crossing a corner region
    ([[0.0, 0.0], [0.05, 0.05]], False),  # short segment nowhere near
])
def test_segment_hits_rect_cases(pts, expect):
    assert _segment_hits_rect(np.asarray(pts, float), (0.1, 0.4, 0.9, 0.6)) is expect


def _refline_polylines(ax):
    """Reference lines on `ax`, in DISPLAY px. `ln.get_transform()`, never `ax.transData`."""
    out = []
    for ln in ax.get_lines():
        if ln.get_gid() != "refline" or not ln.get_visible():
            continue
        xy = np.asarray(ln.get_xydata(), float)
        if len(xy) >= 2:
            out.append(ln.get_transform().transform(xy))
    return out


def test_no_reference_line_crosses_a_text_artist():
    reg = build_default_registry()
    keys = [k.key for k in BUILTIN_PLOTKINDS]     # .key — the OBJECT renders zero artists
    figures = reflines = texts = 0
    own_label_touches = 0
    seen_errors, failures, examined = {}, [], set()

    for f in sorted(EXAMPLES.glob("*.dat")):
        res = analyze_file(load_dat(str(f)), RunConfig.load(), reg)
        for key in keys:
            try:
                fig = render_kind([res], key, PlotSpec(), GlobalStyle())
            except NothingToPlot:
                continue
            except Exception as exc:              # never swallowed: pinned or reported
                seen_errors[(f.name, key)] = type(exc).__name__
                continue
            figures += 1
            examined.add((f.name, key))
            fig.canvas.draw()
            rend = fig.canvas.get_renderer()
            for ax in fig.axes:
                lines = _refline_polylines(ax)
                reflines += len(lines)
                for t in ax.texts:
                    if not t.get_visible() or not str(t.get_text()).strip():
                        continue
                    texts += 1
                    if not lines:
                        continue
                    bb = t.get_window_extent(rend)
                    rect = (bb.x0, bb.y0, bb.x1, bb.y1)
                    if not any(_segment_hits_rect(pl, rect) for pl in lines):
                        continue
                    if (t.get_gid() or "").startswith(OWN_LABEL_GID_PREFIX):
                        own_label_touches += 1     # a label on its OWN line: by design
                        continue
                    txt = t.get_text().replace("\n", " / ")
                    if any(af == f.name and ak == key and txt.startswith(prefix)
                           for (af, ak, prefix) in ALLOWED):
                        continue
                    failures.append(f"{f.name} {key} {txt[:60]!r}")
            plt.close(fig)

    assert seen_errors == EXPECTED_RENDER_ERRORS, (
        f"render-error set changed: {set(seen_errors.items()) ^ set(EXPECTED_RENDER_ERRORS.items())}")
    # prove the scan can see: it rendered real figures, found real reference lines, and
    # looked at real text artists
    assert figures >= MIN_FIGURES, f"scan rendered only {figures} figures — it cannot see"
    assert reflines >= MIN_REFLINES, f"scan found only {reflines} reference lines — it cannot see"
    assert texts >= MIN_TEXTS, f"scan examined only {texts} text artists — it cannot see"
    assert own_label_touches >= 1, (
        "no reference-line label was found sitting on its own line (measured: 4) — the "
        "expected-touch branch is dead, so the gate is no longer distinguishing them")
    assert REPRODUCER in examined, (
        f"{REPRODUCER} did not render — the defect's own reproducer is not being scanned")
    assert not failures, "reference lines crossing text:\n" + "\n".join(failures)
