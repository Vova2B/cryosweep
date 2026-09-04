"""The gate: no text artist may sit on the data — measured, on every shipped example.

Fixed-position placement has now produced the same defect three times (legend, low-T
inset, reference-line labels). This audit stops the class from coming back silently: it
renders every example at its DEFAULT kind, plus the named reference-line reproducers, and
asserts that no visible text artist covers more than the clear-standard share of any
axis' plotted points. Named per-case exceptions carry their measured number; the global
threshold is never loosened.

THE SCAN MUST PROVE IT CAN SEE. A scan whose renders all raise inside a bare
`except: continue` examines nothing and reports a false all-clear (that scan was
written, twice, during this fix's dispatch). So this test asserts the number of figures
rendered and text artists examined against known minimums, and pins the only tolerated
render failures BY NAME.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pytest

from cryosweep_core.plotting.render import (render_kind, default_kind_for,
                                            _axes_points_in_host_frac, NothingToPlot)
from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.registry import build_default_registry
from cryosweep_core.analyzers.dispatch import analyze_file

EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"

# Default kinds that legitimately cannot render for a shipped example (the file carries
# no series for that kind). Pinned BY NAME: any new failure is a test failure, not a skip.
EXPECTED_RENDER_FAILURES = {
    ("hall_field_sweeps.dat", "resistivity_rho_t"),      # field sweeps only, no rho(T)
    ("magnetization_mpms.dat", "inverse_chi"),           # gated: no molar mass/sample mass
}

# The reference-line label reproducers (found by a 55-kind sweep) on top of the defaults.
EXTRA_CASES = [
    ("heat_capacity_multifield.dat", "hc_full_cp_t"),
    ("resistivity_superconductor.dat", "resistivity_rho_t"),
    ("thermal_transport.dat", "tto_lorenz_t"),
]

# Named exceptions: (file, kind, text prefix) -> measured max points allowed. Empty today;
# add a case here, with its measured number, ONLY when a dense panel genuinely has nowhere
# clear — never by loosening the global standard.
ALLOWED = {}


def _cases():
    reg = build_default_registry()
    cases = []
    for f in sorted(EXAMPLES.glob("*.dat")):
        res = analyze_file(load_dat(str(f)), RunConfig.load(), reg)
        cases.append((f.name, default_kind_for((res.data or {}).get("probe")), res))
    for fname, kind in EXTRA_CASES:
        res = analyze_file(load_dat(str(EXAMPLES / fname)), RunConfig.load(), reg)
        cases.append((fname, kind, res))
    return cases


def test_no_text_artist_sits_on_the_data():
    rendered, examined, failures = 0, 0, []
    seen_failures = set()
    for fname, kind, res in _cases():
        try:
            fig = render_kind([res], kind, PlotSpec(), GlobalStyle())
        except NothingToPlot:
            seen_failures.add((fname, kind))
            continue
        rendered += 1
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        for ax in fig.axes:
            if ax.get_label() == "inset":
                continue
            # hide insets while counting: their own magnified points are not hidden data
            insets = [a for a in fig.axes if a.get_label() == "inset"]
            for a in insets:
                a.set_visible(False)
            try:
                pts = _axes_points_in_host_frac(ax)
            finally:
                for a in insets:
                    a.set_visible(True)
            inv = ax.transAxes.inverted()
            for t in ax.texts:
                if not t.get_visible() or not str(t.get_text()).strip():
                    continue
                examined += 1
                bb = t.get_window_extent(rend)
                (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
                x0, x1 = min(x0, x1), max(x0, x1)
                y0, y1 = min(y0, y1), max(y0, y1)
                n = int(((pts[:, 0] >= x0) & (pts[:, 0] <= x1) &
                         (pts[:, 1] >= y0) & (pts[:, 1] <= y1)).sum()) if len(pts) else 0
                total = len(pts)
                cap = next((v for (af, ak, prefix), v in ALLOWED.items()
                            if af == fname and ak == kind
                            and t.get_text().startswith(prefix)), None)
                ok = (n <= cap) if cap is not None else (n < 3 or (total and n / total <= 0.02))
                if not ok:
                    failures.append(f"{fname} {kind} {t.get_text()[:40]!r}: {n}/{total}")
        plt.close(fig)
    assert seen_failures == EXPECTED_RENDER_FAILURES, (
        f"render-failure set changed: {seen_failures ^ EXPECTED_RENDER_FAILURES}")
    # prove the scan can see: it rendered real figures and looked at real artists
    assert rendered >= 10, f"scan rendered only {rendered} figures — it cannot see"
    assert examined >= 10, f"scan examined only {examined} text artists — it cannot see"
    assert not failures, "text artists on data:\n" + "\n".join(failures)
