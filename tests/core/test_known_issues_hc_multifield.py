"""KNOWN-ISSUES #14 / #15 — heat-capacity multifield display defects
(`heat_capacity_multifield.dat`, `hc_lowt_multifield`).

#14 Field setpoint labels printed raw group medians ('0.524968 Oe', '50000.5 Oe',
    '100001 Oe') — six significant figures of instrument noise on a held setpoint.
#15 Four fields x four low-T models drew sixteen fit curves in colours unrelated to their
    data series, several overshooting the axes; which fit belonged to which field could
    not be read off the plot.
"""
import pathlib
import numpy as np
from matplotlib.colors import to_rgba
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.registry import build_default_registry
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.plotting.render import render_kind

ROOT = pathlib.Path(__file__).resolve().parents[2]
_C = {}


def _res():
    if "r" not in _C:
        _C["r"] = analyze_file(load_dat(str(ROOT / "examples" / "heat_capacity_multifield.dat")),
                               RunConfig.load(), build_default_registry())
    return _C["r"]


def test_setpoint_labels_are_rounded_display_values():
    from cryosweep_core.plotting.catalog import series_hc_lowt_multifield
    labels = {s.label for s in series_hc_lowt_multifield(_res())}
    assert labels == {"0 Oe", "50000 Oe", "100000 Oe", "130000 Oe"}


def test_fit_lines_wear_their_field_groups_colour():
    fig = render_kind(_res(), "hc_lowt_multifield")
    ax = fig.axes[0]
    leg = ax.get_legend()
    assert leg is not None
    gcol = {t.get_text(): to_rgba(h.get_color())
            for t, h in zip(leg.get_texts(), leg.legend_handles) if "Oe" in t.get_text()}
    fits = [ln for ln in ax.lines if ln.get_gid() == "fit"]
    assert fits
    for ln in fits:
        field = ln.get_label().split("@")[1]        # "<model>@<field_oe:g>"
        # the display tag rounds the setpoint; recover it the same way
        from cryosweep_core.plotting.catalog import fmt_field_setpoint
        tag = fmt_field_setpoint(float(field))
        assert to_rgba(ln.get_color()) == gcol[tag], (ln.get_label(), tag)


def test_models_are_distinguished_by_linestyle_and_named_in_legend():
    fig = render_kind(_res(), "hc_lowt_multifield")
    ax = fig.axes[0]
    fits = [ln for ln in ax.lines if ln.get_gid() == "fit"]
    by_field = {}
    for ln in fits:
        by_field.setdefault(ln.get_label().split("@")[1], []).append(ln.get_linestyle())
    for field, styles in by_field.items():
        assert len(styles) == len(set(styles)), (field, styles)   # one style per model
    texts = {t.get_text() for t in ax.get_legend().get_texts()}
    assert {"Debye T³", "Debye T³+T⁵", "spin-fl non-int", "spin-fl weak"} <= texts


def test_view_is_framed_by_the_data_not_the_fit_overshoot():
    fig = render_kind(_res(), "hc_lowt_multifield")
    ax = fig.axes[0]
    ys = np.concatenate([np.asarray(ln.get_ydata(), float) for ln in ax.lines
                         if ln.get_gid() is None])
    ys = ys[np.isfinite(ys)]
    dlo, dhi = float(ys.min()), float(ys.max())
    span = dhi - dlo
    lo, hi = ax.get_ylim()
    assert hi <= dhi + 0.15 * span, (hi, dhi)   # fits clip at the frame instead of
    assert lo >= dlo - 0.15 * span, (lo, dlo)   # stretching the axes around themselves
