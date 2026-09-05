"""KNOWN-ISSUES #2 / #3 / #24 — Hall display defects, pinned against shipped examples.

#2  Two estimators (antisym + 0-field+1 fallback) drawn in one R_H(T) panel with no visual
    separation: the ~20% step at the method boundary reads as physics.
#3  Axis offset notation at Hall magnitudes: matplotlib concatenates scale and offset into
    an unreadable `1e-11-2.5e-7` header, so the headline R_H cannot be read off the plot.
#24 The summary's offset third spine carries its `J (A/m²)` label past the figure's right
    edge at the bare GlobalStyle default size.
"""
import pathlib
import pytest
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.registry import build_default_registry
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.plotting.render import render_kind

ROOT = pathlib.Path(__file__).resolve().parents[2]
_REG = build_default_registry()


def _analyze(example, probe, width_mm=None, **hall):
    cfg = RunConfig.load(probe_override=probe)
    for k, v in hall.items():
        setattr(cfg.hall, k, v)
    if width_mm is not None:
        cfg.geometry.width_mm = width_mm
    return analyze_file(load_dat(str(ROOT / "examples" / example)), cfg, _REG)


# ---------------- #2: estimator families must be visually separable -----------------------

def _tdep_result():
    return _analyze("hall_temperature_dependence.dat", "hall_tdep",
                    hall_channel=1, thickness_mm=0.5)


def test_fallback_estimator_is_drawn_open_and_dashed():
    fig = render_kind(_tdep_result(), "hall_tdep_RH_T")
    ax = fig.axes[0]
    by_label = {ln.get_label(): ln for ln in ax.lines if not ln.get_label().startswith("_")}
    anti = by_label["R_H (antisym)"]
    twop = by_label["R_H (0-field+1)"]
    # the fallback is visually secondary: open (hollow) markers + dashed connector, vs the
    # trusted estimator's filled markers + solid line
    assert twop.get_markerfacecolor() == "none"
    assert anti.get_markerfacecolor() != "none"
    assert twop.get_linestyle() != anti.get_linestyle()


@pytest.mark.parametrize("kind", ["hall_tdep_RH_T", "hall_tdep_n_T"])
def test_method_boundary_warning_appears_when_both_estimators_present(kind):
    # n_T matters separately: its wide log tick labels shift the axes (and the centred
    # title) right, so a raw-width fit can pass while the right edge still clips
    fig = render_kind(_tdep_result(), kind)
    ax = fig.axes[0]
    assert "method" in ax.get_title().lower()      # "…steps between estimators are method…"
    # and the warning is fully INSIDE the canvas — a note clipped at an edge warns no one
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    bb = ax.title.get_window_extent(rend)
    fw = fig.get_window_extent(rend)
    assert bb.x0 >= fw.x0 - 0.5 and bb.x1 <= fw.x1 + 0.5, (bb.x0, bb.x1, fw.x1)


def test_no_warning_when_only_one_estimator_family():
    # deselect the fallback series: with a single estimator family on the axes there is no
    # method boundary, so no warning to distract from a clean figure
    from cryosweep_core.plotting.spec import PlotSpec
    fig = render_kind(_tdep_result(), "hall_tdep_RH_T", PlotSpec(curves=["R_H_antisym"]))
    labels = [ln.get_label() for ln in fig.axes[0].lines]
    assert not any("0-field" in l for l in labels)
    assert fig.axes[0].get_title() == ""


# ---------------- #3: no concatenated scale+offset header ---------------------------------

@pytest.mark.parametrize("example,probe,kind,hall", [
    ("hall_field_sweeps.dat", "hall", "hall_rh_t",
     dict(hall_channel=1, thickness_mm=0.5, longitudinal_channel=2)),
    ("hall_temperature_dependence.dat", "hall_tdep", "hall_tdep_RH_T",
     dict(hall_channel=1, thickness_mm=0.5)),
    # Every kind that draws an R_H axis, not just the one the item named: the summary and
    # the twin were rendering a plain "1e-7" header while hall_tdep_RH_T carried mathtext,
    # i.e. the same quantity formatted two ways depending on which kind you opened.
    ("hall_temperature_dependence.dat", "hall_tdep", "hall_tdep_summary",
     dict(hall_channel=1, thickness_mm=0.5, longitudinal_channel=2)),
    ("hall_temperature_dependence.dat", "hall_tdep", "hall_tdep_rh_n_twin",
     dict(hall_channel=1, thickness_mm=0.5)),
])
def test_r_h_axis_never_concatenates_scale_and_offset(example, probe, kind, hall):
    fig = render_kind(_analyze(example, probe, **hall), kind)
    ax = fig.axes[0]
    fmt = ax.yaxis.get_major_formatter()
    assert fmt.get_useOffset() is False            # offset off: ticks carry absolute values
    fig.canvas.draw()
    off = ax.yaxis.get_offset_text().get_text()
    # the header may carry ONE power of ten (the scale), never a second appended exponent
    assert off.count("10^") <= 1 and "e-" not in off, off


# ---------------- #24: the J label stays inside the figure --------------------------------

def test_summary_j_label_inside_figure_at_default_size():
    res = _analyze("hall_mixed_sweeps.dat", "hall_tdep",
                   hall_channel=1, thickness_mm=0.07, longitudinal_channel=2, width_mm=2.0)
    fig = render_kind(res, "hall_tdep_summary")
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    jax = next(a for a in fig.axes if a.yaxis.label.get_text() == "J (A/m²)")
    bb = jax.yaxis.label.get_window_extent(rend)
    fig_w = fig.get_window_extent(rend).x1
    assert bb.x1 <= fig_w + 0.5, (bb.x1, fig_w)    # label fully inside the canvas
    # and the mu axis decorations must still be clear of the J spine (the item-21 fix)
    mu = next(a for a in fig.axes if a.yaxis.label.get_text() == "μ (m²/V·s)")
    sp = jax.spines["right"].get_window_extent(rend)
    assert sp.x0 >= mu.yaxis.label.get_window_extent(rend).x1
