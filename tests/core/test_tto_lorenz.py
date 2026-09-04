"""tto_lorenz_t: L/L0 vs T with the Sommerfeld reference line (spec §4)."""
import matplotlib; matplotlib.use("Agg")       # noqa: E702
import json
import pathlib

import numpy as np
import pytest

from cryosweep_core.config import RunConfig
from cryosweep_core.io.loader import load_dat
from cryosweep_core.plotting.catalog import BUILTIN_PLOTKINDS, series_tto_lorenz_t
from cryosweep_core.plotting.render import _CONNECT_KINDS, _RENDERERS, default_kind_for, render_kind
from cryosweep_core.plotting.spec import GlobalStyle, PlotSpec

FX = pathlib.Path("tests/core/fixtures")
from tests.core.conftest import repo_root

REPO = repo_root()   # the repo root (docs/, skill/ and the real data live there, not in the app folder)



def _run(path):
    from cryosweep_core.analyzers.tto import TTOAnalyzer
    return TTOAnalyzer().analyze(load_dat(str(path)), RunConfig())


def test_kind_registered_with_its_label_and_group_colouring():
    k = {x.key: x for x in BUILTIN_PLOTKINDS}["tto_lorenz_t"]
    assert k.probe == "tto"
    assert k.label == "L/L₀ vs T"
    assert k.group_colored is True
    assert k.default_xscale == "linear" and k.default_yscale == "linear"
    assert "tto_lorenz_t" in _CONNECT_KINDS
    assert "tto_lorenz_t" in _RENDERERS


def test_the_default_kind_for_tto_is_still_the_headline():
    assert default_kind_for("tto") == "tto_summary_t"


def test_series_key_prefix_extends_the_pinned_scheme():
    s = series_tto_lorenz_t(_run(FX / "tto_synth.dat"))
    assert sorted(x.key for x in s) == ["lorenz:0:down", "lorenz:90000:down"]


def test_series_builder_defaults_field_unit_so_pq_compare_can_call_it():
    # pq_compare._render_v2 calls KINDS[kind].series(result) with NO field_unit.
    assert series_tto_lorenz_t(_run(FX / "tto_synth.dat"))


def test_series_values_are_the_analyzers_lorenz_ratio():
    r = _run(FX / "tto_synth.dat")
    s = [x for x in series_tto_lorenz_t(r) if x.key == "lorenz:0:down"][0]
    curve = [c for c in r.data["curves"] if c["field_oe"] == 0.0][0]
    want = np.array([np.nan if v is None else v for v in curve["lorenz_ratio"]], float)
    assert np.allclose(np.asarray(s.y, float), want, equal_nan=True)


def test_series_empty_without_rho_so_the_kind_is_unavailable():
    assert series_tto_lorenz_t(_run(FX / "tto_norho_synth.dat")) == []


def test_render_draws_the_sommerfeld_reference_line_at_one():
    fig = render_kind(_run(FX / "tto_synth.dat"), "tto_lorenz_t", PlotSpec(), GlobalStyle())
    ax = fig.axes[0]
    reflines = [ln for ln in ax.lines if ln.get_gid() == "refline"]
    assert len(reflines) == 1
    assert float(np.asarray(reflines[0].get_ydata(), float)[0]) == pytest.approx(1.0)
    assert ax.get_ylabel() == "L/L₀"
    assert ax.get_xlabel() == "Temperature (K)"


def _refline_labels(ax):
    return [t.get_text() for t in ax.texts if t.get_gid() == "refline-label"]


def test_the_reference_line_is_labelled_when_it_is_in_view(tto_real_path):
    # Meets-or-exceeds the cited gallery reference, whose technique is a *labeled* horizontal
    # reference line (README: "dashed horizontal Dulong–Petit reference line (labeled)").
    fig = render_kind(_run(tto_real_path), "tto_lorenz_t", PlotSpec(), GlobalStyle())
    ax = fig.axes[0]
    bottom, top = ax.get_ylim()
    assert bottom <= 1.0 <= top                       # premise of the guard
    assert _refline_labels(ax) == ["Wiedemann–Franz (L = L₀)"]
    t = [x for x in ax.texts if x.get_gid() == "refline-label"][0]
    assert t.get_position()[1] == pytest.approx(1.0)  # pinned TO the line, not floating
    assert t.get_transform() is ax.get_yaxis_transform()


def test_the_label_scales_with_font_pt(tto_real_path):
    # rcParams is never touched by this renderer, so a literal fontsize="small" would be a
    # FIXED absolute size and under-scale on a 14 pt figure. Pinned to the annotation idiom.
    r = _run(tto_real_path)
    for pt in (9, 14):
        ax = render_kind(r, "tto_lorenz_t", PlotSpec(), GlobalStyle(font_pt=pt)).axes[0]
        t = [x for x in ax.texts if x.get_gid() == "refline-label"][0]
        assert t.get_fontsize() == pytest.approx(pt - 1)


def test_the_label_is_suppressed_when_one_is_out_of_view():
    # The synth fixtures run 1.013-1.030, so the robust view excludes 1.0 and an unguarded
    # label would be drawn on canvas outside the axes.
    fig = render_kind(_run(FX / "tto_synth.dat"), "tto_lorenz_t", PlotSpec(), GlobalStyle())
    ax = fig.axes[0]
    assert ax.get_ylim()[0] > 1.0                     # premise: 1.0 really is out of view
    assert _refline_labels(ax) == []


def test_the_label_is_suppressed_on_an_empty_selection():
    # An empty axes keeps the default 0-1 view, which brackets 1.0: the in-view guard alone
    # would label the explanatory note's blank canvas.
    fig = render_kind(_run(FX / "tto_norho_synth.dat"), "tto_lorenz_t", PlotSpec(),
                      GlobalStyle())
    assert _refline_labels(fig.axes[0]) == []


def test_render_tolerates_an_empty_selection_and_explains_it():
    # Inherited from _tto_single: a rho-less file gets an explanatory note, not a blank plot.
    fig = render_kind(_run(FX / "tto_norho_synth.dat"), "tto_lorenz_t", PlotSpec(),
                      GlobalStyle())
    ax = fig.axes[0]
    assert [ln for ln in ax.lines if ln.get_gid() is None] == []
    notes = [t.get_text() for t in ax.texts]
    assert "requires finite ρ > 0" in notes


def test_lorenz_carries_no_error_band_even_when_asked():
    # I2/§7: L/L0 is DERIVED; uncertainty propagation through kappa*rho/(L0*T) is deferred.
    from cryosweep_core.plotting.render import _TTO_BAND_KINDS
    assert "tto_lorenz_t" not in _TTO_BAND_KINDS
    # The render assertion below is NON-DISCRIMINATING today: force-adding the kind to
    # _TTO_BAND_KINDS still yields no band, because there is no `lorenz_ratio_std` for
    # _tto_draw to take a yerr from. Kept as a FORWARD guard -- it becomes live the moment
    # a propagated sigma is added. The membership assert above is the one with teeth today.
    fig = render_kind(_run(FX / "tto_synth.dat"), "tto_lorenz_t",
                      PlotSpec(error_band=True), GlobalStyle())
    assert [c for ax in fig.axes for c in ax.collections if c.get_gid() == "errband"] == []


def test_real_file_full_lorenz_range_stays_inside_the_axes(tto_real_path):
    # Measured span on the gate file: 1.874 - 7.786. A line-derived robust view must not crop
    # either end of it -- this is the assertion Step 4's measurement exists to satisfy.
    fig = render_kind(_run(tto_real_path), "tto_lorenz_t", PlotSpec(), GlobalStyle())
    ax = fig.axes[0]
    y = np.concatenate([np.asarray(ln.get_ydata(), float) for ln in ax.lines
                        if ln.get_gid() is None])
    y = y[np.isfinite(y)]
    assert y.min() == pytest.approx(1.874, abs=0.01)
    assert y.max() == pytest.approx(7.786, abs=0.01)
    bottom, top = ax.get_ylim()
    assert bottom <= y.min() and top >= y.max()


def test_gallery_manifest_has_a_sixth_tto_entry_for_the_new_kind(tto_real_path):
    m = json.loads((REPO / "docs/superpowers/pq-reference-gallery/"
                           "manifest.json").read_text())
    tto = [e for e in m if e.get("probe") == "tto"]
    assert len(tto) == 6
    e = [x for x in tto if x["id"] == "tto_lorenz_t"][0]
    assert e["v2_kind"] == "tto_lorenz_t"
    assert e["dat"] == str(tto_real_path.relative_to(REPO))
    assert e["reference_images"]
    assert any("L/L₀ = 1" in t for t in e["expected_techniques"])
