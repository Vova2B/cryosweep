import pathlib
import types
import numpy as np
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry
from cryosweep_core.plotting.catalog import BUILTIN_PLOTKINDS
from cryosweep_core.plotting.render import render_kind
from cryosweep_core.plotting.spec import GlobalStyle

FIX = pathlib.Path(__file__).parent / "fixtures"
KINDS = {k.key: k for k in BUILTIN_PLOTKINDS}


def _res(name, probe=None):
    cfg = RunConfig.load(probe_override=probe) if probe else RunConfig.load()
    return analyze_file(load_dat(str(FIX / name)), cfg, build_default_registry())


def _mh_res():
    """Synthetic VSM result carrying M(H) loops (no .dat fixture backs vsm_mh;
    same builder pattern as test_render_vsm_pq3.py)."""
    H = np.linspace(-13000.0, 13000.0, 41)
    loop = {"temperature": 5.0, "field_oe": H.tolist(),
            "moment": np.tanh(H / 3000.0).tolist(), "n_points": 41}
    return types.SimpleNamespace(data={"probe": "vsm", "loops": [loop]})


def test_vsm_mh_x_scaled_and_axis_tesla():
    res = _mh_res()
    oe = KINDS["vsm_mh"].series(res, field_unit="Oe")
    t = KINDS["vsm_mh"].series(res, field_unit="T")
    assert oe and len(oe) == len(t)
    assert np.allclose(np.asarray(t[0].x), np.asarray(oe[0].x) / 1e4)
    assert [s.key for s in oe] == [s.key for s in t]                 # keys invariant
    fig = render_kind(res, "vsm_mh", None, GlobalStyle(field_unit="T"))
    assert "(T)" in fig.axes[0].get_xlabel()
    fig_oe = render_kind(res, "vsm_mh", None, GlobalStyle(field_unit="Oe"))
    assert "(Oe)" in fig_oe.axes[0].get_xlabel()


def test_param_vs_field_axis_and_data_tesla():
    res = _res("hc_multifield_synth.dat")
    oe = KINDS["hc_gamma_vs_field"].series(res, field_unit="Oe")
    t = KINDS["hc_gamma_vs_field"].series(res, field_unit="T")
    assert oe and np.allclose(np.asarray(t[0].x), np.asarray(oe[0].x) / 1e4)
    assert [s.key for s in oe] == [s.key for s in t]                 # keys invariant
    fig = render_kind(res, "hc_gamma_vs_field", None, GlobalStyle(field_unit="T"))
    assert "(T)" in fig.axes[0].get_xlabel()
    fig_oe = render_kind(res, "hc_gamma_vs_field", None, GlobalStyle(field_unit="Oe"))
    assert "(Oe)" in fig_oe.axes[0].get_xlabel()


def test_mr_xlabel_and_data_tesla():
    res = _res("hall_synth.dat", probe="resistivity")
    oe = KINDS["resistivity_mr"].series(res, field_unit="Oe")
    t = KINDS["resistivity_mr"].series(res, field_unit="T")
    assert oe and len(oe) == len(t)
    assert np.allclose(np.asarray(t[0].x), np.asarray(oe[0].x) / 1e4)
    fig = render_kind(res, "resistivity_mr", None, GlobalStyle(field_unit="T"))
    assert "(T)" in fig.axes[0].get_xlabel()
    fig_oe = render_kind(res, "resistivity_mr", None, GlobalStyle(field_unit="Oe"))
    assert "(Oe)" in fig_oe.axes[0].get_xlabel()


def test_mr_h0_refline_stays_zero_both_units():
    res = _res("hall_synth.dat", probe="resistivity")
    for u in ("Oe", "T"):
        fig = render_kind(res, "resistivity_mr", None, GlobalStyle(field_unit=u))
        reflines = [ln for ax in fig.axes for ln in ax.get_lines() if ln.get_gid() == "refline"]
        # any vertical H=0 line has constant x == 0
        assert all(np.allclose(ln.get_xdata(), 0.0) for ln in reflines
                   if len(set(np.round(ln.get_xdata(), 12))) == 1)
