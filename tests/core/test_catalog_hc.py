from cryosweep_core.plotting.catalog import BUILTIN_PLOTKINDS


def test_cp_vs_t_registered():
    kinds = {k.key for k in BUILTIN_PLOTKINDS}
    assert "cp_vs_t" in kinds
    assert "cp_over_t" in kinds        # existing untouched


import numpy as np, pytest
from cryosweep_core.plotting.catalog import get_kind   # added in Task 7
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry
import pathlib
FIX = pathlib.Path(__file__).parent / "fixtures"

def _mf_res():
    return analyze_file(load_dat(str(FIX / "hc_multifield_synth.dat")),
                        RunConfig.load(), build_default_registry())

def test_gamma_vs_field_series_present_and_rising():
    res = _mf_res()
    series = get_kind("hc_gamma_vs_field").series(res)
    deb = next(s for s in series if "Debye T³" in s.label and "T⁵" not in s.label)
    assert deb.x == sorted(deb.x)
    assert deb.y[0] < deb.y[-1]                          # gamma rises with field
    assert deb.yerr is not None

def test_thetaD_vs_field_lattice_models_only():
    res = _mf_res()
    series = get_kind("hc_thetaD_vs_field").series(res)
    labels = {s.label for s in series}
    assert not any("spin" in l.lower() for l in labels)  # no spin-fluct theta_D

def test_param_vs_field_hidden_for_single_field(hc_synth_path):
    res = analyze_file(load_dat(str(hc_synth_path)), RunConfig.load(), build_default_registry())
    assert get_kind("hc_gamma_vs_field").series(res) == []

def test_param_vs_field_renders(tmp_path):
    from cryosweep_core.plotting.render import render_hc_gamma_vs_field   # the concrete renderer from Task 7
    res = _mf_res()
    fig = render_hc_gamma_vs_field([res])
    out = tmp_path / "g.png"; fig.savefig(out); assert out.exists()


def test_lowt_multifield_one_series_per_field():
    res = _mf_res()
    series = get_kind("hc_lowt_multifield").series(res)
    groups = {s.group for s in series}
    assert len(groups) == 3                               # one per field
    s0 = series[0]
    assert len(s0.x) == len(s0.y) > 0                     # Cp/T vs T^2 points
