import pathlib, dataclasses, pytest
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry
from cryosweep_core.plotting.spec import PlotSpec
from cryosweep_core.plotting.catalog import BUILTIN_PLOTKINDS, overlay_series, OverlayFile
from cryosweep_core.plotting.render import render_kind

FIX = pathlib.Path(__file__).parent / "fixtures"
KINDS = {k.key: k for k in BUILTIN_PLOTKINDS}

def _reg(): return build_default_registry()

def _vsm():
    rt = load_dat(str(FIX / "vsm_synth.dat"))
    rt = dataclasses.replace(rt, header=dataclasses.replace(rt.header, molar_mass=200.0, mass_mg=5.0))
    return analyze_file(rt, RunConfig.load(probe_override="vsm"), _reg())

def _hc():
    return analyze_file(load_dat(str(FIX / "hc_synth.dat")),
                        RunConfig.load(probe_override="heatcapacity"), _reg())

def _res_t():    # act_synth: metallic zero-field T ramps + fits, no field loops
    return analyze_file(load_dat(str(FIX / "act_synth.dat")),
                        RunConfig.load(probe_override="resistivity"), _reg())

def _res_h():    # hall_synth as resistivity: field loops w/ rho_zero_field
    return analyze_file(load_dat(str(FIX / "hall_synth.dat")),
                        RunConfig.load(probe_override="resistivity"), _reg())

def _hall():
    return analyze_file(load_dat(str(FIX / "hall_synth.dat")),
                        RunConfig.load(probe_override="hall",
                                       hall={"hall_channel": 1, "thickness_mm": 0.5}), _reg())

def _solid_lines(ax):
    return [l for l in ax.lines if l.get_gid() == "fit"]

def test_chi_t_product_series_single_curve():
    res = _vsm()
    s = KINDS["vsm_chi_t_product"].series(res)
    assert len(s) == 1 and s[0].default_on
    assert len(s[0].x) == len(s[0].y) > 0

def test_chi_t_product_renders():
    fig = render_kind(_vsm(), "vsm_chi_t_product", PlotSpec())
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Temperature (K)" and "χT" in ax.get_ylabel()
    assert len(ax.lines) == 1

def test_hc_c_over_t_linear_series():
    s = KINDS["hc_c_over_t_linear"].series(_hc())
    assert len(s) == 1 and len(s[0].x) == len(s[0].y) > 0

def test_hc_c_over_t_linear_renders_linear_x():
    fig = render_kind(_hc(), "hc_c_over_t_linear", PlotSpec())
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Temperature (K)" and "Cp/T" in ax.get_ylabel()
    assert ax.get_xscale() == "linear" and len(ax.lines) == 1

def test_hall_n_t_series_filters_none_and_sorts():
    s = KINDS["hall_n_t"].series(_hall())
    assert len(s) == 1
    xs = s[0].x
    assert xs == sorted(xs) and all(v is not None for v in s[0].y)

def test_hall_n_t_default_log_y():
    fig = render_kind(_hall(), "hall_n_t", PlotSpec())
    ax = fig.axes[0]
    assert ax.get_yscale() == "log"
    assert ax.get_xlabel() == "Temperature (K)" and "n" in ax.get_ylabel()

def test_hall_n_t_empty_without_thickness():
    res = analyze_file(load_dat(str(FIX / "hall_synth.dat")),
                       RunConfig.load(probe_override="hall", hall={"hall_channel": 1}), _reg())
    assert KINDS["hall_n_t"].series(res) == []

def test_hall_r2_t_series():
    s = KINDS["hall_r2_t"].series(_hall())
    assert len(s) == 1
    assert s[0].x == sorted(s[0].x) and all(0.0 <= v <= 1.0 + 1e-9 for v in s[0].y)

def test_hall_r2_t_renders():
    fig = render_kind(_hall(), "hall_r2_t", PlotSpec())
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Temperature (K)" and "R²" in ax.get_ylabel()
    assert len(ax.lines) == 1

def test_mr_pct_series_normalized():
    import numpy as np
    res = _res_h()
    s = KINDS["resistivity_mr_pct"].series(res)
    assert s, "expected MR% series from field loops"
    # MR% must equal (rho - rho0)/rho0*100 recomputed from the matching raw rho_h curve
    by_key = {}
    for b in res.data["bridges"]:
        for c in b["rho_h_curves"]:
            rho0 = c.get("rho_zero_field")
            if rho0 and rho0 > 0:
                k = f"b{b['channel']}:H:" + ("na" if c["held_temp_k"] is None else f"{c['held_temp_k']:.1f}") \
                    + f":{c['direction']}"
                by_key[k] = (np.asarray(c["rho"], float), rho0)
    for sr in s:
        assert sr.key.startswith("b") and ":H:" in sr.key
        rho, rho0 = by_key[sr.key]
        assert np.allclose(sr.y, (rho - rho0) / rho0 * 100.0)

def test_mr_pct_skips_none_rho_zero_field():
    res = _res_h()
    # null out one loop's rho_zero_field -> that loop is skipped
    b0 = res.data["bridges"][0]
    n_before = len(KINDS["resistivity_mr_pct"].series(res))
    b0["rho_h_curves"][0]["rho_zero_field"] = None
    n_after = len(KINDS["resistivity_mr_pct"].series(res))
    assert n_after == n_before - 1

def test_mr_pct_renders():
    fig = render_kind(_res_h(), "resistivity_mr_pct", PlotSpec())
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Field (Oe)" and "MR" in ax.get_ylabel()
    assert len(ax.lines) >= 1

def test_rho_t2_series_x_is_t_squared():
    res = _res_t()
    s = KINDS["resistivity_rho_t2"].series(res)
    assert s, "expected rho vs T^2 series"
    assert all(sr.key.startswith("b") and ":T:" in sr.key for sr in s)
    # widest ramp default_on, one per bridge
    for b in {sr.group for sr in s}:
        assert sum(1 for sr in s if sr.group == b and sr.default_on) == 1
    # x is T^2 (monotone increasing, max well above the T-axis max of ~300 -> ~90000)
    assert max(s[0].x) > 1000.0

def test_rho_t2_both_fit_lines_default():
    res = _res_t()                                  # 2 metallic bridges, each w/ both fits
    fig = render_kind(res, "resistivity_rho_t2", PlotSpec())   # fit_lines=None -> all
    assert len(_solid_lines(fig.axes[0])) == 4      # 2 bridges x (linear + power_law)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "T² (K²)" and "ρ" in ax.get_ylabel()

def test_rho_t2_fit_lines_independently_toggleable():
    res = _res_t()
    assert len(_solid_lines(render_kind(res, "resistivity_rho_t2", PlotSpec(fit_lines=())).axes[0])) == 0
    assert len(_solid_lines(render_kind(res, "resistivity_rho_t2", PlotSpec(fit_lines=("linear",))).axes[0])) == 2
    assert len(_solid_lines(render_kind(res, "resistivity_rho_t2", PlotSpec(fit_lines=("power_law",))).axes[0])) == 2

def test_rho_t2_fit_line_x_within_window():
    res = _res_t()
    fig = render_kind(res, "resistivity_rho_t2", PlotSpec(fit_lines=("linear",)))
    f = res.data["bridges"][0]["rho_t2_linear"]
    hi2 = f["fit_range"][1] ** 2                    # window max in K^2
    for l in _solid_lines(fig.axes[0]):
        assert max(l.get_xdata()) <= hi2 + 1e-6

def test_rho_t2_power_law_skipped_when_unresolved_or_absent():
    res = _res_t()
    res.data["bridges"][0]["power_law"]["quality_flags"].append("rho0_unresolved")
    res.data["bridges"][1]["power_law"] = None
    # only the two betaT2 (linear) lines survive; both power_law lines skipped
    n = len(_solid_lines(render_kind(res, "resistivity_rho_t2", PlotSpec()).axes[0]))
    assert n == 2

def test_rho_t2_no_fit_lines_in_overlay():
    res = _res_t()
    ov = [OverlayFile(file_id=0, label="A")]
    fig = render_kind([res], "resistivity_rho_t2", PlotSpec(), overlay=ov)
    assert len(_solid_lines(fig.axes[0])) == 0
