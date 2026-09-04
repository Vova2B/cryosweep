import numpy as np
from cryosweep_core.plotting.render import render_kind
from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle

def _heavy_tail_result():
    from cryosweep_core.result import Result, Provenance
    T = list(np.linspace(2, 300, 200))
    # realistic metallic bulk (gentle ramp, ~3-5e-5; MAD>0 so the robust scale is well-defined)
    # plus a 7-point heavy tail at ~1.46e-2; bulk stays < 1e-3 for the assertion.
    rho = list(np.linspace(3e-5, 5e-5, 200))
    rho[-7:] = [1.46e-2] * 7
    data = {"probe": "resistivity", "rho_source": "instrument_column",
            "bridges": [{"channel": 1, "rho_source": "instrument_column",
                         "classification": "metallic",
                         "rho_t_curves": [{"held_field_oe": 0.0, "direction": 1,
                                           "n_points": 200, "classification": "metallic",
                                           "temperature": T, "rho": rho}],
                         "rho_h_curves": []}],
            "capabilities": []}
    return Result(status="ok", data=data,
                  provenance=Provenance(file="x", sha256="ab", app_version=None))

def _clean_result():
    from cryosweep_core.result import Result, Provenance
    T = list(np.linspace(2, 300, 200)); rho = list(np.linspace(4e-6, 9e-5, 200))
    data = {"probe": "resistivity", "rho_source": "instrument_column",
            "bridges": [{"channel": 1, "rho_source": "instrument_column",
                         "classification": "metallic",
                         "rho_t_curves": [{"held_field_oe": 0.0, "direction": 1, "n_points": 200,
                                           "classification": "metallic",
                                           "temperature": T, "rho": rho}],
                         "rho_h_curves": []}], "capabilities": []}
    return Result(status="ok", data=data,
                  provenance=Provenance(file="x", sha256="ab", app_version=None))

def _ylim(fig):
    return fig.axes[0].get_ylim()

# NOTE: "resistivity_rho_t" defaults to linear-y (D3), and robust-view bypasses log-y by
# design. These fixtures exercise the LINEAR robust path the feature governs, rendered with an
# explicit PlotSpec(yscale="linear"). test_robust_view_skipped_on_log_y covers the log-y bypass.
# The ρ axis carries an auto engineering prefix (bulk ~4e-5 Ω·cm -> µΩ·cm, factor 1e6), so the
# absolute ylim below is in the scaled unit; the bulk sits ~30-50, the heavy tail ~1.5e4.
def test_robust_view_narrows_on_heavy_tail():
    r = _heavy_tail_result()
    on = _ylim(render_kind(r, "resistivity_rho_t", PlotSpec(yscale="linear"), GlobalStyle(robust_view=True)))
    off = _ylim(render_kind(r, "resistivity_rho_t", PlotSpec(yscale="linear"), GlobalStyle(robust_view=False)))
    assert on[1] < off[1] / 10
    assert on[1] < 1e3                    # top zoomed to the scaled bulk, far below the ~1.5e4 tail

def test_robust_view_is_noop_on_clean_data():
    r = _clean_result()
    on = _ylim(render_kind(r, "resistivity_rho_t", PlotSpec(yscale="linear"), GlobalStyle(robust_view=True)))
    off = _ylim(render_kind(r, "resistivity_rho_t", PlotSpec(yscale="linear"), GlobalStyle(robust_view=False)))
    assert on == off

def test_explicit_ylim_wins_over_robust_view():
    r = _heavy_tail_result()
    fig = render_kind(r, "resistivity_rho_t", PlotSpec(yscale="linear", ymin=0.0, ymax=0.02),
                      GlobalStyle(robust_view=True))
    assert _ylim(fig) == (0.0, 0.02)

def test_robust_view_skipped_on_log_y():
    r = _heavy_tail_result()
    on = _ylim(render_kind(r, "resistivity_rho_t", PlotSpec(yscale="log"),
                           GlobalStyle(robust_view=True)))
    off = _ylim(render_kind(r, "resistivity_rho_t", PlotSpec(yscale="log"),
                            GlobalStyle(robust_view=False)))
    assert on == off

def _neg_result(rho_vals):
    from cryosweep_core.result import Result, Provenance
    T = list(np.linspace(2, 300, len(rho_vals)))
    data = {"probe": "resistivity", "rho_source": "instrument_column",
            "bridges": [{"channel": 1, "rho_source": "instrument_column", "classification": "metallic",
                         "rho_t_curves": [{"held_field_oe": 0.0, "direction": 1, "n_points": len(rho_vals),
                                           "classification": "metallic",
                                           "temperature": T, "rho": list(rho_vals)}],
                         "rho_h_curves": []}], "capabilities": []}
    return Result(status="ok", data=data,
                  provenance=Provenance(file="x", sha256="ab", app_version=None))

def test_robust_view_handles_negative_data():
    import numpy as np
    # clean all-negative ramp -> no-op
    clean = _neg_result(np.linspace(-9e-5, -4e-6, 200))
    on = _ylim(render_kind(clean, "resistivity_rho_t", PlotSpec(yscale="linear"), GlobalStyle(robust_view=True)))
    off = _ylim(render_kind(clean, "resistivity_rho_t", PlotSpec(yscale="linear"), GlobalStyle(robust_view=False)))
    assert on == off
    # negative bulk with a far low tail -> robust view zooms to the bulk (bottom well above the tail)
    vals = list(np.linspace(-1.1, -0.9, 200)); vals[-7:] = [-50.0] * 7
    tail = _neg_result(vals)
    on2 = _ylim(render_kind(tail, "resistivity_rho_t", PlotSpec(yscale="linear"), GlobalStyle(robust_view=True)))
    off2 = _ylim(render_kind(tail, "resistivity_rho_t", PlotSpec(yscale="linear"), GlobalStyle(robust_view=False)))
    assert on2 != off2
    assert on2[0] > -50.0
