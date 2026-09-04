from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.resistivity import ResistivityAnalyzer

def _real(res_path):
    return ResistivityAnalyzer().analyze(load_dat(res_path), RunConfig())

def test_real_ramp_emits_outlier_diagnostic_with_7_points(res_path):
    r = _real(res_path)
    outs = [d for d in r.diagnostics if d.kind == "outliers"]
    assert outs, "expected at least one outliers diagnostic"
    ramp = max(outs, key=lambda d: d.data.get("n", 0))
    assert ramp.data["n"] == 574
    assert ramp.data["n_outliers"] == 7
    assert ramp.severity == "warning"
    assert "bridge1" in ramp.scope

def test_clean_curves_emit_no_outlier_diagnostic(res_path):
    r = _real(res_path)
    clean = [d for d in r.diagnostics if d.kind == "outliers" and d.data.get("n_outliers", 0) == 0]
    assert clean == [], "diagnostics must only be emitted when n_outliers > 0"


def test_exclusion_off_is_byte_identical_to_today(res_path):
    a = ResistivityAnalyzer().analyze(load_dat(res_path), RunConfig())
    b = ResistivityAnalyzer().analyze(load_dat(res_path),
                                      RunConfig.load(quality={"exclude_outliers": False}))
    assert a.data == b.data

def test_exclusion_on_drops_the_7_outliers_from_curve_and_fit(res_path):
    base = ResistivityAnalyzer().analyze(load_dat(res_path), RunConfig())
    excl = ResistivityAnalyzer().analyze(load_dat(res_path),
                                         RunConfig.load(quality={"exclude_outliers": True}))
    def widest_ramp(res):
        b = res.data["bridges"][0]
        return max(b["rho_t_curves"], key=lambda c: c["n_points"])
    cb, ce = widest_ramp(base), widest_ramp(excl)
    assert ce["n_points"] == cb["n_points"] - 7
    assert max(ce["rho"]) < 1e-3
    eouts = [d for d in excl.diagnostics if d.kind == "outliers" and d.data["n"] == ce["n_points"]]
    assert all(d.data["n_outliers"] == 0 for d in eouts) or not eouts

def test_exclusion_skipped_on_sparse_curve():
    import numpy as np
    from cryosweep_core.analyzers.resistivity import _clean_mask
    from cryosweep_core.config import RunConfig
    rho = np.array([1.0, 1.1, 0.9, 1.0, 50.0])
    cfg = RunConfig.load(quality={"exclude_outliers": True})
    m = _clean_mask(rho, cfg=cfg)
    assert m.all()
