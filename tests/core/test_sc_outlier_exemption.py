"""SC exemption from robust outlier handling (owner-approved 2026-09-01, example-defect D3b).

The resistive-Tc criteria require a near-flat normal state, so an accepted transition's
below-Tc floor ALWAYS sits far outside the median±k·MAD band: without an exemption, the
k=8 rule flags the superconducting state of every accepted transition, and the opt-in
`exclude_outliers` silently removes it from curves, fits and the CSV export. The
superconducting state is a phase, not bad data. Points below a curve's detected
tc_onset_k are therefore exempt from BOTH the outlier diagnostics and the exclusion
mask; everything without a detected transition is byte-identical to the old behaviour."""
import numpy as np
import pathlib
import pytest

from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.resistivity import ResistivityAnalyzer, _clean_mask
from cryosweep_core.analyzers.resistive_tc import detect_resistive_tc

FIX = pathlib.Path(__file__).parent / "fixtures"
SC = str(FIX / "rho_sc_synth.dat")


def _widest_t_curve(res, bridge=0):
    return max(res.data["bridges"][bridge]["rho_t_curves"], key=lambda c: c["n_points"])


def test_below_tc_points_survive_exclusion_in_curve_fits_and_csv(tmp_path):
    """(a) With exclude_outliers=True on a file with a detected transition, the
    superconducting state survives — in the curve, in the Tc fields, and in the CSV."""
    base = ResistivityAnalyzer().analyze(load_dat(SC), RunConfig())
    excl = ResistivityAnalyzer().analyze(load_dat(SC),
                                         RunConfig.load(quality={"exclude_outliers": True}))
    cb, ce = _widest_t_curve(base), _widest_t_curve(excl)
    assert ce["tc_mid_k"] == pytest.approx(8.0, abs=0.01)       # transition still detected
    assert ce["n_points"] == cb["n_points"]                     # nothing dropped on the SC channel
    assert min(ce["temperature"]) == pytest.approx(2.0)         # floor reaches the lowest T
    assert min(ce["rho"]) <= 2e-8                               # ... at the fixture's 1e-8 floor
    # and RRR (an endpoint quantity over the same rows) still sees the floor:
    assert excl.data["bridges"][0]["rrr"] == pytest.approx(base.data["bridges"][0]["rrr"])
    # CSV export inherits the surviving points
    from cryosweep_core.io.export import export_result
    files = export_result(excl, str(tmp_path / "sc"))
    blob = "".join(pathlib.Path(f).read_text() for f in
                   (files.values() if isinstance(files, dict) else files))
    assert "1.0000000000e-08" in blob or "1e-08" in blob


def test_diagnostics_exempt_below_tc_but_not_the_other_bridge():
    """The badge counts no below-Tc point of the SC channel; the (grid-pathological)
    linear bridge-2 flags are untouched by the exemption."""
    res = ResistivityAnalyzer().analyze(load_dat(SC), RunConfig())
    outs = {d.scope: d.data["n_outliers"] for d in res.diagnostics if d.kind == "outliers"}
    assert not any("bridge1" in k for k in outs), f"SC channel flagged: {outs}"
    assert any("bridge2" in k for k in outs)      # normal-metal flags unaffected by the exemption


def test_no_transition_means_byte_identical_masks():
    """(b) No detected transition -> passing T changes nothing (the strict gate)."""
    rng = np.random.default_rng(3)
    T = np.linspace(2, 300, 200)
    rho = 1e-4 * (0.3 + 0.7 * T / 300) * (1 + 0.01 * rng.standard_normal(T.size))
    rho[50] = 5e-3                                  # one genuine spike
    assert detect_resistive_tc(T, rho) is None
    cfg = RunConfig.load(quality={"exclude_outliers": True})
    np.testing.assert_array_equal(_clean_mask(rho, cfg, T=T), _clean_mask(rho, cfg))
    assert not _clean_mask(rho, cfg)[50]            # and the spike is still excluded


def test_normal_state_spike_still_caught_on_sc_curve():
    """(3) The exemption must not weaken the rule above tc_onset_k: a genuine
    normal-state spike on the SC channel is still excluded, while the floor survives."""
    T = np.concatenate([np.linspace(2.0, 6.5, 10), np.linspace(6.5, 9.75, 66)[1:],
                        np.linspace(9.75, 300.0, 60)[1:]])[::-1].copy()
    def rho_of(t):
        if t >= 9.75: return 1e-4
        if t >= 9.0:  return 1e-4 * (0.9 + 0.1 * (t - 9.0) / 0.75)
        if t >= 7.0:  return 1e-4 * (0.1 + 0.4 * (t - 7.0))
        if t >= 6.5:  return max(1e-4 * 0.1 * (t - 6.5) / 0.5, 1e-8)
        return 1e-8
    rng = np.random.default_rng(9)
    # 0.4% instrument noise: an EXACTLY constant plateau has MAD = 0, where outlier_mask's
    # degenerate-input guard disables exclusion entirely (pre-existing, deliberate).
    rho = np.array([rho_of(float(t)) for t in T]) * (1 + 0.004 * rng.standard_normal(T.size))
    j = int(np.argmin(np.abs(T - 150.0)))
    rho[j] = 5e-3                                   # normal-state spike at ~150 K
    tc = detect_resistive_tc(T, rho)
    assert tc is not None
    cfg = RunConfig.load(quality={"exclude_outliers": True})
    m = _clean_mask(rho, cfg, T=T)
    assert not m[j], "normal-state spike must still be excluded"
    below = T < tc["tc_onset_k"]
    assert m[below].all(), "below-Tc points must all survive"


def test_power_law_window_excludes_the_transition():
    """Same gate for the low-T fits: rho0 + A*T^n describes the metallic normal state, and
    pre-fix the window was [2.0, 30] K — the whole transition and floor inside it, yielding
    n = 0.87 at r2 = 0.39 (neither Fermi-liquid nor phonon). The window must start at the
    detected tc_onset_k; the transition-free channel keeps its full window."""
    res = ResistivityAnalyzer().analyze(load_dat(SC), RunConfig())
    b1, b2 = res.data["bridges"]
    c1 = _widest_t_curve(res)
    assert c1["tc_onset_k"] == pytest.approx(9.0)
    assert b1["power_law"]["fit_range"][0] >= c1["tc_onset_k"]     # normal state only
    assert b2["power_law"]["fit_range"][0] == pytest.approx(2.0)   # no transition -> unchanged
    assert b2["power_law"]["params"]["n"] == pytest.approx(1.0, abs=1e-6)
