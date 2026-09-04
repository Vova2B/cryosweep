import numpy as np

from tests.core.conftest import FIX
from cryosweep_core.io.loader import load_dat            # match the import used in test_resistivity_analyzer.py
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.resistivity import ResistivityAnalyzer

RHO_SC = FIX / "rho_sc_synth.dat"


def _analyze(path):
    rt = load_dat(path)
    cfg = RunConfig()
    return ResistivityAnalyzer().analyze(rt, cfg)


def test_rho_sc_fixture_loads_two_bridges():
    res = _analyze(RHO_SC)
    bridges = {b["channel"]: b for b in res.data["bridges"]}
    assert set(bridges) == {1, 2}
    c1 = bridges[1]["rho_t_curves"][0]
    assert c1["held_field_oe"] == 0.0
    # plateau value survives round-trip through the Ohm-m instrument column (x100)
    assert np.isclose(max(c1["rho"]), 1e-4, rtol=1e-6)


from cryosweep_core.analyzers.resistive_tc import detect_resistive_tc


def _sc_ramp():
    """Piecewise-linear drop identical to the fixture math: crossings exactly 9/8/7 K."""
    T = np.concatenate([np.linspace(2.0, 6.5, 10), np.linspace(6.5, 9.75, 66)[1:],
                        np.linspace(9.75, 300.0, 60)[1:]])
    rho_n = 1e-4

    def f(t):
        if t >= 9.75: return rho_n
        if t >= 9.0: return rho_n * (0.9 + 0.1 * (t - 9.0) / 0.75)
        if t >= 7.0: return rho_n * (0.1 + 0.4 * (t - 7.0))
        if t >= 6.5: return max(rho_n * 0.1 * (t - 6.5) / 0.5, 1e-8)
        return 1e-8
    return T, np.array([f(t) for t in T]), rho_n


def test_detector_exact_crossings():
    T, R, rho_n = _sc_ramp()
    out = detect_resistive_tc(T, R)
    assert out is not None
    assert np.isclose(out["tc_onset_k"], 9.0, atol=1e-9)
    assert np.isclose(out["tc_mid_k"], 8.0, atol=1e-9)
    assert np.isclose(out["tc_zero_k"], 7.0, atol=1e-9)
    assert np.isclose(out["tc_rho_normal"], rho_n, rtol=1e-12)
    assert out["tc_low_confidence"] is False


def test_detector_unsorted_input_same_result():
    T, R, _ = _sc_ramp()
    a = detect_resistive_tc(T, R)
    b = detect_resistive_tc(T[::-1], R[::-1])          # cooling order
    assert a == b                                       # byte-identical / deterministic


def test_detector_gates_on_metal():
    T = np.linspace(2, 300, 100)
    assert detect_resistive_tc(T, 5e-5 + 1e-7 * T) is None


def _bloch_grueneisen(T, theta):
    """Normalized Bloch-Grüneisen resistivity ρ_BG(T)/coeff = (T/Θ)^5 ∫_0^{Θ/T} x^5 dx /
    ((e^x-1)(1-e^{-x})). Deterministic trapezoid quadrature. Flat high-T plateau, steep T^5
    falloff at low T — the shape of a clean non-superconducting metal."""
    out = []
    for t in T:
        xs = np.linspace(1e-6, theta / t, 2000)
        integ = np.trapezoid(xs ** 5 / ((np.exp(xs) - 1) * (1 - np.exp(-xs))), xs)
        out.append((t / theta) ** 5 * integ)
    return np.array(out)


def test_detector_declines_clean_metal_bloch_grueneisen():
    """Integrity: a clean non-SC metal with a steep Bloch-Grüneisen T^5 falloff (RRR~1000,
    Θ_D~200 K) coasts below 2%·rho_N at low T without any transition. The drop-floor gate
    alone false-positives (reviewer measured a confident absurd tc_mid~91 K); the NARROWNESS
    gate declines it — the onset→zero span is the whole ramp (rel width ≫ 0.5), not a
    transition."""
    T = np.linspace(2.0, 300.0, 300)
    rho_n = 1e-4
    g = _bloch_grueneisen(T, 200.0)
    rho = rho_n * (g / g[-1]) + rho_n / 1000.0              # normalize 300 K -> rho_n; residual
    # sanity: it does reach below the 2%·rho_N drop floor at low T (so ONLY narrowness saves us)
    assert float(rho[T <= 2.0 + 0.10 * 298.0].min()) < 0.02 * rho_n
    assert detect_resistive_tc(T, rho) is None


def test_detector_gates_on_noise_floor():
    T = np.linspace(2, 300, 100)
    rho = np.full(100, 1e-10)
    rho[:10] = 1e-13                                    # "drop", but rho_N below noise floor
    assert detect_resistive_tc(T, rho) is None


def test_detector_gates_on_short_plateau():
    T = np.linspace(2, 10, 6)                           # top 20% of range holds <5 points
    rho = np.array([1e-8, 1e-8, 5e-5, 1e-4, 1e-4, 1e-4])
    assert detect_resistive_tc(T, rho) is None


def test_detector_low_confidence_on_noisy_plateau():
    T, R, rho_n = _sc_ramp()
    R = R.copy()
    plateau = T >= 240.4                                # top 20% of 2..300
    # deterministic +-35% alternation -> CV > 20%
    idx = np.where(plateau)[0]
    R[idx[::2]] *= 1.35
    R[idx[1::2]] *= 0.65
    out = detect_resistive_tc(T, R)
    assert out is not None and out["tc_low_confidence"] is True


def test_analyzer_tc_oracle_on_synth():
    res = _analyze(RHO_SC)
    bridges = {b["channel"]: b for b in res.data["bridges"]}
    c1 = max(bridges[1]["rho_t_curves"], key=lambda c: c["n_points"])
    assert np.isclose(c1["tc_onset_k"], 9.0, atol=1e-6)
    assert np.isclose(c1["tc_mid_k"], 8.0, atol=1e-6)
    assert np.isclose(c1["tc_zero_k"], 7.0, atol=1e-6)
    assert np.isclose(c1["tc_rho_normal"], 1e-4, rtol=1e-6)
    assert c1["tc_low_confidence"] is False
    for c in bridges[2]["rho_t_curves"]:
        assert c["tc_mid_k"] is None
    caps = {c["name"]: c for c in res.data["capabilities"]}
    assert caps["superconducting_transition"]["applicable"] is True


def test_capability_gated_silent_on_metal():
    res = _analyze(FIX / "act_synth.dat")               # existing featureless fixture
    caps = {c["name"]: c for c in res.data["capabilities"]}
    assert caps["superconducting_transition"]["applicable"] is False
    assert caps["superconducting_transition"]["reason"] == "no resistive drop"
    for b in res.data["bridges"]:
        for c in b["rho_t_curves"]:
            assert c.get("tc_mid_k") is None


def test_analyzer_tc_deterministic():
    a = _analyze(RHO_SC).data
    b = _analyze(RHO_SC).data
    assert a == b
