import numpy as np, pandas as pd, pytest
from types import SimpleNamespace
from cryosweep_core.analyzers.hc import HCAnalyzer
from cryosweep_core.config import RunConfig
from cryosweep_core.fitting.heat_capacity import specific_heat_full
from cryosweep_core.io.loader import load_dat
from tests.core.conftest import require_real


@pytest.fixture(scope="module")
def hc_result():
    return HCAnalyzer().analyze(load_dat(require_real("hc")), RunConfig.load())

class _Hdr:
    title = "synthetic"; app_version = None; n_atoms = 3.0

def _raw(T, cp_mJ, field):
    df = pd.DataFrame({"Sample Temp (Kelvin)": T,
                       "Samp HC (mJ/mole-K)": cp_mJ,
                       "Field (Oe)": field})
    return SimpleNamespace(df=df, header=_Hdr(), path=None)

def _synth():
    T = np.linspace(2.0, 300.0, 150)
    cp = specific_heat_full(T, 220.0, 3.0, 0.01, 60.0, 180.0, 1.0, 1.5)
    return _raw(T, cp * 1e3, np.zeros_like(T))      # mJ; zero field

def test_full_fit_available_and_compared():
    r = HCAnalyzer().analyze(_synth(), RunConfig())
    d = r.data
    assert d["full_fit_available"] is True
    assert d["full_fit"]["ok"] is True
    assert d["full_fit"]["fixed"]["n"] is True
    assert d["full_fit"]["params"]["n"] == 3.0          # n reconciled to header ATOMS
    assert "comparison" in d and d["comparison"]["gamma"]["full"] is not None
    assert len(d["full_temperature"]) == 150 and len(d["full_cp"]) == 150   # FULL group, not low-T subset
    assert len(d["temperature"]) < 150                                      # low-T subset unchanged

def test_full_fit_unavailable_low_t_only():
    T = np.linspace(2.0, 20.0, 40)               # T_max < 50 -> unavailable
    cp = specific_heat_full(T, 220.0, 3.0, 0.01, 60.0, 180.0, 1.0, 1.5)
    r = HCAnalyzer().analyze(_raw(T, cp * 1e3, np.zeros_like(T)), RunConfig())
    assert r.data["full_fit_available"] is False
    assert r.status in ("ok", "low_confidence")        # low-T result preserved, no crash


def test_full_fit_r2_floor_rejects_catastrophic():
    """Absurdly high floor (> 1.0) forces rejection of even a perfect synthetic fit; comparison entries become None."""
    cfg = RunConfig(heatcapacity={"full_min_r2": 1.001})   # impossible to beat -> any real r2 falls below
    r = HCAnalyzer().analyze(_synth(), cfg)
    d = r.data
    # fit attempted but rejected
    assert d["full_fit_available"] is True            # gate: fit was attempted
    assert d["full_fit"]["ok"] is False               # quality: rejected by r² floor
    assert "did not converge" in d["full_fit"]["reason"]
    # garbage params must NOT leak into comparison
    assert d["comparison"]["gamma"]["full"] is None
    assert d["comparison"]["theta_D"]["full"] is None
    assert d["comparison"]["r2"]["full"] is None


def test_full_fit_r2_floor_default_accepts_good_fit():
    """Default floor (0.9) accepts the synthetic near-perfect fit."""
    r = HCAnalyzer().analyze(_synth(), RunConfig())
    d = r.data
    assert d["full_fit"]["ok"] is True
    assert d["comparison"]["gamma"]["full"] is not None
    assert d["comparison"]["theta_D"]["full"] is not None


def _synth_fine():
    """300-pt synthetic (step ~1 K) so low-T range knob has room to narrow."""
    T = np.linspace(2.0, 300.0, 300)
    cp = specific_heat_full(T, 220.0, 3.0, 0.01, 60.0, 180.0, 1.0, 1.5)
    return _raw(T, cp * 1e3, np.zeros_like(T))


def test_lowt_fit_max_k_knob_reduces_point_count():
    """lowt_fit_max_k=6.0 must yield fewer low-T points than the default (~10 K)."""
    raw = _synth_fine()
    r_default = HCAnalyzer().analyze(raw, RunConfig())
    r_narrow = HCAnalyzer().analyze(raw, RunConfig(heatcapacity={"lowt_fit_max_k": 6.0}))
    n_default = len(r_default.data["temperature"])
    n_narrow = len(r_narrow.data["temperature"])
    assert n_narrow < n_default, (
        f"max_k=6.0 gave {n_narrow} pts, default gave {n_default} — knob not wired")


def test_lowt_fit_default_includes_points_up_to_10k():
    """Default RunConfig must include points up to ~10 K (behavior unchanged)."""
    r = HCAnalyzer().analyze(_synth_fine(), RunConfig())
    Tk = np.array(r.data["temperature"])
    assert float(Tk.max()) >= 9.5, f"Expected max T near 10 K, got {Tk.max():.2f} K"


def test_entropy_populated_on_heat_capacity_dat(hc_result):
    d = hc_result.data
    assert d["entropy_available"] is True
    assert len(d["entropy_total"]) == len(d["entropy_temperature"]) > 5
    s = d["entropy_total"]
    assert all(s[i] <= s[i + 1] + 1e-9 for i in range(len(s) - 1))   # monotone
    assert all(abs(x) < 1e30 for x in s)                             # finite
    assert d["n_atoms_available"] in (True, False)
    # DP available iff n_atoms_available
    from cryosweep_core.fitting.entropy import dulong_petit_limit
    assert (dulong_petit_limit(d["n_atoms"]) is not None) == d["n_atoms_available"]


def test_entropy_reference_lattice_self_subtraction(hc_path):
    """Self-reference: the SAME file supplies the lattice -> Cp - Cp = 0 -> magnetic S ~ 0,
    and the source is flagged 'reference'."""
    cfg = RunConfig(heatcapacity={"entropy_lattice_ref_file": str(hc_path)})
    r = HCAnalyzer().analyze(load_dat(hc_path), cfg)
    d = r.data
    assert d["entropy_lattice_source"] == "reference"
    sm = d["entropy_magnetic"]
    assert sm is not None and len(sm) > 5
    assert max(abs(x) for x in sm) < 1e-6, f"self-subtraction magnetic S not ~0: max={max(abs(x) for x in sm)}"


def test_entropy_reference_lattice_load_failure_falls_back_to_fit(hc_path):
    """MIN-2: a non-existent reference file must not fail the analyzer; it falls back to
    the fitted lattice (source='fit') and records a warning naming the failed reference."""
    bad = "Examples of data and prev scripts/DOES_NOT_EXIST_reference.dat"
    cfg = RunConfig(heatcapacity={"entropy_lattice_ref_file": bad})
    r = HCAnalyzer().analyze(load_dat(hc_path), cfg)
    assert r.status in ("ok", "low_confidence")
    assert r.data["entropy_lattice_source"] == "fit"
    assert any(bad in w for w in r.warnings), r.warnings


def test_entropy_rln_j_override(hc_path):
    """entropy_rln_j overrides the auto-suggested plateau with the given J."""
    cfg = RunConfig(heatcapacity={"entropy_rln_j": 1.0})
    r = HCAnalyzer().analyze(load_dat(hc_path), cfg)
    sug = r.data["entropy_rln_suggestion"]
    assert sug is not None
    assert sug["j"] == 1.0
    assert sug["label"] == "R ln3"


def test_field_groups_carry_full_cp_arrays(hc_result):
    """Every field group carries its own sorted full Cp(T); entropy is a dict or None."""
    for g in hc_result.data["field_groups"]:
        assert len(g["full_temperature"]) == len(g["full_cp"]) > 0
        assert list(g["full_temperature"]) == sorted(g["full_temperature"])   # ascending
        assert g["entropy"] is None or isinstance(g["entropy"], dict)
        if isinstance(g["entropy"], dict):
            s = g["entropy"]["s_total"]
            assert all(s[i] <= s[i + 1] + 1e-9 for i in range(len(s) - 1))
