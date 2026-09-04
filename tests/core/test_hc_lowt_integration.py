import numpy as np
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry

def _res(path):
    return analyze_file(load_dat(str(path)), RunConfig.load(), build_default_registry())


def test_config_has_parsimony_field():
    assert RunConfig.load().hc_parsimony_r2 == 0.99


def test_clean_files_choose_debye_t3_unchanged(hc_path):
    res = _res(hc_path)
    assert res.status == "ok"
    assert res.data["model"] == "debye_t3"
    assert res.data["fit"]["params"]["theta_D"] == __import__("pytest").approx(54.7, rel=1e-2)
    assert len(res.data["lowt_fits"]) == 4


def test_2_65K_upturn_now_ok_via_spin_fluctuation(hc_lowt_path):
    res = _res(hc_lowt_path)
    assert res.data["model"] == "spin_fluct_noninteracting"
    assert res.status == "ok"                                 # good fit (R^2~0.98) drives status
    assert res.data["fit"]["r2"] > 0.95
    assert np.isnan(res.data["fit"]["params"]["theta_D"])      # beta<0 -> no Debye theta_D
    assert any("spin-fluctuation" in w.lower() or "theta" in w.lower() or "β" in w
               for w in res.warnings)


def test_lattice_model_with_negative_beta_stays_low_confidence(tmp_path):
    # perfect-linear Cp/T vs T^2 with beta<0 -> chosen debye_t3 (R^2~1) -> lattice inadequate.
    p = tmp_path / "neg.dat"
    rows = ["[Header]", "BYAPP,HeatCapacity,1,1", "INFO,150,MOLWGHT:Formula Weight (g/mole)",
            "INFO,2,ATOMS:Atoms per Formula Unit", "[Data]",
            "Sample Temp (Kelvin),Field (Oersted),Samp HC (mJ/mole-K)"]
    for i in range(17):
        t = 2.0 + 0.5 * i
        rows.append(f"{t},0.0,{(0.2*t - 5e-4*t**3)*1000:.6f}")
    p.write_text("\n".join(rows) + "\n")
    res = analyze_file(load_dat(str(p)), RunConfig.load(), build_default_registry())
    assert res.data["model"] == "debye_t3"
    assert res.status == "low_confidence"
    assert np.isnan(res.data["fit"]["params"]["theta_D"])
