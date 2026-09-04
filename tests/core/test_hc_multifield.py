"""Multi-field heat-capacity datasets must fit the ZERO-FIELD low-T lattice term.

Real bug (the low-T heat-capacity file): the HC analyzer selected fit data via sweep-segmentation, which
fragments a multi-field dataset's zero-field ramp and loses its low-T points -> "<5 low-T
points" / garbage fit. The analyzer must select the lowest-|field| group directly.
"""
import pathlib
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry

FIX = pathlib.Path(__file__).parent / "fixtures"
# oracle from make_hc_multifield.py
GAMMA, BETA, THETA_D = 0.005, 2.0e-4, 268.87368901339767


def _res():
    return analyze_file(load_dat(str(FIX / "hc_multifield_synth.dat")),
                        RunConfig.load(), build_default_registry())


def test_multifield_recovers_zero_field_lattice_fit():
    import pytest
    res = _res()
    assert res.data["probe"] == "heatcapacity"
    assert res.status == "ok", (res.status, res.confidence)
    p = res.data["fit"]["params"]
    assert p["gamma"] == pytest.approx(GAMMA, rel=1e-3)
    assert p["beta"] == pytest.approx(BETA, rel=1e-3)
    assert p["theta_D"] == pytest.approx(THETA_D, rel=1e-3)


def test_multifield_selects_zero_field_group():
    res = _res()
    # field_setpoint must be the ~0 Oe group, not 1 T / 3 T
    assert abs(res.data["field_setpoint"]) < 50.0, res.data["field_setpoint"]
    # enough low-T points recovered (17 zero-field temperatures, all <= 10 K)
    assert len(res.data["temperature"]) >= 15


def test_field_groups_built_for_three_fields():
    import pytest
    res = _res()
    fg = res.data["field_groups"]
    assert len(fg) == 3
    fields = [g["field_oe"] for g in fg]
    assert fields == sorted(fields)                       # ascending
    assert fg[0]["is_primary"] is True
    assert abs(fg[0]["field_oe"]) < 50                    # ~0 Oe primary
    assert fg[1]["field_oe"] == pytest.approx(10000, rel=1e-3)
    assert fg[2]["field_oe"] == pytest.approx(30000, rel=1e-3)


def test_field_groups_gamma_rises_with_field():
    import pytest
    # fixture: Cp/T = (gamma + 0.02*f/10000) + beta*T^2 -> gamma 0.005, 0.025, 0.065
    res = _res()
    def g_of(group):
        d = next(f for f in group["fits"] if f["key"] == "debye_t3")
        return d["params"]["gamma"]
    fg = res.data["field_groups"]
    assert g_of(fg[0]) == pytest.approx(0.005, abs=2e-4)
    assert g_of(fg[1]) == pytest.approx(0.025, abs=2e-4)
    assert g_of(fg[2]) == pytest.approx(0.065, abs=2e-4)


def test_primary_scalar_result_unchanged_with_field_groups():
    import pytest
    # field_groups is additive: the primary scalar fit is still the zero-field debye_t3 oracle
    res = _res()
    p = res.data["fit"]["params"]
    assert p["gamma"] == pytest.approx(GAMMA, rel=1e-3)
    assert p["theta_D"] == pytest.approx(THETA_D, rel=1e-3)
    assert res.data["model"] == "debye_t3"


def test_single_field_data_has_empty_field_groups(hc_synth_path):
    from cryosweep_core.io.loader import load_dat
    from cryosweep_core.config import RunConfig
    from cryosweep_core.analyzers.dispatch import analyze_file
    from cryosweep_core.registry import build_default_registry
    res = analyze_file(load_dat(str(hc_synth_path)), RunConfig.load(), build_default_registry())
    assert res.data["field_groups"] == []                 # < 2 fields -> empty


def test_negative_beta_fit_is_flagged_not_reported_as_ok(tmp_path):
    """A sample whose low-T Cp/T vs T^2 has NEGATIVE slope (low-T upturn: nuclear/magnetic
    Schottky) yields beta<=0 -> no valid Debye theta_D. The analyzer must flag it
    low_confidence with a warning rather than report status ok with theta_D = NaN
    (real case: the low-T heat-capacity file once the loader/selection bugs are fixed)."""
    import math
    from cryosweep_core.io.loader import load_dat
    from cryosweep_core.config import RunConfig
    from cryosweep_core.analyzers.dispatch import analyze_file
    from cryosweep_core.registry import build_default_registry
    p = tmp_path / "neg_beta.dat"
    lines = ["[Header]", "BYAPP,HeatCapacity,1,1", "INFO,150,MOLWGHT:Formula Weight (g/mole)",
             "INFO,2,ATOMS:Atoms per Formula Unit", "[Data]",
             "Sample Temp (Kelvin),Field (Oersted),Samp HC (mJ/mole-K)"]
    for i in range(17):
        t = 2.0 + 0.5 * i
        cp = 0.2 * t - 5e-4 * t**3                 # Cp/T = 0.2 - 5e-4 T^2 -> beta<0
        lines.append(f"{t},0.0,{cp*1000:.6f}")
    p.write_text("\n".join(lines) + "\n")
    res = analyze_file(load_dat(str(p)), RunConfig.load(), build_default_registry())
    assert res.status == "low_confidence", res.status
    assert any("beta" in w.lower() or "θ_d" in w.lower() or "debye" in w.lower() for w in res.warnings), res.warnings


def test_lowt_upturn_warning_flags_nuclear_schottky(tmp_path):
    # zero field: clean lattice; 1 T: add a rising 1/T^2-like upturn at the lowest temps
    from cryosweep_core.io.loader import load_dat
    from cryosweep_core.config import RunConfig
    from cryosweep_core.analyzers.dispatch import analyze_file
    from cryosweep_core.registry import build_default_registry
    p = tmp_path / "upturn.dat"
    lines = ["[Header]", "BYAPP,HeatCapacity,1,1",
             "INFO,150,MOLWGHT:Formula Weight (g/mole)",
             "INFO,2,ATOMS:Atoms per Formula Unit", "[Data]",
             "Sample Temp (Kelvin),Field (Oersted),Samp HC (mJ/mole-K)"]
    import numpy as np
    for t in np.linspace(2.0, 10.0, 17):
        cp0 = 0.005 * t + 2.0e-4 * t**3
        lines.append(f"{t},0.0,{cp0*1000:.6f}")
        cp1 = 0.005 * t + 2.0e-4 * t**3 + 0.01 / t        # +alpha/T^2 in Cp/T -> upturn
        lines.append(f"{t},10000.0,{cp1*1000:.6f}")
    p.write_text("\n".join(lines) + "\n")
    res = analyze_file(load_dat(str(p)), RunConfig.load(), build_default_registry())
    g1 = res.data["field_groups"][1]                       # the 1 T group
    assert any("upturn" in w.lower() or "schottky" in w.lower() for w in g1["warnings"]), g1["warnings"]


def test_cross_field_gamma_rise_is_consistent_no_false_flag():
    res = _res()                                           # clean fixture, monotonic gamma
    # theta_D ~ field-independent on the fixture -> no theta_D drift warning
    assert not any("theta_d drift" in w.lower() for w in res.warnings), res.warnings
