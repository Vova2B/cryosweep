# tests/core/test_hall_unresolved_decline.py
"""When sigma >= |R_H| the +-1sigma interval on R_H contains zero. n = 1/(e|R_H|) then has
no finite upper bound and the carrier SIGN -- the thing a Hall measurement exists to
determine -- is undetermined in principle. So n, carrier type and mobility are withheld,
with a machine-readable reason; R_H and its sigma stay visible because the fit did happen
and hiding it would hide the evidence for the decline.

Same discipline as resistivity's n_unresolved and Hall's own antisym_r_h_missing.
"""
import numpy as np
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.hall import HallAnalyzer

_HDR = ("[Header]\nBYAPP, Resistivity\nINFO, unres_synth, SAMPLE\n[Data]\n"
        "Temperature (K),Magnetic Field (Oe),Bridge 1 Resistance (Ohms),"
        "Bridge 1 Resistivity (Ohm-m),Bridge 1 Std. Dev. (Ohm-m),"
        "Bridge 2 Resistance (Ohms),Bridge 2 Resistivity (Ohm-m)\n")
_RATIO = 1000.0

def _write(tmp_path, hall_slope, sd_rho, name):
    rows = []
    for b in np.arange(-20000.0, 20000.1, 250.0):
        rxy = 1e-3 + hall_slope * (b / 1e4)
        rows.append(f"10.0000,{b:.1f},{rxy:.10e},{rxy / _RATIO:.10e},{sd_rho:.10e},"
                    f"{1e-3:.10e},{1e-6:.10e}")
    rows += [f"{t:.4f},20000.0,1.5e-3,1.5e-6,{sd_rho:.10e},1e-3,1e-6"
             for t in np.arange(11.0, 51.0, 2.0)]
    p = tmp_path / name
    p.write_text(_HDR + "\n".join(rows) + "\n")
    return p

def _analyze(path):
    return HallAnalyzer().analyze(load_dat(path), RunConfig(
        hall={"hall_channel": 1, "thickness_mm": 0.1, "longitudinal_channel": 2}))

def test_resolved_point_publishes_everything(tmp_path):
    res = _analyze(_write(tmp_path, 5e-4, 1e-12, "resolved.dat"))
    assert [p["temperature"] for p in res.data["points"]] == [10.0]
    p = res.data["points"][0]
    assert p["carrier_n"] is not None and p["carrier_type"] is not None
    assert "r_h_unresolved" not in p["derived_flags"]
    assert p["withheld"] is None

def test_unresolved_point_withholds_n_type_and_mobility(tmp_path):
    # tiny Hall slope, huge instrument noise -> sigma_inst >= |R_H|
    res = _analyze(_write(tmp_path, 1e-9, 1e-4, "unresolved.dat"))
    p = res.data["points"][0]
    assert p["R_H"] is not None, "R_H itself stays visible"
    assert p["r_h_sigma_instrument"] is not None
    assert p["carrier_n"] is None
    assert p["carrier_type"] is None
    assert p["mobility"] is None
    assert p["carrier_n_sigma"] is None and p["mobility_sigma"] is None
    assert "r_h_unresolved" in p["derived_flags"]

def test_withheld_values_are_kept_for_inspection(tmp_path):
    res = _analyze(_write(tmp_path, 1e-9, 1e-4, "unresolved2.dat"))
    p = res.data["points"][0]
    w = p["withheld"]
    assert w is not None
    assert w["carrier_n"] is not None and w["carrier_type"] is not None
    # what was withheld equals what would have been published
    assert np.isclose(w["carrier_n"], 1.0 / (1.602176634e-19 * abs(p["R_H"])), rtol=1e-9)

def test_a_point_with_no_R_H_at_all_is_also_unresolved(tmp_path):
    """A missing R_H is not a resolved R_H. It keeps the reason it already carries."""
    from cryosweep_core.analyzers.hall import resolved_sigma
    class _P:
        R_H = None; r_h_sigma = None; r_h_sigma_instrument = None
    assert resolved_sigma(_P()) is None


# ---- Task 5 follow-on 1: hall_tempdep gains the same rho_xx provenance hall.py has -------

def test_tempdep_stamps_rho_xx_field_oe_on_success():
    """field_sweep_points has stamped rho_xx_field_oe on success since Task 1;
    hall_tempdep's own per-point rho_fn call had nowhere to record it until this task
    gave HallTDepPoint the field. _sigma_mu_J is the call site (grep
    'mobility_sigma_instrument' in hall_tempdep.py)."""
    from cryosweep_core.analyzers.hall_tempdep import _sigma_mu_J, HallTDepPoint

    def rho_fn(temp):
        return (2e-6, 12.5) if temp == 5.0 else (None, None)

    ok = HallTDepPoint(temperature=5.0, R_H=-5e-8)
    _sigma_mu_J(ok, rho_fn, rho_reason=None, long_channel=2)
    assert ok.rho_xx == 2e-6 and ok.rho_xx_field_oe == 12.5
    assert ok.derived_flags == []


def test_tempdep_flags_the_per_point_rho_xx_decline():
    """A setpoint whose own zero-field row is missing (per-point decline within an
    otherwise-covered file) must carry rho_xx_no_zero_field, exactly as field_sweep_points
    does -- not stay silent (the pre-Task-5 behaviour)."""
    from cryosweep_core.analyzers.hall_tempdep import _sigma_mu_J, HallTDepPoint

    def rho_fn(temp):
        return (2e-6, 12.5) if temp == 5.0 else (None, None)

    bad = HallTDepPoint(temperature=10.0, R_H=-5e-8)
    _sigma_mu_J(bad, rho_fn, rho_reason=None, long_channel=2)
    assert bad.rho_xx is None and bad.rho_xx_field_oe is None
    assert bad.derived_flags == ["rho_xx_no_zero_field"]


def test_tempdep_flags_the_file_level_rho_xx_reason_when_no_rho_fn():
    from cryosweep_core.analyzers.hall import _RHO_XX_CHANNEL_MISSING
    from cryosweep_core.analyzers.hall_tempdep import _sigma_mu_J, HallTDepPoint

    p = HallTDepPoint(temperature=5.0, R_H=-5e-8)
    _sigma_mu_J(p, None, rho_reason=_RHO_XX_CHANNEL_MISSING, long_channel=3)
    assert p.derived_flags == [_RHO_XX_CHANNEL_MISSING]


# ---- Task 5 follow-on 2: hall_tempdep's mobility-gap reason climbs the evidence ladder ----

def test_hall_tempdep_capability_names_temperature_misalignment_not_missing_zero_field(
        tmp_path, hall_tdep_synth_path):
    """Mirrors test_capability_reason_names_temperature_misalignment_not_missing_zero_field_data
    (test_hall_rho_xx_zero_field.py) on the hall_tempdep side. Before this follow-on,
    _capabilities called _mobility_gap_reason(long_source, rho_reason) with no `points`,
    so it could never climb past the generic 'cause not established per point' rung --
    even once HallTDepPoint carried the per-point evidence to name the real cause."""
    long_hdr = ("[Header]\nBYAPP, Resistivity\nINFO, long_only, SAMPLE\n[Data]\n"
                "Temperature (K),Magnetic Field (Oe),Bridge 2 Resistivity (Ohm-m)\n")
    long_rows = [f"{T:.4f},{b:.1f},{1e-6:.10e}"
                 for T in (500.0, 600.0) for b in (-100.0, 0.0, 100.0)]
    long_path = tmp_path / "long_only.dat"
    long_path.write_text(long_hdr + "\n".join(long_rows) + "\n")

    from cryosweep_core.analyzers.hall_tempdep import HallTempDepAnalyzer
    rt = load_dat(hall_tdep_synth_path)
    cfg = RunConfig(hall={"hall_channel": 1, "thickness_mm": 0.05,
                          "longitudinal_channel": 2, "longitudinal_file": str(long_path)})
    res = HallTempDepAnalyzer().analyze(rt, cfg)
    assert res.data["longitudinal_source"].startswith("file:")
    assert res.data["points"]                          # Hall fits still resolve fine
    for p in res.data["points"]:
        assert p["mobility"] is None
        assert "rho_xx_no_zero_field" in p["derived_flags"]
    caps = {c["name"]: c for c in res.data["capabilities"]}
    assert caps["mobility"]["applicable"] is False
    reason = caps["mobility"]["reason"].lower()
    assert "cause not established per point" not in reason   # the old, vague rung
    assert "no |h| <" not in reason           # must not claim missing zero-field data (false)
    assert "temp" in reason                   # must name the real cause: temperature misalignment


# ---- Task 5 Step 7: real-file oracle -----------------------------------------------------

def test_real_file_tempdep_declines_measured_fraction(hall_real_path):
    if not hall_real_path.exists():
        import pytest
        pytest.skip("real Hall measurement file gitignored/absent")
    from cryosweep_core.analyzers.hall_tempdep import HallTempDepAnalyzer
    res = HallTempDepAnalyzer().analyze(load_dat(hall_real_path), RunConfig(
        hall={"hall_channel": 1, "thickness_mm": 0.07, "longitudinal_channel": 2}))
    pts = res.data["points"]
    assert len(pts) == 138
    declined = [p for p in pts if "r_h_unresolved" in p["derived_flags"]]
    # measured 2026-09-10 (oracle re-confirmed post Task 4b): 72/138 declined, 66 still
    # publish carrier_n -- see task-5 report for the full derivation.
    assert len(declined) == 72
    assert sum(1 for p in pts if p["carrier_n"] is not None) == 66
    assert all(p["carrier_n"] is None and p["carrier_type"] is None and p["mobility"] is None
               for p in declined)
    assert all(p["R_H"] is not None and p["r_h_sigma_instrument"] is not None for p in declined)
    assert all(p["withheld"] is not None and p["withheld"]["carrier_n"] is not None
               for p in declined)


def test_real_file_field_sweep_declines_nothing(hall_real_path):
    """The field-sweep probe averages 136-186 antisymmetrised points into each T's fit,
    against 1-3 field values for a temp-dep point -- its relative instrument sigma never
    approaches the sigma >= |R_H| threshold on this file. A test asserting a decline here
    would be wrong and would fail (task-5 brief, Global Constraints)."""
    if not hall_real_path.exists():
        import pytest
        pytest.skip("real Hall measurement file gitignored/absent")
    res = HallAnalyzer().analyze(load_dat(hall_real_path), RunConfig(
        hall={"hall_channel": 1, "thickness_mm": 0.07, "longitudinal_channel": 2}))
    pts = res.data["points"]
    assert len(pts) == 9
    assert all("r_h_unresolved" not in p["derived_flags"] for p in pts)
    assert all(p["withheld"] is None for p in pts)
    assert all(p["carrier_n"] is not None for p in pts)
