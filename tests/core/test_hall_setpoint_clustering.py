"""KNOWN-ISSUES #19 (2026-09-02): drifting temperature setpoints must not fragment
a field loop. On the real Hall file the 200 K loop arrived as segments at 199.8521 /
199.9904 / 199.9945 K; round(T, 1) binning split them across the 199.9/200.0 edge
into a 46-point fragment (R_H None, carrier_n fabricated from Stage A) beside the
real 136-point group. Clustering the setpoints actually present has no edge."""
import numpy as np
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.hall import HallAnalyzer

_HDR = ("[Header]\nBYAPP, Resistivity\nINFO, drift_synth, SAMPLE\n[Data]\n"
        "Temperature (K),Magnetic Field (Oe),Bridge 1 Resistance (Ohms),"
        "Bridge 2 Resistance (Ohms),Bridge 2 Resistivity (Ohm-m)\n")


def _row(T, B_oe):
    # pure odd Hall signal: R_xy = 1e-3 + 5e-4 * B_T  (slope 5e-4 Ohm/T)
    r = 1e-3 + 5e-4 * (B_oe / 1e4)
    return f"{T:.4f},{B_oe:.1f},{r:.10e},{1e-3:.10e},{1e-6:.10e}"


def _write_drifting_dat(tmp_path):
    rows = []
    # 1) full ± loop held near 10 K (drifts 9.98 within the same physical setpoint)
    rows += [_row(9.98, b) for b in np.arange(-20000.0, 20000.1, 500.0)]
    # 2) T ramp at held B (separates the field segments)
    rows += [_row(t, 20000.0) for t in np.arange(15.0, 200.0, 5.0)]
    # 3) NEGATIVE half-loop at 199.8521 K  -> old code bins to 199.9
    rows += [_row(199.8521, b) for b in np.arange(-20000.0, -400.0, 500.0)]
    # 4) T ramp back down at held B
    rows += [_row(t, -500.0) for t in np.arange(195.0, 10.0, -5.0)]
    # 5) POSITIVE half-loop back near 10 K (10.02: same physical setpoint as 9.98)
    rows += [_row(10.02, b) for b in np.arange(0.0, 20000.1, 500.0)]
    # 6) T ramp up again
    rows += [_row(t, 20000.0) for t in np.arange(15.0, 200.0, 5.0)]
    # 7) POSITIVE half-loop at 199.9904 K  -> old code bins to 200.0
    rows += [_row(199.9904, b) for b in np.arange(20000.0, -100.0, -500.0)]
    p = tmp_path / "drifting_setpoints.dat"
    p.write_text(_HDR + "\n".join(rows) + "\n")
    return p


def test_drifting_setpoint_is_one_point_not_two(tmp_path):
    rt = load_dat(_write_drifting_dat(tmp_path))
    cfg = RunConfig(hall={"hall_channel": 1, "thickness_mm": 0.1,
                          "longitudinal_channel": 2})
    res = HallAnalyzer().analyze(rt, cfg)
    temps = sorted(p["temperature"] for p in res.data["points"])
    # ONE physical setpoint -> ONE point, labelled with the round number: the drifting
    # ~200 K segments merge (no 199.9 fragment), the ~10 K halves merge to 10.0.
    assert temps == [10.0, 200.0], temps
    for p in res.data["points"]:
        # each merged group now spans both field signs -> Stage B runs everywhere
        assert p["antisymmetrized"] is True
        assert p["R_H"] is not None


# ---- same latent defect on FIELD setpoints (hall_tempdep fixed-field grouping) ----

def _write_drifting_field_dat(tmp_path):
    # the same 40 kOe hold arrives as two ramps whose field medians straddle the
    # integer-rounding edge — the measured VSM values from commit 3d722ff.
    def _r(t, b):
        return f"{t:.4f},{b:.4f},{1e-3 + 1e-5 * t:.10e},{1e-3:.10e},{1e-6:.10e}"
    rows = [_r(2.0 + i, 40000.887) for i in range(59)]           # warm ramp at hold #1
    # field excursion at held T so the segmenter separates the two ramps (a 1.3 Oe
    # step alone is within the field drift tolerance and would merge them)
    rows += [_r(61.0, b) for b in list(range(40000, 9999, -1000))
             + list(range(10000, 40001, 1000))]
    rows += [_r(61.0 - i, 39999.586) for i in range(59)]          # cool ramp at hold #2
    p = tmp_path / "drifting_fields.dat"
    p.write_text(_HDR + "\n".join(rows) + "\n")
    return p


def test_drifting_field_setpoint_is_one_curve_not_two(tmp_path):
    from cryosweep_core.io.columns import canonicalize_columns
    from cryosweep_core.analyzers.hall_tempdep import _interp_fixed_field_curves
    rt = load_dat(_write_drifting_field_dat(tmp_path))
    df, cmap = canonicalize_columns(rt.df, rt.header)
    curves = _interp_fixed_field_curves(df, cmap, RunConfig(), 1, 1.0)
    # one physical field -> one curve, labelled with the round number
    assert sorted(curves) == [40000.0], sorted(curves)
