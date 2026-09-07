# tests/core/test_hall_rho_xx_zero_field.py
"""rho_xx for mobility must be a ZERO-FIELD quantity.

_long_rho_xx used to mean over every row at a temperature with no field mask. On a
magnetoresistive longitudinal channel that averages the whole field loop into the
denominator of mu = |R_H|/rho_xx. Measured on the real Hall file: the mean sat 60.8%
above the |H| < 50 Oe value at 2 K, so mu was 38% low, and the reported rho_xx was
non-monotonic (2 K above 5 K).
"""
import numpy as np
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.hall import HallAnalyzer

_HDR = ("[Header]\nBYAPP, Resistivity\nINFO, mr_synth, SAMPLE\n[Data]\n"
        "Temperature (K),Magnetic Field (Oe),Bridge 1 Resistance (Ohms),"
        "Bridge 2 Resistance (Ohms),Bridge 2 Resistivity (Ohm-m)\n")

# rho_xx(H) = rho0 * (1 + 1.0 * (H/1e4)^2): a strongly magnetoresistive longitudinal
# channel. At |H| <= 50 Oe it is rho0 to 2.5e-5 relative; averaged over a +-2 T loop it
# is ~1.33 * rho0. The two answers differ by a third, which is the point of the test.
_RHO0 = 1e-6

def _row(T, B_oe):
    rxy = 1e-3 + 5e-4 * (B_oe / 1e4)
    rho = _RHO0 * (1.0 + (B_oe / 1e4) ** 2)
    return f"{T:.4f},{B_oe:.1f},{rxy:.10e},{rho * 1e3:.10e},{rho:.10e}"

def _write(tmp_path, include_zero=True):
    rows = []
    for T in (10.0, 50.0):
        fields = np.arange(-20000.0, 20000.1, 250.0)
        if not include_zero:                       # drop everything inside +-50 Oe
            fields = fields[np.abs(fields) > 50.0]
        rows += [_row(T, b) for b in fields]
        # 20-row separator. It MUST exceed StabilityCfg.window (16, config.py:41): the
        # rolling-activity classifier uses a centred window of that size, so a shorter
        # separator lets the neighbouring loop's field activity bleed across the gap.
        # The two loops then merge into one block, the fixed-axis drift guard rejects
        # the merged stretch, and the T=10 point silently disappears.
        rows += [_row(t, 20000.0) for t in np.arange(T + 1, T + 40, 2.0)]  # ramp separator
    p = tmp_path / ("mr.dat" if include_zero else "mr_nozero.dat")
    p.write_text(_HDR + "\n".join(rows) + "\n")
    return p

def _analyze(path):
    cfg = RunConfig(hall={"hall_channel": 1, "thickness_mm": 0.1,
                          "longitudinal_channel": 2})
    return HallAnalyzer().analyze(load_dat(path), cfg)

def test_rho_xx_is_the_zero_field_value_not_the_loop_average(tmp_path):
    res = _analyze(_write(tmp_path))
    pts = {p["temperature"]: p for p in res.data["points"]}
    assert set(pts) >= {10.0, 50.0}
    for T, p in pts.items():
        # zero-field value, not the ~1.33x loop average
        assert abs(p["rho_xx"] - _RHO0) / _RHO0 < 0.01, (T, p["rho_xx"])

def test_point_records_the_field_its_rho_xx_came_from(tmp_path):
    res = _analyze(_write(tmp_path))
    for p in res.data["points"]:
        assert p["rho_xx_field_oe"] is not None
        assert abs(p["rho_xx_field_oe"]) <= 50.0

def test_mobility_declines_when_no_near_zero_field_row_exists(tmp_path):
    res = _analyze(_write(tmp_path, include_zero=False))
    for p in res.data["points"]:
        assert p["mobility"] is None
        assert p["rho_xx"] is None
        assert "rho_xx_no_zero_field" in p["derived_flags"]
