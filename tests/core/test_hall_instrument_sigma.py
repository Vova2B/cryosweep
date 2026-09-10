"""The field-sweep `hall` analyzer computed NO instrument sigma, only the residual
(fit-scatter) one. Consequence measured on the real Hall file: `hall` reports
status ok / warnings [] while `hall-tdep` reports every one of its 138 points as
instrument noise -- one file, two commands, opposite verdicts. The 50% noise warning
could never fire on the `hall` path because the quantity it tests did not exist.
"""
import numpy as np
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.hall import HallAnalyzer

_HDR = ("[Header]\nBYAPP, Resistivity\nINFO, sig_synth, SAMPLE\n[Data]\n"
        "Temperature (K),Magnetic Field (Oe),Bridge 1 Resistance (Ohms),"
        "Bridge 1 Resistivity (Ohm-m),Bridge 1 Std. Dev. (Ohm-m),"
        "Bridge 2 Resistance (Ohms),Bridge 2 Resistivity (Ohm-m)\n")
_RATIO = 1000.0          # R/rho, constant -> the constancy gate passes

def _row(T, B_oe, sd_rho):
    rxy = 1e-3 + 5e-4 * (B_oe / 1e4)
    return (f"{T:.4f},{B_oe:.1f},{rxy:.10e},{rxy / _RATIO:.10e},{sd_rho:.10e},"
            f"{1e-3:.10e},{1e-6:.10e}")

def _write(tmp_path, sd_rho):
    rows = []
    for T in (10.0, 50.0):
        rows += [_row(T, b, sd_rho) for b in np.arange(-20000.0, 20000.1, 250.0)]
        rows += [_row(t, 20000.0, sd_rho) for t in np.arange(T + 1, T + 40, 2.0)]
    p = tmp_path / f"sig_{sd_rho:.0e}.dat"
    p.write_text(_HDR + "\n".join(rows) + "\n")
    return p

def _analyze(path):
    cfg = RunConfig(hall={"hall_channel": 1, "thickness_mm": 0.1,
                          "longitudinal_channel": 2})
    return HallAnalyzer().analyze(load_dat(path), cfg)

def test_instrument_sigma_is_computed_and_named_apart_from_residual():
    """Both families present, both non-None, and NOT the same number."""
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        res = _analyze(_write(pathlib.Path(d), 1e-11))
    # assert WHICH setpoints are present before looping: a `for p in points` over an empty
    # list passes while asserting nothing, and a short separator can silently drop a setpoint
    assert {p["temperature"] for p in res.data["points"]} == {10.0, 50.0}
    for p in res.data["points"]:
        assert p["r_h_sigma"] is not None
        assert p["r_h_sigma_instrument"] is not None
        assert p["carrier_n_sigma_instrument"] is not None
        assert p["r_h_sigma"] != p["r_h_sigma_instrument"]

def test_noise_warning_fires_on_the_hall_path_when_instrument_sigma_is_large():
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        res = _analyze(_write(pathlib.Path(d), 1e-3))     # absurd instrument noise
    assert {p["temperature"] for p in res.data["points"]} == {10.0, 50.0}
    assert res.warnings, "a noise-dominated field sweep must warn on the `hall` path too"
    assert any("instrument" in w for w in res.warnings)

def test_absent_std_column_leaves_instrument_sigma_None_without_error():
    import tempfile, pathlib
    hdr = _HDR.replace("Bridge 1 Std. Dev. (Ohm-m),", "")
    rows = []
    for b in np.arange(-20000.0, 20000.1, 250.0):
        rxy = 1e-3 + 5e-4 * (b / 1e4)
        rows.append(f"10.0000,{b:.1f},{rxy:.10e},{rxy / _RATIO:.10e},{1e-3:.10e},{1e-6:.10e}")
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "nostd.dat"
        p.write_text(hdr + "\n".join(rows) + "\n")
        res = _analyze(p)
    assert res.data["points"], "one whole-file field sweep must still yield a point"
    for pt in res.data["points"]:
        assert pt["r_h_sigma_instrument"] is None
        assert pt["r_h_sigma"] is not None          # residual family unaffected
