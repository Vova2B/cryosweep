import json, numpy as np, pandas as pd, pytest, dataclasses
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.acms import ACMSAnalyzer

def _run(path, molar=None, mass=None):
    rt = load_dat(path)
    if molar is not None or mass is not None:
        rt = dataclasses.replace(rt, header=dataclasses.replace(rt.header, molar_mass=molar, mass_mg=mass))
    return ACMSAnalyzer().analyze(rt, RunConfig())

def test_real_file_ok_three_groups_dropped_stray(acms_real_path):
    r = _run(acms_real_path)
    assert r.status == "ok"
    d = r.data
    amps = sorted({round(c["amplitude_oe"], 4) for c in d["curves"]})
    assert amps == [0.0498, 0.1, 0.3013]                 # 0.4979 stray dropped
    assert any(round(g["amplitude_oe"], 4) == 0.4979 and g["n_points"] == 1
               for g in d["dropped_groups"])

def test_real_file_declines_sc(acms_real_path):
    # physics-integrity: the real file has no diamagnetic drop -> detector must decline.
    d = _run(acms_real_path).data
    assert d["sc_transition"] is None
    cap = next(c for c in d["capabilities"] if c["name"] == "superconducting_screening")
    assert cap["applicable"] is False and cap["reason"] == "no diamagnetic drop"


def test_real_file_declines_chipp_peak(acms_real_path):
    # physics-integrity: featureless real file has no chi'' peak -> detector must decline.
    d = _run(acms_real_path).data
    assert d["chi_dprime_peaks"] == []
    cap = next(c for c in d["capabilities"] if c["name"] == "chi_dprime_peak")
    assert cap["applicable"] is False and cap["reason"] == "no chi'' peak"


def test_main_group_segments_to_up_and_down(acms_real_path):
    # min_len=15 is empirically load-bearing on the real file: min_len=10 -> 4 ramps,
    # min_len>=20 -> 1 (up-ramp swallowed). Assert the SET of directions, not exact order/count.
    curves = _run(acms_real_path).data["curves"]
    main = [c for c in curves if round(c["amplitude_oe"], 4) == 0.0498]
    assert {c["direction"] for c in main} == {"up", "down"}

def test_chi_prime_linearity_across_amplitudes(acms_real_path):
    curves = _run(acms_real_path).data["curves"]
    for c in curves:
        med = float(np.median(c["chi_prime"]))
        assert med == pytest.approx(-4.02e-12, rel=0.05)

def test_per_row_amplitude_normalization(acms_real_path):
    r = _run(acms_real_path); c = r.data["curves"][0]
    assert all(np.isfinite(x) for x in c["chi_prime"])   # amp>0 filtered -> no NaN

def test_molar_ladder_off_by_default_on_when_supplied(acms_real_path):
    assert _run(acms_real_path).data["curves"][0]["chi_prime_molar"] is None
    on = _run(acms_real_path, molar=200.0, mass=5.0).data["curves"][0]
    assert on["chi_prime_molar"] is not None

def test_analyze_twice_byte_identical(acms_real_path):
    a = json.dumps(_run(acms_real_path).data, sort_keys=True); b = json.dumps(_run(acms_real_path).data, sort_keys=True)
    assert a == b

def test_no_nonfinite_in_json(acms_real_path):
    s = json.dumps(_run(acms_real_path).data)
    assert "NaN" not in s and "Infinity" not in s

def test_partial_nan_mdc_stays_none(tmp_path):
    # M-DC finite on some rows only -> curve.m_dc must be None (never a NaN-carrying list)
    p = tmp_path / "partial_mdc.dat"
    head = ("[Header]\nTITLE,partial_mdc\nBYAPP,ACMS,1.0,1.1\n[Data]\n"
            "Comment,Time Stamp (sec),Temperature (K),Magnetic Field (Oe),Frequency (Hz),"
            "Amplitude (Oe),M-DC (emu),M-Std.Dev. (emu),M' (emu),M'' (emu)\n")
    rows = []
    for i in range(30):
        t = 1.0 + 0.1 * i
        mdc = f"{1e-6:.3e}" if i % 2 == 0 else ""           # every other row empty -> NaN
        rows.append(f",0,{t:.3f},0.0,477.0,0.05,{mdc},1e-9,-2e-13,1e-14\n")
    p.write_text(head + "".join(rows))
    r = _run(str(p))
    assert all(c["m_dc"] is None for c in r.data["curves"])
    assert "NaN" not in json.dumps(r.data)
