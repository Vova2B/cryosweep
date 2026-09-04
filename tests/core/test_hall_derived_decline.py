"""KNOWN-ISSUES #20 (2026-09-02): carrier_n and mobility derive from the TRUSTED
Stage B (antisymmetrized) R_H. When Stage B declines (no ± field overlap), the old
code silently fell back to the untrusted Stage A raw fit — publishing a carrier
density and mobility beside an empty R_H cell. Under the project's decline
discipline the derived quantities are withheld and a machine-readable reason is
carried (same idiom as the resistivity power-law decline / power_law_flags)."""
import csv
import numpy as np
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.hall import HallAnalyzer

_HDR = ("[Header]\nBYAPP, Resistivity\nINFO, decline_synth, SAMPLE\n[Data]\n"
        "Temperature (K),Magnetic Field (Oe),Bridge 1 Resistance (Ohms),"
        "Bridge 2 Resistance (Ohms),Bridge 2 Resistivity (Ohm-m)\n")


def _row(T, B_oe):
    r = 1e-3 + 5e-4 * (B_oe / 1e4)
    return f"{T:.4f},{B_oe:.1f},{r:.10e},{1e-3:.10e},{1e-6:.10e}"


def _write_one_sided_dat(tmp_path):
    rows = []
    # full ± loop at 10 K -> Stage B runs, derived quantities published
    rows += [_row(10.0, b) for b in np.arange(-20000.0, 20000.1, 500.0)]
    rows += [_row(t, 20000.0) for t in np.arange(15.0, 200.0, 5.0)]   # T ramp separator
    # POSITIVE-ONLY sweep at 200 K -> no ± overlap -> Stage B declines, R_H None
    rows += [_row(200.0, b) for b in np.arange(500.0, 20000.1, 500.0)]
    p = tmp_path / "one_sided.dat"
    p.write_text(_HDR + "\n".join(rows) + "\n")
    return p


def _analyze(tmp_path):
    rt = load_dat(_write_one_sided_dat(tmp_path))
    cfg = RunConfig(hall={"hall_channel": 1, "thickness_mm": 0.1,
                          "longitudinal_channel": 2})
    return HallAnalyzer().analyze(rt, cfg)


def test_derived_quantities_withheld_without_trusted_R_H(tmp_path):
    res = _analyze(tmp_path)
    pts = {p["temperature"]: p for p in res.data["points"]}
    ok, bad = pts[10.0], pts[200.0]
    # trusted group: everything published, no flags
    assert ok["R_H"] is not None and ok["carrier_n"] is not None
    assert ok["mobility"] is not None and ok["derived_flags"] == []
    # declined group: Stage A stays visible for transparency, Stage C is withheld
    assert bad["R_H"] is None and bad["R_H_raw"] is not None
    assert bad["carrier_n"] is None
    assert bad["carrier_type"] is None
    assert bad["mobility"] is None
    assert bad["derived_flags"] == ["antisym_r_h_missing"]


def test_declined_csv_cells_are_blank_with_flag(tmp_path):
    from cryosweep_core.io.export import export_result
    res = _analyze(tmp_path)
    outs = export_result(res, tmp_path / "out")
    with open(outs["points"]) as f:
        rows = {float(r["temperature (K)"]): r for r in csv.DictReader(f)}
    bad = rows[200.0]
    assert bad["R_H (m^3/C)"] == "" and bad["carrier_n (1/m^3)"] == ""
    assert bad["carrier_type"] == "" and bad["mobility (m^2/Vs)"] == ""
    assert bad["derived_flags"] == "antisym_r_h_missing"
    ok = rows[10.0]
    assert ok["carrier_n (1/m^3)"] != "" and ok["derived_flags"] == ""
