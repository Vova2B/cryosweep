import numpy as np, pandas as pd
from cryosweep_core.io.loader import load_dat


def test_trailing_commas_do_not_shift_columns(tmp_path):
    """QD .dat rows often carry trailing empty fields (more values than header names).
    pandas would auto-promote the leading columns to an index, shifting every named column
    (real bug: the low-T heat-capacity file -> 'Sample Temp' held Samp-HC-Err values). The loader must align
    names to the first fields and drop the trailing empties instead."""
    p = tmp_path / "trailing.dat"
    # 4 header names; data rows carry 3 EXTRA trailing commas (7 fields vs 4 names).
    rows = ["[Header]", "BYAPP,HeatCapacity", "[Data]",
            "Sample Temp (Kelvin),Samp HC (mJ/mole-K),Field (Oersted),Pressure (Torr)"]
    for t in (50.0, 40.0, 30.0):
        rows.append(f"{t},{t*10:.1f},0.0,1e-5,,,")     # value for col4 then 3 empty trailers
    p.write_text("\n".join(rows) + "\n")
    rt = load_dat(str(p))
    assert "Sample Temp (Kelvin)" in rt.df.columns
    st = pd.to_numeric(rt.df["Sample Temp (Kelvin)"], errors="coerce").to_numpy(float)
    hc = pd.to_numeric(rt.df["Samp HC (mJ/mole-K)"], errors="coerce").to_numpy(float)
    assert list(st) == [50.0, 40.0, 30.0], f"temperature column shifted: {st}"
    assert list(hc) == [500.0, 400.0, 300.0], f"HC column shifted: {hc}"


def test_load_hc(hc_path):
    rt = load_dat(hc_path)
    assert "Sample Temp (Kelvin)" in rt.df.columns
    assert any("Addenda HC" in c for c in rt.df.columns)   # mangled-µ column present
    assert rt.df.shape[0] > 50
    assert rt.header.app == "HeatCapacity"

def test_load_res(res_path):
    rt = load_dat(res_path)
    assert "Temperature (K)" in rt.df.columns
    assert "Magnetic Field (Oe)" in rt.df.columns
    assert "Bridge 1 Resistivity (Ohm-m)" in rt.df.columns
