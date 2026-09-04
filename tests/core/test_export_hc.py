# tests/core/test_export_hc.py
import csv, numpy as np, pandas as pd, pathlib
from pathlib import Path
from types import SimpleNamespace
from cryosweep_core.analyzers.hc import HCAnalyzer
from cryosweep_core.config import RunConfig
from cryosweep_core.fitting.heat_capacity import specific_heat_full
from cryosweep_core.io.export import export_result
from cryosweep_core.io.loader import load_dat
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry

FIX = pathlib.Path(__file__).parent / "fixtures"

class _Hdr: title="s"; app_version=None; n_atoms=3.0

def _result():
    T = np.linspace(2.0, 300.0, 150)
    cp = specific_heat_full(T, 220.0, 3.0, 0.01, 60.0, 180.0, 1.0, 1.5)
    df = pd.DataFrame({"Sample Temp (Kelvin)": T, "Samp HC (mJ/mole-K)": cp*1e3,
                       "Field (Oe)": np.zeros_like(T)})
    return HCAnalyzer().analyze(SimpleNamespace(df=df, header=_Hdr(), path=None), RunConfig())

def test_hc_export_writes_all_tables(tmp_path):
    out = export_result(_result(), tmp_path / "hc")
    for key in ("fit_params", "comparison", "model_curves", "points", "meta"):
        assert key in out and Path(out[key]).exists()
    rows = list(csv.DictReader(open(out["comparison"])))
    assert any(r["param"] == "gamma" for r in rows)


def test_field_dependence_csv_written(tmp_path):
    res = analyze_file(load_dat(str(FIX / "hc_multifield_synth.dat")),
                       RunConfig.load(), build_default_registry())
    out = export_result(res, tmp_path / "hc")
    fd = pathlib.Path(out["field_dependence"])
    assert fd.exists()
    rows = list(csv.DictReader(fd.open()))
    assert {"field_oe", "model", "param", "value", "sigma"} <= set(rows[0])
    gammas = [r for r in rows if r["param"] == "gamma" and r["model"] == "debye_t3"]
    assert len(gammas) == 3                               # one per field


def test_single_field_no_field_dependence_file(tmp_path, hc_synth_path):
    res = analyze_file(load_dat(str(hc_synth_path)), RunConfig.load(), build_default_registry())
    out = export_result(res, tmp_path / "hc")
    assert "field_dependence" not in out                  # only when >=2 field groups


def test_entropy_columns_exported(tmp_path, hc_path):
    res = analyze_file(load_dat(str(hc_path)), RunConfig.load(), build_default_registry())
    assert res.data.get("entropy_available")              # fixture yields S(T)
    out = export_result(res, tmp_path / "hc")
    ep = pathlib.Path(out["entropy"])
    assert ep.exists()
    rows = list(csv.DictReader(ep.open()))
    header = set(rows[0])
    assert {"T", "S_total"} <= header
    # column length aligns to entropy_temperature
    assert len(rows) == len(res.data["entropy_temperature"])
    # magnetic present as a list on this fixture
    if res.data.get("entropy_magnetic") is not None:
        assert "S_magnetic" in header
    # never write non-finite / None literals into the entropy columns
    cols = ("T", "S_total") + (("S_magnetic",) if "S_magnetic" in header else ())
    for r in rows:
        for c in cols:
            assert r[c].strip().lower() not in ("nan", "none", "inf", "-inf")


def test_entropy_csv_carries_rln_verdict(tmp_path, hc_path):
    # 2026-08-10 spec §7: the entropy CSV appends the Rln match verdict AFTER the
    # existing columns — repeated on every row (file-level scalar, tidy-long), so any
    # row read alone carries it. Name-keyed readers safe; exact-width readers break.
    res = analyze_file(load_dat(str(hc_path)), RunConfig.load(), build_default_registry())
    sug = res.data.get("entropy_rln_suggestion")
    assert sug and "matched" in sug                       # Task-11 verdict present
    out = export_result(res, tmp_path / "hc")
    rows = list(csv.DictReader(pathlib.Path(out["entropy"]).open()))
    hdr = list(rows[0].keys())
    assert hdr[-3:] == ["rln_label", "rln_matched", "rln_rel_err"]  # appended last
    for r in (rows[0], rows[-1]):                         # constant on every row
        assert r["rln_label"] == sug["label"]
        assert r["rln_matched"] == str(bool(sug["matched"]))
    if sug.get("rel_err") is not None:
        # _cell writes %.6g -> compare at that precision
        assert abs(float(rows[0]["rln_rel_err"]) - sug["rel_err"]) < 1e-4 * max(1.0, abs(sug["rel_err"]))
