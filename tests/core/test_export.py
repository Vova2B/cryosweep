import json, csv, pathlib
from cryosweep_core.analyzers.mag import VSMAnalyzer
from cryosweep_core.io.loader import load_dat
from cryosweep_core.io.export import export_result
from cryosweep_core.config import RunConfig

FIX = pathlib.Path(__file__).parent / "fixtures" / "vsm_synth.dat"

def test_export_csv_and_sidecar(tmp_path):
    rt = load_dat(FIX); res = VSMAnalyzer().analyze(rt, RunConfig.load())
    out = export_result(res, tmp_path / "vsm", fmt="csv")
    points = pathlib.Path(out["points"])
    rows = list(csv.DictReader(points.open()))
    assert "temperature (K)" in rows[0] and "inv_chi (mol*Oe/emu)" in rows[0]
    fitp = list(csv.DictReader(pathlib.Path(out["fit_params"]).open()))
    assert any(r["param"] == "C" for r in fitp)
    assert all("value" in r and "sigma" in r for r in fitp)
    meta = json.loads(pathlib.Path(out["meta"]).read_text())
    assert meta["sha256"] == res.provenance.sha256
    assert meta["unit_system"] in ("CGS", "SI")

def test_export_hc_points(tmp_path, hc_path):
    from cryosweep_core.analyzers.hc import HCAnalyzer
    rt = load_dat(hc_path); res = HCAnalyzer().analyze(rt, RunConfig.load())
    out = export_result(res, tmp_path / "hc", fmt="csv")
    # HC-specific exporter now produces 5 tables
    for key in ("fit_params", "comparison", "model_curves", "points", "meta"):
        assert key in out and pathlib.Path(out[key]).exists()
    rows = list(csv.DictReader(pathlib.Path(out["points"]).open()))
    hdr = rows[0]
    assert "temperature (K)" in hdr
    assert "cp (J/(mol*K))" in hdr
    fitp = list(csv.DictReader(pathlib.Path(out["fit_params"]).open()))
    # theta_D appears in low-T model params column
    assert any(r["param"] == "theta_D" for r in fitp)

def test_export_json(tmp_path):
    rt = load_dat(FIX); res = VSMAnalyzer().analyze(rt, RunConfig.load())
    out = export_result(res, tmp_path / "vsm", fmt="json")
    payload = json.loads(pathlib.Path(out["points"]).read_text())
    assert len(payload) == len(res.data["temperature"])


def test_export_cw_ladder_columns_appended(tmp_path):
    # 2026-08-10 spec §7: the CW CSV grows ladder columns AFTER the existing four.
    # Name-keyed readers stay safe (asserted here); the pre-existing param rows keep
    # their exact four cells (blank ladder cells), so old-column reads are unchanged.
    rt = load_dat(FIX); res = VSMAnalyzer().analyze(rt, RunConfig.load())
    assert res.data.get("cw_ladder")                      # vsm_synth carries a ladder
    out = export_result(res, tmp_path / "vsm", fmt="csv")
    fitp = list(csv.DictReader(pathlib.Path(out["fit_params"]).open()))
    hdr = list(fitp[0].keys())
    assert hdr[:4] == ["param", "value", "sigma", "unit"]           # existing first
    assert hdr[4:] == ["rung_tmin_k", "r2", "n_points",
                       "sigma_kind", "flags"]                       # appended after
    # original param rows untouched: ladder cells blank/absent there
    crow = next(r for r in fitp if r["param"] == "C")
    assert not (crow.get("rung_tmin_k") or "").strip()
    # one theta + one mu_eff row per rung, tagged by the rung's T_min, cells filled
    rungs = res.data["cw_ladder"]
    trows = [r for r in fitp if r["param"].startswith("theta(T>=")]
    murows = [r for r in fitp if r["param"].startswith("mu_eff(T>=")]
    assert len(trows) == len(murows) == len(rungs)
    assert float(trows[0]["value"]) == rungs[0]["theta_k"]
    assert float(trows[0]["rung_tmin_k"]) == rungs[0]["tmin_k"]
    assert trows[0]["unit"] == "K"
    # the spread rows ride along when the analyzer measured a spread
    if res.data.get("theta_spread_k") is not None:
        srow = next(r for r in fitp
                    if r["param"] == "theta_window_spread_not_an_error_bar")
        assert float(srow["value"]) == res.data["theta_spread_k"]


def test_fit_params_csv_distinguishes_sigma_from_spread_and_is_not_ragged(tmp_path):
    """F2 + F9 (final-review).

    F2: a reader was taking `theta`, a column literally NAMED `sigma`, and publishing
    theta = -50.27 +- 0.99 K — while `fit.quality_flags` (holding `window_sensitive`) reached
    no column and no row of this file, and the 12.7 K window spread landed 12 rows later as a
    param row with a BLANK sigma cell, i.e. structurally another fitted parameter. U3 requires
    "spread != error bar" to hold in EVERY rendering, and this is the VSM probe's primary
    numeric export.

    F9: every row is padded to the header width. The regenerated goldens were ragged
    ({7: 11, 4: 5} fields); pandas/DictReader cope, numpy.genfromtxt and fixed-width
    importers do not."""
    rt = load_dat(FIX); res = VSMAnalyzer().analyze(rt, RunConfig.load())
    out = export_result(res, tmp_path / "vsm", fmt="csv")
    path = pathlib.Path(out["fit_params"])
    rows = list(csv.reader(path.open()))
    # F9: ragged-free — one field count for the whole file
    assert len({len(r) for r in rows}) == 1, {len(r) for r in rows}
    fitp = list(csv.DictReader(path.open()))
    # F2: the sigma cell says what family it is, on EVERY row that has one
    theta = next(r for r in fitp if r["param"] == "theta")
    assert theta["sigma_kind"] == "fit_scatter_stat"
    # F2: the spread is named so it cannot be read as a parameter with an unknown error bar
    spread = next(r for r in fitp
                  if r["param"] == "theta_window_spread_not_an_error_bar")
    assert spread["sigma"] == "" and spread["sigma_kind"] == "window_spread"
    # F2: quality_flags reach a column
    flags = set(res.data["fit"].get("quality_flags") or [])
    assert set(filter(None, theta["flags"].split(";"))) == flags
