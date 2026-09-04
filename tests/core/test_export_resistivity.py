import pytest
import csv
from pathlib import Path
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.resistivity import ResistivityAnalyzer
from cryosweep_core.io.export import export_result

def _analyze(res_path):
    return ResistivityAnalyzer().analyze(load_dat(res_path), RunConfig())


def test_export_mr_percent_sidecar(tmp_path, res_path):
    res = _analyze(res_path)
    out = export_result(res, str(tmp_path / "res"), fmt="csv")
    mp = Path(out["mr_percent"])
    assert mp.name.endswith(".mr_percent.csv")
    header = mp.read_text().splitlines()[0]
    assert header == "channel,held_temp_k,max_abs_field_oe,direction,mr_percent_at_max_field,low_confidence"


def test_derived_csv_has_tc_columns(tmp_path, res_path):
    res = _analyze(res_path)
    out = export_result(res, str(tmp_path / "res"), fmt="csv")
    header = Path(out["derived"]).read_text().splitlines()[0]
    # F1 (final-review): the four honesty columns are appended AFTER the Tc trio, so the
    # original 13 keep their names and order.
    assert "tc_onset_k,tc_mid_k,tc_zero_k" in header
    assert header.endswith(
        "rrr_std,power_law_n_sigma,power_law_n_spread,power_law_flags")


def test_derived_csv_carries_the_window_sensitivity_beside_the_number(tmp_path, res_path):
    """F1 (final-review): .derived.csv wrote power_law_n and rrr BARE while the GUI, for the
    identical file, said 'WINDOW-SENSITIVE: n(15->30 K) = 3.04->0.649'. The CSV is the surface
    the owner publishes from — a bare n there is the exact failure this slice exists to end."""
    res = _analyze(res_path)
    out = export_result(res, str(tmp_path / "res"), fmt="csv")
    with open(out["derived"]) as f:
        rows = {r["channel"]: r for r in csv.DictReader(f)}
    r1 = rows["1"]
    assert float(r1["power_law_n"]) == pytest.approx(0.649, rel=1e-2)
    # the spread and the flag ride in the SAME row as the number they qualify
    assert float(r1["power_law_n_spread"]) > 0.05
    assert "window_sensitive" in r1["power_law_flags"].split(";")
    assert float(r1["power_law_n_sigma"]) > 0
    assert float(r1["rrr"]) == pytest.approx(18.52, rel=1e-2)
    assert float(r1["rrr_std"]) > 0

def test_export_writes_curves_derived_capabilities(tmp_path, res_path):
    res = _analyze(res_path)
    stem = tmp_path / "res_out"
    out = export_result(res, str(stem), fmt="csv")
    assert "curves" in out and "derived" in out and "capabilities" in out
    # curves: long format with a row per physical point
    with open(out["curves"]) as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert {"bridge", "curve_type", "x", "rho_ohm_cm"} <= set(rows[0].keys())
    assert {r["curve_type"] for r in rows} == {"rho_T", "rho_H"}
    # derived: one row per bridge with RRR (Ch2 routed out as Hall-wired at defaults)
    with open(out["derived"]) as f:
        drows = {r["channel"]: r for r in csv.DictReader(f)}
    assert set(drows) == {"1"}
    assert float(drows["1"]["rrr"]) > 1.0
    # capabilities table present
    with open(out["capabilities"]) as f:
        caps = {r["name"]: r for r in csv.DictReader(f)}
    assert caps["RRR"]["applicable"] == "True"
