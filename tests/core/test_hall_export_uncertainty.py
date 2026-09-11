# tests/core/test_hall_export_uncertainty.py
"""Four sigma families are computed and shown in the GUI, and ALL of them died at the CSV
boundary -- as did the run-level warning that says the R_H are noise. The CSV is the
surface papers get written from, so a co-author must be able to reconstruct what was
fitted, over what window, with which uncertainty, from the CSV alone.

Also covers the field-window ladder export (controller audit after Task 7): a lone spread
number next to two columns literally named "sigma" is the exact accident the VSM ladder
export's own code comment records happening for real (a reader publishing
theta = -50.27 +- 0.99 K because a spread sat in a sigma-shaped column). The points-CSV
spread column is therefore named with the established, deliberately ugly convention, and
the rungs themselves -- including which sigma family backed each one, and which rungs were
excluded as unresolved -- go to a sibling .hall_ladder.csv, never silently.
"""
import csv
import pathlib
import numpy as np
import pytest
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.hall import HallAnalyzer
from cryosweep_core.analyzers.hall_tempdep import HallTempDepAnalyzer
from cryosweep_core.io.export import export_result


def _rows(path):
    """Read a cryosweep CSV, skipping the leading '#' comment block."""
    with open(path) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    return list(csv.DictReader(lines))


def _export(tmp_path, dat, out_name="out"):
    res = HallAnalyzer().analyze(load_dat(dat), RunConfig(
        hall={"hall_channel": 1, "thickness_mm": 0.1, "longitudinal_channel": 2}))
    return res, export_result(res, str(tmp_path / out_name), fmt="csv")


def _export_tdep(tmp_path, dat, out_name="td"):
    res = HallTempDepAnalyzer().analyze(load_dat(dat), RunConfig(
        hall={"hall_channel": 1, "thickness_mm": 0.1, "longitudinal_channel": 2}))
    return res, export_result(res, str(tmp_path / out_name), fmt="csv")


# ---- Steps 1-4: sigma, provenance, warnings (field-sweep .points.csv) -------------------

def test_both_sigma_families_are_exported_and_named_apart(tmp_path, hall_synth_path):
    _res, out = _export(tmp_path, hall_synth_path)
    hdr = _rows(out["points"])[0].keys()
    assert any("sigma" in h and "instrument" in h for h in hdr)
    assert any("sigma" in h and "instrument" not in h for h in hdr)


def test_provenance_columns_let_a_reader_reconstruct_the_fit(tmp_path, hall_synth_path):
    _res, out = _export(tmp_path, hall_synth_path)
    hdr = set(_rows(out["points"])[0].keys())
    for col in ("thickness_m", "geometry_sign", "n_points", "rho_xx_field_oe",
                "derived_flags"):
        assert any(col in h for h in hdr), col


def test_existing_columns_keep_their_names_and_order(tmp_path, hall_synth_path):
    _res, out = _export(tmp_path, hall_synth_path)
    with open(out["points"]) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    header = next(csv.reader(lines))
    assert header[:11] == ["temperature (K)", "R_H (m^3/C)", "R_H_raw (m^3/C)",
                           "slope (Ohm/T)", "r2", "antisymmetrized",
                           "carrier_n (1/m^3)", "carrier_type", "rho_xx (Ohm*m)",
                           "mobility (m^2/Vs)", "derived_flags"]


def test_run_warnings_reach_both_the_comment_block_and_a_sibling_file(
        tmp_path, hall_synth_path):
    res, out = _export(tmp_path, hall_synth_path)
    res = res.model_copy(update={"warnings": ["treat these R_H as noise"]})
    out = export_result(res, str(tmp_path / "warned"), fmt="csv")
    head = pathlib.Path(out["points"]).read_text().splitlines()[0]
    assert head.startswith("#") and "noise" in head
    assert "noise" in pathlib.Path(out["warnings"]).read_text()


def test_withheld_values_are_blank_cells_never_zero(tmp_path, hall_synth_path):
    """decline_unresolved() withholds carrier_n/carrier_type/mobility on a point whose
    sigma >= |R_H|; the withheld copy must reach its own *_withheld column, and the live
    columns for that point must be blank, not 0 / "0" / "" that a spreadsheet would coerce
    to zero."""
    res, out = _export(tmp_path, hall_synth_path)
    withheld_pts = [p for p in res.data["points"] if p["withheld"] is not None]
    if not withheld_pts:
        pytest.skip("this fixture resolves every point; covered by the tdep case instead")
    rows = _rows(out["points"])
    row = next(r for r in rows if float(r["temperature (K)"]) == withheld_pts[0]["temperature"])
    assert row["carrier_n (1/m^3)"] == "" and row["mobility (m^2/Vs)"] == ""
    assert row["carrier_n_withheld (1/m^3)"] != ""


# ---- hall_tdep: same sigma/provenance discipline, no ladder ----------------------------

def test_hall_tdep_columns_keep_their_names_and_order(tmp_path, hall_tdep_synth_path):
    _res, out = _export_tdep(tmp_path, hall_tdep_synth_path)
    with open(out["points"]) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    header = next(csv.reader(lines))
    assert header[:12] == ["temperature (K)", "R_H (m^3/C)", "r_h_method", "r2",
                           "antisym_points", "carrier_n (1/m^3)", "carrier_type",
                           "rho_xx (Ohm*m)", "mobility (m^2/Vs)", "low_confidence",
                           "excitation (uA)", "current_density_J (A/m^2)"]


def test_hall_tdep_gains_sigma_flags_and_withheld_columns(tmp_path, hall_tdep_synth_path):
    res, out = _export_tdep(tmp_path, hall_tdep_synth_path)
    hdr = set(_rows(out["points"])[0].keys())
    assert any("sigma" in h and "instrument" in h for h in hdr)
    assert any("sigma" in h and "instrument" not in h for h in hdr)
    assert "derived_flags" in hdr, "hall_tdep had no flags column at all before this"
    for col in ("thickness_m", "geometry_sign", "carrier_n_withheld"):
        assert any(col in h for h in hdr), col

    withheld_pts = [p for p in res.data["points"] if p["withheld"] is not None]
    assert withheld_pts, "hall_tdep_synth is measured (2026-09) to withhold some points"
    rows = _rows(out["points"])
    row = next(r for r in rows
               if float(r["temperature (K)"]) == withheld_pts[0]["temperature"])
    assert row["carrier_n (1/m^3)"] == ""
    assert row["carrier_n_withheld (1/m^3)"] != ""
    assert "r_h_unresolved" in row["derived_flags"]


def test_hall_tdep_never_gains_a_ladder_column_or_sibling(tmp_path, hall_tdep_synth_path):
    """Scope guard (controller audit): hall_tdep has no ladder concept at all. Task 7
    measured 0 differing envelope leaves on this probe and the export must match."""
    _res, out = _export_tdep(tmp_path, hall_tdep_synth_path)
    with open(out["points"]) as f:
        header = next(csv.reader(ln for ln in f if not ln.startswith("#")))
    assert not any("window_spread" in h or "r_h_spread" in h for h in header)
    assert "hall_ladder" not in out
    assert not (tmp_path / "td.hall_ladder.csv").exists()


# ---- Section (d): the field-window ladder needs more than one column -------------------

def test_ladder_spread_column_uses_the_not_an_error_bar_name(tmp_path, hall_synth_path):
    """A column called r_h_spread sitting between two columns called r_h_sigma* is the
    exact accident the VSM ladder export exists to prevent for theta/mu_eff. Pin the
    ugly, self-describing name verbatim and forbid the bare name from ever reappearing."""
    _res, out = _export(tmp_path, hall_synth_path)
    with open(out["points"]) as f:
        header = next(csv.reader(ln for ln in f if not ln.startswith("#")))
    assert "r_h_window_spread_not_an_error_bar (m^3/C)" in header
    assert not any(h == "r_h_spread (m^3/C)" for h in header)


def test_ladder_sibling_is_written_with_sigma_kind_and_survives_unresolved_rungs(
        tmp_path, hall_synth_path):
    """hall_synth.dat carries no Std. Dev. column, so every rung's sigma is residual
    (fit-scatter) -- sigma_kind must say so, not be guessed from magnitude."""
    res, out = _export(tmp_path, hall_synth_path)
    assert "hall_ladder" in out
    lp = pathlib.Path(out["hall_ladder"])
    assert lp.exists() and lp.name == "out.hall_ladder.csv"
    rows = list(csv.DictReader(lp.open()))
    assert rows, "at least one point on this fixture reaches a ladder"
    assert set(rows[0].keys()) == {"temperature (K)", "f", "R_H (m^3/C)", "sigma (m^3/C)",
                                    "sigma_kind", "r2", "n_points", "unresolved"}
    assert all(r["sigma_kind"] == "residual" for r in rows)
    # every rung the analyzer computed for a laddered point reaches the CSV, including
    # ones excluded from the spread -- not just the resolved ones.
    ladder_pts = [p for p in res.data["points"] if p["r_h_ladder"]]
    n_rungs_expected = sum(len(p["r_h_ladder"]) for p in ladder_pts)
    assert len(rows) == n_rungs_expected


def test_ladder_sibling_names_instrument_sigma_kind_and_marks_unresolved_rungs(tmp_path):
    """Reuses the field-window-ladder test's own instrument-sigma fixture shape (Std.
    Dev. column large enough to swamp R_H): every rung must come back unresolved AND the
    sibling CSV must say the sigma backing that verdict was instrument, not residual."""
    hdr = ("[Header]\nBYAPP, Resistivity\nINFO, ladder_instsig, SAMPLE\n[Data]\n"
           "Temperature (K),Magnetic Field (Oe),Bridge 1 Resistance (Ohms),"
           "Bridge 1 Resistivity (Ohm-m),Bridge 1 Std. Dev. (Ohm-m),"
           "Bridge 2 Resistance (Ohms),Bridge 2 Resistivity (Ohm-m)\n")
    sd = 5e-5   # Ohm-m; ratio (R/rho)=1000 -> instrument sigma_R = SD*1000 = 5e-2 Ohm/row
    rows = []
    for b in np.arange(-20000.0, 20000.1, 200.0):
        rxy = 1e-3 + 5e-4 * (b / 1e4)
        rows.append(f"10.0000,{b:.1f},{rxy:.10e},{rxy / 1000:.10e},"
                    f"{sd:.10e},{1e-3:.10e},{1e-6:.10e}")
    rows += [f"{t:.4f},20000.0,1.5e-3,1.5e-6,{sd:.10e},1e-3,1e-6"
             for t in np.arange(11.0, 51.0, 2.0)]
    p = tmp_path / "hall_export_instsig.dat"
    p.write_text(hdr + "\n".join(rows) + "\n")
    _res, out = _export(tmp_path, p, out_name="instsig_out")
    lp = pathlib.Path(out["hall_ladder"])
    ladder_rows = list(csv.DictReader(lp.open()))
    assert ladder_rows
    assert all(r["sigma_kind"] == "instrument" for r in ladder_rows)
    assert all(r["unresolved"] == "True" for r in ladder_rows)


def test_ladder_thin_flag_round_trips_through_the_existing_flags_column(tmp_path):
    """ladder_thin needs no new column (it rides in derived_flags, already exported) --
    but a test must cover that it actually reaches the CSV cell."""
    hdr = ("[Header]\nBYAPP, Resistivity\nINFO, ladder_thin, SAMPLE\n[Data]\n"
           "Temperature (K),Magnetic Field (Oe),Bridge 1 Resistance (Ohms),"
           "Bridge 1 Resistivity (Ohm-m),Bridge 1 Std. Dev. (Ohm-m),"
           "Bridge 2 Resistance (Ohms),Bridge 2 Resistivity (Ohm-m)\n")
    sd = 2e-6   # fix round 1's measured boundary: exactly 2 of 4 rungs resolve
    rows = []
    for b in np.arange(-20000.0, 20000.1, 200.0):
        rxy = 1e-3 + 5e-4 * (b / 1e4)
        rows.append(f"10.0000,{b:.1f},{rxy:.10e},{rxy / 1000:.10e},"
                    f"{sd:.10e},{1e-3:.10e},{1e-6:.10e}")
    rows += [f"{t:.4f},20000.0,1.5e-3,1.5e-6,{sd:.10e},1e-3,1e-6"
             for t in np.arange(11.0, 51.0, 2.0)]
    p = tmp_path / "hall_export_thin.dat"
    p.write_text(hdr + "\n".join(rows) + "\n")
    res, out = _export(tmp_path, p, out_name="thin_out")
    assert "ladder_thin" in res.data["points"][0]["derived_flags"]
    row = _rows(out["points"])[0]
    assert "ladder_thin" in row["derived_flags"]


def test_gated_run_has_no_ladder_sibling_and_a_blank_spread_column(tmp_path, hall_synth_path):
    """No --thickness -> r_h_ladder/r_h_spread are None on every point (thickness gates
    R_H itself); the sibling ladder file must not be written at all."""
    res = HallAnalyzer().analyze(load_dat(hall_synth_path), RunConfig(
        hall={"hall_channel": 1, "longitudinal_channel": 2}))   # no thickness_mm
    assert res.status == "gated"
    out = export_result(res, str(tmp_path / "gated"), fmt="csv")
    assert "hall_ladder" not in out
    assert not (tmp_path / "gated.hall_ladder.csv").exists()
    row = _rows(out["points"])[0]
    assert row["r_h_window_spread_not_an_error_bar (m^3/C)"] == ""
