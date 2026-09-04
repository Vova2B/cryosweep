import json, pathlib
import numpy as np
import pytest
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.hall import HallAnalyzer
from cryosweep_core.plotting.render import render_kind
from cryosweep_core.result import Result, Provenance

ROOT = pathlib.Path(__file__).resolve().parents[2]
from tests.core.conftest import repo_root

REPO = repo_root()   # the repo root (docs/, skill/ and the real data live there, not in the app folder)

MANIFEST = REPO / "docs/superpowers/pq-reference-gallery/manifest.json"


def _dat(rel):
    """Manifest dat: repo-root-relative for real data; fixtures live under the app after the split."""
    p = REPO / rel
    return p if p.exists() else ROOT / rel


def _manifest_entry(entry_id):
    from tests.core.conftest import require_manifest
    for e in json.loads(require_manifest().read_text()):
        if e.get("id") == entry_id:
            return e
    raise KeyError(entry_id)


def _fit_lines(ax):
    return [ln for ln in ax.lines if ln.get_gid() == "fit"]


def test_rxy_vs_b_draws_mirror_fits_for_antisym_points(hall_synth_path):
    # hall_synth.dat: 3 temperatures, all antisymmetrized (full ±loop).
    res = HallAnalyzer().analyze(
        load_dat(hall_synth_path),
        RunConfig(hall={"hall_channel": 1, "thickness_mm": 0.5}))
    fig = render_kind(res, "hall_rxy_vs_B")
    ax = fig.axes[0]
    fits = _fit_lines(ax)
    # 3 T x (one +B-branch line + one -B-branch line) = 6 mirror fit lines
    assert len(fits) == 6
    assert all(ln.get_linestyle() == "--" for ln in fits)


def _raw_only_result():
    # A HallTempPoint with a raw sweep but NOT antisymmetrized -> no fit line.
    data = {"probe": "hall", "hall_channel": 1, "thickness_m": 5e-4, "geometry_sign": 1,
            "points": [{"temperature": 10.0, "n_points": 3, "antisymmetrized": False,
                        "slope_ohm_per_T": None, "asym_intercept_ohm": None,
                        "field_raw_T": [-2.0, 0.0, 2.0], "R_xy_raw": [-1e-3, 0.0, 1e-3]}],
            "capabilities": []}
    return Result(status="ok", confidence=1.0, data=data,
                  provenance=Provenance(file="x", sha256="0", app_version="", config={}))


def test_rxy_vs_b_no_fit_for_raw_only_point():
    fig = render_kind(_raw_only_result(), "hall_rxy_vs_B")
    assert _fit_lines(fig.axes[0]) == []


def _blownup_slope_result():
    # A point whose antisym slope is pathological (garbage sparse-T fit): the mirror line
    # would shoot off-scale (near-vertical) if not envelope-clipped. Must draw NO line.
    data = {"probe": "hall", "hall_channel": 1, "thickness_m": 5e-4, "geometry_sign": 1,
            "points": [{"temperature": 300.0, "n_points": 6, "antisymmetrized": True,
                        "slope_ohm_per_T": -8.0e5, "asym_intercept_ohm": 5e-7, "r2": 0.0,
                        "field_raw_T": [-6.0, -3.0, -0.1, 0.1, 3.0, 6.0],
                        "R_xy_raw": [9.5e-4, 9.4e-4, 9.6e-4, 9.5e-4, 9.5e-4, 9.6e-4]}],
            "capabilities": []}
    return Result(status="ok", confidence=1.0, data=data,
                  provenance=Provenance(file="x", sha256="0", app_version="", config={}))


def test_rxy_vs_b_clips_blownup_slope_no_vertical_line():
    # Regression: a blown-up antisym slope must be envelope-clipped away (no near-vertical
    # artifact overlaying the raw band). Visual gate caught this on real 300 K data.
    fig = render_kind(_blownup_slope_result(), "hall_rxy_vs_B")
    assert _fit_lines(fig.axes[0]) == []


def test_hall_raw_vs_asym_fit_count_unchanged(hall_synth_path):
    # Guard: A must NOT touch hall_raw_vs_asym (still 9 fit lines: 3 T x 3).
    res = HallAnalyzer().analyze(
        load_dat(hall_synth_path),
        RunConfig(hall={"hall_channel": 1, "thickness_mm": 0.5}))
    fig = render_kind(res, "hall_raw_vs_asym")
    assert len(_fit_lines(fig.axes[0])) == 9


def test_rxy_vs_b_uses_manifest_runconfig():
    entry = _manifest_entry("hall_rxy_vs_B")
    dat = _dat(entry["dat"])
    if not dat.exists():
        pytest.skip("gallery reference .dat not present")
    res = HallAnalyzer().analyze(load_dat(dat), RunConfig(**entry["runconfig"]))
    fig = render_kind(res, "hall_rxy_vs_B")
    # at least one antisymmetrized T -> at least one mirror fit line drawn
    assert len(_fit_lines(fig.axes[0])) >= 1


def test_two_point_coverage_does_not_deflate_confidence():
    """Regression (final-review): extending R_H(T) coverage with flagged 2-point points must NOT
    lower status/confidence — `frac` is computed over the trusted antisym points only, so a file
    with all-good antisym fits + a long 2-point tail stays confident, not blended down."""
    from cryosweep_core.analyzers.hall_tempdep import HallTempDepAnalyzer
    e = _manifest_entry("hall_tdep_RH_T")
    dat = _dat(e["dat"])
    if not dat.exists():
        pytest.skip("tdep fixture not present")
    rc = dict(e["runconfig"]); hall = rc.pop("hall", None)
    res = HallTempDepAnalyzer().analyze(load_dat(dat), RunConfig.load(hall=hall, **rc))
    pts = res.data["points"]
    n_anti = sum(1 for p in pts if p.get("R_H") is not None and p.get("r_h_method") != "2point")
    n_2pt = sum(1 for p in pts if p.get("r_h_method") == "2point")
    assert n_anti > 0 and n_2pt > 0                       # both methods present (coverage extended)
    # confidence reflects the antisym fraction (1.0 here), NOT antisym/(antisym+2point) ~0.6
    assert res.confidence == 1.0 and res.status == "ok"
    assert res.confidence_parts["antisym_fraction"] == 1.0


def _two_panel_result_with_rxx():
    # two T's, each with a signed-B longitudinal sweep that VARIES with B
    def pt(T, off):
        B = [-2.0, -1.0, 0.0, 1.0, 2.0]
        return {"temperature": T, "n_points": 5, "antisymmetrized": True,
                "slope_ohm_per_T": -6e-4, "asym_intercept_ohm": 0.0,
                "field_raw_T": B, "R_xy_raw": [-6e-4 * b for b in B],
                "field_rxx_T": B, "R_xx_raw": [off + 3e-4 * b * b for b in B],
                "rho_xx": None}
    data = {"probe": "hall", "hall_channel": 1, "thickness_m": 5e-4, "geometry_sign": 1,
            "points": [pt(10.0, 1e-3), pt(100.0, 2e-3)], "capabilities": []}
    return Result(status="ok", confidence=1.0, data=data,
                  provenance=Provenance(file="x", sha256="0", app_version="", config={}))


def _two_panel_result_rho_only():
    data = {"probe": "hall", "hall_channel": 1, "thickness_m": 5e-4, "geometry_sign": 1,
            "points": [{"temperature": 10.0, "n_points": 3, "antisymmetrized": True,
                        "slope_ohm_per_T": -6e-4, "asym_intercept_ohm": 0.0,
                        "field_raw_T": [-2.0, 0.0, 2.0], "R_xy_raw": [1.2e-3, 0.0, -1.2e-3],
                        "field_rxx_T": [], "R_xx_raw": [], "rho_xx": 1e-6}],
            "capabilities": []}
    return Result(status="ok", confidence=1.0, data=data,
                  provenance=Provenance(file="x", sha256="0", app_version="", config={}))


def test_two_panel_right_plots_zero_subtracted_rxx_when_present():
    from cryosweep_core.plotting.catalog import series_hall_two_panel
    res = _two_panel_result_with_rxx()
    keys = {s.key for s in series_hall_two_panel(res)}
    assert {"rxxb:10.0K", "rxxb:100.0K"} <= keys
    fig = render_kind(res, "hall_two_panel")
    ax_right = fig.axes[1]
    ys = ax_right.lines[0].get_ydata()
    # zero-subtracted: value at B=0 is 0 for each T series
    assert abs(min(abs(v) for v in ys)) < 1e-12
    # two T series drawn on the right panel (one per temperature)
    assert len([ln for ln in ax_right.lines if ln.get_gid() != "refline"]) == 2


def test_two_panel_right_falls_back_to_rho_xx_t():
    from cryosweep_core.plotting.catalog import series_hall_two_panel
    res = _two_panel_result_rho_only()
    keys = {s.key for s in series_hall_two_panel(res)}
    assert "rhoxx" in keys and not any(k.startswith("rxxb:") for k in keys)
    fig = render_kind(res, "hall_two_panel")
    assert fig.axes[1].get_title() == "Longitudinal"
