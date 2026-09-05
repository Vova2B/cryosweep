"""KNOWN-ISSUES item 21 — excitation current reported; J = I/(w·t) as a gated capability.

The owner's decision is narrower than the issue text: report the instrument's excitation
current I directly (derivable from the file alone — it answers "was the drive constant,
and low enough not to heat the sample"), and make current density J a capability that
activates only when sample width AND thickness are both supplied, through the EXISTING
--width-mm / SampleGeometry route. An ungated J on unset geometry would be
scale-arbitrary — the exact failure the resistivity geometry-unset warning names.

On the shipped real example the excitation is constant at 7999.997 µA, so the I and J
curves are flat horizontal lines. That is the correct result, not a bug.
"""
import numpy as np
import pathlib
import pytest

from cryosweep_core.io.loader import load_dat
from cryosweep_core.io.columns import canonicalize_columns
from cryosweep_core.config import RunConfig
from cryosweep_core.registry import build_default_registry
from cryosweep_core.analyzers.dispatch import analyze_file

EXAMPLE = pathlib.Path(__file__).resolve().parents[2] / "examples" / "hall_mixed_sweeps.dat"
_REG = build_default_registry()


def _run(width_mm=None, thickness_mm=0.07):
    cfg = RunConfig.load(probe_override="hall_tdep")
    cfg.hall.hall_channel = 1
    cfg.hall.thickness_mm = thickness_mm
    cfg.hall.longitudinal_channel = 2
    if width_mm is not None:
        cfg.geometry.width_mm = width_mm
    return analyze_file(load_dat(str(EXAMPLE)), cfg, _REG)


def _cap(result, name):
    return next(c for c in result.data["capabilities"] if c["name"] == name)


# ---------------- canonicalization ----------------

def test_excitation_columns_canonicalize():
    rt = load_dat(str(EXAMPLE))
    _df, cmap = canonicalize_columns(rt.df, rt.header)
    assert "excitation_ch1" in cmap.logical
    assert "excitation_ch2" in cmap.logical
    assert cmap.unit["excitation_ch1"] == "uA"


# ---------------- I is reported from the file alone ----------------

def test_excitation_current_reported_per_point_and_constant_here():
    pts = _run().data["points"]
    ex = [p["excitation_uA"] for p in pts if p["excitation_uA"] is not None]
    assert len(ex) >= 100, "excitation should be reported on essentially every point"
    assert all(abs(v - 7999.997) < 0.5 for v in ex), (min(ex), max(ex))
    # constant drive -> flat line IS the correct result; nothing may manufacture variation
    assert max(ex) - min(ex) < 0.1


# ---------------- J activates only on width AND thickness ----------------

def test_j_absent_without_width_and_capability_says_why():
    r = _run(width_mm=None)
    assert all(p["current_density_J"] is None for p in r.data["points"])
    cap = _cap(r, "current_density")
    assert cap["applicable"] is False
    assert "width" in cap["reason"].lower()


def test_j_computed_with_width_and_thickness():
    r = _run(width_mm=2.0, thickness_mm=0.07)
    pts = [p for p in r.data["points"] if p["current_density_J"] is not None]
    assert len(pts) >= 100
    # J[A/m^2] = I[uA] / (w[mm] * t[mm]); uA/mm^2 == A/m^2 exactly
    expect = 7999.997 / (2.0 * 0.07)
    for p in pts:
        assert p["current_density_J"] == pytest.approx(
            p["excitation_uA"] / (2.0 * 0.07), rel=1e-12)
    assert np.median([p["current_density_J"] for p in pts]) == pytest.approx(expect, rel=1e-4)
    cap = _cap(r, "current_density")
    assert cap["applicable"] is True


def test_capability_carries_the_measured_vs_requested_caveat():
    # the column is what the instrument REPORTS, which need not equal the requested drive
    cap = _cap(_run(width_mm=2.0), "current_density")
    assert "requested" in cap["reason"].lower()


def test_j_absent_without_thickness_even_with_width():
    r = _run(width_mm=2.0, thickness_mm=None)
    assert all(p["current_density_J"] is None for p in r.data["points"])


# ---------------- CSV: the columns exist and are named ----------------

def test_hall_tdep_export_carries_excitation_and_j(tmp_path):
    import csv
    from cryosweep_core.io.export import export_result
    r = _run(width_mm=2.0)
    paths = export_result(r, tmp_path / "td")
    with open(paths["points"]) as f:
        rows = list(csv.DictReader(f))
    assert rows, "the hall_tdep points CSV must not be an empty shell"
    assert "excitation (uA)" in rows[0]
    assert "current_density_J (A/m^2)" in rows[0]
    assert "R_H (m^3/C)" in rows[0]
    vals = [float(x["excitation (uA)"]) for x in rows if x["excitation (uA)"]]
    assert vals and all(abs(v - 7999.997) < 0.5 for v in vals)


# ---------------- plot gates ----------------

def test_summary_series_include_j_only_when_computed():
    from cryosweep_core.plotting.catalog import BUILTIN_PLOTKINDS
    kind = next(k for k in BUILTIN_PLOTKINDS if k.key == "hall_tdep_summary")
    keys_without = {s.key for s in kind.series(_run())}
    keys_with = {s.key for s in kind.series(_run(width_mm=2.0))}
    assert "j" not in keys_without
    assert "j" in keys_with


def test_summary_renders_three_axes_with_j_two_without():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cryosweep_core.plotting.render import render_kind
    from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle
    fig2 = render_kind([_run()], "hall_tdep_summary", PlotSpec(), GlobalStyle())
    fig3 = render_kind([_run(width_mm=2.0)], "hall_tdep_summary", PlotSpec(), GlobalStyle())
    assert len(fig3.axes) == len(fig2.axes) + 1, (len(fig2.axes), len(fig3.axes))
    plt.close(fig2); plt.close(fig3)


def test_j_t_kind_renders_the_flat_line_honestly():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cryosweep_core.plotting.render import render_kind
    from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle
    fig = render_kind([_run(width_mm=2.0)], "hall_tdep_J_T", PlotSpec(), GlobalStyle())
    ys = np.concatenate([l.get_ydata() for ax in fig.axes for l in ax.lines
                         if l.get_gid() not in ("refline", "fit")])
    ys = ys[np.isfinite(ys)]
    assert len(ys) >= 100
    assert (ys.max() - ys.min()) / ys.mean() < 1e-4       # flat, as measured — no variation
    plt.close(fig)


# ---------------- AHE: deferred on data grounds, reported as such ----------------

def test_anomalous_hall_is_a_deferred_capability_on_the_hall_probe():
    cfg = RunConfig.load(probe_override="hall")
    cfg.hall.hall_channel = 1
    cfg.hall.thickness_mm = 0.07
    cfg.hall.longitudinal_channel = 2
    r = analyze_file(load_dat(str(EXAMPLE)), cfg, _REG)
    cap = next(c for c in r.data["capabilities"] if c["name"] == "anomalous_hall")
    assert cap["applicable"] is False
    assert "m(h)" in cap["reason"].lower() or "magnetization" in cap["reason"].lower()
