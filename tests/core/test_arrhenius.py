"""Activated transport — the Arrhenius fit, and the factor-of-two trap it refuses.

An Arrhenius fit (OLS of ln rho vs 1/T) measures an ACTIVATION ENERGY E_a. For an
intrinsic semiconductor rho ~ exp(+E_g/2k_BT), so the gap is E_g = 2*E_a — but for
extrinsic conduction the activation is a donor/acceptor level and the factor is 1 (or
1/2 under compensation), and TRANSPORT DATA ALONE CANNOT TELL THE REGIMES APART. The fit
therefore reports E_a as measured (meV, with sigma and a window-ladder spread) and never
silently converts to a gap: the only gap field is named `e_g_assuming_intrinsic_mev`, so
the assumption travels in the name.

Ground truth and controls (no shipped example was insulating before this — measured, the
whole corpus held ONE insulating channel, a bad metal the fit must DECLINE on):
  rho_semi_synth.dat  planted E_a = 60 meV     -> must recover it
  rho_vrh_synth.dat   Mott VRH (T0 = 1e6 K)    -> Arrhenius must flag window_sensitive
  rho_weak_synth.dat  ~25% linear fall         -> insulating but 0.29 e-folds: DECLINE
"""
import numpy as np
import pathlib
import pytest

from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.registry import build_default_registry
from cryosweep_core.analyzers.dispatch import analyze_file

FIX = pathlib.Path(__file__).parent / "fixtures"
_REG = build_default_registry()
_CACHE = {}


def _bridge1(fname):
    if fname not in _CACHE:
        _CACHE[fname] = analyze_file(load_dat(str(FIX / fname)),
                                     RunConfig.load(probe_override="resistivity"), _REG)
    return _CACHE[fname].data["bridges"][0]


def _cap(fname, name):
    return next(c for c in _CACHE[fname].data["capabilities"] if c["name"] == name)


# ---------------- ground truth: the planted E_a is recovered ----------------

def test_planted_ea_recovered_in_mev():
    b = _bridge1("rho_semi_synth.dat")
    fit = b["arrhenius"]
    assert fit is not None
    ea = fit["params"]["e_a_mev"]
    assert ea == pytest.approx(60.0, rel=1e-6)          # exact synthetic data
    assert fit["params"]["e_g_assuming_intrinsic_mev"] == pytest.approx(120.0, rel=1e-6)
    assert fit["r2"] > 0.999999
    assert not fit["quality_flags"]


def test_no_silent_gap_anywhere():
    # the ONLY gap-like key permitted anywhere in the payload is the explicitly-named one
    import json
    b = _bridge1("rho_semi_synth.dat")
    keys = set()
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                keys.add(k); walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(b)
    gapish = {k for k in keys if "e_g" in k.lower() or "gap" in k.lower()}
    assert gapish <= {"e_g_assuming_intrinsic_mev"}, gapish


def test_ladder_and_spread_reported():
    b = _bridge1("rho_semi_synth.dat")
    ladder = b["arrhenius_ladder"]
    assert ladder and len(ladder) >= 3
    spread = b["arrhenius_ea_spread_mev"]
    assert spread is not None and spread < 0.01          # exact data: numerically flat
    cap = _cap("rho_semi_synth.dat", "activated_transport")
    assert cap["applicable"] is True
    assert "2" in cap["reason"] and "intrinsic" in cap["reason"].lower()


def test_metallic_channel_gets_no_arrhenius():
    r = _CACHE["rho_semi_synth.dat"]
    b2 = r.data["bridges"][1]
    assert b2["classification"] == "metallic"
    assert b2.get("arrhenius") is None


# ---------------- the VRH impostor flags itself ----------------

def test_vrh_sample_is_window_sensitive():
    b = _bridge1("rho_vrh_synth.dat")
    fit = b["arrhenius"]
    assert fit is not None, "VRH data still fits SOME Arrhenius slope — it must report one"
    assert "window_sensitive" in fit["quality_flags"]
    # the drift is the physics: E_a moves substantially across rungs on non-Arrhenius data
    assert b["arrhenius_ea_spread_mev"] > 3 * (fit["sigma"].get("e_a_mev") or 0.0)


def test_alt_models_reported_with_the_note_r2_cannot_choose():
    b = _bridge1("rho_vrh_synth.dat")
    alt = b["arrhenius_alt_models"]
    assert {m["model"] for m in alt["models"]} == {"arrhenius", "mott_vrh_3d", "efros_shklovskii"}
    assert all(0 <= m["r2"] <= 1 for m in alt["models"])
    assert "cannot" in alt["note"].lower()               # r² does not select a mechanism


# ---------------- decline discipline ----------------

def test_weak_insulator_declines_on_rho_span():
    b = _bridge1("rho_weak_synth.dat")
    assert b["classification"] == "insulating"
    fit = b["arrhenius"]
    assert fit is not None                               # the attempt is reported…
    assert "insufficient_rho_span" in fit["quality_flags"]   # …but flagged as non-measurement
    cap = _cap("rho_weak_synth.dat", "activated_transport")
    assert cap["applicable"] is False
    assert "e-fold" in cap["reason"] or "span" in cap["reason"]


def test_declined_fit_leaves_blank_csv_cells(tmp_path):
    import csv
    from cryosweep_core.io.export import export_result
    paths = export_result(_CACHE["rho_weak_synth.dat"], tmp_path / "weak")
    with open(paths["derived"]) as f:
        rows = list(csv.DictReader(f))
    r1 = rows[0]
    assert r1["arrhenius_ea_mev"] == ""                 # blank, not a number
    assert r1["e_g_assuming_intrinsic_mev"] == ""
    assert "insufficient_rho_span" in r1["arrhenius_flags"]


def test_fitted_csv_carries_ea_and_named_gap(tmp_path):
    import csv
    from cryosweep_core.io.export import export_result
    paths = export_result(_CACHE["rho_semi_synth.dat"], tmp_path / "semi")
    with open(paths["derived"]) as f:
        rows = list(csv.DictReader(f))
    r1 = rows[0]
    assert float(r1["arrhenius_ea_mev"]) == pytest.approx(60.0, rel=1e-6)
    assert float(r1["e_g_assuming_intrinsic_mev"]) == pytest.approx(120.0, rel=1e-6)


# ---------------- the real corpus channel: the honest answer is a decline ----------------

def test_real_insulating_channel_declines():
    from tests.core.conftest import require_real
    p = require_real("act")
    r = analyze_file(load_dat(str(p)), RunConfig.load(probe_override="resistivity"), _REG)
    b1 = r.data["bridges"][0]
    assert b1["classification"] == "insulating"
    fit = b1["arrhenius"]
    assert fit is not None
    # measured 2026-09-05: rho changes 1.3x (0.28 e-folds), full-window r2 = 0.10, E_a
    # drifts 0.054 -> 7.96 meV across rungs — a bad metal, not activated conduction
    assert "insufficient_rho_span" in fit["quality_flags"]


# ---------------- the figure ----------------

def test_arrhenius_plot_kind_renders_fit_and_annotation():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cryosweep_core.plotting.render import render_kind
    from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle
    fig = render_kind([_CACHE["rho_semi_synth.dat"]], "resistivity_arrhenius",
                      PlotSpec(), GlobalStyle())
    ax = fig.axes[0]
    assert "1000/T" in ax.get_xlabel()
    assert any(l.get_gid() == "fit" for l in ax.lines)
    txt = " ".join(t.get_text() for t in ax.texts)
    assert "60.0" in txt and "meV" in txt
    assert "intrinsic" in txt                            # the assumption is ON the figure
    plt.close(fig)


def test_arrhenius_plot_declines_with_note_on_weak_insulator():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cryosweep_core.plotting.render import render_kind
    from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle
    fig = render_kind([_CACHE["rho_weak_synth.dat"]], "resistivity_arrhenius",
                      PlotSpec(), GlobalStyle())
    ax = fig.axes[0]
    assert not any(l.get_gid() == "fit" for l in ax.lines)   # no fit line for a non-measurement
    txt = " ".join(t.get_text() for t in ax.texts)
    assert "declined" in txt.lower()
    plt.close(fig)


def test_spread_floor_is_not_finely_tuned():
    """The 1 meV window_sensitive floor is deliberately loose (the 1/chi-guard rule): the
    verdict on all three fixtures is identical for any floor from 0.5 to 2 meV. If this
    ever fails, the noise/physics separation has narrowed and the rule needs rethinking,
    not retuning."""
    import cryosweep_core.fitting.transport as tr
    import numpy as np

    def verdicts(floor):
        old = tr.ARRHENIUS_SPREAD_FLOOR_MEV
        tr.ARRHENIUS_SPREAD_FLOOR_MEV = floor
        try:
            out = {}
            for fname in ("rho_semi_synth.dat", "rho_vrh_synth.dat", "rho_weak_synth.dat"):
                r = analyze_file(load_dat(str(FIX / fname)),
                                 RunConfig.load(probe_override="resistivity"), _REG)
                fit = r.data["bridges"][0]["arrhenius"]
                out[fname] = "window_sensitive" in (fit["quality_flags"] or [])
            return out
        finally:
            tr.ARRHENIUS_SPREAD_FLOOR_MEV = old

    v1 = verdicts(0.5)
    assert verdicts(1.0) == v1
    assert verdicts(2.0) == v1
    assert v1["rho_vrh_synth.dat"] is True          # the impostor flags at every floor
    assert v1["rho_semi_synth.dat"] is False        # exact data never flags
