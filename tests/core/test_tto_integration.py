"""End-to-end TTO integration: real-file dispatch, the D9 --probe resistivity pin, and the
gallery manifest entries. Every number here is from the spec's measured oracle band."""
import matplotlib; matplotlib.use("Agg")       # noqa: E702  (module level, as the sibling
# plot tests do: calling matplotlib.use() *inside* a test fails once another module has
# already switched the backend)
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.config import RunConfig
from cryosweep_core.detect.probe import detect_probe
from cryosweep_core.io.columns import canonicalize_columns
from cryosweep_core.io.loader import load_dat
from cryosweep_core.registry import build_default_registry

ROOT = pathlib.Path(__file__).resolve().parents[2]
from tests.core.conftest import repo_root

REPO = repo_root()   # the repo root (docs/, skill/ and the real data live there, not in the app folder)

TTO_KINDS = ("tto_summary_t", "tto_kappa_t", "tto_seebeck_t", "tto_zt_t", "tto_wf_t")


def _analyze(path):
    rt = load_dat(str(path))
    return analyze_file(rt, RunConfig(), build_default_registry())


def test_auto_detection_routes_the_real_file_to_tto(tto_real_path):
    rt = load_dat(str(tto_real_path))
    df, _ = canonicalize_columns(rt.df, rt.header)
    score, key = detect_probe(rt.header, set(df.columns), build_default_registry())
    assert (score, key) == (1.0, "tto")


def test_end_to_end_dispatch_shape(tto_real_path):
    r = _analyze(tto_real_path)
    assert r.status == "ok" and r.data["probe"] == "tto"
    assert len(r.data["curves"]) == 1
    c = r.data["curves"][0]
    assert c["n_points"] == 976 and c["direction"] == "down"
    assert r.data["dropped_groups"] == []
    assert r.data["n_error_rows"] == 6
    assert any("6 rows carry instrument error codes (kept)" == w for w in r.warnings)


def test_end_to_end_oracle_numbers(tto_real_path):
    d = _analyze(tto_real_path).data
    assert 1.45 <= d["rrr"]["rrr"] <= 1.46         # NOT the 1.476 raw-extrema ratio
    assert d["rrr"]["classification"] == "metallic"
    # L/L0 at T_high. Index the lorenz array POSITIONALLY against `t` (they are parallel and
    # None-holed together); filtering the list first and then indexing it with an index into
    # the UNfiltered `t` is only correct while the real file happens to have no holes.
    t = d["curves"][0]["t"]
    lorenz = d["curves"][0]["lorenz_ratio"]
    assert len(lorenz) == len(t)
    lr_at_thigh = lorenz[int(np.argmax(t))]
    assert lr_at_thigh is not None
    assert 2.0 <= lr_at_thigh <= 2.2
    assert 3.92e-4 <= d["summary"]["zt_peak"] <= 3.93e-4
    assert d["summary"]["pf_at_thigh"] == pytest.approx(4.645e-6, rel=1e-3)
    assert min(v for v in d["curves"][0]["kappa_ph"] if v is not None) > 0.0


def test_capabilities_on_the_real_file(tto_real_path):
    caps = {c["name"]: c["applicable"] for c in _analyze(tto_real_path).data["capabilities"]}
    for name in ("thermal_conductivity", "seebeck", "wiedemann_franz", "power_factor",
                 "figure_of_merit", "rrr"):
        assert caps[name] is True, name
    for name in ("callaway_fit", "boundary_scattering_fit", "diffusive_seebeck",
                 "kappa_field_sweep"):
        assert caps[name] is False, name


def test_determinism_and_json_validity_on_the_real_file(tto_real_path):
    a = json.dumps(_analyze(tto_real_path).data, sort_keys=True, allow_nan=False)
    b = json.dumps(_analyze(tto_real_path).data, sort_keys=True, allow_nan=False)
    assert a == b


def test_every_tto_kind_backs_the_real_file(tto_real_path):
    from cryosweep_core.plotting.catalog import BUILTIN_PLOTKINDS
    kinds = {k.key: k for k in BUILTIN_PLOTKINDS}
    r = _analyze(tto_real_path)
    for key in TTO_KINDS:
        assert kinds[key].series(r), key         # called with NO field_unit, as pq_compare does


def test_kappa_legend_stays_clear_of_the_lowt_inset_at_gui_font_size(tto_real_path):
    """Closing visual gate (font_pt=14, the GUI's on-screen size): the lone 'cooling' legend
    on `tto_kappa_t` is placed inside with loc='best', which is blind to the low-T inset
    (a separate Axes). At 14 pt 'best' put it exactly under the inset — invisible. The
    gallery renders at 9 pt, where it lands top-left, so `pq_compare` cannot see this."""
    import matplotlib.pyplot as plt
    from cryosweep_core.plotting.render import render_kind
    from cryosweep_core.plotting.spec import GlobalStyle, PlotSpec
    r = _analyze(tto_real_path)
    for pt in (9, 14):
        fig = render_kind(r, "tto_kappa_t", PlotSpec(), GlobalStyle(font_pt=pt))
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        assert len(fig.axes) == 2, "host + low-T inset expected"
        leg = fig.axes[0].get_legend()
        assert leg is not None
        assert not leg.get_window_extent(rend).overlaps(
            fig.axes[1].get_window_extent(rend)), f"legend hidden under inset at {pt} pt"
        plt.close(fig)


def test_explicit_probe_resistivity_on_a_tto_file_still_works(tto_real_path):
    # D9 pin: adding "sample temp. (k)" to _TEMP made ResistivityAnalyzer viable on TTO files.
    # Auto-detection always routes them to `tto`; this explicit view is a legitimate,
    # differently-conventioned reading and must stay available. Its RRR may differ from tto's.
    #
    # PINNED AT THE REAL SEAM, deliberately: the --probe flag travels
    # RunConfig.probe_override -> registry.get_analyzer(override) (dispatch.py:25-30).
    # Calling ResistivityAnalyzer().analyze(...) directly would bypass detection, the registry
    # and the flag entirely, so a regression in the override plumbing — or a change that let
    # auto-detection win over an explicit --probe — would sail straight past the assertion.
    # (The RunConfig field name is `probe_override`; dispatch reads it via
    # getattr(cfg, "probe_override", None). Do not guess a different spelling.)
    cfg = RunConfig(probe_override="resistivity")
    r = analyze_file(load_dat(str(tto_real_path)), cfg, build_default_registry())
    assert r.status == "ok"
    assert r.data["probe"] == "resistivity"
    # ...and the default (no override) still lands on tto, so the override is a real choice
    assert analyze_file(load_dat(str(tto_real_path)), RunConfig(),
                        build_default_registry()).data["probe"] == "tto"


def test_cli_analyze_reports_tto(tto_real_path):
    out = subprocess.run([sys.executable, "-m", "cryosweep_cli", "analyze", str(tto_real_path)],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    payload = json.loads(out.stdout)
    assert payload["data"]["probe"] == "tto"
    assert payload["data"]["rrr"]["classification"] == "metallic"


def test_manifest_has_an_entry_for_every_tto_kind(tto_real_path):
    entries = json.loads((REPO / "docs/superpowers/pq-reference-gallery/manifest.json").read_text())
    by_id = {e["id"]: e for e in entries}
    rel = str(tto_real_path.relative_to(REPO))
    for key in TTO_KINDS:
        e = by_id[key]
        assert e["probe"] == "tto"
        assert e["v2_kind"] == key
        assert e["dat"] == rel


# ---- integrity slice (2026-08-03): every number measured on the real gate file -------------

def test_real_file_kappa_ph_fit_oracle(tto_real_path):
    kf = _analyze(tto_real_path).data["kappa_ph_fit"]
    assert kf is not None
    assert kf["n"] == pytest.approx(2.0266, rel=1e-3)
    assert kf["n_sigma"] == pytest.approx(0.0062, rel=1e-3)     # 4 s.f.; measured 0.0062041
    assert kf["b"] == pytest.approx(3.5341e-3, rel=1e-3)
    assert kf["r2"] == pytest.approx(0.99927, rel=1e-4)
    assert kf["n_points"] == 163
    assert kf["window_k"] == [pytest.approx(2.025, abs=1e-3), 10.0]
    assert kf["n_spread"] == pytest.approx(0.7121, rel=1e-3)
    assert kf["n_loglog"] == pytest.approx(2.0078, rel=1e-3)
    assert kf["n_method_delta"] == pytest.approx(0.018781, rel=1e-3)   # measured 0.0187807
    assert len(kf["ladder"]) == 5
    cf = [e for e in kf["ladder"] if e["method"] == "curve_fit"]
    assert [e["cutoff_k"] for e in cf] == [10.0, 15.0, 20.0, 30.0]
    assert cf[0]["n"] == pytest.approx(2.0266, rel=1e-3)
    assert cf[-1]["n"] == pytest.approx(1.3145, rel=1e-3)


def test_real_file_fit_declares_itself_window_sensitive(tto_real_path):
    # E4/I8: no real kappa(T) is one power law from 10 to 30 K, so on real files this flag is
    # EXPECTED to fire. Its ABSENCE is the informative case (the synth fixture provides that).
    kf = _analyze(tto_real_path).data["kappa_ph_fit"]
    assert "window_sensitive" in kf["quality_flags"]
    assert "ladder_incomplete" not in kf["quality_flags"]
    assert kf["n_spread"] > 3 * kf["n_sigma"]
    assert kf["n_spread"] > 0.05
    # the honest reading: the window moves n ~115x more than the statistical sigma does
    assert kf["n_spread"] / kf["n_sigma"] > 50


def test_real_file_kappa_ph_capability_is_applicable(tto_real_path):
    caps = {c["name"]: c for c in _analyze(tto_real_path).data["capabilities"]}
    assert caps["kappa_ph_power_fit"]["applicable"] is True


def test_real_file_integrity_warnings_all_fire_with_their_measured_text(tto_real_path):
    w = _analyze(tto_real_path).warnings
    assert ("20 rows have ΔT/T > 5% (max 11.72% at 2.025 K) — "
            "kappa there is averaged over a wide T window") in w
    assert ("S changes sign 11 times between 10.186 K and 11.910 K (a 1.725 K window) — "
            "the low-T sign structure oscillates from point to point") in w
    assert not any("classification_uncertain" in x for x in w)


def test_the_sign_oscillation_bound_is_twenty_kelvin_everywhere_in_the_slice():
    # The 11 crossings on this file are the only ones below 12 K, so a 12 K bound would pass
    # the oracle above and the disagreement would stay SILENT. Pinned at source level.
    import inspect
    import cryosweep_core.analyzers.tto as T
    assert T._S_OSC_MAX_T_K == 20.0
    assert "12" not in inspect.getsource(T._seebeck_oscillation_warning)


def test_real_file_rrr_uncertainty_and_zt_peak_uncertainty_oracles(tto_real_path):
    d = _analyze(tto_real_path).data
    assert d["rrr"]["rrr_std"] == pytest.approx(0.01742, rel=1e-3)
    assert d["rrr"]["classification"] == "metallic"
    assert d["summary"]["zt_peak_std"] == pytest.approx(1.59828e-5, rel=1e-4)
    # 4.07 % relative -- the reason the ZT peak needed an uncertainty at all
    assert d["summary"]["zt_peak_std"] / d["summary"]["zt_peak"] == pytest.approx(0.0407,
                                                                                  rel=1e-2)


def test_real_file_envelope_stays_json_safe_and_deterministic(tto_real_path):
    a = json.dumps(_analyze(tto_real_path).data, sort_keys=True, allow_nan=False)
    b = json.dumps(_analyze(tto_real_path).data, sort_keys=True, allow_nan=False)
    assert a == b


def test_all_six_tto_kinds_render_on_the_real_file_with_and_without_bands(tto_real_path):
    from cryosweep_core.plotting.render import render_kind
    from cryosweep_core.plotting.spec import GlobalStyle, PlotSpec
    r = _analyze(tto_real_path)
    for kind in TTO_KINDS + ("tto_lorenz_t",):
        for spec in (PlotSpec(), PlotSpec(error_band=True)):
            assert render_kind(r, kind, spec, GlobalStyle()).axes


def test_real_file_summary_csv_carries_the_twenty_columns(tmp_path, tto_real_path):
    from cryosweep_core.io.export import export_result
    import csv as _csv
    out = export_result(_analyze(tto_real_path), tmp_path / "real")
    with open(out["tto_summary"], newline="") as f:
        rows = list(_csv.DictReader(f))
    assert len(rows[0]) == 20
    assert float(rows[0]["kappa_ph_n"]) == pytest.approx(2.0266, rel=1e-3)
    assert float(rows[0]["rrr_std"]) == pytest.approx(0.01742, rel=1e-3)
    assert "window_sensitive" in rows[0]["kappa_ph_flags"]
