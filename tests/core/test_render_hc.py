import copy
import math
import pathlib
import pytest
from tests.core.conftest import real_data
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry
from cryosweep_core.plotting.spec import PlotSpec
from cryosweep_core.plotting.render import render_for, render_kind

FIX = pathlib.Path(__file__).parent / "fixtures"

def _hc():
    return analyze_file(load_dat(str(FIX / "hc_synth.dat")),
                        RunConfig.load(probe_override="heatcapacity"), build_default_registry())

def test_render_for_hc_cp_over_t():
    fig = render_for(_hc(), PlotSpec())
    ax = fig.axes[0]
    assert ax.get_xlabel() == "T² (K²)" and "Cp/T" in ax.get_ylabel()
    assert len(ax.lines) >= 1

def test_hc_overlays_all_fitted_models():
    res = _hc()
    n_ok = sum(1 for f in res.data["lowt_fits"] if f["ok"])
    on = render_kind(res, "cp_over_t", PlotSpec(fit_line=True)).axes[0].lines
    off = render_kind(res, "cp_over_t", PlotSpec(fit_line=False)).axes[0].lines
    assert n_ok >= 1
    assert len(on) == len(off) + n_ok                      # one fit line per fitted model
    labels = [l.get_label() for l in on]
    assert any("R²" in str(s) for s in labels)             # fit labels carry R^2


def test_hc_fit_lines_subset_toggle():
    res = _hc()
    one = render_kind(res, "cp_over_t",
                      PlotSpec(fit_line=True, fit_lines=("debye_t3",))).axes[0].lines
    off = render_kind(res, "cp_over_t", PlotSpec(fit_line=False)).axes[0].lines
    assert len(one) == len(off) + 1                        # only debye_t3 drawn


# ---- PQ-5 Task 4: hc_entropy_vs_t --------------------------------------------
HC_RUNCONFIG = {"probe_override": "heatcapacity"}   # copied verbatim from manifest.json


def _hc_real(path):
    return analyze_file(load_dat(path),
                        RunConfig.load(**HC_RUNCONFIG), build_default_registry())


def test_entropy_kind_renders(hc_path):
    res = _hc_real(hc_path)
    assert res.data.get("entropy_available")
    fig = render_kind(res, "hc_entropy_vs_t")
    ax = fig.axes[0]
    assert "S" in ax.get_ylabel()                          # has an S(T) label
    # total-entropy line has data
    assert any(line.get_ydata().size for line in ax.get_lines())


def test_entropy_magnetic_series_has_no_none(hc_path):
    from cryosweep_core.plotting.catalog import get_kind
    res = _hc_real(hc_path)
    series = get_kind("hc_entropy_vs_t").series(res)
    mag = [s for s in series if s.role == "magnetic"]
    if res.data.get("entropy_magnetic") is not None:
        assert mag, "magnetic series expected when entropy_magnetic present"
        for s in mag:
            assert all(v is not None for v in s.y)         # None entries dropped
            assert len(s.x) == len(s.y)


def _reflines(fig):
    return [ln for axx in fig.axes for ln in axx.get_lines() if ln.get_gid() == "refline"]


def _hc_meaningful_magnetic(path):
    """The real heat-capacity file with a SYNTHESIZED magnetic S(T) rising to ~0.5*Rln, so the
    meaningful-magnetic gate (max finite magnetic > 0.05*Rln) fires -> twin-axis path."""
    res = _hc_real(path)
    d = res.data
    T = d["entropy_temperature"]
    rln = float(d["entropy_rln_suggestion"]["value"])
    n = len(T)
    d["entropy_magnetic"] = [0.5 * rln * (i / (n - 1)) for i in range(n)]
    return res


def test_entropy_twin_when_magnetic_meaningful(hc_path):
    # meaningful magnetic -> TWO y-axes (twin), magnetic series on the right axis, Rln refline drawn
    res = _hc_meaningful_magnetic(hc_path)
    fig = render_kind(res, "hc_entropy_vs_t")
    assert len(fig.axes) == 2, "expected a twinned (2-axis) figure for meaningful magnetic S"
    host, tax = fig.axes[0], fig.axes[1]
    # magnetic (dashed) line lives on the twin (right) axis, not the host
    dashed = [ln for ln in tax.get_lines()
              if ln.get_gid() != "refline" and ln.get_linestyle() in ("--", "dashed")]
    assert dashed, "magnetic series expected on the right (twin) axis"
    assert _reflines(fig), "Rln reference line expected on the twin axis"
    # and the total (solid) series is on the host axis
    assert any(ln.get_ydata().size for ln in host.get_lines()
               if ln.get_gid() != "refline")


def test_entropy_full_curve_visible_not_clipped(hc_path):
    # robust view clips a cumulative monotone S_total; the fix shows the full y-range so the
    # entire headline curve (~523 J/mol·K here) is inside the frame.
    res = _hc_real(hc_path)
    fig = render_kind(res, "hc_entropy_vs_t")
    ax = fig.axes[0]
    smax = max(float(v) for v in res.data["entropy_total"])
    assert ax.get_ylim()[1] >= smax, (ax.get_ylim(), smax)


def test_entropy_annotation_reports_total_not_negative_rln(hc_path):
    # annotation headlines the TOTAL saturation; on this fixture S_magnetic ~ 0 so it must NOT
    # print a misleading negative "% R ln2".
    res = _hc_real(hc_path)
    fig = render_kind(res, "hc_entropy_vs_t")
    ax = fig.axes[0]
    texts = [t.get_text() for t in ax.texts]
    ann = next((t for t in texts if "S_total" in t), None)
    assert ann is not None, texts
    total_int = str(int(float(res.data["entropy_total"][-1])))
    assert total_int in ann, (ann, total_int)
    # no negative "% R ln2" (or "% Rln") should appear anywhere in the annotation text
    for t in texts:
        assert "-" not in t or "% R" not in t, t


# DEFECT 1 (task-9 vizfix): the auto-suggested Rln(2J+1) reference line must be drawn ONLY when
# it is on-scale. On a tiny-entropy fixture (S_total ~ 0.3 J/mol·K, no magnetic entropy) the Rln
# value (~5.76) sits ~18x above the axes top; drawing a line + text label there pushed
# savefig(bbox_inches="tight") into a broken ~600x6600 vertical sliver.
def _hc_tiny_entropy(path):
    return analyze_file(load_dat(path),
                        RunConfig.load(**HC_RUNCONFIG), build_default_registry())


def test_entropy_refline_skipped_when_offscale(hc_lowmass_path):
    res = _hc_tiny_entropy(hc_lowmass_path)
    d = res.data
    # precondition: rln suggestion present but far above the total-entropy range; no magnetic S
    sug = d.get("entropy_rln_suggestion")
    assert sug and sug.get("value") is not None
    smax = max(float(v) for v in d["entropy_total"])
    assert sug["value"] > 5 * smax                       # genuinely off-scale
    assert d.get("entropy_magnetic") is None
    fig = render_kind(res, "hc_entropy_vs_t")
    ax = fig.axes[0]
    reflines = [ln for ln in ax.get_lines() if ln.get_gid() == "refline"]
    assert not reflines, "off-scale Rln reference line must be skipped"
    # no Rln label text should sit above the axes top either
    assert all(t.get_position()[1] <= ax.get_ylim()[1] for t in ax.texts
               if t.get_transform() == ax.get_yaxis_transform())


def test_entropy_offscale_case_saves_with_sane_aspect(tmp_path, hc_lowmass_path):
    # locks the bbox_inches="tight" blowup: with the line/label gated, height/width stays sane.
    res = _hc_tiny_entropy(hc_lowmass_path)
    fig = render_kind(res, "hc_entropy_vs_t")
    out = tmp_path / "hcn_entropy.png"
    fig.savefig(str(out), bbox_inches="tight", facecolor="white")
    from PIL import Image
    w, h = Image.open(str(out)).size
    assert h / w < 2.0, (w, h)                            # was ~619x6685 before the fix


def test_entropy_single_axis_when_magnetic_trivial(tmp_path, hc_path):
    # The real heat-capacity file: S_total ~ 523 but magnetic S ~ 0 (lattice-dominated) -> NON-meaningful
    # magnetic. Fall back to single-axis total-only: NO twin, NO Rln line, and a sane bbox.
    res = _hc_real(hc_path)
    d = res.data
    mag = d.get("entropy_magnetic")
    mfin = [float(v) for v in mag if v is not None and math.isfinite(float(v))] if mag is not None else []
    rln = float(d["entropy_rln_suggestion"]["value"])
    assert not (mfin and max(mfin) > 0.05 * rln)          # precondition: trivial magnetic
    fig = render_kind(res, "hc_entropy_vs_t")
    assert len(fig.axes) == 1, "non-meaningful magnetic must stay single-axis"
    assert not _reflines(fig), "no Rln line in the single-axis fallback"
    # total curve visible + sane saved dimensions under bbox_inches='tight'
    smax = max(float(v) for v in d["entropy_total"])
    assert fig.axes[0].get_ylim()[1] >= smax
    out = tmp_path / "hc_single.png"
    fig.savefig(str(out), bbox_inches="tight", facecolor="white")
    from PIL import Image
    w, h = Image.open(str(out)).size
    assert 0.4 < h / w < 2.0, (w, h)


def test_entropy_hc_n_none_magnetic_single_axis_sane(tmp_path, hc_lowmass_path):
    # HC_N: entropy_magnetic is None -> single axis, no refline, no bbox blowup.
    res = _hc_tiny_entropy(hc_lowmass_path)
    assert res.data.get("entropy_magnetic") is None
    fig = render_kind(res, "hc_entropy_vs_t")
    assert len(fig.axes) == 1
    assert not _reflines(fig)
    out = tmp_path / "hcn_none.png"
    fig.savefig(str(out), bbox_inches="tight", facecolor="white")
    from PIL import Image
    w, h = Image.open(str(out)).size
    assert 0.4 < h / w < 2.0, (w, h)


# ---- PQ-5 Task 5: hc_full_cp_t (journal Cp(T) + Dulong–Petit + low-T inset) ---------
def test_full_cp_t_renders_with_dp_and_overlay(hc_path):
    res = _hc_real(hc_path)
    fig = render_kind(res, "hc_full_cp_t")
    ax = fig.axes[0]
    reflines = [ln for ln in ax.get_lines() if ln.get_gid() == "refline"]
    d = res.data
    from cryosweep_core.fitting.entropy import dulong_petit_limit
    if d["n_atoms_available"]:
        assert reflines, "Dulong-Petit line expected when n_atoms known"
        assert dulong_petit_limit(d["n_atoms"]) is not None
    else:
        assert not reflines, "no DP line when n_atoms unknown"
    # solid-red DE fit line drawn when full_fit ok
    if (d.get("full_fit") or {}).get("ok"):
        assert any(ln.get_gid() == "fit" for ln in ax.get_lines()), "DE fit overlay expected"
    # low-T inset axis present (>= 2 axes on the figure)
    assert len(fig.axes) >= 2


# ---- PQ-5 Task 6: cp_over_t + hc_c_over_t_linear annotation + fit-window shade ------
def _has_shade(ax):
    return bool(ax.patches) or any(
        getattr(c, "get_alpha", lambda: None)() for c in ax.collections)


def test_cp_over_t_has_annotation_and_shade(hc_path):
    res = _hc_real(hc_path)
    fig = render_kind(res, "cp_over_t")
    ax = fig.axes[0]
    txt = " ".join(t.get_text() for t in ax.texts)
    assert ("γ" in txt) or ("gamma" in txt.lower())
    assert ("θ" in txt) or ("theta" in txt.lower())
    # chosen model on this fixture is a lattice model -> theta_D is a finite number
    theta_D = (res.data["fit"]["params"] or {}).get("theta_D")
    if theta_D is not None and math.isfinite(theta_D):
        assert "n/a" not in txt, txt
    assert _has_shade(ax), "fit-window shade expected"


def test_hc_c_over_t_linear_has_annotation_and_shade(hc_path):
    res = _hc_real(hc_path)
    fig = render_kind(res, "hc_c_over_t_linear")
    ax = fig.axes[0]
    txt = " ".join(t.get_text() for t in ax.texts)
    assert ("γ" in txt) or ("gamma" in txt.lower())
    assert ("θ" in txt) or ("theta" in txt.lower())
    assert _has_shade(ax), "fit-window shade expected"


def test_theta_d_shows_na_for_spin_fluct_model(hc_path):
    # spin-fluctuation beta is not a lattice property -> theta_D is undefined ("n/a").
    # The real heat-capacity file chooses a lattice model, so synthesize the spin-fluct branch.
    res = _hc_real(hc_path)
    syn = copy.deepcopy(res)
    syn.data["model"] = "spin_fluct_weak"
    syn.data["fit"]["params"]["theta_D"] = float("nan")
    for kind in ("cp_over_t", "hc_c_over_t_linear"):
        fig = render_kind(syn, kind)
        txt = " ".join(t.get_text() for t in fig.axes[0].texts)
        assert "n/a" in txt, (kind, txt)


def test_hc_full_cp_t_legend_clears_inset_at_large_font():
    """matplotlib's loc='best' is blind to inset_axes children (they are separate Axes
    overlaying the host), so on hc_full_cp_t the legend lands under the low-T inset at
    12pt+. Measured before the fix: 27.9% covered at 12pt, 62.3% at 16pt, with the
    '13 T' and 'Debye' rows unreadable. Invisible at the 9pt gallery baseline, which is
    why only a font sweep could see it.
    """
    if real_data("hc") is None:
        pytest.skip("local-only measurement file for key 'hc' is not available")
    import sys
    TOOLS = pathlib.Path(__file__).resolve().parents[2] / "tools"
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    import matplotlib.pyplot as plt
    import pq_compare
    from cryosweep_core.plotting.spec import GlobalStyle

    entry = [e for e in pq_compare._load_manifest() if e.get("id") == "hc_full_cp_t"][0]
    base = pq_compare.STYLE
    try:
        for pt in (12.0, 16.0):
            pq_compare.STYLE = base.model_copy(update={"font_pt": pt})
            fig, status = pq_compare._render_v2(entry)
            assert fig is not None, status
            fig.canvas.draw()
            fails = [f for f in pq_compare._check_fig(entry, fig, status) if "OCCLUSION" in f]
            plt.close(fig)
            assert not fails, (pt, fails)
    finally:
        pq_compare.STYLE = base
