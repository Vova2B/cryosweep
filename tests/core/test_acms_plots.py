import matplotlib; matplotlib.use("Agg")
import pathlib
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.acms import ACMSAnalyzer
from cryosweep_core.plotting.catalog import series_acms_chi_t
from cryosweep_core.plotting.render import render_kind

FX = pathlib.Path("tests/core/fixtures")


def _run(p):
    return ACMSAnalyzer().analyze(load_dat(p), RunConfig())


def test_series_split_prime_and_dprime(acms_real_path):
    r = _run(acms_real_path)
    s = series_acms_chi_t(r)
    assert any(x.key.startswith("chip:") for x in s)
    assert any(x.key.startswith("chipp:") for x in s)


def test_legend_label_full_precision_frequency(acms_real_path):
    # frequency at FULL precision (477, never 2sf-rounded 480); amplitude 2sf (0.0498 -> 0.05)
    s = series_acms_chi_t(_run(acms_real_path))
    assert s[0].label == "477 Hz, 0.05 Oe"


def test_headline_renders_two_panels_on_real_file(acms_real_path):
    fig = render_kind(_run(acms_real_path), "acms_chi_t")
    assert len(fig.axes) >= 2


def test_headline_renders_sc_marker_on_synth():
    fig = render_kind(_run(str(FX / "acms_sc_synth.dat")), "acms_chi_t")
    assert fig is not None


def test_single_panel_kinds_render(acms_real_path):
    r = _run(acms_real_path)
    for kind in ("acms_chi_prime_t", "acms_chi_dprime_t"):
        assert render_kind(r, kind) is not None


def test_mdc_kind_empty_without_mdc_present_with(acms_real_path):
    from cryosweep_core.plotting.catalog import series_acms_mdc_t
    assert series_acms_mdc_t(_run(acms_real_path)) == []
    assert series_acms_mdc_t(_run(str(FX / "acms_featureless_synth.dat"))) != []


def test_sci_offset_does_not_crash_with_small_values(acms_real_path):
    fig = render_kind(_run(acms_real_path), "acms_chi_prime_t")
    fig.canvas.draw()   # forces formatter evaluation on ~1e-12 magnitudes


def _legend_labels(fig):
    labels = []
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is not None:
            labels += [t.get_text() for t in leg.get_texts()]
    for leg in fig.legends:
        labels += [t.get_text() for t in leg.get_texts()]
    return labels


def test_lone_direction_key_suppressed_single_direction():
    # peak synth is all-'up': the direction role key must NOT appear in the legend
    fig = render_kind(_run(str(FX / "acms_peak_synth.dat")), "acms_chi_prime_t")
    assert "up" not in _legend_labels(fig)


def test_direction_keys_present_two_directions(acms_real_path):
    # real file has up AND down ramps: both direction keys must appear
    labels = _legend_labels(render_kind(_run(acms_real_path), "acms_chi_prime_t"))
    assert "up" in labels and "down" in labels


def _reflines(fig):
    return [ln for ax in fig.axes for ln in ax.lines if ln.get_gid() == "refline"]


def test_per_curve_tf_markers_on_peak_synth():
    # one T_f marker per curve with a detected peak, each in its group's colour
    fig = render_kind(_run(str(FX / "acms_peak_synth.dat")), "acms_chi_dprime_t")
    marks = _reflines(fig)
    assert len(marks) == 3
    xs = sorted(ln.get_xdata()[0] for ln in marks)
    assert abs(xs[0] - 3.5) < 0.1 and abs(xs[1] - 4.0) < 0.1 and abs(xs[2] - 4.5) < 0.1
    assert len({ln.get_color() for ln in marks}) == 3   # group colours, not one grey


def test_tf_markers_knob_switches_all_off():
    from cryosweep_core.plotting.spec import PlotSpec
    fig = render_kind(_run(str(FX / "acms_peak_synth.dat")), "acms_chi_dprime_t",
                      spec=PlotSpec(tc_marker=False))
    assert _reflines(fig) == []


def _axes_texts(fig):
    return [t.get_text() for ax in fig.axes for t in ax.texts]


def test_sc_marker_low_confidence_text_present():
    # a low-confidence SC detection must render the "(low confidence)" text beside the Tc marker
    r = _run(str(FX / "acms_sc_synth.dat"))
    r.data["sc_transition"]["low_confidence"] = True
    fig = render_kind(r, "acms_chi_t")
    assert any("(low confidence)" in t for t in _axes_texts(fig))


def test_sc_marker_full_confidence_no_low_conf_text():
    # golden full-confidence synth (tc_mid ~ 5.000): NO "(low confidence)" text on the figure
    r = _run(str(FX / "acms_sc_synth.dat"))
    assert r.data["sc_transition"]["low_confidence"] is False
    fig = render_kind(r, "acms_chi_t")
    assert not any("(low confidence)" in t for t in _axes_texts(fig))


def test_tf_marker_low_confidence_text_present():
    # a low-confidence chi'' peak must render the "(low confidence)" text beside its T_f marker
    r = _run(str(FX / "acms_peak_synth.dat"))
    marked = False
    for c in r.data.get("curves") or []:
        p = c.get("peak")
        if p and p.get("t_f_k") is not None:
            p["low_confidence"] = True; marked = True
    assert marked
    fig = render_kind(r, "acms_chi_dprime_t")
    assert any("(low confidence)" in t for t in _axes_texts(fig))


def test_tf_marker_full_confidence_no_low_conf_text():
    fig = render_kind(_run(str(FX / "acms_peak_synth.dat")), "acms_chi_dprime_t")
    assert not any("(low confidence)" in t for t in _axes_texts(fig))
