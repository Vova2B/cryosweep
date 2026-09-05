"""The hc_lowt_multifield fit-line checkboxes must key exactly what the renderer draws.

The AxisStrip used to derive the key's field part from the DISPLAY group tag
(`"50000.5 Oe".split()[0]`). That happened to match the renderer's raw
`{field_oe:g}` until KNOWN-ISSUES #14/#7 display-rounded the tags ("50 kOe") — after
which every derived key ("debye_t3@50") missed the real key ("debye_t3@50000.5"), so
unchecking ANY box built a fit_lines set that matched nothing and silently dropped every
fit line. The keys now come from the Series key ("mf:<field_oe:g>"), the same raw value
the renderer uses; the display tag is only the checkbox caption.
"""
import os; os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pathlib
from PySide6.QtWidgets import QApplication
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.registry import build_default_registry
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.plotting.spec import PlotSpec
from cryosweep_core.plotting.catalog import BUILTIN_PLOTKINDS, series_hc_lowt_multifield
from cryosweep_gui.plot_controls import AxisStrip

_app = QApplication.instance() or QApplication([])
ROOT = pathlib.Path(__file__).resolve().parents[2]
_KIND = {k.key: k for k in BUILTIN_PLOTKINDS}


def _result():
    return analyze_file(load_dat(str(ROOT / "examples" / "heat_capacity_multifield.dat")),
                        RunConfig.load(), build_default_registry())


def _render_fit_labels(res, spec):
    from cryosweep_core.plotting.render import render_kind
    fig = render_kind(res, "hc_lowt_multifield", spec)
    return {ln.get_label() for ln in fig.axes[0].lines if ln.get_gid() == "fit"}


def test_checkbox_keys_match_the_renderers_fit_line_keys():
    res = _result()
    series = series_hc_lowt_multifield(res)
    spec = PlotSpec(kind="hc_lowt_multifield")
    c = AxisStrip(series, spec, _KIND["hc_lowt_multifield"])
    drawn = _render_fit_labels(res, PlotSpec(kind="hc_lowt_multifield"))
    assert drawn                                       # premise: fits exist
    assert set(c._mf_fit_cbs) == drawn                 # every checkbox names a real fit key


def test_unchecking_one_box_drops_exactly_that_fit():
    res = _result()
    series = series_hc_lowt_multifield(res)
    spec = PlotSpec(kind="hc_lowt_multifield")
    c = AxisStrip(series, spec, _KIND["hc_lowt_multifield"])
    all_keys = set(c._mf_fit_cbs)
    victim = sorted(all_keys)[0]
    c._mf_fit_cbs[victim].setChecked(False)            # fires _commit_mf_fit_lines
    drawn = _render_fit_labels(res, spec)
    assert drawn == all_keys - {victim}                # not empty, not unchanged


def test_checkbox_captions_show_the_display_tag():
    series = series_hc_lowt_multifield(_result())
    spec = PlotSpec(kind="hc_lowt_multifield")
    c = AxisStrip(series, spec, _KIND["hc_lowt_multifield"])
    captions = " | ".join(cb.text() for cb in c._mf_fit_cbs.values())
    assert "50 kOe" in captions and "0 Oe" in captions
    assert "50000.5" not in captions                   # raw medians stay out of the UI
