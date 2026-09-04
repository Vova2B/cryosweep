import os; os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
from cryosweep_core.plotting.spec import PlotSpec
from cryosweep_core.plotting.catalog import BUILTIN_PLOTKINDS
from cryosweep_gui.plot_controls import AxisStrip
_app = QApplication.instance() or QApplication([])

_KIND = {k.key: k for k in BUILTIN_PLOTKINDS}

def test_cp_vs_t_model_toggle_sets_fit_line():
    spec = PlotSpec(kind="cp_vs_t")
    c = AxisStrip([], spec, _KIND["cp_vs_t"])     # AxisStrip(series, spec, kind)
    c.set_model_visible(False)
    assert spec.fit_line is False

def test_cp_over_t_model_toggles_set_fit_lines():
    spec = PlotSpec(kind="cp_over_t")
    c = AxisStrip([], spec, _KIND["cp_over_t"])
    # all four checkboxes start checked (spec.fit_lines is None => all on)
    c._lowt_cbs["spin_fluct_weak"].setChecked(False)   # toggling fires _commit_lowt_fit_lines
    assert spec.fit_lines is not None and "spin_fluct_weak" not in spec.fit_lines
    assert "debye_t3" in spec.fit_lines
