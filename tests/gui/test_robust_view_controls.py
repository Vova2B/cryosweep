import numpy as np
from cryosweep_gui.plot_controls import AxisStrip
from cryosweep_core.plotting.spec import PlotSpec
from cryosweep_core.plotting.catalog import BUILTIN_PLOTKINDS

# This codebase has no pytest-qt; widget tests use the session `qapp` fixture
# (see conftest.py) and construct widgets directly.

_KIND = {k.key: k for k in BUILTIN_PLOTKINDS}["resistivity_rho_t"]

def test_axisstrip_has_robust_view_checkbox_default_checked(qapp):
    spec = PlotSpec()
    strip = AxisStrip(series=[], spec=spec, kind=_KIND)
    assert strip._robust_cb.isChecked()
    strip._robust_cb.setChecked(False)
    assert spec.robust_view is False

def test_resistivity_panel_exclude_outliers_override(qapp):
    from cryosweep_gui.inputs.resistivity import ResistivityInputPanel
    p = ResistivityInputPanel()
    assert "quality" not in p.build_overrides()
    p.exclude_cb.setChecked(True)
    ov = p.build_overrides()
    assert ov["quality"]["exclude_outliers"] is True
