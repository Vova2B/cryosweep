import os; os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pathlib
from PySide6.QtWidgets import QApplication
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry
from cryosweep_core.plotting.catalog import get_kind
from cryosweep_core.plotting.spec import PlotSpec
from cryosweep_gui.plot_controls import AxisStrip

_app = QApplication.instance() or QApplication([])
FIX = pathlib.Path(__file__).parents[1] / "core" / "fixtures"


def _mf():
    return analyze_file(load_dat(str(FIX / "hc_multifield_synth.dat")),
                        RunConfig.load(), build_default_registry())


def test_multifield_axisstrip_has_fitline_controls():
    res = _mf()
    kind = get_kind("hc_lowt_multifield")
    series = kind.series(res)
    strip = AxisStrip(series, PlotSpec(), kind)
    # the strip exposes per-(model,field) fit-line checkboxes for this kind
    assert hasattr(strip, "_mf_fit_cbs") and len(strip._mf_fit_cbs) > 0
