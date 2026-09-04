import dataclasses, pathlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry
from cryosweep_core.plotting.spec import PlotLayout, PlotEntry
from cryosweep_core.plotting.catalog import OverlayFile
FIX = pathlib.Path(__file__).resolve().parents[1] / "core" / "fixtures"

def _vsm():
    rt = load_dat(str(FIX / "vsm_synth.dat"))
    rt = dataclasses.replace(rt, header=dataclasses.replace(rt.header, molar_mass=200.0, mass_mg=5.0))
    return analyze_file(rt, RunConfig.load(probe_override="vsm"), build_default_registry())

def test_overlay_card_two_lines_filegrouped_checklist(qapp):
    from cryosweep_gui.output_panel import OutputPanel
    r = _vsm()
    p = OutputPanel()
    ov = [OverlayFile(0, "A"), OverlayFile(1, "B")]
    p.show_result([r, r], PlotLayout(plots=[PlotEntry(kind="inverse_chi")]), overlay=ov)
    card = p._cards[0]
    assert len([ln for ln in card.figure.axes[0].lines if ln.get_marker() != "None"]) == 2
    # the curve checklist is grouped by file (2 group headers "— A —", "— B —")
    texts = [card.strip.checklist._list.item(i).text() for i in range(card.strip.checklist._list.count())]
    assert "— A —" in texts and "— B —" in texts

def test_single_result_backcompat(qapp, vsm_path):
    # existing one-arg show_result(result, layout) still works (overlay=None, A/B)
    from cryosweep_gui.output_panel import OutputPanel
    r = _vsm()
    p = OutputPanel()
    p.show_result(r, PlotLayout(plots=[PlotEntry(kind="inverse_chi")]))
    assert len(p.findChildren(FigureCanvasQTAgg)) == 1 and p.last_figure is not None
