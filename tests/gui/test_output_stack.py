import dataclasses, pathlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry
from cryosweep_core.plotting.catalog import BUILTIN_PLOTKINDS, build_default_layout

FIX = pathlib.Path(__file__).resolve().parents[1] / "core" / "fixtures"

def _vsm():
    rt = load_dat(str(FIX / "vsm_synth.dat"))
    rt = dataclasses.replace(rt, header=dataclasses.replace(rt.header, molar_mass=200.0, mass_mg=5.0))
    return analyze_file(rt, RunConfig.load(probe_override="vsm"), build_default_registry())

def _vsm_layout(res):
    return build_default_layout([k for k in BUILTIN_PLOTKINDS if k.probe == "vsm"], res)

def test_stack_renders_one_canvas_per_enabled_kind(qapp):
    from cryosweep_gui.output_panel import OutputPanel
    res = _vsm()
    p = OutputPanel()
    p.show_result(res, _vsm_layout(res))                 # 4 backed VSM kinds
    assert len(p.findChildren(FigureCanvasQTAgg)) == 4
    assert p.last_figure is not None                      # back-compat: first card's figure

def test_stack_no_accumulation_across_repeated_shows(qapp):
    from cryosweep_gui.output_panel import OutputPanel
    res = _vsm(); lay = _vsm_layout(res)
    p = OutputPanel()
    for _ in range(4):
        p.show_result(res, lay)
    assert len(p.findChildren(FigureCanvasQTAgg)) == 4    # not 16

def test_failed_kind_becomes_placeholder_card_not_crash(qapp):
    from cryosweep_gui.output_panel import OutputPanel
    from cryosweep_core.plotting.spec import PlotLayout, PlotEntry, PlotSpec
    res = _vsm()
    lay = PlotLayout(plots=[PlotEntry(kind="inverse_chi", spec=PlotSpec(curves=[]))])  # [] -> render raises
    p = OutputPanel()
    p.show_result(res, lay)                               # must not raise
    assert len(p.findChildren(FigureCanvasQTAgg)) == 0
    assert p.placeholder_shown
