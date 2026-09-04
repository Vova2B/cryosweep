import pathlib
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry
from cryosweep_core.plotting.spec import PlotLayout, PlotEntry, PlotSpec
from cryosweep_core.plotting.catalog import OverlayFile, overlay_series, BUILTIN_PLOTKINDS
from cryosweep_core.plotting.presets import reconcile_overlay_layout

FIX = pathlib.Path(__file__).parent / "fixtures"
KINDS = {k.key: k for k in BUILTIN_PLOTKINDS}

def _act():
    return analyze_file(load_dat(str(FIX / "act_synth.dat")),
                        RunConfig.load(probe_override="resistivity"), build_default_registry())

def test_overlay_reconcile_keeps_union_keys_resets_stale():
    reg = build_default_registry(); r = _act()
    ov = [OverlayFile(0, "A"), OverlayFile(1, "B")]
    valid = [s.key for s in overlay_series(KINDS["resistivity_rho_t"], [r, r], ov)]  # 0::.. and 1::..
    lay = PlotLayout(plots=[
        PlotEntry(kind="resistivity_rho_t", spec=PlotSpec(curves=[valid[0]])),     # valid file-qualified -> kept
        PlotEntry(kind="resistivity_rho_t", spec=PlotSpec(curves=["9::ghost"])),   # stale -> None
    ])
    out = reconcile_overlay_layout(lay, [r, r], reg, ov)
    assert out.plots[0].spec.curves == [valid[0]]
    assert out.plots[1].spec.curves is None

def test_overlay_reconcile_drops_wrong_probe_kind():
    reg = build_default_registry(); r = _act()
    ov = [OverlayFile(0, "A")]
    lay = PlotLayout(plots=[PlotEntry(kind="inverse_chi")])   # vsm kind vs resistivity result
    out = reconcile_overlay_layout(lay, [r], reg, ov)
    assert all(e.kind != "inverse_chi" for e in out.plots)    # no vsm kinds leak through
    backed = {k.key for k in reg.plot_kinds_for("resistivity") if k.series(r)}
    assert {e.kind for e in out.plots} == backed              # heals to the backed default
