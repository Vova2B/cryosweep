import json, subprocess, sys
from cryosweep_core.plotting.spec import GlobalStyle, PlotLayout, PlotEntry, PlotSpec
from cryosweep_core.plotting.presets import NamedPreset, PresetStore, load_store, save_store

def _store():
    return PresetStore(
        global_style=GlobalStyle(marker="s"),
        last_used={"vsm": PlotLayout(plots=[PlotEntry(kind="inverse_chi")])},
        presets=[NamedPreset(name="paper", probe="vsm",
                             layout=PlotLayout(plots=[PlotEntry(kind="vsm_moment_t", spec=PlotSpec(yscale="log"))]))])

def test_roundtrip(tmp_path):
    p = tmp_path / "s.json"
    assert save_store(_store(), p) is True
    s = load_store(p)
    assert s.global_style.marker == "s"
    assert s.last_used["vsm"].plots[0].kind == "inverse_chi"
    assert s.presets[0].name == "paper" and s.presets[0].layout.plots[0].spec.yscale == "log"

def test_missing_and_garbage_return_defaults(tmp_path):
    assert load_store(tmp_path / "nope.json").presets == []          # missing
    (tmp_path / "g.json").write_text("not json{{{")
    assert load_store(tmp_path / "g.json").global_style.marker == "o"  # garbage -> defaults
    (tmp_path / "arr.json").write_text("[1,2,3]")
    assert load_store(tmp_path / "arr.json").presets == []           # non-dict

def test_element_wise_salvage(tmp_path):
    raw = {"global_style": {"dpi": -5},                             # invalid (gt=0) -> salvage to default
           "last_used": {"vsm": {"plots": [{"kind": "inverse_chi"}]},
                         "bad": {"plots": "not-a-list"}},           # one good, one bad
           "presets": [{"name": "ok", "probe": "vsm", "layout": {"plots": []}},
                       {"name": "bad", "probe": "vsm", "layout": 12345}]}  # one good, one bad
    p = tmp_path / "s.json"; p.write_text(json.dumps(raw))
    s = load_store(p)
    assert s.global_style.dpi == 300                                # corrupt style -> default GlobalStyle()
    assert "vsm" in s.last_used and "bad" not in s.last_used         # dropped the bad value
    assert [x.name for x in s.presets] == ["ok"]                     # dropped the bad preset

def test_save_unwritable_returns_false(tmp_path):
    assert save_store(_store(), tmp_path) is False                  # path is a directory -> OSError -> False

def test_presets_module_is_matplotlib_and_qt_free():
    # subprocess (not in-process) so render tests in the same `pytest tests/core` run can't
    # pollute sys.modules and false-fail this — mirrors tests/core/test_qt_free.py.
    code = ("import sys, cryosweep_core.plotting.presets; "
            "assert 'matplotlib' not in sys.modules and not {'PySide6', 'PyQt6'} & sys.modules.keys(); "
            "print('CLEAN')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert "CLEAN" in out.stdout, (out.stdout + out.stderr)

import pathlib, dataclasses
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry
from cryosweep_core.plotting.presets import reconcile_layout
FIXD = pathlib.Path(__file__).parent / "fixtures"

def _act():
    return analyze_file(load_dat(str(FIXD / "act_synth.dat")),
                        RunConfig.load(probe_override="resistivity"), build_default_registry())

def test_reconcile_resets_stale_curves_keeps_valid():
    reg = build_default_registry(); res = _act()
    valid = [s.key for s in reg.plot_kinds_for("resistivity")[0].series(res)]
    lay = PlotLayout(plots=[
        PlotEntry(kind="resistivity_rho_t", spec=PlotSpec(curves=["b9:T:0:0"])),   # stale -> None
        PlotEntry(kind="resistivity_rho_t", spec=PlotSpec(curves=valid[:1])),       # valid -> kept
        PlotEntry(kind="unknown_kind", spec=PlotSpec()),                            # unknown -> dropped
    ])
    out = reconcile_layout(lay, res, reg)
    assert out.plots[0].spec.curves is None
    assert out.plots[1].spec.curves == valid[:1]
    assert all(e.kind != "unknown_kind" for e in out.plots)        # unknown dropped

def test_reconcile_wrong_probe_drops_all():
    reg = build_default_registry(); res = _act()                   # resistivity result
    vsm_layout = PlotLayout(plots=[PlotEntry(kind="inverse_chi")])
    out = reconcile_layout(vsm_layout, res, reg)
    assert all(e.kind != "inverse_chi" for e in out.plots)         # no vsm kinds leak through
    backed = {k.key for k in reg.plot_kinds_for("resistivity") if k.series(res)}
    assert {e.kind for e in out.plots} == backed                   # heals to the backed default


def test_save_failure_cleans_up_tmp(tmp_path):
    # os.replace fails (target is a directory) -> False AND no stray .tmp left behind
    d = tmp_path / "adir"; d.mkdir()
    assert save_store(_store(), d) is False
    assert not (tmp_path / "adir.tmp").exists()

def test_fit_lines_survives_save_load(tmp_path):
    from cryosweep_core.plotting.presets import PresetStore, save_store, load_store
    from cryosweep_core.plotting.spec import PlotLayout, PlotEntry, PlotSpec
    store = PresetStore()
    store.last_used["resistivity"] = PlotLayout(plots=[
        PlotEntry(kind="resistivity_rho_t2", spec=PlotSpec(fit_lines=("linear",)))])
    p = tmp_path / "presets.json"
    assert save_store(store, p) is True
    back = load_store(p)
    assert back.last_used["resistivity"].plots[0].spec.fit_lines == ("linear",)

def test_old_layout_without_fit_lines_loads(tmp_path):
    import json
    from cryosweep_core.plotting.presets import load_store
    p = tmp_path / "old.json"
    p.write_text(json.dumps({
        "version": 1,
        "last_used": {"vsm": {"plots": [{"kind": "inverse_chi", "spec": {"fit_line": True}}]}},
    }))
    store = load_store(p)
    assert store.last_used["vsm"].plots[0].spec.fit_lines is None

def test_reconcile_appends_newly_backed_kinds():
    """Regression (owner 2026-07-09): a last_used layout saved while R_H kinds were unbacked
    (no thickness) pinned the Hall tabs to 3 kinds forever. Kinds backed NOW but not known at
    save time must be appended."""
    reg = build_default_registry(); res = _act()
    backed = [k.key for k in reg.plot_kinds_for("resistivity") if k.series(res)]
    assert len(backed) >= 2
    saved = PlotLayout(plots=[PlotEntry(kind=backed[0])], known=[backed[0]])   # only [0] existed at save
    out = reconcile_layout(saved, res, reg)
    assert [e.kind for e in out.plots][0] == backed[0]                # saved order preserved first
    assert {e.kind for e in out.plots} == set(backed)                 # newly backed appended
    assert out.known == backed                                        # refreshed for the next save

def test_reconcile_respects_deliberate_uncheck():
    """A kind the user unchecked (present in known, absent from plots) must NOT be resurrected."""
    reg = build_default_registry(); res = _act()
    backed = [k.key for k in reg.plot_kinds_for("resistivity") if k.series(res)]
    saved = PlotLayout(plots=[PlotEntry(kind=backed[0])], known=backed)        # user pruned the rest
    out = reconcile_layout(saved, res, reg)
    assert [e.kind for e in out.plots] == [backed[0]]

def test_reconcile_recovers_old_stores_without_known():
    """Stores written before PlotLayout.known existed (known=None) can't distinguish 'unbacked
    then' from 'unchecked'; recover by treating the saved plots as the known set, so missing
    backed kinds reappear once (the owner's stuck Hall layout heals on first load)."""
    reg = build_default_registry(); res = _act()
    backed = [k.key for k in reg.plot_kinds_for("resistivity") if k.series(res)]
    saved = PlotLayout(plots=[PlotEntry(kind=backed[0])])                      # known=None (old format)
    out = reconcile_layout(saved, res, reg)
    assert {e.kind for e in out.plots} == set(backed)
