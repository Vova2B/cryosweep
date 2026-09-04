from cryosweep_core.plotting.presets import builtin_presets, BUILTIN_PRESETS
from cryosweep_core.plotting.catalog import BUILTIN_PLOTKINDS

_JOURNAL = {
    "vsm": ["vsm_moment_t", "vsm_chi_t", "vsm_mh"],
    "resistivity": ["resistivity_rho_t", "resistivity_mr_pct"],
    "hall": ["hall_asym_vs_B", "hall_rh_t"],
    "heatcapacity": ["hc_full_cp_t", "cp_over_t", "hc_entropy_vs_t"],
}


def test_journal_pins_per_probe():
    for probe, kinds in _JOURNAL.items():
        j = next(p for p in builtin_presets(probe) if p.name == "Journal")
        assert [e.kind for e in j.layout.plots] == kinds
        assert j.probe == probe


def test_all_plots_lists_every_backed_kind_in_order():
    for probe in ("vsm", "resistivity", "hall", "heatcapacity", "hall_tdep"):
        allp = next(p for p in builtin_presets(probe) if p.name == "All plots")
        expected = [k.key for k in BUILTIN_PLOTKINDS if k.probe == probe]
        assert [e.kind for e in allp.layout.plots] == expected


def test_hall_tdep_has_no_journal():
    names = [p.name for p in builtin_presets("hall_tdep")]
    assert "All plots" in names and "Journal" not in names


def test_returns_fresh_copies():
    a = builtin_presets("vsm")[0]
    a.layout.plots.clear()
    b = builtin_presets("vsm")[0]
    assert b.layout.plots, "builtin_presets must return independent copies"
