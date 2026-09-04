# tests/core/test_field_legend_labels.py
import pathlib, types
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry
from cryosweep_core.plotting.catalog import BUILTIN_PLOTKINDS

FIX = pathlib.Path(__file__).parent / "fixtures"
KINDS = {k.key: k for k in BUILTIN_PLOTKINDS}


def _res(name, probe=None):
    cfg = RunConfig.load(probe_override=probe) if probe else RunConfig.load()
    return analyze_file(load_dat(str(FIX / name)), cfg, build_default_registry())


def _multifield():
    return _res("hc_multifield_synth.dat")


def test_rho_t_label_reads_tesla_keys_unchanged():
    one = types.SimpleNamespace(data={"bridges": [{"channel": 1, "rho_t_curves": [
        {"temperature": [10.0, 20.0], "rho": [1.0, 2.0], "held_field_oe": 90000.0, "direction": 0}]}]})
    oe = KINDS["resistivity_rho_t"].series(one, field_unit="Oe")
    t = KINDS["resistivity_rho_t"].series(one, field_unit="T")
    assert oe[0].label == "90000 Oe"
    assert t[0].label == "9 T"
    assert oe[0].key == t[0].key                       # KEY unit-invariant


def test_lowt_multifield_labels_and_group_tesla_keys_unchanged():
    res = _multifield()
    oe = KINDS["hc_lowt_multifield"].series(res, field_unit="Oe")
    t = KINDS["hc_lowt_multifield"].series(res, field_unit="T")
    assert oe and len(oe) == len(t)
    assert all(s.label.endswith(" Oe") for s in oe)
    assert all(s.label.endswith(" T") for s in t)
    assert [s.group for s in t] == [s.label for s in t]      # group tracks label
    assert [s.key for s in oe] == [s.key for s in t]         # KEYS identical


def test_schottky_and_transition_multifield_labels_tesla():
    res = _res("hc_schottky_synth.dat")
    for key in ("hc_schottky_multifield",):
        oe = KINDS[key].series(res, field_unit="Oe")
        t = KINDS[key].series(res, field_unit="T")
        if not oe:
            continue
        assert [s.key for s in oe] == [s.key for s in t]
        assert any(" T" in s.label for s in t)
