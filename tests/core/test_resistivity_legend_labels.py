"""Resistivity legend labels must disambiguate by channel when >1 bridge contributes.

Real Hall/resistivity data exposed two legend defects, both the same root cause: the four
resistivity series labels omitted the bridge/channel identifier, so
  - rho(T)/rho(T^2): Ch1 and Ch2 curves at the same field both read "90000 Oe"
  - rho(H)/MR%:      2 bridges x N temps -> each temperature label appears twice
Fix: prefix the label with "Ch{ch} " when more than one bridge contributes
curves of that kind; single-bridge plots stay unprefixed (no clutter).
"""
import pathlib, types
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry
from cryosweep_core.plotting.catalog import BUILTIN_PLOTKINDS

FIX = pathlib.Path(__file__).parent / "fixtures"
KINDS = {k.key: k for k in BUILTIN_PLOTKINDS}


def _res(name):
    return analyze_file(load_dat(str(FIX / name)),
                        RunConfig.load(probe_override="resistivity"), build_default_registry())


def _stub(bridges):
    return types.SimpleNamespace(data={"bridges": bridges})


def test_multi_bridge_rho_t_labels_carry_channel_and_are_unique():
    # act_synth has 2 bridges, both rho(T) at 0 Oe -> without a channel prefix both read "0 Oe".
    res = _res("act_synth.dat")
    s = KINDS["resistivity_rho_t"].series(res)
    assert len(s) == 2
    labels = [sr.label for sr in s]
    assert len(set(labels)) == len(labels), f"labels must be unique, got {labels}"
    assert any(l.startswith("Ch1") for l in labels) and any(l.startswith("Ch2") for l in labels)
    assert all("Oe" in l for l in labels)


def test_multi_bridge_rho_t2_labels_carry_channel():
    res = _res("act_synth.dat")
    labels = [sr.label for sr in KINDS["resistivity_rho_t2"].series(res)]
    assert len(set(labels)) == len(labels)
    assert any(l.startswith("Ch1") for l in labels) and any(l.startswith("Ch2") for l in labels)


def test_single_bridge_label_has_no_channel_prefix():
    # one bridge -> a channel prefix would only add clutter; label stays "<field> Oe".
    one = _stub([{"channel": 1, "rho_t_curves": [
        {"temperature": [10.0, 20.0, 30.0], "rho": [1.0, 2.0, 3.0],
         "held_field_oe": 0.0, "direction": 0}]}])
    s = KINDS["resistivity_rho_t"].series(one)
    assert len(s) == 1
    assert s[0].label == "0 Oe"


def test_multi_bridge_mr_and_mr_pct_dedupe_via_channel_prefix():
    # 2 bridges, each one rho(H) loop at 50 K -> labels must be "Ch1 50.0 K" / "Ch2 50.0 K".
    def _bridge(ch):
        return {"channel": ch, "rho_h_curves": [
            {"field": [-1000.0, 0.0, 1000.0], "rho": [2.0, 1.0, 2.0],
             "held_temp_k": 50.0, "direction": 0, "rho_zero_field": 1.0}]}
    two = _stub([_bridge(1), _bridge(2)])
    for kind in ("resistivity_mr", "resistivity_mr_pct"):
        labels = [sr.label for sr in KINDS[kind].series(two)]
        assert len(labels) == 2, kind
        assert len(set(labels)) == 2, f"{kind} labels not unique: {labels}"
        assert {"Ch1 50.0 K", "Ch2 50.0 K"} == set(labels), f"{kind}: {labels}"


def test_single_bridge_mr_label_has_no_channel_prefix():
    one = _stub([{"channel": 1, "rho_h_curves": [
        {"field": [-1000.0, 0.0, 1000.0], "rho": [2.0, 1.0, 2.0],
         "held_temp_k": 50.0, "direction": 0, "rho_zero_field": 1.0}]}])
    assert KINDS["resistivity_mr"].series(one)[0].label == "50.0 K"
