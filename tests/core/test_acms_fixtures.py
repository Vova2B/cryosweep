import pathlib, numpy as np
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.acms import ACMSAnalyzer

FX = pathlib.Path(__file__).parent / "fixtures"


def _run(name):
    return ACMSAnalyzer().analyze(load_dat(str(FX / name)), RunConfig())


def test_sc_synth_fires_on_one_declines_on_other():
    r = _run("acms_sc_synth.dat")
    assert r.data["sc_transition"] is not None
    assert abs(r.data["sc_transition"]["tc_mid_k"] - 5.0) < 0.3
    # nonzero normal-state baseline in the fixture -> full-confidence detection
    assert r.data["sc_transition"]["low_confidence"] is False
    assert r.data["sc_transition"]["reasons"] == []
    fired = [c for c in r.data["curves"] if c["sc"] is not None]
    assert all(round(c["amplitude_oe"]) == 1 for c in fired)     # featureless 3.0 Oe group silent


def test_peak_synth_three_frequencies_each_peak():
    r = _run("acms_peak_synth.dat")
    tfs = sorted(p["t_f_k"] for p in r.data["chi_dprime_peaks"])
    assert len(tfs) == 3
    for got, want in zip(tfs, [3.5, 4.0, 4.5]):
        assert abs(got - want) < 0.2


def test_featureless_declines_and_has_mdc():
    r = _run("acms_featureless_synth.dat")
    assert r.data["sc_transition"] is None and r.data["chi_dprime_peaks"] == []
    assert any(c["m_dc"] for c in r.data["curves"])


def test_real_subset_drops_and_logs_stray():
    r = _run("acms_real_subset.dat")
    assert any(round(g["amplitude_oe"], 4) == 0.4979 for g in r.data["dropped_groups"])


def test_real_subset_header_carries_no_identity():
    """See tests/core/test_tto_fixtures.py — same rule, ACMS header (spec §2b)."""
    head = (FX / "acms_real_subset.dat").read_text(encoding="latin-1").split("[Data]")[0]
    assert "TITLE,acms_real_subset.dat" in head
    assert "FILEOPENTIME,0.00,09/01/2026,12:00 am" in head
    assert "INFO,,anonymized" in head
    assert "BYAPP,ACMS,1.0,1.1" in head
    assert "Quantum Design" in head
