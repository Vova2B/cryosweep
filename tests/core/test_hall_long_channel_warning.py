# tests/core/test_hall_long_channel_warning.py
"""`--long-channel` equal to `--hall-channel` reads rho_xx from the TRANSVERSE wiring.

Measured on the real Hall file: with both set to the same bridge the median mobility came
out 0.008734 against 0.0002382 m^2/Vs with the longitudinal bridge -- a factor of 37 --
with `mobility applicable: true` and ZERO warnings. A user may have a genuine reason to do
this (a single-bridge van der Pauw arrangement, a deliberate check), so it warns rather
than declines, and says what the number then is.
"""
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.hall import HallAnalyzer
from cryosweep_core.analyzers.hall_tempdep import HallTempDepAnalyzer


def _same(probe, path, ch=1):
    cfg = RunConfig(hall={"hall_channel": ch, "thickness_mm": 0.1, "longitudinal_channel": ch})
    return probe().analyze(load_dat(path), cfg)


def _warned(res):
    return [w for w in res.warnings if "transverse" in w.lower() and "long-channel" in w]


def test_field_sweep_warns_when_long_channel_is_the_hall_channel(hall_synth_path):
    res = _same(HallAnalyzer, hall_synth_path)
    assert _warned(res), res.warnings
    assert "mobility" in _warned(res)[0]
    # warned, not declined: the number is still published for a user who meant it
    assert any(p["mobility"] is not None for p in res.data["points"])


def test_tempdep_warns_when_long_channel_is_the_hall_channel(hall_tdep_std_synth_path):
    res = _same(HallTempDepAnalyzer, hall_tdep_std_synth_path)
    assert _warned(res), res.warnings
    assert any(p["mobility"] is not None for p in res.data["points"])


def test_distinct_channels_do_not_warn(hall_synth_path, hall_tdep_std_synth_path):
    for probe, path in ((HallAnalyzer, hall_synth_path), (HallTempDepAnalyzer, hall_tdep_std_synth_path)):
        cfg = RunConfig(hall={"hall_channel": 1, "thickness_mm": 0.1, "longitudinal_channel": 2})
        assert not _warned(probe().analyze(load_dat(path), cfg))


def test_same_channel_number_in_a_separate_file_is_not_the_same_wiring(hall_synth_path,
                                                                        hall_long_synth_path):
    # --long-file ch1 beside --hall-channel 1: a different file, so a different bridge
    cfg = RunConfig(hall={"hall_channel": 1, "thickness_mm": 0.1, "longitudinal_channel": 1,
                          "longitudinal_file": str(hall_long_synth_path)})
    res = HallAnalyzer().analyze(load_dat(hall_synth_path), cfg)
    assert not _warned(res), res.warnings
