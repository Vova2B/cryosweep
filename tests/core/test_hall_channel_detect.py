# tests/core/test_hall_channel_detect.py
import pytest
from cryosweep_core.io.loader import load_dat
from cryosweep_core.io.columns import canonicalize_columns
from cryosweep_core.config import RunConfig
from cryosweep_core.detect.sweeps import segment_sweeps
from cryosweep_core.detect.hall_channel import (
    detect_hall_channel, hall_field_sweep_applicable, hall_tdep_applicable, _clear_winner)


def _ctx(path):
    rt = load_dat(path)
    df, cmap = canonicalize_columns(rt.df, rt.header)
    return df, cmap, segment_sweeps(df, cmap, RunConfig())


def test_clear_winner_threshold():
    assert _clear_winner([(0.806, 1), (0.0, 2)]) == (1, 0.806)     # runaway winner
    assert _clear_winner([(0.53, 2), (0.073, 1)]) == (2, 0.53)     # 7.3x runner-up
    assert _clear_winner([(0.5, 1), (0.4, 2)]) is None             # < 2x runner-up -> ambiguous
    assert _clear_winner([(0.03, 1), (0.0, 2)]) is None            # below 0.05 floor
    assert _clear_winner([]) is None


def test_detect_channel_on_synth(hall_synth_path):
    df, cmap, segs = _ctx(hall_synth_path)
    ch, frac = detect_hall_channel(df, cmap, segs)
    assert ch == 1
    assert frac == pytest.approx(0.806, abs=0.05)


def test_applicability_on_synth(hall_synth_path):
    df, cmap, segs = _ctx(hall_synth_path)
    assert hall_field_sweep_applicable(cmap, segs) is True
    assert hall_tdep_applicable(segs) is False                     # no temp-ramp segments


def test_tdep_only_file_applicability_and_no_channel(hall_tdep_synth_path):
    # a temp-ramp-only file (no field loops): Temp-Dep Hall applicable, field-sweep NOT,
    # and channel detection returns None (nothing to antisymmetrize over field) -> ask.
    df, cmap, segs = _ctx(hall_tdep_synth_path)
    assert hall_field_sweep_applicable(cmap, segs) is False
    assert hall_tdep_applicable(segs) is True
    assert detect_hall_channel(df, cmap, segs) is None


def test_detect_and_applicability_on_real_file(hall_real_path):
    if not hall_real_path.exists():
        pytest.skip("real Hall measurement file not present (gitignored)")
    df, cmap, segs = _ctx(hall_real_path)
    ch, frac = detect_hall_channel(df, cmap, segs)
    assert ch == 1 and frac > 0.1
    assert hall_field_sweep_applicable(cmap, segs) is True
    assert hall_tdep_applicable(segs) is True                      # ±40k/±90k field ramps


def test_detect_longitudinal_channel_on_synth(hall_synth_path):
    """Owner 2026-07-09: mobility never appeared because the longitudinal channel had to be
    typed by hand. The companion bridge (data present, not the Hall channel, most even-in-B)
    is detectable the same way the Hall channel is."""
    from cryosweep_core.detect.hall_channel import detect_longitudinal_channel
    df, cmap, segs = _ctx(hall_synth_path)
    assert detect_longitudinal_channel(df, cmap, segs, hall_channel=1) == 2

def test_detect_longitudinal_channel_none_when_no_companion(hall_long_synth_path):
    from cryosweep_core.detect.hall_channel import detect_longitudinal_channel
    df, cmap, segs = _ctx(hall_long_synth_path)
    # only ch2 carries data in this fixture -> no companion for hall_channel=2
    assert detect_longitudinal_channel(df, cmap, segs, hall_channel=2) is None

def test_detect_longitudinal_channel_on_real_file(hall_real_path):
    if not hall_real_path.exists():
        pytest.skip("real Hall measurement file not present (gitignored)")
    from cryosweep_core.detect.hall_channel import detect_longitudinal_channel
    df, cmap, segs = _ctx(hall_real_path)
    assert detect_longitudinal_channel(df, cmap, segs, hall_channel=1) == 2   # Ch2 = MR/R_xx
