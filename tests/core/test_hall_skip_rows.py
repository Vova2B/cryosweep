"""Task 4b: a user-controlled leading-row skip for the Hall probes.

Some PPMS runs write a first data row taken before the measurement bridge has settled.
It is not a noisy reading -- it is not a reading at all. Measured on the real
resistivity-option file (channel 2): row 0 carries R = -4.0e6 Ohm against a file median
of ~1e-4 Ohm, and the field-sweep Hall fit's `np.isfinite`-only mask lets it straight
into the 300 K fit, moving the published R_H by a factor of 5e5 (see
docs/physics-reference.md, "Leading-row skip", and KNOWN-ISSUES).

`skip_rows` drops leading rows of the file before either Hall analyzer runs. It defaulted
to an unconditional 1 until 2026-09-14 and now defaults to "auto", which drops a row only
where it is PROVABLY corrupt; an explicit integer still drops exactly that many and turns
detection off. The one thing this design must not do is skip silently: the count is always
in the envelope, auto says what it dropped and why, and an explicit count is told when a
row it dropped looked PHYSICAL (naming --skip-rows 0 to get it back).

The tests below are in two halves -- the operator-count semantics first, then auto mode.
"""
import tempfile
import pathlib

import numpy as np
import pytest

from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.hall import HallAnalyzer

_HDR = ("[Header]\nBYAPP, Resistivity\nINFO, skiprows_synth, SAMPLE\n[Data]\n"
        "Temperature (K),Magnetic Field (Oe),Bridge 1 Resistance (Ohms),"
        "Bridge 1 Resistivity (Ohm-m),Bridge 1 Std. Dev. (Ohm-m)\n")
_RATIO = 1000.0             # R/rho, held exactly constant -> row_sigma_R's constancy gate passes
_SLOPE = -5.0e-4            # Ohm/T
_INTERCEPT = 1.0e-3         # Ohm
_SD_TYPICAL = 1.0e-6        # Ohm-m (raw column; ratio-scaled to Ohm by row_sigma_R)
_T = 10.0
_FIELDS_OE = np.arange(-20000.0, 20000.1, 5000.0)      # 9-point field sweep


def _rxy(b_oe):
    return _SLOPE * (b_oe / 1e4) + _INTERCEPT


def _row(T, b_oe, r, sd):
    rho = r / _RATIO
    return f"{T:.4f},{b_oe:.1f},{r:.10e},{rho:.10e},{sd:.10e}"


def _build(row0_R, row0_SD):
    """A clean 9-point field sweep at one T, with one extra row PREPENDED at the sweep's
    own starting field (a duplicate setpoint, matching how a real pre-settling read
    repeats the first commanded field) -- everything downstream of `row0_R`/`row0_SD` is
    the well-behaved reference sweep."""
    rows = [_row(_T, _FIELDS_OE[0], row0_R, row0_SD)]
    rows += [_row(_T, b, _rxy(b), _SD_TYPICAL) for b in _FIELDS_OE]
    return _HDR + "\n".join(rows) + "\n"


def _write(tmp_path, name, text):
    p = pathlib.Path(tmp_path) / name
    p.write_text(text)
    return p


def _analyze(path, **hall):
    hall.setdefault("hall_channel", 1)
    hall.setdefault("thickness_mm", 0.1)
    return HallAnalyzer().analyze(load_dat(path), RunConfig(hall=hall))


# unphysical row0: 7 orders of magnitude off the sweep's own values (both R and its
# reported std) -- the measured real-file separation (docs/physics-reference.md).
_BAD_R, _BAD_SD = 1.0e4, 1.0e4
# physical row0: a plausible repeat of the sweep's own first point, +-typical noise.
_GOOD_R, _GOOD_SD = _rxy(_FIELDS_OE[0]) * 1.01, _SD_TYPICAL * 0.9


# ------------------------------------------- (a) the count is always in the envelope --
def test_a_dropped_row_is_always_visible_in_the_envelope():
    with tempfile.TemporaryDirectory() as d:
        p = _write(d, "a.dat", _build(_BAD_R, _BAD_SD))
        r = _analyze(p)                      # no skip_rows override -> HallCfg default
    assert r.data["skipped_rows"] == 1
    # the leading (bad) row is gone from Stage A's own count -- one row dropped, no more
    assert r.data["points"][0]["n_points"] == len(_FIELDS_OE)


# ---------------------------------------------------- (b) --skip-rows 0 is a true off --
def test_skip_rows_zero_restores_every_row():
    with tempfile.TemporaryDirectory() as d:
        p = _write(d, "b.dat", _build(_BAD_R, _BAD_SD))
        r0 = _analyze(p, skip_rows=0)
        r1 = _analyze(p, skip_rows=1)
    assert r0.data["skipped_rows"] == 0
    # skip_rows=0 takes the untouched dataframe -- Stage A sees every row the file has,
    # never fewer than the skip_rows=1 run (segmentation may itself trim an unphysical
    # leading jump, so this is >=, not a fixed difference of exactly one).
    assert r0.data["points"][0]["n_points"] >= r1.data["points"][0]["n_points"]
    assert r0.data["points"][0]["n_points"] > len(_FIELDS_OE) - 1


# -------------------------- (c) an explicit count over a good row triggers the reversal --
def test_an_explicit_skip_warns_when_the_dropped_row_looks_physical():
    """The reversal warning belongs to the EXPLICIT path now: auto would have kept this
    row, so only an operator who typed a count can be in the position it warns about."""
    with tempfile.TemporaryDirectory() as d:
        p = _write(d, "c.dat", _build(_GOOD_R, _GOOD_SD))
        r = _analyze(p, skip_rows=1)
    hits = [w for w in r.warnings if "looks physical" in w]
    assert len(hits) == 1
    w = hits[0]
    assert "skipped 1 leading data row" in w
    assert "--skip-rows 0" in w
    # both quantities that grounded the judgement are named, so the warning is auditable
    assert "|R| = " in w and "reported std " in w


# --------------------------------- (d) an unphysical leading row stays silent --------
def test_default_skip_is_silent_when_the_dropped_row_is_unphysical():
    with tempfile.TemporaryDirectory() as d:
        p = _write(d, "d.dat", _build(_BAD_R, _BAD_SD))
        r = _analyze(p)                      # default skip_rows=1
    assert not any("looks physical" in w for w in r.warnings)
    assert r.data["skipped_rows"] == 1        # the count is still reported -- never silent


def test_no_judgement_possible_neither_acts_nor_speaks():
    """The leading row carries no usable R (blank -- the instrument logged nothing at all,
    not even a bad number) and there is no std-dev column either. No evidence supports a
    verdict either way, so auto must not drop it and must not claim anything about it:
    absence of evidence is not evidence. Under the superseded unconditional default this
    row was dropped anyway and only the JUDGEMENT was withheld."""
    hdr_no_sd = ("[Header]\nBYAPP, Resistivity\nINFO, skiprows_synth, SAMPLE\n[Data]\n"
                 "Temperature (K),Magnetic Field (Oe),Bridge 1 Resistance (Ohms),"
                 "Bridge 1 Resistivity (Ohm-m)\n")
    rows = [f"{_T:.4f},{_FIELDS_OE[0]:.1f},,"]                    # row0: R and rho both blank
    rows += [f"{_T:.4f},{b:.1f},{_rxy(b):.10e},{_rxy(b) / _RATIO:.10e}" for b in _FIELDS_OE]
    with tempfile.TemporaryDirectory() as d:
        p = _write(d, "e.dat", hdr_no_sd + "\n".join(rows) + "\n")
        r = _analyze(p)
    assert r.data["skipped_rows"] == 0
    assert not any("leading data row" in w for w in r.warnings)


# --------------------------------------- (e) the real file recovers the sound value --
def test_real_resistivity_file_recovers_the_sound_300k_point(res_path):
    """Measured on the real resistivity-option file (task 4b brief): including the
    unphysical row 0 gives n=302, R_H=-1.4346e-04, r2=0.00213 at 300 K; excluding it
    (dropping it) gives n=301, R_H=-2.7424e-10, r2=0.66948 -- landing on
    the trend set by the 200 K neighbour (R_H=-2.8812e-10, r2=0.99888)."""
    r_default = _analyze(res_path, hall_channel=2, thickness_mm=0.1)   # "auto" detects it
    r_unfiltered = _analyze(res_path, hall_channel=2, thickness_mm=0.1, skip_rows=0)

    assert r_default.data["skipped_rows"] == 1
    assert r_unfiltered.data["skipped_rows"] == 0

    p300 = next(p for p in r_default.data["points"] if p["temperature"] == 300.0)
    assert p300["n_points"] == 301
    assert p300["R_H"] == pytest.approx(-2.7424e-10, rel=1e-3)
    assert p300["r2"] == pytest.approx(0.66948, abs=2e-4)

    p200 = next(p for p in r_default.data["points"] if p["temperature"] == 200.0)
    assert p200["R_H"] == pytest.approx(-2.8812e-10, rel=1e-3)
    assert p200["r2"] == pytest.approx(0.99888, abs=1e-3)

    # --skip-rows 0 reproduces the OLD (pre-4b) pathological numbers EXACTLY -- proving
    # the flag controls the behaviour, not that the data quietly changed underneath.
    p300_old = next(p for p in r_unfiltered.data["points"] if p["temperature"] == 300.0)
    assert p300_old["n_points"] == 302
    assert p300_old["R_H"] == pytest.approx(-1.4346e-04, rel=1e-3)
    assert p300_old["r2"] == pytest.approx(0.00213, abs=2e-4)

    # the row is plainly unphysical (10-11 orders of magnitude off), so the default did
    # its job silently -- no reversal warning, just the always-present count.
    assert not any("looks physical" in w for w in r_default.warnings)


# ------------------------------------- (g) skip_rows > 1 names one row, on purpose --
def test_multi_row_skip_reports_one_row_not_every_physical_one():
    """`skip_rows` > 1 is a smaller use case than the default of 1, and the warning
    deliberately names only the FIRST physical-looking row it finds rather than
    enumerating all of them. Pinned because the alternative (one warning per skipped
    physical row) is a defensible design someone might later prefer -- and if they do,
    they should change this test on purpose rather than discover the behaviour by
    accident. The remedy is identical whichever row is named: --skip-rows 0, then pick a
    smaller N.

    The file here has three perfectly ordinary leading rows, so all three are physical."""
    with tempfile.TemporaryDirectory() as d:
        p = _write(d, "f.dat", _build(_GOOD_R, _GOOD_SD))
        r = _analyze(p, skip_rows=3)

    assert r.data["skipped_rows"] == 3
    warns = [w for w in r.warnings if "looks physical" in w]
    assert len(warns) == 1, "one warning, however many skipped rows were physical"
    assert "skipped 3 leading data row" in warns[0]
    assert "--skip-rows 0" in warns[0]


# =========================== auto mode (2026-09-14) ==================================
#
# `skip_rows` now defaults to "auto": leading rows are dropped only where they are
# PROVABLY corrupt, by the same 1e6 ratio the reversal warning already used to judge
# them. The measured evidence for making the threshold act rather than only warn: across
# every resistivity-format file available here, row 0 sits between 0.26x and 14.7x the
# file median EXCEPT on the one corrupted file, where it sits at 1.16e12x (channel 1) and
# 1.32e11x (channel 2). Ten empty orders of magnitude separate the two populations, and
# the threshold sits in the middle of that gap.
#
# The old unconditional default of 1 dropped a good row on every other file, moving R_H
# by up to 9.6%. An explicit integer still means exactly what it said -- and turns
# detection off, because the operator has spoken.

from cryosweep_core.analyzers.hall_sigma import corrupt_leading_rows, AUTO_SCAN_CAP
from cryosweep_core.io.columns import canonicalize_columns


def _corrupt_count(path):
    rt = load_dat(str(path))
    df, cmap = canonicalize_columns(rt.df, rt.header)
    return corrupt_leading_rows(df, cmap)


def test_auto_is_the_default_and_drops_a_provably_corrupt_leading_row():
    with tempfile.TemporaryDirectory() as d:
        p = _write(d, "auto_bad.dat", _build(_BAD_R, _BAD_SD))
        r = _analyze(p)                                  # no override -> HallCfg default
    assert r.data["skipped_rows"] == 1
    assert r.data["points"][0]["n_points"] == len(_FIELDS_OE)


def test_auto_keeps_a_physical_leading_row():
    """THE point of auto mode. A leading row that looks like ordinary data is data, and
    the analyzer must reach exactly the result it would have reached with --skip-rows 0."""
    with tempfile.TemporaryDirectory() as d:
        p = _write(d, "auto_good.dat", _build(_GOOD_R, _GOOD_SD))
        r_auto = _analyze(p)
        r_off = _analyze(p, skip_rows=0)
    assert r_auto.data["skipped_rows"] == 0
    assert r_auto.data["points"][0]["n_points"] == r_off.data["points"][0]["n_points"]
    assert r_auto.data["points"][0]["R_H"] == r_off.data["points"][0]["R_H"]


def test_auto_says_why_it_dropped_a_row():
    """Dropping data silently is the one thing this design must never do. When auto acts
    it names the evidence that convinced it and the flag that reverses it."""
    with tempfile.TemporaryDirectory() as d:
        p = _write(d, "auto_why.dat", _build(_BAD_R, _BAD_SD))
        r = _analyze(p)
    hits = [w for w in r.warnings if "dropped 1 leading data row" in w]
    assert len(hits) == 1
    assert "--skip-rows 0" in hits[0]
    assert "times the file median" in hits[0]


def test_auto_is_silent_when_it_drops_nothing():
    with tempfile.TemporaryDirectory() as d:
        p = _write(d, "auto_quiet.dat", _build(_GOOD_R, _GOOD_SD))
        r = _analyze(p)
    assert not any("leading data row" in w for w in r.warnings)


def test_an_explicit_count_turns_detection_off():
    """--skip-rows 0 means keep everything, even a row auto would have dropped. The
    operator's explicit choice outranks the detector; that is what makes it reversible."""
    with tempfile.TemporaryDirectory() as d:
        p = _write(d, "auto_off.dat", _build(_BAD_R, _BAD_SD))
        r0 = _analyze(p, skip_rows=0)
        r2 = _analyze(p, skip_rows=2)
    assert r0.data["skipped_rows"] == 0
    assert r2.data["skipped_rows"] == 2          # exactly 2 -- never 2 plus a detection


def test_auto_never_drops_more_than_the_scan_cap():
    """A bounded blast radius is the safety property that makes a detector acceptable
    here: however broken the head of a file is, auto can never eat into the measurement."""
    # The clean tail has to outnumber the corrupt head, or the baseline median this
    # judges against is itself corrupt and nothing is detectable -- which is exactly the
    # regime the cap protects, not one it can measure.
    rows = [_row(_T, _FIELDS_OE[0], _BAD_R, _BAD_SD) for _ in range(AUTO_SCAN_CAP + 10)]
    rows += [_row(_T, b, _rxy(b), _SD_TYPICAL) for b in _FIELDS_OE] * 7
    with tempfile.TemporaryDirectory() as d:
        p = _write(d, "auto_cap.dat", _HDR + "\n".join(rows) + "\n")
        assert _corrupt_count(p) == AUTO_SCAN_CAP


def test_auto_declines_to_judge_without_evidence():
    """No usable R and no std-dev column: nothing supports a verdict either way. Auto
    drops nothing -- it acts only on proof, never on the absence of it."""
    hdr_no_sd = ("[Header]\nBYAPP, Resistivity\nINFO, skiprows_synth, SAMPLE\n[Data]\n"
                 "Temperature (K),Magnetic Field (Oe),Bridge 1 Resistance (Ohms),"
                 "Bridge 1 Resistivity (Ohm-m)\n")
    rows = [f"{_T:.4f},{_FIELDS_OE[0]:.1f},,"]
    rows += [f"{_T:.4f},{b:.1f},{_rxy(b):.10e},{_rxy(b) / _RATIO:.10e}" for b in _FIELDS_OE]
    with tempfile.TemporaryDirectory() as d:
        p = _write(d, "auto_blind.dat", hdr_no_sd + "\n".join(rows) + "\n")
        r = _analyze(p)
    assert r.data["skipped_rows"] == 0


def test_auto_finds_the_corrupt_row_on_the_real_file(res_path):
    """The file auto exists for. Both channels' row 0 is corrupt; the count is 1, not
    more -- row 1 is already ordinary (8.6x and 3.4x the median)."""
    assert _corrupt_count(res_path) == 1
    r = _analyze(res_path, hall_channel=2, thickness_mm=0.1)
    assert r.data["skipped_rows"] == 1
    p300 = next(p for p in r.data["points"] if p["temperature"] == 300.0)
    assert p300["R_H"] == pytest.approx(-2.7424e-10, rel=1e-3)


def test_auto_leaves_the_real_hall_file_untouched(hall_real_path):
    """The regression the old default caused: this file's row 0 is ordinary (|R| 13x the
    median only because |R_xy| varies across a field sweep, with its reported sigma at
    0.89x), and dropping it moved R_H at 300 K by 3.3%. Auto must keep it."""
    assert _corrupt_count(hall_real_path) == 0
    r_auto = _analyze(hall_real_path, hall_channel=1, thickness_mm=0.07, longitudinal_channel=2)
    r_off = _analyze(hall_real_path, hall_channel=1, thickness_mm=0.07,
                     longitudinal_channel=2, skip_rows=0)
    assert r_auto.data["skipped_rows"] == 0
    assert [p["R_H"] for p in r_auto.data["points"]] == [p["R_H"] for p in r_off.data["points"]]


def test_skip_rows_rejects_a_negative_count():
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        RunConfig(hall={"skip_rows": -1})
