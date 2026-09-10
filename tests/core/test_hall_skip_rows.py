"""Task 4b: a user-controlled leading-row skip for the Hall probes.

Some PPMS runs write a first data row taken before the measurement bridge has settled.
It is not a noisy reading -- it is not a reading at all. Measured on the real
resistivity-option file (channel 2): row 0 carries R = -4.0e6 Ohm against a file median
of ~1e-4 Ohm, and the field-sweep Hall fit's `np.isfinite`-only mask lets it straight
into the 300 K fit, moving the published R_H by a factor of 5e5 (see
docs/physics-reference.md, "Leading-row skip", and KNOWN-ISSUES).

The fix is an OPERATOR-CONTROLLED skip, not a detector: `skip_rows` (default 1) drops
the first N rows of the file before either Hall analyzer runs. No threshold decides
whether to skip -- the operator does. The one thing this design must not do is skip
silently: the count is always in the envelope, and a reversal warning fires when the
row the default dropped looks PHYSICAL (naming --skip-rows 0 to get it back).
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


# ------------------------------------------------------------------ (a) default is 1 --
def test_skip_rows_defaults_to_1_and_is_visible_in_the_envelope():
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


# ---------------------------------- (c) a good leading row triggers the reversal warn --
def test_default_skip_warns_when_the_dropped_row_looks_physical():
    with tempfile.TemporaryDirectory() as d:
        p = _write(d, "c.dat", _build(_GOOD_R, _GOOD_SD))
        r = _analyze(p)                      # default skip_rows=1
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


def test_no_judgement_possible_says_nothing():
    """The skipped row itself carries no usable R (blank -- the instrument logged
    nothing at all, not even a bad number) and there is no std-dev column either: no
    evidence supports a verdict either way, so the reversal warning must say nothing
    rather than guess. The count is still reported -- only the JUDGEMENT is withheld."""
    hdr_no_sd = ("[Header]\nBYAPP, Resistivity\nINFO, skiprows_synth, SAMPLE\n[Data]\n"
                 "Temperature (K),Magnetic Field (Oe),Bridge 1 Resistance (Ohms),"
                 "Bridge 1 Resistivity (Ohm-m)\n")
    rows = [f"{_T:.4f},{_FIELDS_OE[0]:.1f},,"]                    # row0: R and rho both blank
    rows += [f"{_T:.4f},{b:.1f},{_rxy(b):.10e},{_rxy(b) / _RATIO:.10e}" for b in _FIELDS_OE]
    with tempfile.TemporaryDirectory() as d:
        p = _write(d, "e.dat", hdr_no_sd + "\n".join(rows) + "\n")
        r = _analyze(p)
    assert r.data["skipped_rows"] == 1
    assert not any("looks physical" in w for w in r.warnings)


# --------------------------------------- (e) the real file recovers the sound value --
def test_real_resistivity_file_recovers_the_sound_300k_point(res_path):
    """Measured on the real resistivity-option file (task 4b brief): including the
    unphysical row 0 gives n=302, R_H=-1.4346e-04, r2=0.00213 at 300 K; excluding it
    (skip_rows=1, the default) gives n=301, R_H=-2.7424e-10, r2=0.66948 -- landing on
    the trend set by the 200 K neighbour (R_H=-2.8812e-10, r2=0.99888)."""
    r_default = _analyze(res_path, hall_channel=2, thickness_mm=0.1)
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
