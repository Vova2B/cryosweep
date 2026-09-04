"""Synthetic + trimmed-real ACMS fixtures (BYAPP,ACMS QD format). Committed beside the .dat.

acms_sc_synth.dat        two groups at 477 Hz: amp 1.0 Oe carries a diamagnetic chi' step with
                         oracle tc_mid = 5.000 K (onset 5.5) + a chi'' peak inside the window;
                         amp 3.0 Oe is featureless (detector fires on one, declines on the other).
acms_peak_synth.dat      three frequencies (100/477/1000 Hz), each a chi'' Gaussian peak at a
                         known T_f (3.5/4.0/4.5 K) -> multi-group grouping x per-curve peaks.
acms_featureless_synth.dat  flat chi'/chi'' + noise (both detectors decline; determinism fixture).
                         Carries a finite M-DC column (only fixture exercising acms_mdc_t).
acms_real_subset.dat     header + every ~10th real row, PLUS the 0.4979 Oe stray row (drop-and-log
                         contract stays testable on the subset)."""
import numpy as np

_HEADER = ("[Header]\nTITLE,{title}\nBYAPP,ACMS,1.0,1.1\nINFO,HARMONICS,1\n"
           "INFO, , {title}\n[Data]\n")
_COLS = ("Comment,Time Stamp (sec),Temperature (K),Magnetic Field (Oe),Frequency (Hz),"
         "Amplitude (Oe),M-DC (emu),M-Std.Dev. (emu),M' (emu),M'' (emu)\n")


def _row(t, field, freq, amp, mdc, mp, mpp):
    mdc_s = "" if mdc is None else f"{mdc:.8e}"
    return (f",0,{t:.5f},{field:.4f},{freq:.1f},{amp:.4f},{mdc_s},1e-9,"
            f"{mp:.8e},{mpp:.8e}\n")


def _write(path, title, rows):
    with open(path, "w") as f:
        f.write(_HEADER.format(title=title)); f.write(_COLS)
        for r in rows:
            f.write(_row(*r))


def _sc_group(amp, sc):
    rng = np.random.default_rng(0 if sc else 1)
    t = np.linspace(0.4, 8.0, 220)
    rows = []
    for tk in t:
        if sc:
            # nonzero normal-state baseline (-1e-12) avoids the zero-baseline relative-spread
            # misfire in _detect_sc; diamagnetic step below Tc, 50% of the drop at 5.0 K.
            chi = -1.0e-12 - 1e-11 * (1.0 / (1 + np.exp((tk - 5.0) / 0.2)))
            mpp = 3e-12 * np.exp(-((tk - 5.25) ** 2) / (0.2 ** 2)) + 1e-13
        else:
            chi = -4e-12; mpp = 2.4e-13
        mp = (chi + rng.normal(0, 2e-14)) * amp
        rows.append((tk, -0.1, 477.0, amp, None, mp, (mpp + rng.normal(0, 5e-15)) * amp))
    return rows


def write_all(d):
    import pathlib
    d = pathlib.Path(d)
    _write(d / "acms_sc_synth.dat", "acms_sc_synth", _sc_group(1.0, True) + _sc_group(3.0, False))
    # peak fixture: three frequencies, each a Gaussian chi'' peak at a known T_f
    rows = []
    tf = {100.0: 3.5, 477.0: 4.0, 1000.0: 4.5}
    rng = np.random.default_rng(2)
    for freq, tfk in tf.items():
        t = np.linspace(1.0, 8.0, 200)
        for tk in t:
            mpp = 5e-12 * np.exp(-((tk - tfk) ** 2) / (0.3 ** 2)) + 1e-13
            rows.append((tk, -0.1, freq, 1.0, None, (-4e-12) * 1.0,
                         (mpp + rng.normal(0, 3e-14)) * 1.0))
    _write(d / "acms_peak_synth.dat", "acms_peak_synth", rows)
    # featureless + M-DC
    rng = np.random.default_rng(4); t = np.linspace(0.4, 5.0, 220); rows = []
    for tk in t:
        rows.append((tk, -0.1, 477.0, 0.05, 1e-6 + 1e-8 * tk,
                     (-4e-12 + rng.normal(0, 2e-14)) * 0.05,
                     (2.4e-13 + rng.normal(0, 1e-14)) * 0.05))
    _write(d / "acms_featureless_synth.dat", "acms_featureless_synth", rows)
    return {"sc_tc_mid": 5.000, "sc_amp": 1.0, "peak_tf": tf,
            "featureless_has_mdc": True}


def write_sc_example(path, molar_mass=250.0, mass_mg=15.0):
    """examples/ file ONLY — never a committed fixture. The same two 477 Hz groups as
    acms_sc_synth (identical rows: same seeds/laws), plus what a complete instrument header
    would carry: MOLWGHT/MASS INFO lines (so molar normalization is capability-on out of the
    box instead of demanding --molar-mass/--mass-mg) and a populated M-DC column
    (M_dc = chi * H_dc at the -0.1 Oe bias), so the dc_magnetization capability applies too."""
    header = ("[Header]\nTITLE,{title}\nBYAPP,ACMS,1.0,1.1\nINFO,HARMONICS,1\n"
              f"INFO,{molar_mass},MOLWGHT:Molecular Weight (g/mol)\n"
              f"INFO,{mass_mg},MASS:Sample Mass (mg)\n"
              "INFO, , {title}\n[Data]\n").format(title="acms_sc_example")
    rows = _sc_group(1.0, True) + _sc_group(3.0, False)
    with open(path, "w") as f:
        f.write(header); f.write(_COLS)
        for (t, field, freq, amp, mdc, mp, mpp) in rows:
            chi = mp / amp                      # recover chi from the group's own scaling
            f.write(_row(t, field, freq, amp, chi * field, mp, mpp))
    return {"molar_mass": molar_mass, "mass_mg": mass_mg}


# Neutral publication stamp: the shipped subsets carry the repo's sanitisation date, never the
# real acquisition date. The FIRST field (the QD time base) is zeroed too — it is the ONLY thing
# that ever tied the absolute "Time Stamp (sec)" column to a calendar date, so with it gone that
# column is a bare session counter and needs no rebasing.
import pathlib as _pl, sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))  # _anonymize is a sibling:
# these generators are both run as scripts AND imported as modules by the test suite,
# and only the script form puts this directory on sys.path.
from _anonymize import (NEUTRAL_TIME, anonymize_header, assert_no_identity_leak,
                        identity_values, split_at_data, write_subset)   # noqa: F401

# ACMS's probe-specific rule: the sample line is the free-text `INFO,,<sample>` between
# INFO,HARMONICS,1 and the run of empty INFO,, lines. BYAPP is load-bearing (detection scores
# 1.0 on it, 0.2 without) and is left untouched.
_SAMPLE_RULES = ((lambda ln: ln.startswith("INFO,,") and ln[len("INFO,,"):].strip(),
                  "INFO,,anonymized"),)
_SAMPLE_LINE = (lambda ln: ln[len("INFO,,"):] if ln.startswith("INFO,,") else None)


def _anonymize_header(lines, title):
    return anonymize_header(lines, title, _SAMPLE_RULES)


def _assert_no_identity_leak(head, src_lines, sample_line):
    return assert_no_identity_leak(head, src_lines, sample_line)


def write_real_subset(src, dst):
    import pathlib
    lines = pathlib.Path(src).read_text().splitlines()
    di = next(i for i, ln in enumerate(lines) if ln.strip() == "[Data]")
    head = _anonymize_header(lines[:di + 2], "acms_real_subset.dat")   # header + column line
    _assert_no_identity_leak(head, lines,
                             lambda ln: ln[len("INFO,,"):] if ln.startswith("INFO,,") else None)
    body = lines[di + 2:]
    kept = body[::10]
    stray = next((ln for ln in body if ln.split(",")[5:6] == ["0.4979"]
                  or (len(ln.split(",")) > 5 and ln.split(",")[5].startswith("0.4979"))), None)
    if stray and stray not in kept:
        kept.append(stray)
    pathlib.Path(dst).write_text("\n".join(head + kept) + "\n")


if __name__ == "__main__":
    import pathlib
    here = pathlib.Path(__file__).parent
    print(write_all(here))
    # The real file is NOT in the repo; it is resolved through the untracked real_data_map.json
    # (see tests/core/conftest.py). Guard the subset regeneration so a machine without it gets
    # good synthetic fixtures and a clear message, not a FileNotFoundError traceback.
    import sys
    sys.path.insert(0, str(here.parent))        # tests/core, for conftest's real_data()
    from conftest import real_data
    src = real_data("acms")
    if src is not None:
        write_real_subset(src, here / "acms_real_subset.dat")
    else:
        print("real ACMS file unavailable -> acms_real_subset.dat not regenerated "
              "(the committed copy is left untouched)")
