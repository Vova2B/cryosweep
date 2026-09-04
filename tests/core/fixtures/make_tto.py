"""Synthetic + trimmed-real TTO fixtures (BYAPP,THERMAL_TRANSPORT QD format). Committed
beside the .dat files.

tto_synth.dat        150-pt cooling sweep at 0 Oe + a 30-pt group at 90000 Oe. rho, S and
                     kappa follow closed-form laws chosen so kappa_ph == 1.0 EXACTLY
                     everywhere and RRR == 494/59 under the median-of-5 endpoint convention.
                     3 rows carry Error (code) = 16.
tto_gap_synth.dat    Seebeck column entirely empty + 3 rows with kappa <= 0 (drop-and-warn
                     contract + `seebeck` capability inapplicable).
tto_norho_synth.dat  Resistivity and ZT columns entirely empty (`wiedemann_franz` and
                     `power_factor` inapplicable; the WF plot kind must return []).
tto_powerlaw_synth.dat  150-pt cooling sweep 30->2 K at 0 Oe with rho CONSTANT at 1e-5 Ohm*m,
                     so kappa_ph = kappa - L0*T/rho is an EXACT power law with n = 3.0 and
                     B = 1e-3. 43 points on the primary <=10 K rung. rho = 1e-5 (not 1e-7)
                     keeps median kappa_e/kappa over T<=10 K at 0.0646, so kappa_e_dominant
                     does NOT fire and `quality_flags == []` is itself the assertion (M6).
                     Seebeck and ZT are blank; no error codes. RRR == 1.0 exactly.
tto_deltat_synth.dat 40-pt cooling sweep 30->2 K at 0 Oe, the ONLY fixture with a populated
                     `Delta Temp. (K)` column: 0.01 K everywhere except row 30 (0.9 K at
                     T = 8.4615 K -> 10.64 %). Rows 5-7 carry kappa = -1 so the D6 filter drops
                     3 rows BEFORE that one, which is what makes a `[keep]`-less read of the
                     column report a DIFFERENT temperature (M4). Seebeck and ZT blank.
tto_real_subset.dat  Header + every 4th body row of the real file (n_error_rows oracle = 2).

WRITE FORMAT: every numeric cell is "%.8e" (repo precedent, make_acms.py:21-23). The kappa_ph
oracle tolerance (1e-6) is pinned to this precision — measured round-trip error
max|kappa - L0*T/rho - 1| is 2.2526e-07 over the whole file (both field groups; the 30-point
90000 Oe sweep carries the worst point) and 4.8521e-08 over the 150-point 0 Oe sweep alone.
%.8e is the LEAST precise format that still clears 1e-6: %.7e gives 1.9133e-06 and %.6e gives
1.8355e-05, both outside it. Do not change one without the other.
ENCODING: latin-1 everywhere (the column header carries a micro sign).
"""
import numpy as np

L0 = 2.443e-8              # Sommerfeld Lorenz number, W*Ohm*K^-2

_HEADER = ("[Header]\n"
           "TITLE,{title}\n"
           "BYAPP,THERMAL_TRANSPORT,1.0,1.1\n"
           "INFO,PPMS Thermal Transport Option Version: Release 1.1.5 Build 5,APPNAME\n"
           "INFO,TTO_SYNTH,SAMPLE_MATERIAL\n"
           "INFO,bulk,SAMPLE_COMMENT\n"
           "INFO,2.5,SAMPLE_VLEAD_SEPARATION\n"
           "INFO,2.5,SAMPLE_ILEAD_SEPARATION\n"
           "INFO,2.8565,SAMPLE_CROSS_SECTION\n"
           "INFO,38.545,SAMPLE_SURFACE_AREA\n"
           "INFO,0.3,SAMPLE_EMISSIVITY\n"
           "[Data]\n")

# The real file's column header line, verbatim (63 columns, micro signs included).
_COLS = ("Comment,Time Stamp (sec),Status (code),Error (code),"
         "Magnetic Field (Oe),Sample Temp. (K),Conductivity (W/K-m),"
         "Cond. Std.Dev.,Seebeck Coef. (µV/K),Seebeck Std.Dev.,"
         "Resistivity (Ohm-m),Resist Std.Dev.,Figure of Merit ZT,"
         "Merit Std.Dev.,Delta Temp. (K),Conductance (W/K),"
         "Raw Conductance (W/K),Seebeck Volt. (µV),Resistance (Ohm),"
         "Min. Temp. (K),Max. Temp. (K),Temp. Rise (K),Req. Htr Power (W),"
         "Heater Power (W),Rad. Loss (W),Cond. Pwr. (W),Heater Current (mA),"
         "Res. Drive (mA),Res. Freq (Hz),Period (sec),Period Ratio,tau1 (sec),"
         "tau2 (sec),Seebeck Gain,Resist. Gain,System Temp (K),"
         "Sample Position (deg),Bridge 1 Resistance (ohms),"
         "Bridge 1 Excitation (µA),Bridge 2 Resistance (ohms),"
         "Bridge 2 Excitation (µA),Bridge 3 Resistance (ohms),"
         "Bridge 3 Excitation (µA),Bridge 4 Resistance (ohms),"
         "Bridge 4 Excitation (µA),Signal 1 Vin (V),Signal 2 Vin (V),"
         "Digital Inputs (code),Drive 1 Iout (mA),Drive 1 Ipower (W),"
         "Drive 2 Iout (mA),Drive 2 Ipower (W),Pressure (),Map 20 (),"
         "Map 21 (),Map 22 (),Map 23 (),Map 24 (),Map 25 (),Map 26 (),"
         "Map 27 (),Map 28 (),Map 29 ()\n")

_N_COLS = 63
# 0-based column indices of the fields we populate (all others stay empty cells).
_I_ERR, _I_FIELD, _I_TEMP = 3, 4, 5
_I_KAPPA, _I_KAPPA_SD, _I_SEE, _I_SEE_SD = 6, 7, 8, 9
_I_RHO, _I_RHO_SD, _I_ZT, _I_ZT_SD = 10, 11, 12, 13
_I_DT = 14                                   # Delta Temp. (K) -- populated by ONE fixture


def _f(v):
    """%.8e, or an empty cell for None/non-finite (missing values are blank, never NaN)."""
    if v is None:
        return ""
    v = float(v)
    return "" if not np.isfinite(v) else f"{v:.8e}"


def _row(err, field, t, kappa, seebeck, rho, zt, delta_t=None):
    cells = [""] * _N_COLS
    cells[1] = "0"                    # Time Stamp (sec)
    cells[2] = "5906"                 # Status (code)
    cells[_I_ERR] = str(int(err))
    cells[_I_FIELD] = f"{field:.4f}"
    cells[_I_TEMP] = _f(t)
    cells[_I_KAPPA] = _f(kappa)
    cells[_I_KAPPA_SD] = _f(None if kappa is None else 0.01 * abs(kappa))
    cells[_I_SEE] = _f(seebeck)
    cells[_I_SEE_SD] = _f(None if seebeck is None else 0.01 * abs(seebeck))
    cells[_I_RHO] = _f(rho)
    cells[_I_RHO_SD] = _f(None if rho is None else 0.01 * abs(rho))
    cells[_I_ZT] = _f(zt)
    cells[_I_ZT_SD] = _f(None if zt is None else 0.01 * abs(zt))
    cells[_I_DT] = _f(delta_t)               # NEW -- None -> "" -> byte-identical to today
    return ",".join(cells) + "\n"


def _write(path, title, rows):
    with open(path, "w", encoding="latin-1", newline="") as f:
        f.write(_HEADER.format(title=title))
        f.write(_COLS)
        for r in rows:
            f.write(r)


def _rho(t):
    return 1e-8 * (1.0 + 9.0 * t / 300.0)


def _seebeck(t):
    return 0.01 * t                              # microvolt per kelvin


def _kappa(t):
    return 1.0 + L0 * t / _rho(t)                # kappa_ph == 1.0 by construction


def _zt(t):
    s_volts = _seebeck(t) * 1e-6
    return s_volts ** 2 * t / (_rho(t) * _kappa(t))


_RHO_PL = 1e-5                               # Ohm*m, CONSTANT -> RRR == 1.0 exactly


def _kappa_powerlaw(t):
    """kappa = kappa_e + 1e-3*T^3, so kappa_ph is an exact cube with B = 1e-3."""
    return L0 * t / _RHO_PL + 1.0e-3 * t ** 3


def _sweep(temps, field, err_at=()):
    rows = []
    for i, t in enumerate(temps):
        rows.append(_row(16 if i in err_at else 0, field, t,
                         _kappa(t), _seebeck(t), _rho(t), _zt(t)))
    return rows


def write_all(d):
    import pathlib
    d = pathlib.Path(d)

    t_main = np.linspace(300.0, 2.0, 150)        # exact 2 K spacing
    t_hifield = np.linspace(300.0, 2.0, 30)
    _write(d / "tto_synth.dat", "tto_synth",
           _sweep(t_main, 0.0, err_at={40, 41, 42}) + _sweep(t_hifield, 90000.0))

    # gap fixture: no Seebeck at all, plus three unphysical kappa <= 0 rows
    gap = []
    for i, t in enumerate(t_main):
        kappa = -1.0 if i in (10, 11, 12) else _kappa(t)
        gap.append(_row(0, 0.0, t, kappa, None, _rho(t), _zt(t)))
    _write(d / "tto_gap_synth.dat", "tto_gap_synth", gap)

    # no-rho fixture: rho and ZT blank -> WF/power-factor/ZT all inapplicable
    norho = [_row(0, 0.0, t, _kappa(t), _seebeck(t), None, None) for t in t_main]
    _write(d / "tto_norho_synth.dat", "tto_norho_synth", norho)

    # exact-power-law fixture: the kappa_ph fit's positive oracle (n = 3.000000 at every rung)
    t_pl = np.linspace(30.0, 2.0, 150)       # PINNED grid: the oracle is not reproducible
    powerlaw = [_row(0, 0.0, t, _kappa_powerlaw(t), None, _RHO_PL, None) for t in t_pl]
    _write(d / "tto_powerlaw_synth.dat", "tto_powerlaw_synth", powerlaw)

    # DeltaT/T alignment fixture (M4/I3): 3 unphysical kappa <= 0 rows are dropped BEFORE
    # grouping, and the ONE oversized DeltaT sits far enough after them that an unfiltered
    # read of the column pairs it with a different temperature. 40 points, 0 Oe.
    t_dt = np.linspace(30.0, 2.0, 40)        # PINNED grid: t[30] = 8.4615 K is the oracle
    dt_rows = []
    for i, t in enumerate(t_dt):
        kappa = -1.0 if i in (5, 6, 7) else _kappa(t)
        dt_rows.append(_row(0, 0.0, t, kappa, None, _rho(t), None,
                            delta_t=(0.9 if i == 30 else 0.01)))
    _write(d / "tto_deltat_synth.dat", "tto_deltat_synth", dt_rows)

    return {"rrr_synth": 494 / 59, "kappa_ph_synth": 1.0, "n_error_rows_synth": 3,
            "pf_at_300k_synth": 9e-5, "n_groups_synth": 2}


def _rho_ex(t):
    """examples/ only: dirty-metal rho(T) in Ohm*m (RRR = 2)."""
    return 1e-5 * (0.5 + 0.5 * (t / 300.0) ** 2)


def _kappa_ph_ex(t):
    """examples/ only: phonon kappa ~ T^3 at low T, peak near 33 K, ~1/T decay above."""
    return 5.0e-3 * t ** 3 / (1.0 + (t / 25.0) ** 4)


def write_example(path, title="thermal_transport_example"):
    """examples/ file ONLY — never a committed fixture. Same format/laws family as tto_synth,
    but with a measurement-like grid (extra 0.5 K steps below 12 K -> >=10 finite kappa_ph > 0
    points below 10 K, so the kappa_ph power-law fit runs instead of declining) and a kappa
    that is genuinely phonon-dominated below 10 K (the file's advertised feature): kappa_ph
    follows an approximate T^3 law at low T (effective n ~ 2.9 on the <=10 K window), while
    kappa_e = L0*T/rho stays small (dirty metal, rho ~ 1e-5 Ohm*m). A 30-pt 9 T sweep with
    kappa_ph suppressed x0.85 keeps the two-group grouping the synth fixture also exercises."""
    t_lo = np.arange(2.0, 12.0 + 1e-9, 0.5)                 # 21 pts; 17 of them < 10 K
    t_hi = np.arange(15.0, 300.0 + 1e-9, 3.0)
    t_main = np.concatenate([t_hi[::-1], t_lo[::-1]])       # cooling: 300 -> 2
    def rows(temps, field, ph_scale):
        out = []
        for t in temps:
            kappa = L0 * t / _rho_ex(t) + ph_scale * _kappa_ph_ex(t)
            s_uv = 0.01 * t
            zt = (s_uv * 1e-6) ** 2 * t / (_rho_ex(t) * kappa)
            out.append(_row(0, field, t, kappa, s_uv, _rho_ex(t), zt))
        return out
    t_hifield = np.linspace(300.0, 2.0, 30)
    _write(path, title, rows(t_main, 0.0, 1.0) + rows(t_hifield, 90000.0, 0.85))
    return {"n_low_pts": int((t_main < 10.0).sum())}


import pathlib as _pl, sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))  # _anonymize is a sibling:
# these generators are both run as scripts AND imported as modules by the test suite,
# and only the script form puts this directory on sys.path.
from _anonymize import (NEUTRAL_TIME, anonymize_header, assert_no_identity_leak,
                        identity_values, split_at_data, write_subset)   # noqa: F401

# TTO's probe-specific rules. TITLE -> sample["name"] and SAMPLE_MATERIAL -> sample["material"]
# are the two fields that carry identity into the JSON envelope; both reach the CSV/GUI
# surfaces. SAMPLE_COMMENT reaches no result field, but it is operator free text on a real
# instrument — the value that happened to ship was the harmless "bulk", which is luck, not a
# rule — so it is neutralised too rather than passed through unread. BYAPP and the five
# SAMPLE_* geometry lines are load-bearing and are left untouched.
_SAMPLE_RULES = (
    (lambda ln: ln.startswith("INFO,") and ln.endswith(",SAMPLE_MATERIAL"),
     "INFO,anonymized,SAMPLE_MATERIAL"),
    (lambda ln: ln.startswith("INFO,") and ln.endswith(",SAMPLE_COMMENT"),
     "INFO,anonymized,SAMPLE_COMMENT"),
)
_SAMPLE_LINE = (lambda ln: ln.split(",")[1] if ln.startswith("INFO,")
                and ln.endswith(",SAMPLE_MATERIAL") else None)


def _anonymize_header(lines, title):
    return anonymize_header(lines, title, _SAMPLE_RULES)


def _assert_no_identity_leak(head, src_lines, sample_line):
    return assert_no_identity_leak(head, src_lines, sample_line)


def write_real_subset(src, dst):
    """Header + every 4th body row. latin-1 on BOTH ends: the acms precedent's bare
    read_text() raises UnicodeDecodeError 0xb5 on the micro sign in this file."""
    import pathlib
    lines = pathlib.Path(src).read_text(encoding="latin-1").splitlines()
    di = next(i for i, ln in enumerate(lines) if ln.strip() == "[Data]")
    head = _anonymize_header(lines[:di + 2], "tto_real_subset.dat")   # header + the column line
    _assert_no_identity_leak(head, lines,
                             lambda ln: ln.split(",")[1] if ln.startswith("INFO,")
                             and ln.endswith(",SAMPLE_MATERIAL") else None)
    body = [ln for ln in lines[di + 2:] if ln.strip()]
    kept = body[::4]
    pathlib.Path(dst).write_text("\n".join(head + kept) + "\n", encoding="latin-1")


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
    src = real_data("tto")
    if src is not None:
        write_real_subset(src, here / "tto_real_subset.dat")
    else:
        print("real Thermal Transport file unavailable -> tto_real_subset.dat not regenerated "
              "(the committed copy is left untouched)")
