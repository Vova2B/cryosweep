"""Resistivity rrr_std from the instrument Std. Dev. columns (2026-08-10 spec §4).

Std-column canonicalization (QD `Bridge N Std. Dev. (Ohm-m)` -> rho_std_bridge{N};
ACT `Res. Std.Dev. chN` -> rho_std_ch{N}) + rrr_std via the shared uncertainty helpers,
computed on the same ramp rows and endpoint policy the shipped RRR uses.
"""
import json
import math
import pathlib

import numpy as np
import pandas as pd
import pytest

from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.config import RunConfig
from cryosweep_core.fitting.uncertainty import MEDIAN_SE
from cryosweep_core.io.columns import canonicalize_columns
from cryosweep_core.io.loader import load_dat
from cryosweep_core.registry import build_default_registry

FIX = pathlib.Path("tests/core/fixtures")


def _analyze(path):
    return analyze_file(load_dat(str(path)), RunConfig.load(), build_default_registry())


class _H:  # minimal header stand-in for canonicalize_columns
    info: dict = {}


def test_std_columns_canonicalize_with_units():
    df = pd.DataFrame({
        "Temperature (K)": [1.0], "Magnetic Field (Oe)": [0.0],
        "Bridge 1 Resistivity (Ohm-m)": [1e-6],
        "Bridge 1 Std. Dev. (Ohm-m)": [1e-8],
        "Res. ch2 (ohm-cm)": [1e-4],
        "Res. Std.Dev. ch2": [1e-6],
    })
    cmap, _ = None, None
    df2, cmap = canonicalize_columns(df, _H())
    assert cmap.logical["rho_std_bridge1"] == "Bridge 1 Std. Dev. (Ohm-m)"
    assert cmap.unit["rho_std_bridge1"] == "Ohm-m"
    assert cmap.logical["rho_std_ch2"] == "Res. Std.Dev. ch2"
    assert cmap.unit["rho_std_ch2"] == "Ohm-cm"


_HEADER = (
    "[Header]\n"
    "TITLE,rrr_std_synth\n"
    "BYAPP, Resistivity, 2.0, 1.0\n"
    "INFO, , Sample1 Name\n"
    "INFO, 1, Sample1 Cross Section\n"
    "INFO, 1, Sample1 Length\n"
    "[Data]\n"
)
_COLS = ("Comment,Time Stamp (sec),Status (code),Temperature (K),Magnetic Field (Oe),"
         "Sample Position (degrees),Bridge 1 Resistivity (Ohm-m),Bridge 1 Std. Dev. (Ohm-m),"
         "Number of Readings,Bridge 1 Resistance (Ohms)\n")


def _write_constant_relstd_dat(path, n=100, rel=0.01):
    """Clean 300 -> 2 K zero-field ramp, rho linear in T, std = rel * rho everywhere."""
    T = np.linspace(300.0, 2.0, n)
    rho_ohm_cm = 1e-5 * (1.0 + 9.0 * T / 300.0)           # ~1e-4 at 300 K, ~1.06e-5 at 2 K
    rho_ohm_m = rho_ohm_cm / 100.0
    std_ohm_m = rel * rho_ohm_m
    rows = [f",{i},,{T[i]:.6f},0.0000,90.0,{rho_ohm_m[i]:.10e},{std_ohm_m[i]:.10e},25,"
            f"{rho_ohm_m[i]:.10e}" for i in range(n)]
    path.write_text(_HEADER + _COLS + "\n".join(rows) + "\n")
    return T, rho_ohm_cm


def test_rrr_std_constant_relative_std_closed_form(tmp_path):
    p = tmp_path / "rrr_std_synth.dat"
    _write_constant_relstd_dat(p, rel=0.01)
    res = _analyze(p)
    br = next(b for b in res.data["bridges"] if b["channel"] == 1)
    assert br["rrr"] is not None
    assert br["rrr_std"] is not None and br["rrr_std"] > 0.0
    # Closed form for constant RELATIVE std (monotone rho): at each endpoint the median-of-5
    # std is rel * (median-of-5 rho), so sigma_ep/rho_ep = rel * MEDIAN_SE / sqrt(5) exactly,
    # and rrr_std = rrr * sqrt(2) * rel * MEDIAN_SE / sqrt(5).
    expected = br["rrr"] * math.sqrt(2.0) * 0.01 * MEDIAN_SE / math.sqrt(5.0)
    assert br["rrr_std"] == pytest.approx(expected, rel=1e-9)
    json.dumps(res.data, allow_nan=False)


def test_rrr_std_none_when_std_column_absent():
    res = _analyze(FIX / "rho_sc_synth.dat")           # fixture has no Std. Dev. column
    for b in res.data["bridges"]:
        assert b["rrr_std"] is None


def test_real_qd_rrr_std(res_path):
    res = _analyze(res_path)
    br = next(b for b in res.data["bridges"] if b["channel"] == 1)
    assert br["rrr"] is not None and br["rrr_std"] is not None
    # Build-measured through the shipped path (2026-08-10). The spec's manual anchor was
    # RRR 26.60 +- 0.80 on ALL zero-field rows; the analyzer's own widest-ramp selection
    # gives the values pinned here (sanity: within ~2x of the anchor).
    assert br["rrr_std"] == pytest.approx(RRR_STD_QD_B1, rel=1e-3)
    assert 0.0 < br["rrr_std"] / br["rrr"] < 0.5       # sane relative sigma
    json.dumps(res.data, allow_nan=False)


# Pinned at build time through the shipped path (2026-08-10): bridge 1 RRR = 18.52 (the
# shipped widest-ramp value — spec §11 quotes the same 18.52), rrr_std = 0.3494 (1.89 %
# relative; the spec's manual all-zero-field-rows anchor was 3.0 % — within the ~2x sanity
# band, the ramp-subset difference the spec anticipated).
RRR_STD_QD_B1 = 0.3494
