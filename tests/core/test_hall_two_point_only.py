# tests/core/test_hall_two_point_only.py
"""Say when the only published value came from the untrusted estimator.

Measured on the real Hall file, channel 2: the sigma >= |R_H| decline withholds all 137
trusted `antisym` carrier densities and publishes exactly ONE, from the `2point` sparse
fallback -- with nothing in the envelope marking that the single surviving number rests
on the estimator the analyzer itself trusts least. The same envelope reported
`antisym_fraction: 1.0`, because the 2point points were excluded from that fraction's
denominator: the headline diagnostic contradicted the situation.

`antisym_fraction` is now taken over the PUBLISHED points -- of the carrier densities this
result reports, what fraction rests on a trusted antisym fit with enough pairs -- and is
None (never 0.0 or 1.0 asserted over nothing) when no point published. Its old basis
("trusted antisym points only, so that a 2-point tail cannot deflate status") was a rule
about a quantity that DROVE status; it no longer does, and a diagnostic whose denominator
excludes exactly the points that were published is not a diagnostic of the published result.
"""
import numpy as np
import pytest

from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.hall_tempdep import HallTempDepAnalyzer
from tests.core.conftest import real_data

_HDR = ("[Header]\nBYAPP, Resistivity\nINFO, twopt_only_synth, SAMPLE\n[Data]\n"
        "Temperature (K),Magnetic Field (Oe),Bridge 1 Resistance (Ohms),"
        "Bridge 1 Resistivity (Ohm-m),Bridge 1 Std. Dev. (Ohm-m)\n")
_RATIO = 1000.0


def _write(tmp_path):
    """+-10000 Oe pair over 2-40 K with HUGE instrument sigma (every antisym point is
    withheld), plus a 0 Oe and an unpaired +30000 Oe ramp over 2-70 K with tiny sigma:
    above 40 K only the 2point fallback exists, and it resolves."""
    rows = []
    for b, sign, tmax, sd in ((10000.0, 1.0, 40, 1e-4), (-10000.0, -1.0, 40, 1e-4),
                              (0.0, 0.0, 70, 1e-12), (30000.0, 1.0, 70, 1e-12)):
        rxy = 1e-3 + sign * 5e-4 * (abs(b) / 1e4)
        for i in range(tmax - 1):
            T = 2.0 + i
            rows.append(f"{T:.4f},{b:.1f},{rxy:.10e},{rxy / _RATIO:.10e},{sd:.10e}")
    p = tmp_path / "twopt_only.dat"
    p.write_text(_HDR + "\n".join(rows) + "\n")
    return p


def _analyze(path, ch=1):
    return HallTempDepAnalyzer().analyze(load_dat(path), RunConfig(
        hall={"hall_channel": ch, "thickness_mm": 0.1}))


def test_only_two_point_published_is_named_and_counted(tmp_path):
    res = _analyze(_write(tmp_path))
    pts = res.data["points"]
    published = [p for p in pts if p["carrier_n"] is not None]
    withheld_anti = [p for p in pts if p["r_h_method"] == "antisym" and p["carrier_n"] is None]
    assert published and all(p["r_h_method"] == "2point" for p in published)
    assert withheld_anti, "fixture must withhold every antisym point"
    assert "two_point_only_published" in res.data["flags"]
    w = [w for w in res.warnings if "2point" in w]
    assert w, res.warnings
    assert str(len(published)) in w[0] and str(len(withheld_anti)) in w[0]
    assert "antisym" in w[0]


def test_antisym_fraction_is_over_the_published_points(tmp_path):
    res = _analyze(_write(tmp_path))
    # every published point is 2point -> none of them rests on an antisym fit
    assert res.confidence_parts["antisym_fraction"] == 0.0


def test_antisym_fraction_is_none_when_nothing_published(hall_tdep_synth_path):
    # the noiseless fixture publishes nothing (residual sigma is float noise, no std
    # column): a fraction asserted over zero points is not a fraction
    res = HallTempDepAnalyzer().analyze(load_dat(hall_tdep_synth_path), RunConfig(
        hall={"hall_channel": 1, "thickness_mm": 0.05}))
    assert all(p["carrier_n"] is None for p in res.data["points"])
    assert res.confidence_parts["antisym_fraction"] is None
    assert "two_point_only_published" not in res.data["flags"]


def test_no_flag_when_an_antisym_point_publishes(hall_tdep_std_synth_path):
    res = HallTempDepAnalyzer().analyze(load_dat(hall_tdep_std_synth_path), RunConfig(
        hall={"hall_channel": 1, "thickness_mm": 0.05}))
    assert any(p["carrier_n"] is not None and p["r_h_method"] == "antisym"
               for p in res.data["points"])
    assert "two_point_only_published" not in res.data["flags"]
    assert not any("2point" in w for w in res.warnings)
    assert res.confidence_parts["antisym_fraction"] == 1.0


def test_real_file_channel_2_names_its_single_two_point_survivor():
    path = real_data("hall")
    if path is None:
        pytest.skip("local-only measurement file for key 'hall' is not available")
    res = _analyze(path, ch=2)
    pts = res.data["points"]
    published = [p for p in pts if p["carrier_n"] is not None]
    assert len(published) == 1 and published[0]["r_h_method"] == "2point"
    assert sum(1 for p in pts if p["r_h_method"] == "antisym") == 137
    assert "two_point_only_published" in res.data["flags"]
    assert any("137" in w and "2point" in w for w in res.warnings)
    assert res.confidence_parts["antisym_fraction"] == 0.0
