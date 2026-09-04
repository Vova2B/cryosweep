"""1/χ must not be dominated by the reciprocal of near-zero susceptibility.

On a real multi-field M(T) one 40 kOe branch has χ crossing zero 18 times above 200 K
(χ_min = −1.08e-05, about 1e-6 of the sample's typical χ). 1/χ there swings between
−1.0e6 and +1.4e6, which drew vertical stripes across the panel and flattened every real
curve onto the axis. Those points are 1/noise, not a measurement.

Masking is DISPLAY-only: `result.data["inv_chi"]` and the exported CSV still carry every
value. This module pins both that the mask fires where it must and that it does not fire
on the other real files (whose gallery figures must stay byte-identical).
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.config import RunConfig
from cryosweep_core.io.loader import load_dat
from cryosweep_core.plotting.catalog import (BUILTIN_PLOTKINDS, _chi_noise_floor,
                                        _mask_reciprocal_noise)
from cryosweep_core.registry import build_default_registry

from tests.core.conftest import require_real

KINDS = {k.key: k for k in BUILTIN_PLOTKINDS}


def _result(key, molar, mass):
    rt = load_dat(str(require_real(key)))
    rt = dataclasses.replace(rt, header=dataclasses.replace(
        rt.header, molar_mass=molar, mass_mg=mass))
    return analyze_file(rt, RunConfig.load(), build_default_registry())


def _inv_values(result, kind):
    out = []
    for s in KINDS[kind].series(result):
        if s.role == "inv_chi" or s.key.startswith("curve"):
            out.extend(s.y)
    return out


def test_mask_helper_preserves_length_and_order():
    inv = [1.0, 2.0, 3.0]
    chi = [1.0, 1e-9, 1.0]
    out = _mask_reciprocal_noise(inv, chi, 1e-6)
    assert out == [1.0, None, 3.0], out


def test_mask_helper_is_a_noop_without_a_floor_or_on_length_mismatch():
    inv = [1.0, 2.0]
    assert _mask_reciprocal_noise(inv, [1.0, 1e-9], None) == inv
    assert _mask_reciprocal_noise(inv, [1.0], 1e-6) == inv
    assert _mask_reciprocal_noise(inv, None, 1e-6) == inv


def test_floor_is_none_when_no_susceptibility_present():
    class _R:
        data = {"t_blocks": [], "chi_molar_cgs": []}

    assert _chi_noise_floor(_R()) is None


def test_divergent_branch_is_blanked_on_the_real_file():
    r = _result("vsm_mt", 200.0, 5.0)
    vals = _inv_values(r, "inverse_chi")
    blanked = sum(1 for v in vals if v is None)
    finite = [v for v in vals if v is not None and math.isfinite(v)]
    assert blanked > 0, "the divergent branch was not blanked"
    assert max(abs(v) for v in finite) < 1000.0, max(abs(v) for v in finite)
    # the underlying DATA is untouched -- masking is display-only
    raw = [v for b in r.data["t_blocks"] for v in b["inv_chi"] if v is not None]
    assert max(abs(v) for v in raw) > 1e5, "data should still carry the raw reciprocals"


def test_clean_files_are_untouched():
    """These two drive gallery figures that must stay byte-identical."""
    for key, molar, mass in (("mpms", 683.22, 12.0), ("vsm", 300.0, 1.1)):
        r = _result(key, molar, mass)
        for kind in ("inverse_chi", "vsm_chi_t"):
            vals = _inv_values(r, kind)
            assert not [v for v in vals if v is None], (key, kind)


def test_threshold_is_not_finely_tuned():
    """The mask must not depend on the exact constant.

    The noise points sit orders of magnitude below the real ones, so the drop count is
    identical across two decades of threshold. If this ever becomes sensitive, the
    separation has narrowed and the rule needs rethinking rather than retuning.
    """
    import cryosweep_core.plotting.catalog as cat

    r = _result("vsm_mt", 200.0, 5.0)
    counts = set()
    original = cat._INV_CHI_NOISE_REL
    try:
        for rel in (1e-4, 1e-3, 1e-2):
            cat._INV_CHI_NOISE_REL = rel
            vals = _inv_values(r, "inverse_chi")
            counts.add(sum(1 for v in vals if v is None))
    finally:
        cat._INV_CHI_NOISE_REL = original
    assert len(counts) == 1, f"drop count varies with the threshold: {counts}"


def test_masked_series_still_carry_their_real_points():
    """Blanking must remove only the divergent points, not whole curves."""
    r = _result("vsm_mt", 200.0, 5.0)
    for s in KINDS["inverse_chi"].series(r):
        finite = [v for v in s.y if v is not None and np.isfinite(v)]
        assert finite, f"series {s.key!r} was blanked entirely"
