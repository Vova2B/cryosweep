"""Pin the resistivity probe's share of the multi-field block recovery.

The recovery pass in `find_blocks` was built for multi-field M(T), but it is probe-agnostic
and therefore changed a SHIPPED probe too: on this two-channel file each channel gained 8
`rho_h_curves` (0 -> 8) and `magnetoresistance` flipped applicable False -> True. The
recovered curves are genuine field sweeps (0 -> 9 T at T = 2, 4, 6 ... 50 K, each held to
better than 0.02 K), so this is an improvement -- but it arrived as a side effect of a VSM
change and was NOT covered by the VSM tests. Pinned here so it cannot silently regress.

Skips in a public checkout: the file is local-only and resolved through the gitignored
key map, so no measurement filename appears in any tracked file.
"""
from __future__ import annotations

from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.config import RunConfig
from cryosweep_core.io.loader import load_dat
from cryosweep_core.registry import build_default_registry

from tests.core.conftest import require_real


def _analyze():
    return analyze_file(load_dat(str(require_real("res_2ch"))),
                        RunConfig.load(), build_default_registry())


def test_both_channels_keep_their_recovered_field_sweeps():
    r = _analyze()
    assert r.status == "ok", (r.status, r.errors)
    bridges = r.data["bridges"]
    assert len(bridges) == 2, len(bridges)
    for b in bridges:
        assert len(b.get("rho_h_curves") or []) == 8, (b["channel"], len(b.get("rho_h_curves") or []))


def test_magnetoresistance_stays_applicable():
    r = _analyze()
    caps = {c["name"]: c["applicable"] for c in r.data["capabilities"]}
    assert caps["magnetoresistance"] is True, caps


def test_recovered_curves_are_genuine_isothermal_field_sweeps():
    """Each recovered curve must hold ONE temperature while the field really sweeps.

    This is what distinguishes a genuine recovery from the block splitter merely slicing a
    ramp into pieces -- the failure mode the span-dominance guard exists to prevent.
    """
    r = _analyze()
    for b in r.data["bridges"]:
        temps = sorted(c["held_temp_k"] for c in b["rho_h_curves"])
        assert len(set(temps)) == len(temps), f"ch{b['channel']}: duplicate held temps {temps}"
        for c in b["rho_h_curves"]:
            fields = [f for f in (c.get("field") or []) if f is not None]
            assert len(fields) >= 5, (b["channel"], c["held_temp_k"], len(fields))
            # a genuine isothermal sweep: the field really moves, at one held temperature
            assert max(fields) - min(fields) > 10000.0, (
                b["channel"], c["held_temp_k"], max(fields) - min(fields))
