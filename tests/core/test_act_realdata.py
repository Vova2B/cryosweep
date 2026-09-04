import pathlib
import numpy as np
import pytest
from cryosweep_core.io.loader import load_dat
from cryosweep_core.io.columns import canonicalize_columns
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.detect.probe import detect_probe
from cryosweep_core.registry import build_default_registry

def _bridge(res, ch):
    return next((b for b in res.data["bridges"] if b["channel"] == ch), None)

def test_real_act_resistivity(act_real_path):
    if not pathlib.Path(act_real_path).exists():
        pytest.skip("real ACT file not present (gitignored)")
    rt = load_dat(str(act_real_path)); df, _ = canonicalize_columns(rt.df, rt.header)
    score, key = detect_probe(rt.header, set(df.columns), build_default_registry())
    assert key == "resistivity" and score >= 0.5
    res = analyze_file(rt, RunConfig.load(), build_default_registry())
    assert res.status in ("ok", "low_confidence")          # actual: ok (conf ~0.99)
    assert res.data["rho_source"] == "instrument_column"
    b1, b2 = _bridge(res, 1), _bridge(res, 2)
    assert b1 is not None and b2 is not None
    # strict unit-range: rho values in Ohm-cm (a x100 bug -> ~3e-2). Actual ch1 ~3-4e-4.
    rho1 = np.array([r for c in b1["rho_t_curves"] for r in c["rho"]], float)
    assert 1e-5 < float(np.nanmax(rho1)) < 1e-2
    assert float(np.nanmin(rho1)) > 0
    # ch2 metallic (provisional band, tighten with paper values). Actual RRR ~4.7-4.9.
    # NOTE: ch1 is non-monotonic (rho_max near ~25 K) -> RRR<1, classified insulating; do NOT assert ch1 metallic.
    assert b2["rrr"] is not None and b2["rrr"] > 1.0
    # no superconducting collapse in 1.82-340 K
    rho2 = np.array([r for c in b2["rho_t_curves"] for r in c["rho"]], float)
    assert float(np.nanmin(rho2)) > 1e-6
