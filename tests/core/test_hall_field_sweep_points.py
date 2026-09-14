# tests/core/test_hall_field_sweep_points.py
from cryosweep_core.io.loader import load_dat
from cryosweep_core.io.columns import canonicalize_columns
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.hall import HallAnalyzer, field_sweep_points, _long_rho_xx
from cryosweep_core.analyzers.hall_sigma import resolve_skip_rows


def test_field_sweep_points_matches_analyze(hall_synth_path):
    rt = load_dat(hall_synth_path)
    cfg = RunConfig(hall={"hall_channel": 1, "thickness_mm": 0.5})
    res = HallAnalyzer().analyze(rt, cfg)
    pts_via_analyze = res.data["points"]
    df, cmap = canonicalize_columns(rt.df, rt.header)
    hc = cfg.hall
    # HallAnalyzer.analyze() drops leading rows BEFORE calling field_sweep_points --
    # replicate that step here so this equivalence check still compares like for like,
    # rather than the pre-skip dataframe against the post-skip result. Through the SAME
    # resolver, never a local re-reading of the flag: `skip_rows` defaults to "auto", so a
    # hand-rolled `df.iloc[hc.skip_rows:]` is not merely duplicated, it is a TypeError.
    n_skip, _ = resolve_skip_rows(hc.skip_rows, df, cmap)
    df = df.iloc[n_skip:].reset_index(drop=True)
    rho_fn, rho_reason = _long_rho_xx(df, cmap, hc.longitudinal_channel, None, None, cfg)
    pts = field_sweep_points(df, cmap, cfg, hc, hc.thickness_mm * 1e-3, rho_fn, rho_reason)
    assert [p.model_dump() for p in pts] == pts_via_analyze
