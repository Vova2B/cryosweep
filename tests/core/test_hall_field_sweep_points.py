# tests/core/test_hall_field_sweep_points.py
from cryosweep_core.io.loader import load_dat
from cryosweep_core.io.columns import canonicalize_columns
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.hall import HallAnalyzer, field_sweep_points, _long_rho_xx


def test_field_sweep_points_matches_analyze(hall_synth_path):
    rt = load_dat(hall_synth_path)
    cfg = RunConfig(hall={"hall_channel": 1, "thickness_mm": 0.5})
    res = HallAnalyzer().analyze(rt, cfg)
    pts_via_analyze = res.data["points"]
    df, cmap = canonicalize_columns(rt.df, rt.header)
    hc = cfg.hall
    # task 4b: HallAnalyzer.analyze() drops the first `hc.skip_rows` rows (default 1)
    # BEFORE calling field_sweep_points -- replicate that step here so this equivalence
    # check still compares like for like, rather than the pre-skip dataframe against the
    # post-skip result.
    df = df.iloc[hc.skip_rows:].reset_index(drop=True)
    rho_fn, rho_reason = _long_rho_xx(df, cmap, hc.longitudinal_channel, None, None, cfg)
    pts = field_sweep_points(df, cmap, cfg, hc, hc.thickness_mm * 1e-3, rho_fn, rho_reason)
    assert [p.model_dump() for p in pts] == pts_via_analyze
