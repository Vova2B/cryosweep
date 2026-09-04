import numpy as np, pandas as pd
from cryosweep_core.io.loader import load_dat
from cryosweep_core.analyzers.acms import _cluster_1d, group_rows


def _cols(rt):
    df = rt.df
    def f(n): return next(c for c in df.columns if n.lower() in c.lower())
    g = lambda n: pd.to_numeric(df[f(n)], errors="coerce").to_numpy(float)
    return g("Frequency (Hz)"), g("Amplitude (Oe)"), g("Magnetic Field (Oe)")


def test_cluster_1d_merges_within_reltol_splits_beyond():
    vals = np.array([0.0498, 0.0500, 0.1000, 0.3013])
    labels, reps = _cluster_1d(vals, 0.05)
    assert labels.tolist() == [0, 0, 1, 2]
    assert reps[0] == 0.0499  # median of the two members


def test_field_abs_tol_near_zero():
    labels, reps = _cluster_1d(np.array([-0.096, -0.05, 0.02, 1000.0]), 0.01, abs_tol=5.0)
    assert labels.tolist() == [0, 0, 0, 1]


def test_real_file_yields_four_raw_groups(acms_real_path):
    freq, amp, field = _cols(load_dat(acms_real_path))
    m = np.isfinite(freq) & (amp > 0)
    groups = group_rows(freq[m], amp[m], field[m])
    amps = [round(g["amplitude_oe"], 4) for g in groups]
    sizes = [g["idx"].size for g in groups]
    assert amps == [0.0498, 0.1, 0.3013, 0.4979]
    assert sizes == [1953, 186, 186, 1]
    assert all(round(g["frequency_hz"]) == 477 for g in groups)
