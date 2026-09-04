import numpy as np, pandas as pd
from scipy.stats import linregress
from cryosweep_core.io.loader import load_dat
from cryosweep_core.analyzers.hc import HCAnalyzer
from cryosweep_core.fitting.heat_capacity import debye_temp_from_beta
from cryosweep_core.config import RunConfig

def test_hc_lowt_matches_independent_linregress(hc_path):
    rt = load_dat(hc_path); df = rt.df
    cfg = RunConfig.load()
    # independent ground truth: read the RAW columns by their literal names (NOT via cmap,
    # so this oracle can't share a canonicalization bug with the analyzer)
    T = pd.to_numeric(df["Sample Temp (Kelvin)"], errors="coerce").to_numpy(float)
    cp = pd.to_numeric(df["Samp HC (mJ/mole-K)"], errors="coerce").to_numpy(float) * 1e-3
    F = pd.to_numeric(df["Field (Oersted)"], errors="coerce").to_numpy(float)
    # match the analyzer's selection: lowest-|field| group (zero-field Debye), |field| binned
    # to 1 kOe, then the low-T (<=10 K) subset.
    m = np.isfinite(T) & np.isfinite(cp) & np.isfinite(F) & (T > 0)
    bins = np.round(np.abs(F[m]) / 1000.0)
    grp = bins == bins.min()
    Tk, Ck = T[m][grp], cp[m][grp]
    assert abs(np.median(F[m][grp])) < 1000.0                 # it IS the ~zero-field group
    low = Tk <= 10.0
    Tk, Ck = Tk[low], Ck[low]
    r = linregress(Tk**2, Ck/Tk)
    n_atoms = rt.header.n_atoms or 1.0
    exp_theta = debye_temp_from_beta(r.slope, n_atoms)
    # analyzer must reproduce it
    res = HCAnalyzer().analyze(rt, cfg)
    p = res.data["fit"]["params"]
    assert abs(p["gamma"] - r.intercept) < 1e-9
    assert abs(p["beta"] - r.slope) < 1e-12
    assert abs(p["theta_D"] - exp_theta) < 1e-6
    # sanity: a real physical Debye temperature (tens to hundreds of K)
    assert 5.0 < p["theta_D"] < 1000.0
