# tests/core/test_render_cp_vs_t.py
import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd
from types import SimpleNamespace
from cryosweep_core.analyzers.hc import HCAnalyzer
from cryosweep_core.config import RunConfig
from cryosweep_core.fitting.heat_capacity import specific_heat_full
from cryosweep_core.plotting.render import render_kind

class _Hdr: title="s"; app_version=None; n_atoms=3.0

def _result():
    T = np.linspace(2.0, 300.0, 150)
    cp = specific_heat_full(T, 220.0, 3.0, 0.01, 60.0, 180.0, 1.0, 1.5)
    df = pd.DataFrame({"Sample Temp (Kelvin)": T, "Samp HC (mJ/mole-K)": cp*1e3,
                       "Field (Oe)": np.zeros_like(T)})
    return HCAnalyzer().analyze(SimpleNamespace(df=df, header=_Hdr(), path=None), RunConfig())

def test_render_cp_vs_t_has_model_line():
    fig = render_kind(_result(), "cp_vs_t")
    ax = fig.axes[0]
    # data points + a model fit line (gid='fit')
    assert any(getattr(l, "get_gid", lambda: None)() == "fit" for l in ax.get_lines())
