import numpy as np, pandas as pd
from types import SimpleNamespace
from cryosweep_core.analyzers.hc import HCAnalyzer
from cryosweep_core.config import RunConfig
from cryosweep_core.fitting.heat_capacity import specific_heat_full
from cryosweep_core.reports import build_report

class _Hdr: title="s"; app_version=None; n_atoms=3.0

def test_hc_report_has_all_sections():
    T = np.linspace(2.0, 300.0, 150)
    cp = specific_heat_full(T, 220.0, 3.0, 0.01, 60.0, 180.0, 1.0, 1.5)
    df = pd.DataFrame({"Sample Temp (Kelvin)": T, "Samp HC (mJ/mole-K)": cp*1e3,
                       "Field (Oe)": np.zeros_like(T)})
    r = HCAnalyzer().analyze(SimpleNamespace(df=df, header=_Hdr(), path=None), RunConfig())
    md = build_report(r)["markdown"]
    assert "Low-T fits" in md and "Full-range" in md and "Comparison" in md


def test_report_has_field_dependence_section():
    import pathlib
    from cryosweep_core.io.loader import load_dat
    from cryosweep_core.config import RunConfig
    from cryosweep_core.analyzers.dispatch import analyze_file
    from cryosweep_core.registry import build_default_registry
    from cryosweep_core.reports import build_report
    FIX = pathlib.Path(__file__).parent / "fixtures"
    res = analyze_file(load_dat(str(FIX / "hc_multifield_synth.dat")),
                       RunConfig.load(), build_default_registry())
    md = build_report(res)["markdown"]
    assert "## Field dependence" in md
