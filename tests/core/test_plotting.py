import pathlib
import pytest
from cryosweep_core.analyzers.mag import VSMAnalyzer
from cryosweep_core.io.loader import load_dat
from cryosweep_core.plotting.spec import PlotSpec
from cryosweep_core.plotting.render import render_inverse_chi
from cryosweep_core.result import Result, Provenance
from cryosweep_core.config import RunConfig

FIX = pathlib.Path(__file__).parent / "fixtures" / "vsm_synth.dat"


def test_render_inverse_chi_no_sweep_raises_valueerror():
    # Bug 6: low_confidence no-sweep result lacks temperature/inv_chi -> clear ValueError
    res = Result(status="low_confidence", confidence=0.2,
                 data={"probe": "vsm", "reason": "no temperature sweep found"},
                 provenance=Provenance(file="x", sha256="0", app_version=None))
    with pytest.raises(ValueError):
        render_inverse_chi(res, PlotSpec())

def test_render_inverse_chi(tmp_path):
    rt = load_dat(FIX); res = VSMAnalyzer().analyze(rt, RunConfig.load())
    fig = render_inverse_chi(res, PlotSpec())
    assert fig.get_axes()
    ax = fig.get_axes()[0]
    assert len(ax.lines) >= 1                      # data + fit
    out = tmp_path / "p.png"; fig.savefig(out); assert out.stat().st_size > 1000

def test_plotspec_defaults():
    from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle
    p = PlotSpec()
    assert p.curves is None and p.fit_line is True and p.xscale is None
    st = GlobalStyle()
    assert st.width_mm > 0 and st.font_pt > 0
    assert "width_mm" in GlobalStyle.model_json_schema()["properties"]
