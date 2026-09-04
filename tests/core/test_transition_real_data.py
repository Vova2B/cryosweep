import pytest
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.hc import HCAnalyzer


def _tch(path, form):
    cfg = RunConfig(heatcapacity={"transitions_enabled": True, "transition_form": form})
    res = HCAnalyzer().analyze(load_dat(str(path)), cfg)
    d = res.data
    return (d.tc_h if hasattr(d, "tc_h") else d["tc_h"]), res


def test_real_multifield_hc_recovers_field_suppressed_tn(hc_fields_path):
    tch, _ = _tch(hc_fields_path, "lambda")
    assert len(tch) >= 5                                   # >=5 of 6 groups determined
    z = min(tch, key=lambda p: abs(p["field_oe"]))
    assert abs(z["Tc"] - 203.0) <= 10.0
    by_h = sorted(tch, key=lambda p: abs(p["field_oe"]))
    tcs = [p["Tc"] for p in by_h]
    assert all(tcs[i + 1] <= tcs[i] + 0.5 for i in range(len(tcs) - 1))  # monotone decline (0.5 K slack)


@pytest.mark.parametrize("form", ["lambda", "jump"])
def test_featureless_declines_all_groups(form, hc_lowt_path):
    tch, _ = _tch(hc_lowt_path, form)
    assert tch == []
