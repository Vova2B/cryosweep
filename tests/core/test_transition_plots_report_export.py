import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
from cryosweep_core.result import Result, Provenance
from cryosweep_core.plotting import catalog
from cryosweep_core.plotting.catalog import get_kind
from cryosweep_core.plotting import render
from cryosweep_core.reports import build_report
from cryosweep_core.io.export import export_result


def _prov():
    return Provenance(file="x", sha256="ab", app_version=None)


def _result(enabled):
    from cryosweep_core.fitting import transitions as tr
    T = np.linspace(2.0, 30.0, 60)
    C = tr.background(T, 0.01, 2e-4) + tr.lambda_anomaly(T, 10.0, 0.110, 0.05, 0.08)
    g = tr.fit_transition(T, C, form="lambda", universality="ising3d")
    fg = [{"field_oe": 0.0, "status": "ok", "transition": g, "fits": [],
           "n_lowt": 0, "is_primary": True},
          {"field_oe": 10000.0, "status": "ok", "transition": g, "fits": [],
           "n_lowt": 0, "is_primary": False}]
    return Result(status="ok", data={"probe": "heatcapacity", "transitions_enabled": enabled,
                                      "field_groups": fg}, provenance=_prov())


def test_series_empty_when_disabled():
    r = _result(enabled=False)
    assert catalog.series_hc_tc_vs_field(r) == []
    assert catalog.series_hc_transition_multifield(r) == []
    assert catalog.series_hc_transition_signal(r) == []


def test_series_present_when_enabled():
    r = _result(enabled=True)
    assert catalog.series_hc_tc_vs_field(r)               # >=1 series
    assert catalog.series_hc_transition_multifield(r)
    assert catalog.series_hc_transition_signal(r)


def test_series_empty_when_single_group():
    from cryosweep_core.fitting import transitions as tr
    T = np.linspace(2.0, 30.0, 60)
    C = tr.background(T, 0.01, 2e-4) + tr.lambda_anomaly(T, 10.0, 0.110, 0.05, 0.08)
    g = tr.fit_transition(T, C, form="lambda", universality="ising3d")
    fg = [{"field_oe": 0.0, "status": "ok", "transition": g}]
    r = Result(status="ok", data={"probe": "heatcapacity", "transitions_enabled": True,
                                  "field_groups": fg}, provenance=_prov())
    assert catalog.series_hc_tc_vs_field(r) == []


def test_plotkinds_registered():
    for k in ("hc_tc_vs_field", "hc_transition_multifield", "hc_transition_signal"):
        assert get_kind(k).probe == "heatcapacity"
    assert get_kind("hc_transition_multifield").group_colored is True
    assert get_kind("hc_transition_signal").group_colored is True


def test_render_transition_smoke():
    r = _result(enabled=True)
    for fn in (render.render_hc_tc_vs_field,
               render.render_hc_transition_multifield,
               render.render_hc_transition_signal):
        fig = fn([r])
        assert fig is not None
        assert len(fig.axes) >= 1


def test_report_has_transition_section_when_enabled():
    r = _result(enabled=True)
    rep = build_report(r)
    assert "Transitions (opt-in)" in rep["markdown"]


def test_report_no_transition_section_when_disabled():
    r = _result(enabled=False)
    rep = build_report(r)
    assert "Transitions (opt-in)" not in rep["markdown"]


def test_export_writes_transitions_csv(tmp_path):
    r = _result(enabled=True)
    stem = tmp_path / "out"
    files = export_result(r, stem)
    assert "transitions" in files
    assert pathlib.Path(files["transitions"]).exists()
    text = pathlib.Path(files["transitions"]).read_text()
    assert "field_oe" in text and "Tc" in text


def test_export_no_transitions_csv_when_disabled(tmp_path):
    r = _result(enabled=False)
    stem = tmp_path / "out"
    files = export_result(r, stem)
    assert "transitions" not in files
    assert not pathlib.Path(str(stem) + ".transitions.csv").exists()
