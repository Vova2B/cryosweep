"""KNOWN-ISSUES #7 (owner decision, 2026-09-05): large fields label in kOe, unit system
unchanged — `90 kOe`, not `90000 Oe` and not `9 T`.

The rule lives in ONE place, `fmt_field`'s Oe path: five call sites used to bypass it with
an inline `... if unit == "T" else f"...Oe"` ternary, so a fix to `fmt_field` alone would
have produced legends half in kOe and half in raw Oe. These tests therefore pin the OUTPUT
of every formerly-bypassing site, not just `fmt_field` itself.
"""
import pathlib
import pytest
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.registry import build_default_registry
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.plotting import catalog as C
from cryosweep_core.result import Result, Provenance

ROOT = pathlib.Path(__file__).resolve().parents[2]
_REG = build_default_registry()


def _analyzed(path, **kw):
    return analyze_file(load_dat(str(ROOT / path)), RunConfig.load(**kw), _REG)


# ---------------- the rule itself (single source of truth) --------------------------------

@pytest.mark.parametrize("oe,label", [
    (90000, "90 kOe"), (100000, "100 kOe"), (130000, "130 kOe"), (45500, "45.5 kOe"),
    (10000, "10 kOe"), (-90000, "-90 kOe"),
    (9999, "9999 Oe"), (1000, "1000 Oe"), (500, "500 Oe"), (0, "0 Oe"),
])
def test_fmt_field_koe_rule(oe, label):
    assert C.fmt_field(oe) == label


def test_threshold_is_a_named_constant_with_the_low_field_rationale():
    # 10 kOe leaves the low-field regime (Curie-Weiss fits, the MPMS 1000 Oe oracle)
    # untouched; a bare literal would invite drive-by retuning.
    assert C.FIELD_KOE_THRESHOLD_OE == 10000.0


def test_tesla_path_is_unchanged():
    assert C.fmt_field(90000, "T") == "9 T"
    assert C.fmt_field(500, "T") == "0.05 T"
    assert C.fmt_field(137000, "T") == "13.7 T"


# ---------------- the five formerly-bypassing sites ---------------------------------------

def test_site_fmt_field_setpoint_hc_multifield_legend():
    # catalog fmt_field_setpoint (site 1), via series_hc_lowt_multifield
    res = _analyzed("examples/heat_capacity_multifield.dat")
    labels = {s.label for s in C.series_hc_lowt_multifield(res)}
    assert labels == {"0 Oe", "50 kOe", "100 kOe", "130 kOe"}


def test_site_vsm_ramp_split_legend():
    # _split_from_tblocks (site 2), via the multifield VSM example's inverse_chi labels
    res = _analyzed("examples/magnetization_vsm_multifield.dat",
                    molar_mass=200.0, mass_mg=5.0)
    labels = [s.label for s in C.series_inverse_chi(res)]
    assert "100 kOe ↑" in labels and "40 kOe ↓" in labels
    assert "100 Oe ↑" in labels and "5000 Oe ↓" in labels    # below threshold: untouched
    import re
    assert not any(re.search(r"\d{5,} Oe", l) for l in labels)   # no 5+ digit Oe label left


def _rho_result(held_field_oe):
    curves = [{"temperature": [10.0, 200.0], "rho": [1.0, 2.0],
               "held_field_oe": held_field_oe, "direction": 1}]
    data = {"probe": "resistivity", "rho_source": "instrument_column",
            "bridges": [{"channel": 1, "rho_source": "instrument_column",
                         "classification": "metallic",
                         "rho_t_curves": curves, "rho_h_curves": []}],
            "capabilities": []}
    return Result(status="ok", data=data,
                  provenance=Provenance(file="x", sha256="ab", app_version=None))


@pytest.mark.parametrize("builder", [C.series_resistivity_rho_t, C.series_resistivity_rho_t2])
def test_sites_resistivity_held_field_labels(builder):
    # series_resistivity_rho_t / _rho_t2 (sites 3 and 4)
    assert [s.label for s in builder(_rho_result(90000.0))] == ["90 kOe"]
    assert [s.label for s in builder(_rho_result(500.0))] == ["500 Oe"]


def test_site_tto_field_ls_label():
    # tto_field_ls_label (site 5) + the TTO curve labels that share the convention
    assert C.tto_field_ls_label(90000.0) == "90 kOe"
    assert C.tto_field_ls_label(500.0) == "500 Oe"
    res = _analyzed("examples/thermal_transport.dat")
    labels = [s.label for s in C.series_tto_kappa_t(res)]
    assert "90 kOe, cooling" in labels and "0 Oe, cooling" in labels
