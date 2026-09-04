"""TTO detection + canonicalization (spec §1). Every literal here was measured against the
real Thermal Transport file; do not loosen one without re-measuring."""
import pytest

from cryosweep_core.io.columns import canonicalize_columns, _norm
from cryosweep_core.io.loader import load_dat
from cryosweep_core.registry import build_default_registry

def _cols_and_map(path):
    rt = load_dat(str(path))
    df, cmap = canonicalize_columns(rt.df, rt.header)
    return rt, df, cmap


def test_tto_detector_scores_one_on_real_file(tto_real_path):
    rt, df, _ = _cols_and_map(tto_real_path)
    from cryosweep_core.detect.probe import TTODetector
    assert TTODetector.key == "tto"
    assert TTODetector.matches(rt.header, set(df.columns)) == 1.0


def test_every_other_builtin_detector_scores_zero_on_the_real_file(tto_real_path):
    # Ranking regression: auto-detection must route TTO files to `tto`, unambiguously.
    rt, df, _ = _cols_and_map(tto_real_path)
    reg = build_default_registry()
    scores = {d.key: d.matches(rt.header, set(df.columns)) for d in reg.detectors()}
    assert scores["tto"] == 1.0
    assert all(v == 0.0 for k, v in scores.items() if k != "tto"), scores


def test_strong_fingerprint_alone_scores_without_byapp():
    # A hypothetical BYAPP-less TTO export must still be recognised at 0.8.
    from cryosweep_core.detect.probe import TTODetector
    from cryosweep_core.model import HeaderMeta
    h = HeaderMeta(app=None, app_version=None, title=None, info={}, info_rows=(),
                   channels={}, molar_mass=None, n_atoms=None, mass_mg=None,
                   data_line=0, raw_lines=(), bare_csv=False)
    cols = {"Sample Temp. (K)", "Conductivity (W/K-m)", "Figure of Merit ZT"}
    assert TTODetector.matches(h, cols) == pytest.approx(0.8)


def test_strong_fingerprint_needs_BOTH_fragments_not_just_one():
    # Specificity, not mere presence: reducing `strong_fingerprint` to a single token survived
    # the suite, because every positive case carries both fragments. A κ column on its own is
    # not a TTO file (a heat-capacity or bare thermal export could carry one), so it must NOT
    # collect the +0.6.
    from cryosweep_core.detect.probe import TTODetector
    from cryosweep_core.model import HeaderMeta
    h = HeaderMeta(app=None, app_version=None, title=None, info={}, info_rows=(),
                   channels={}, molar_mass=None, n_atoms=None, mass_mg=None,
                   data_line=0, raw_lines=(), bare_csv=False)
    assert TTODetector.matches(h, {"Conductivity (W/K-m)"}) == pytest.approx(0.2)
    assert TTODetector.matches(h, {"Figure of Merit ZT"}) == pytest.approx(0.0)


def test_detector_fingerprints_are_spelled_the_raw_way():
    # The raw name carries a micro sign; `_norm` strips it. A fingerprint literal spelled the
    # `_norm` way would be silently dead (matches nothing) and the ranking test would pass
    # vacuously. Pin the asymmetry.
    assert _norm("Seebeck Coef. (µV/K)") == "seebeck coef. (v/k)"
    assert "seebeck coef. (v/k)" not in "Seebeck Coef. (µV/K)".lower()
    assert "seebeck coef." in "Seebeck Coef. (µV/K)".lower()
    # ...and exercise TTODetector itself, so re-spelling the weak fingerprint the `_norm` way
    # actually breaks a test. Without this the ranking test stays green on a dead literal:
    # `_fp` is any-of, and "conductivity (w/k-m)" alone would carry the +0.2.
    from cryosweep_core.detect.probe import TTODetector
    from cryosweep_core.model import HeaderMeta
    h = HeaderMeta(app=None, app_version=None, title=None, info={}, info_rows=(),
                   channels={}, molar_mass=None, n_atoms=None, mass_mg=None,
                   data_line=0, raw_lines=(), bare_csv=False)
    assert TTODetector.matches(h, {"Seebeck Coef. (µV/K)"}) == pytest.approx(0.2)


def test_canonicalizes_the_eight_tto_columns(tto_real_path):
    _, _, cmap = _cols_and_map(tto_real_path)
    assert cmap.logical["kappa"] == "Conductivity (W/K-m)"
    assert cmap.unit["kappa"] == "W/K-m"
    assert cmap.logical["kappa_std"] == "Cond. Std.Dev."
    assert cmap.logical["seebeck"] == "Seebeck Coef. (µV/K)"
    assert cmap.unit["seebeck"] == "uV/K"
    assert cmap.logical["seebeck_std"] == "Seebeck Std.Dev."
    assert cmap.logical["rho_tto"] == "Resistivity (Ohm-m)"
    assert cmap.unit["rho_tto"] == "Ohm-m"
    assert cmap.logical["rho_tto_std"] == "Resist Std.Dev."
    assert cmap.logical["zt"] == "Figure of Merit ZT"
    assert cmap.unit["zt"] == "1"
    assert cmap.logical["zt_std"] == "Merit Std.Dev."


def test_sample_temp_dot_k_now_maps_to_temperature(tto_real_path):
    # Trap 1: before this slice the TTO file canonicalized with NO temperature mapping.
    _, _, cmap = _cols_and_map(tto_real_path)
    assert cmap.logical["temperature"] == "Sample Temp. (K)"
    assert cmap.unit["temperature"] == "K"
    assert cmap.logical["field"] == "Magnetic Field (Oe)"


def test_resistivity_ch1_mapping_on_the_tto_file_is_unchanged(tto_real_path):
    # Trap 2: `Resistivity (Ohm-m)` already maps to resistivity_ch1 via the generic
    # ^resistivity rule. rho_tto is a SECOND, additive logical name for the same column.
    _, _, cmap = _cols_and_map(tto_real_path)
    assert cmap.logical["resistivity_ch1"] == "Resistivity (Ohm-m)"
    assert cmap.unit["resistivity_ch1"] == "Ohm-m"


def test_temp_variant_addition_does_not_touch_hc_family_files(hc_path):
    # Byte-identity guard for the _TEMP addition: the HC-family files use
    # `Sample Temp (Kelvin)` (no dot, spelled "Kelvin"), so they are unaffected.
    rt = load_dat(str(hc_path))
    _, cmap = canonicalize_columns(rt.df, rt.header)
    assert cmap.logical["temperature"] == "Sample Temp (Kelvin)"
