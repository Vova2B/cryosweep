"""Shared identifiers must have exactly one spelling.

This repo's recurring structural defect is one fact constructed in several places: the
`model@field` fit-line key was built by two independent f-strings (plot_controls vs render)
and rounding one of them shipped a live bug — unchecking one checkbox dropped all 16 fit
lines. `fmt_field` and `NON_DATA_GIDS` are the precedent for the cure: one function or
constant whose docstring says it IS the source of truth, and a test pinning that every
consumer binds THAT object rather than a private respelling.

The `is`-identity assertions are deliberate: `==` would stay green if a consumer re-typed
the same value locally, which is exactly the defect class this file exists to prevent.
"""
import numpy as np


# ---- Fact A: the hc_lowt_multifield identity keys ('mf:<token>', '<model>@<token>') ----

def test_lowt_key_helpers_exist_and_round_trip():
    from cryosweep_core.plotting.catalog import (
        lowt_field_token, lowt_series_key, lowt_token_from_series_key, lowt_fit_key)
    # Fields that expose formatter drift: display-rounded medians, sci-notation magnitudes.
    for f in (0.524968, 499.9, 50000.5, 40000.887, 100001.0, 1e5, 123456.789, 0.0):
        tok = lowt_field_token(f)
        assert tok == f"{f:g}"                       # the shipped spelling, byte-for-byte
        assert lowt_series_key(f) == f"mf:{tok}"
        assert lowt_token_from_series_key(lowt_series_key(f)) == tok
        # The checkbox path (token recovered from the series key) and the renderer path
        # (token formatted from the raw float) MUST land on the same fit key.
        assert (lowt_fit_key("debye_t3", lowt_token_from_series_key(lowt_series_key(f)))
                == lowt_fit_key("debye_t3", lowt_field_token(f)))


def test_lowt_key_construction_sites_use_the_helpers():
    """No site may rebuild 'mf:...' or '...@...' inline; grep the three known consumers."""
    import inspect
    import cryosweep_core.plotting.catalog as catalog
    import cryosweep_core.plotting.render as render
    import cryosweep_gui.plot_controls as plot_controls
    # exactly ONE spelling of 'mf:' — the helper itself
    assert inspect.getsource(catalog).count('f"mf:') == 1, \
        "catalog spells the series key outside lowt_series_key"
    rsrc = inspect.getsource(render)
    assert "lowt_fit_key(" in rsrc and '@{g[' not in rsrc, "render rebuilds the fit key inline"
    psrc = inspect.getsource(plot_controls)
    assert "lowt_fit_key(" in psrc and '@{fnum}' not in psrc, \
        "plot_controls rebuilds the fit key inline"
    assert "lowt_token_from_series_key(" in psrc, "plot_controls re-splits the series key inline"


# ---- Fact B: the low-T model registry (keys, order, labels, lattice subset) ----

def test_lowt_model_registry_is_exported_by_the_fitting_module():
    from cryosweep_core.fitting import heat_capacity as hcfit
    assert hcfit.LOWT_MODEL_KEYS == ("debye_t3", "debye_t3_t5",
                                     "spin_fluct_noninteracting", "spin_fluct_weak")
    assert set(hcfit.LOWT_MODEL_LABELS) == set(hcfit.LOWT_MODEL_KEYS)
    assert hcfit.LOWT_LATTICE_KEYS == ("debye_t3", "debye_t3_t5")
    # Derived from the _LOWT_MODELS registry itself, not a parallel literal.
    assert hcfit.LOWT_MODEL_KEYS == tuple(m["key"] for m in hcfit._LOWT_MODELS)
    assert hcfit.LOWT_MODEL_LABELS == {m["key"]: m["label"] for m in hcfit._LOWT_MODELS}


def test_lowt_model_consumers_bind_the_registry_objects():
    from cryosweep_core.fitting import heat_capacity as hcfit
    import cryosweep_core.plotting.catalog as catalog
    import cryosweep_core.plotting.render as render
    assert catalog._ALL is hcfit.LOWT_MODEL_KEYS
    assert catalog._LATTICE is hcfit.LOWT_LATTICE_KEYS
    assert catalog._MODEL_LABELS is hcfit.LOWT_MODEL_LABELS
    assert render._LOWT_FIT_KEYS is hcfit.LOWT_MODEL_KEYS
    assert render._LOWT_MODEL_NAMES is hcfit.LOWT_MODEL_LABELS
    # render's model->linestyle is one fact too: _LOWT_MODEL_LS must derive from _LOWT_FIT_STYLE.
    assert render._LOWT_MODEL_LS == {k: ls for k, (_c, ls) in render._LOWT_FIT_STYLE.items()}
    import inspect
    import cryosweep_gui.plot_controls as plot_controls
    import cryosweep_core.analyzers.hc as hc
    import cryosweep_core.io.export as export
    assert "LOWT_MODEL_KEYS" in inspect.getsource(plot_controls)
    assert "LOWT_MODEL_LABELS" in inspect.getsource(plot_controls)
    for mod in (hc, export, catalog):
        assert '("debye_t3", "debye_t3_t5")' not in inspect.getsource(mod), \
            f"{mod.__name__} respells the lattice subset"


# ---- Fact C: the Oe<->T conversion factor ----

def test_oe_per_t_is_single_sourced():
    from cryosweep_core.units import OE_PER_T
    assert OE_PER_T == 10000.0
    from cryosweep_core.analyzers import hall, hall_tempdep
    assert hall._OE_PER_T is OE_PER_T
    assert hall_tempdep._OE_PER_T is OE_PER_T
    import inspect
    import cryosweep_core.fitting.heat_capacity as hcfit
    import cryosweep_core.plotting.catalog as catalog
    import cryosweep_core.plotting.render as render
    for mod, needle in ((hcfit, "/ 1e4"), (catalog, "/ 1e4"), (render, "* 1e4")):
        assert needle not in inspect.getsource(mod), \
            f"{mod.__name__} respells the Oe->T conversion as a literal"


def test_field_koe_threshold_is_not_the_conversion_factor():
    """FIELD_KOE_THRESHOLD_OE (a display threshold) numerically equals OE_PER_T today —
    coincidence, not the same fact. They must stay separate names; merging them would couple
    a display choice to a physical constant."""
    from cryosweep_core.units import OE_PER_T
    from cryosweep_core.plotting.catalog import FIELD_KOE_THRESHOLD_OE
    assert FIELD_KOE_THRESHOLD_OE == OE_PER_T          # documents the coincidence
    assert FIELD_KOE_THRESHOLD_OE is not OE_PER_T      # and pins that they stay distinct


# ---- Fact D: the zero-field convention (|H| < 50 Oe counts as held zero field) ----

def test_zero_field_threshold_is_single_sourced():
    from cryosweep_core.units import ZERO_FIELD_OE
    assert ZERO_FIELD_OE == 50.0
    from cryosweep_core.analyzers import resistivity, tto
    assert resistivity._ZERO_FIELD_OE is ZERO_FIELD_OE
    assert tto._ZERO_FIELD_OE is ZERO_FIELD_OE
    import inspect
    import cryosweep_core.analyzers.acms as acms
    assert "50.0" not in inspect.getsource(acms._detect_sc), \
        "acms respells the zero-field threshold inline"
    assert "ZERO_FIELD_OE" in inspect.getsource(acms)
