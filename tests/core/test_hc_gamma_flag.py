"""KNOWN-ISSUES item 9 — a flagged-unphysical γ must say so ON THE FIGURE.

γ < 0 is the measured value, so it is shown, not blanked (this is not the decline case);
the figure — the artifact that travels into a talk without the status bar — must carry
the same verdict the analyzer's warnings already state. The channel is the established
one: a machine-readable `gamma_negative` in the fit's `quality_flags`, which the
annotation reads. The prose warnings in hc.py are unchanged.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pytest

from cryosweep_core.fitting.heat_capacity import fit_lowt_models
from cryosweep_core.plotting.render import render_kind
from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.registry import build_default_registry
from cryosweep_core.analyzers.dispatch import analyze_file

EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"
_REG = build_default_registry()
_RESULTS = {}


def _result(fname):
    if fname not in _RESULTS:
        _RESULTS[fname] = analyze_file(load_dat(str(EXAMPLES / fname)), RunConfig.load(), _REG)
    return _RESULTS[fname]


def _lowt_annotation(fig):
    texts = [t for ax in fig.axes for t in ax.texts if "γ = " in t.get_text()]
    assert len(texts) == 1, "expected exactly one low-T annotation"
    return texts[0].get_text()


# ---------------- the flag, at the fit level ----------------

def _synthetic(gamma):
    T = np.linspace(2.0, 10.0, 40)
    beta = 2.0e-4
    cp = (gamma + beta * T ** 2) * T          # exact debye_t3 form
    return T, cp


def test_negative_gamma_sets_the_quality_flag():
    T, cp = _synthetic(gamma=-0.008)
    out = fit_lowt_models(T, cp, n_atoms=1.0)
    fr = out["chosen"]
    assert fr is not None and fr.params["gamma"] < 0
    assert "gamma_negative" in fr.quality_flags


def test_positive_gamma_leaves_the_flag_clear():
    T, cp = _synthetic(gamma=+0.008)
    out = fit_lowt_models(T, cp, n_atoms=1.0)
    fr = out["chosen"]
    assert fr is not None and fr.params["gamma"] > 0
    assert "gamma_negative" not in fr.quality_flags


# ---------------- the figure says it ----------------

@pytest.mark.parametrize("kind", ["cp_over_t", "hc_c_over_t_linear"])
def test_reproducer_annotation_says_unphysical(kind):
    # heat_capacity_multifield.dat: γ = -8.3e-03, both prose warnings fire, and before this
    # fix the annotation printed the number plainly. Both users of _hc_lowt_annotation.
    fig = render_kind([_result("heat_capacity_multifield.dat")], kind,
                      PlotSpec(), GlobalStyle())
    text = _lowt_annotation(fig)
    gamma_line = text.splitlines()[0]
    assert "γ = " in gamma_line and "(unphysical)" in gamma_line, gamma_line
    plt.close(fig)


def test_healthy_gamma_annotation_is_untagged():
    fig = render_kind([_result("heat_capacity.dat")], "cp_over_t", PlotSpec(), GlobalStyle())
    text = _lowt_annotation(fig)
    assert "unphysical" not in text, text
    plt.close(fig)
