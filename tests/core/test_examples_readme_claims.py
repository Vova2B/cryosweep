"""Every number `examples/README.md` advertises must be a number the analyzer actually produces.

Added 2026-09-01 after the README was found claiming that magnetization_mpms.dat, given
200 g/mol and 5 mg, produces "the same numbers as the VSM file". It does not - it is a
different synthetic sample (theta = -30 K, C = 1.5) and the VSM file is theta = -10 K, C = 0.5.
Nothing tied the prose to the output, so the claim could rot silently; a reader running the
example would have concluded the app was broken.

These are DOCUMENTATION tests: they fail when the docs and the code disagree, without saying
which one is wrong. Whichever moved, they have to be reconciled by hand.
"""
import csv
import dataclasses
import pathlib
import pytest

from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.config import RunConfig
from cryosweep_core.io.loader import load_dat
from cryosweep_core.registry import build_default_registry

EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"
README = EXAMPLES / "README.md"


def _analyze(name, **header_overrides):
    rt = load_dat(str(EXAMPLES / name))
    if header_overrides:
        rt = dataclasses.replace(rt, header=dataclasses.replace(rt.header, **header_overrides))
    return analyze_file(rt, RunConfig(), build_default_registry()).data


def _readme():
    return README.read_text(encoding="utf-8")


def test_vsm_example_matches_its_advertised_curie_weiss():
    p = _analyze("magnetization_vsm.dat")["fit"]["params"]
    assert p["theta"] == pytest.approx(-10.0, abs=0.05)
    assert p["C"] == pytest.approx(0.5, abs=0.005)
    assert "theta = -10 K, C = 0.5" in _readme()


def test_mpms_example_is_a_different_sample_and_the_readme_says_so():
    """The exact regression: the README claimed these two files agree. They do not."""
    # 10 mg, not 5: tests/core/fixtures/make_mpms.py:5 builds the file at MASS_MG = 10.0,
    # so only this pair recovers the sample it encodes (5 mg doubles C and scales mu_eff by sqrt(2)).
    mpms = _analyze("magnetization_mpms.dat", molar_mass=200.0, mass_mg=10.0)["fit"]["params"]
    vsm = _analyze("magnetization_vsm.dat")["fit"]["params"]
    assert mpms["theta"] == pytest.approx(-30.0, abs=0.05)
    assert mpms["C"] == pytest.approx(1.5, abs=0.02)
    assert mpms["mu_eff"] == pytest.approx(3.46, abs=0.02)
    assert mpms["theta"] != pytest.approx(vsm["theta"], abs=1.0), "the files disagree by design"
    txt = _readme()
    assert "theta = -30 K, C = 1.5, mu_eff = 3.46" in txt
    assert "same numbers as the VSM file" not in txt, "the disproved claim came back"


def test_mpms_example_gates_without_the_inputs_the_readme_names():
    """The README tells the reader to supply 200 g/mol and 10 mg. That must be why it gates."""
    res = analyze_file(load_dat(str(EXAMPLES / "magnetization_mpms.dat")),
                       RunConfig(), build_default_registry())
    assert res.status == "gated"
    assert {g.need for g in res.gate} == {"molar_mass", "sample_mass"}
    assert "200 g/mol, 10 mg" in _readme()


def test_heat_capacity_example_matches_its_advertised_gamma_and_debye_temperature():
    f = _analyze("heat_capacity.dat")["fit"]   # the low-T Cp/T vs T^2 fit
    assert f["params"]["gamma"] == pytest.approx(0.01, abs=0.0005)
    assert f["params"]["theta_D"] == pytest.approx(227, abs=2)
    assert "gamma = 0.01, theta_D ~ 227 K" in _readme()


def test_acms_example_matches_its_advertised_tc():
    scs = [c["sc"] for c in _analyze("ac_susceptibility.dat")["curves"] if c.get("sc")]
    assert scs, "the screening step must be detected"
    assert scs[0]["tc_mid_k"] == pytest.approx(5.0, abs=0.1)
    assert "Tc mid 5.0 K" in _readme()


def test_resistivity_example_matches_its_advertised_tc_and_rrr(tmp_path):
    """Read the shipped CSV surface, which is what a user actually consumes."""
    from cryosweep_cli.__main__ import main
    stem = tmp_path / "r"
    main(["export", str(EXAMPLES / "resistivity_superconductor.dat"), "--out", str(stem)])
    rows = list(csv.DictReader(open(f"{stem}.derived.csv")))
    ch1 = next(r for r in rows if r.get("tc_mid_k"))
    assert float(ch1["tc_onset_k"]) == pytest.approx(8.8, abs=0.1)
    assert float(ch1["tc_mid_k"]) == pytest.approx(8.0, abs=0.1)
    assert float(ch1["tc_zero_k"]) == pytest.approx(7.5, abs=0.1)
    assert float(ch1["rrr"]) == pytest.approx(86.7, abs=1.0)
    assert "onset 8.8 / mid 8.0 / zero 7.5 K" in _readme()


def test_every_shipped_example_has_a_readme_section():
    txt = _readme()
    missing = [p.name for p in sorted(EXAMPLES.glob("*.dat")) if f"## {p.name}" not in txt]
    assert not missing, f"undocumented example file(s): {missing}"
