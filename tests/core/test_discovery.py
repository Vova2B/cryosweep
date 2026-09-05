from cryosweep_core.discovery import discover
from cryosweep_core.registry import build_default_registry


def test_discover_shapes():
    d = discover(build_default_registry())
    assert {"probes", "fits", "plots", "observables"} <= set(d)
    assert any(p["key"] == "vsm" for p in d["probes"])
    cw = next(f for f in d["fits"] if f["key"] == "curie_weiss")
    assert cw["params"] == ["C", "theta", "mu_eff"]
    # probe carries its analyzer's needs (typed Need -> dict)
    vsm = next(p for p in d["probes"] if p["key"] == "vsm")
    assert any(n["key"] == "molar_mass" for n in vsm["needs"])
    # all three plot keys are discoverable
    plot_keys = {pl["key"] for pl in d["plots"]}
    assert "inverse_chi" in plot_keys
    assert "cp_over_t" in plot_keys
    assert "resistivity_rho_t" in plot_keys


def test_registry_exposes_all_kinds():
    r = build_default_registry()
    assert "vsm" in r.detector_keys()
    assert "vsm" in r.analyzer_keys()
    assert "curie_weiss" in r.fitmodel_keys()
    a = r.get_analyzer("vsm")
    assert a is not None
    f = r.get_fitmodel("curie_weiss")
    assert f.params == ["C", "theta", "mu_eff"]
    # accessors never KeyError on unknown
    assert r.get_analyzer("nope") is None and r.get_fitmodel("nope") is None


def test_plots_are_registry_driven_per_probe():
    from cryosweep_core.discovery import discover
    from cryosweep_core.registry import build_default_registry
    plots = discover(build_default_registry())["plots"]
    by_key = {p["key"]: p for p in plots}
    # legacy keys preserved
    assert {"inverse_chi", "cp_over_t", "resistivity_rho_t"} <= set(by_key)
    # new shape: each entry carries label/probe/scales
    assert by_key["resistivity_rho_t"]["probe"] == "resistivity"
    assert by_key["resistivity_rho_t"]["default_yscale"] == "linear"   # D3: ρ(T) headline is linear
    assert by_key["hall_n_t"]["default_yscale"] == "log"   # non-default scale still flows through discovery
    assert by_key["hall_rh_t"]["probe"] == "hall"          # hall now present (was missing in the stub)


def test_plots_sorted_and_includes_sp1_kinds():
    d = discover(build_default_registry())
    keys = [p["key"] for p in d["plots"]]
    assert keys == sorted(keys)                      # deterministic ordering
    expected = {"vsm_chi_t_product", "resistivity_mr_pct", "resistivity_rho_t2",
                "hall_n_t", "hall_r2_t", "hc_c_over_t_linear"}
    assert expected <= set(keys)


# ---- KNOWN-ISSUES #10: discovery commands become file-aware ------------------------------
# `cryosweep plots <file>` used to ignore the file entirely — probes/fits/plots/observables
# all emitted the byte-identical global registry dump, so `plots` on a resistivity file
# listed ACMS kinds and there was no way to ask which kinds a file can actually draw.

def _analyzed(path):
    from cryosweep_core.io.loader import load_dat
    from cryosweep_core.config import RunConfig
    from cryosweep_core.analyzers.dispatch import analyze_file
    return analyze_file(load_dat(path), RunConfig.load(), build_default_registry())


def test_discover_for_filters_plots_to_the_detected_probe():
    from cryosweep_core.discovery import discover_for
    res = _analyzed("examples/resistivity_semiconductor.dat")
    d = discover_for(build_default_registry(), res)
    assert d["probe"] == "resistivity"
    probes = {p["probe"] for p in d["plots"]}
    assert probes == {"resistivity"}                      # no ACMS kinds on a resistivity file
    # every entry says whether THIS file can draw it
    assert all(isinstance(p["available"], bool) for p in d["plots"])
    # the headline kind is drawable; a kind whose series is empty on this file is not lied about
    by_key = {p["key"]: p for p in d["plots"]}
    assert by_key["resistivity_rho_t"]["available"] is True


def test_discover_for_keeps_global_sections_and_envelope_keys():
    from cryosweep_core.discovery import discover_for
    res = _analyzed("examples/thermal_transport.dat")
    d = discover_for(build_default_registry(), res)
    base = discover(build_default_registry())
    # fits/observables/probes are registry facts, not file facts: unchanged
    assert d["fits"] == base["fits"] and d["observables"] == base["observables"]
    assert d["probes"] == base["probes"]
    assert d["probe"] == "tto"


def test_cli_plots_with_file_differs_from_global_dump():
    import json, subprocess, sys, pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    def run(*args):
        return subprocess.run([sys.executable, "-m", "cryosweep_cli", *args],
                              capture_output=True, text=True, cwd=root)
    bare = run("plots")
    with_file = run("plots", "examples/resistivity_semiconductor.dat")
    assert bare.returncode == 0 and with_file.returncode == 0
    assert bare.stdout != with_file.stdout                # the file is no longer ignored
    d = json.loads(with_file.stdout)
    assert d["probe"] == "resistivity"
    assert {p["probe"] for p in d["plots"]} == {"resistivity"}
    # the no-file dump is byte-identical to before (global registry dump, no probe key)
    b = json.loads(bare.stdout)
    assert "probe" not in b and "file" not in b
