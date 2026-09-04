import json, subprocess, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
FIX = str(ROOT / "tests/core/fixtures/vsm_synth.dat")
# The skill now SHIPS inside the app tree, so this test must pass from a standalone
# checkout of the app directory alone -- no repo_root() escape (Risk 4).



def _run(*args):
    return subprocess.run([sys.executable, "-m", "cryosweep_cli", *args], capture_output=True, text=True, cwd=ROOT)


def test_cli_analyze_hc(hc_path):
    r = _run("analyze", str(hc_path))
    assert r.returncode in (0, 11), r.stderr
    d = json.loads(r.stdout)
    assert d["data"]["probe"] == "heatcapacity"
    assert d["data"]["fit"]["params"]["theta_D"] > 0


def test_cli_probes_lists_hc_analyzer():
    d = json.loads(_run("probes").stdout)
    hc = next(p for p in d["probes"] if p["key"] == "heatcapacity")
    assert hc["has_analyzer"] is True


def test_schema_hc():
    r = _run("schema", "analyze:hc"); assert r.returncode == 0
    assert "cp_over_t" in json.loads(r.stdout)["properties"]


def test_cli_probes():
    r = _run("probes"); assert r.returncode == 0
    assert any(p["key"] == "vsm" for p in json.loads(r.stdout)["probes"])


def test_cli_fits():
    r = _run("fits"); assert r.returncode == 0
    assert any(f["key"] == "curie_weiss" for f in json.loads(r.stdout)["fits"])


def test_cli_schema():
    r = _run("schema", "result"); assert r.returncode == 0
    assert "properties" in json.loads(r.stdout)


def test_cli_schema_analyze_vsm():
    r = _run("schema", "analyze:vsm"); assert r.returncode == 0
    assert "inv_chi" in json.loads(r.stdout)["properties"]


SKILL = ROOT / "skill/cryosweep/SKILL.md"


def test_skill_documents_contract():
    """The original ten contract tokens, renamed where they name the command, PLUS every
    probe the registry actually ships -- derived, not hard-coded, so the guide cannot
    silently understate the product again (it claimed 'magnetization/VSM today' while
    seven probes shipped)."""
    t = SKILL.read_text()
    for token in ("cryosweep analyze", "cryosweep schema", "cryosweep probes", "gated",
                  "low_confidence", "exit", "10", "11", "--molar-mass", "pipeline"):
        assert token in t, token

    from cryosweep_core.registry import build_default_registry
    from cryosweep_core.discovery import discover
    for probe in (p["key"] for p in discover(build_default_registry())["probes"]):
        assert probe in t, "SKILL.md does not mention shipped probe %r" % probe


def test_agent_recovery_from_gated(tmp_path):
    import tests.core.fixtures.make_vsm as mk
    p = tmp_path / "nomol.dat"; mk.write_vsm(p)
    p.write_text("\n".join(l for l in p.read_text().splitlines() if "MOLWGHT" not in l))
    r1 = _run("analyze", str(p))
    assert r1.returncode == 10
    gate = json.loads(r1.stdout)["gate"]
    flag = next(g["remedy"]["flag"] for g in gate if g["need"] == "molar_mass")
    r2 = _run("analyze", str(p), flag, "200.0")
    assert r2.returncode == 0
    assert abs(json.loads(r2.stdout)["data"]["fit"]["params"]["C"] - 0.5) < 0.02
