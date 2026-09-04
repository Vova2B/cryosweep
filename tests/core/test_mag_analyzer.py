import pathlib
from cryosweep_core.io.loader import load_dat
from cryosweep_core.io.columns import canonicalize_columns
from cryosweep_core.detect.probe import detect_probe
from cryosweep_core.registry import build_default_registry
from cryosweep_core.analyzers.mag import VSMAnalyzer
from cryosweep_core.config import RunConfig

FIX = pathlib.Path(__file__).parent / "fixtures" / "vsm_synth.dat"

def test_vsm_fixture_loads():
    rt = load_dat(FIX)
    assert rt.header.app == "VSM"
    assert rt.header.molar_mass == 200.0
    assert "Moment (emu)" in rt.df.columns
    assert rt.df.shape[0] == 300

def test_vsm_columns_and_detect():
    rt = load_dat(FIX)
    df, cmap = canonicalize_columns(rt.df, rt.header)
    assert cmap.logical["moment"] == "Moment (emu)"
    assert cmap.logical["temperature"] == "Temperature (K)"
    assert cmap.logical["field"] == "Magnetic Field (Oe)"
    reg = build_default_registry()
    score, key = detect_probe(rt.header, set(df.columns), reg)
    assert key == "vsm" and score >= 0.8

def test_vsm_analyze_recovers_curie_weiss():
    rt = load_dat(FIX)
    res = VSMAnalyzer().analyze(rt, RunConfig.load())
    assert res.status == "ok"
    fit = res.data["fit"]
    assert abs(fit["params"]["C"] - 0.5) < 0.02
    assert abs(fit["params"]["theta"] + 10.0) < 0.5
    assert abs(fit["params"]["mu_eff"] - 1.999) < 0.05
    assert res.provenance.sha256

def test_vsm_si_mu_eff_matches_cgs():
    # Bug 1: mu_eff is unit-system-independent physics. Both CGS and SI must give ~1.999.
    rt = load_dat(FIX)
    res_cgs = VSMAnalyzer().analyze(rt, RunConfig.load(unit_system="CGS"))
    res_si = VSMAnalyzer().analyze(rt, RunConfig.load(unit_system="SI"))
    mu_cgs = res_cgs.data["fit"]["params"]["mu_eff"]
    mu_si = res_si.data["fit"]["params"]["mu_eff"]
    assert abs(mu_cgs - 1.999) < 0.02
    assert abs(mu_si - 1.999) < 0.02
    # exported C unit reflects the unit system
    # PQ-3 Task 3: CGS Curie-constant unit reconciled to the physically-correct string
    # (χ_molar is emu/(mol*Oe) -> C = χ*(T-θ) is emu*K/(mol*Oe)).
    assert res_cgs.data["fit"]["units"]["C"] == "emu*K/(mol*Oe)"
    assert res_si.data["fit"]["units"]["C"] == "m^3*K/mol"


def test_vsm_zero_field_no_crash(tmp_path):
    # Bug 2: a field=0 fixture must return a Result, never an unhandled crash.
    import tests.core.fixtures.make_vsm as mk
    p = tmp_path / "zerofield.dat"
    mk.write_vsm(p)
    # rewrite all field values to 0 while keeping moment finite (chi -> inf -> inv_chi -> 0)
    lines = p.read_text().splitlines()
    # find [Data] marker, then header row, then data rows
    di = next(i for i, l in enumerate(lines) if l.strip().lower().startswith("[data]"))
    hdr = lines[di + 1].split(",")
    fcol = next(i for i, c in enumerate(hdr) if "Field" in c)
    out = lines[: di + 2]
    for l in lines[di + 2:]:
        if not l.strip():
            out.append(l); continue
        cells = l.split(",")
        if len(cells) > fcol:
            cells[fcol] = "0"
        out.append(",".join(cells))
    p.write_text("\n".join(out))
    rt = load_dat(p)
    res = VSMAnalyzer().analyze(rt, RunConfig.load())   # must not raise
    assert res.status in ("error", "low_confidence", "gated")
    assert res.warnings or res.errors


def test_vsm_gated_without_molar_mass(tmp_path):
    import tests.core.fixtures.make_vsm as mk
    p = tmp_path / "nomol.dat"
    # write a fixture then strip the MOLWGHT line
    mk.write_vsm(p)
    text = "\n".join(l for l in p.read_text().splitlines() if "MOLWGHT" not in l)
    p.write_text(text)
    rt = load_dat(p)
    res = VSMAnalyzer().analyze(rt, RunConfig.load())
    assert res.status == "gated"
    assert any(g.need == "molar_mass" for g in res.gate)


def test_vsm_on_non_vsm_file_returns_error_not_keyerror():
    import pathlib
    from cryosweep_core.io.loader import load_dat
    from cryosweep_core.config import RunConfig
    from cryosweep_core.analyzers.mag import VSMAnalyzer
    fix = pathlib.Path(__file__).resolve().parent / "fixtures" / "hall_synth.dat"
    res = VSMAnalyzer().analyze(load_dat(fix), RunConfig.load())  # hall file has no 'moment' column
    assert res.status == "error"
    assert res.data.get("probe") == "vsm"
    assert any("moment" in e for e in res.errors)
