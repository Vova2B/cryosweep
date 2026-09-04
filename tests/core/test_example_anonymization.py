"""The two anonymized real-derived examples must carry no identity (owner-approved 2026-09-01).

`examples/` used to be synthetic-only, so it was leak-proof by construction. Two files are now
subsets of real measurements, and these tests are what replaces "by construction": they run on
the SHIPPED bytes, so they hold on any clone, with or without the private source data.

Header anonymisation alone was NOT enough, which is why the body checks exist. Measured on the
real sources before the fix:
  * both Time Stamp columns are absolute seconds since 1900-01-01 and decoded to the real
    acquisition instant (2025-01-11 12:43:14 and 2023-08-19 23:09:39);
  * the heat-capacity Comment column carried
    "CALFILE: C:\\QDDYNA~1\\...\\Puck1659.cal|Addenda #51 measured on 8/16/2023 ..." —
    a lab filesystem path, a calibration-puck serial, an addenda number and a date.
Neither column is read by any analyzer, so scrubbing them changes no result (pinned below).
"""
import pathlib
import re
import pytest

EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"
REAL = ("magnetization_vsm_multifield.dat", "heat_capacity_multifield.dat")

# The ONE date allowed anywhere: the neutral publication stamp the anonymizer writes.
NEUTRAL_DATE = "09/01/2026"


def _read(name):
    return (EXAMPLES / name).read_text(encoding="latin-1")


def _split(txt):
    lines = txt[txt.index("[Data]"):].splitlines()
    return lines[1].split(","), lines[2:]


@pytest.mark.parametrize("name", REAL)
def test_shipped_example_exists_and_is_example_sized(name):
    p = EXAMPLES / name
    assert p.exists(), f"{name} must ship — it is committed, not generated on demand"
    assert p.stat().st_size < 400_000


@pytest.mark.parametrize("name", REAL)
def test_no_lab_paths_serials_or_calibration_text(name):
    """The body-level leak that header anonymisation cannot reach.

    Patterns are the leak SHAPES, not bare words: "Puck Temp (Kelvin)", "Puck Resist (Ohms)"
    and "Addenda HC (uJ/K)" are legitimate QD column names in this very file, so the puck
    serial and the addenda number have to be matched with their digits attached.
    """
    txt = _read(name)
    for pattern in (r"CALFILE", r"QDDYNA", r"TempCal", r"\.cal\b",
                    r"Puck\d", r"Addenda\s*#", r"\\"):
        assert not re.search(pattern, txt), f"{name} still carries {pattern!r}"


@pytest.mark.parametrize("name", REAL)
def test_no_calendar_date_other_than_the_neutral_stamp(name):
    dates = set(re.findall(r"\d{1,2}/\d{1,2}/\d{2,4}", _read(name)))
    assert dates <= {NEUTRAL_DATE}, f"{name} carries real date(s) {sorted(dates - {NEUTRAL_DATE})}"


@pytest.mark.parametrize("name", REAL)
def test_time_stamp_is_rebased_so_it_decodes_to_no_instant(name):
    """An absolute QD counter (seconds since 1900-01-01) IS the acquisition timestamp."""
    cols, body = _split(_read(name))
    ti = next(i for i, c in enumerate(cols) if "time stamp" in c.strip().lower())
    first = float(body[0].split(",")[ti])
    assert first == 0.0, f"{name}: Time Stamp starts at {first}, which decodes to a real instant"
    # ... and the column is still a usable relative clock, not blanked
    assert float(body[-1].split(",")[ti]) > 0.0


@pytest.mark.parametrize("name", REAL)
def test_sample_identity_fields_are_neutralised(name):
    """Every INFO row that names the sample, the operator or the instrument reads 'anonymized'."""
    txt = _read(name)
    head = txt[:txt.index("[Data]")].splitlines()
    for ln in head:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 3 or parts[0].upper() != "INFO":
            continue
        key = parts[2].split(":", 1)[0].strip().upper()
        if key.startswith("SAMPLE_") or key.endswith("_SERIAL_NUMBER"):
            assert parts[1] == "anonymized", f"{name}: {key} still carries {parts[1]!r}"
    assert f"TITLE,{name}" in txt, "TITLE must be the neutral example filename"


def test_comment_column_keeps_only_benign_instrument_warnings():
    """Allowlist, not blocklist: the surviving comments are instrument warnings, nothing else."""
    cols, body = _split(_read("heat_capacity_multifield.dat"))
    ci = next(i for i, c in enumerate(cols) if c.strip().lower().startswith("comment"))
    vals = {ln.split(",")[ci].strip() for ln in body if ln.split(",")[ci].strip()}
    assert vals, "the column should still be exercised, not blanked wholesale"
    for v in vals:
        assert v == "anonymized" or re.match(r"^Error: Warning: [A-Za-z0-9 .,;:()\-]+$", v), v


def test_the_examples_still_analyze_to_their_documented_numbers():
    """Scrubbing Time Stamp and Comment must not move a single fitted number: neither column is
    canonicalized or read. These are also the values examples/README.md advertises."""
    from cryosweep_core.io.loader import load_dat
    from cryosweep_core.analyzers.dispatch import analyze_file
    from cryosweep_core.registry import build_default_registry
    from cryosweep_core.config import RunConfig
    reg = build_default_registry()

    d = analyze_file(load_dat(str(EXAMPLES / "magnetization_vsm_multifield.dat")),
                     RunConfig(), reg).data
    assert d["probe"] == "vsm"
    assert sorted({round(b["field_oe"]) for b in d["t_blocks"]}) == [100, 5000, 40000, 100000]
    assert len(d["loops"]) == 4
    p = d["fit"]["params"]
    assert p["mu_eff"] == pytest.approx(3.70, abs=0.02)
    assert p["theta"] == pytest.approx(-31.0, abs=0.5)
    assert d["fit"]["r2"] == pytest.approx(0.987, abs=0.002)

    d = analyze_file(load_dat(str(EXAMPLES / "heat_capacity_multifield.dat")),
                     RunConfig(), reg).data
    assert d["probe"] == "heatcapacity"
    assert [round(g["field_oe"]) for g in d["field_groups"]] == [1, 50001, 100001, 130000]
    assert d["full_fit"]["params"]["theta_D"] == pytest.approx(118.85, abs=0.5)
    assert d["full_fit"]["r2"] > 0.999
