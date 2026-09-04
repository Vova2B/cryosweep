"""The version is written in four files and they must agree.

Added 2026-09-02, when they did not: `pyproject.toml` and `CITATION.cff` said 0.1.0 while
`cryosweep_core/__init__.py` said 0.0.1 — and `test_smoke.py` pinned the wrong one, so the
disagreement was load-bearing and still green. Nothing compared them to each other.

A release bump touches all four plus the CHANGELOG date. This is what makes that a mechanical
edit instead of a thing to remember: change one, the suite tells you about the other three.
"""
import pathlib
import re
import tomllib
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _pyproject_version():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]


def _citation():
    t = (ROOT / "CITATION.cff").read_text()
    ver = re.search(r'^version:\s*"?([^"\s]+)"?\s*$', t, re.M)
    rel = re.search(r'^date-released:\s*"?(\d{4}-\d{2}-\d{2})"?\s*$', t, re.M)
    assert ver and rel, "CITATION.cff must carry both version and date-released"
    return ver.group(1), rel.group(1)


def _changelog_top():
    """The newest CHANGELOG heading: `## <version> — <date>` (or `— unreleased`)."""
    for ln in (ROOT / "CHANGELOG.md").read_text().splitlines():
        m = re.match(r"^##\s+(\S+)\s+—\s+(.+?)\s*$", ln)
        if m:
            return m.group(1), m.group(2)
    pytest.fail("CHANGELOG.md has no `## <version> — <date>` heading")


def test_all_four_version_strings_agree():
    import cryosweep_core
    py = _pyproject_version()
    cff_ver, _ = _citation()
    cl_ver, _ = _changelog_top()
    assert cryosweep_core.__version__ == py, "cryosweep_core/__init__.py disagrees with pyproject"
    assert cff_ver == py, "CITATION.cff disagrees with pyproject"
    assert cl_ver == py, "the newest CHANGELOG entry disagrees with pyproject"


def test_citation_and_changelog_dates_agree():
    """A stale date-released is how CITATION.cff quietly starts describing an old release."""
    _, cff_date = _citation()
    _, cl_date = _changelog_top()
    if cl_date == "unreleased":
        pytest.skip("CHANGELOG top entry is still unreleased; no date to compare")
    assert cff_date == cl_date


def test_the_version_is_a_plain_release_number():
    v = _pyproject_version()
    assert re.fullmatch(r"\d+\.\d+\.\d+", v), f"{v!r} is not MAJOR.MINOR.PATCH"


def test_cli_reports_the_same_version():
    """`cryosweep --version` must exist and agree with the package.

    .github/ISSUE_TEMPLATE/bug_report.md asks reporters for their "cryosweep version", so a
    command that answers that has to exist. It must also work with NO subcommand, even though
    `command` is a required positional -- argparse fires a `version` action during parsing,
    before it checks required args.
    """
    import subprocess
    import sys
    import cryosweep_core
    r = subprocess.run([sys.executable, "-m", "cryosweep_cli", "--version"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, f"--version exited {r.returncode}: {r.stderr[:200]}"
    assert r.stdout.strip() == f"cryosweep {cryosweep_core.__version__}"
