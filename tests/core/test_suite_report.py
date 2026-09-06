"""The CI completeness gate must itself be able to fail.

`tools/suite_report.py` exists because CI ran 209 silent skips for its whole history behind a
green tick. A gate that cannot fail would recreate exactly that, one level up — so these pin
both directions: it fires on the shapes that mean coverage vanished, and it stays quiet on the
shape CI legitimately has.
"""
import subprocess
import sys
import pathlib

TOOL = pathlib.Path(__file__).resolve().parents[2] / "tools" / "suite_report.py"


def _xml(tmp_path, passed, skipped, reason="requires real measurement data"):
    body = "".join(f'<testcase name="p{i}"/>' for i in range(passed))
    body += "".join(f'<testcase name="s{i}"><skipped message="{reason}"/></testcase>'
                    for i in range(skipped))
    p = tmp_path / "r.xml"
    p.write_text(f"<testsuites><testsuite>{body}</testsuite></testsuites>")
    return p


def _run(xml, max_skipped=215, min_total=2000):
    return subprocess.run([sys.executable, str(TOOL), str(xml),
                           "--max-skipped", str(max_skipped), "--min-total", str(min_total)],
                          capture_output=True, text=True)


def test_the_shape_ci_actually_has_passes(tmp_path):
    """1908/209 measured on the 0.4.0 release commit. A gate that failed this would be
    replaced within a day, and we would be back to no gate."""
    r = _run(_xml(tmp_path, 1908, 209))
    assert r.returncode == 0, r.stderr
    assert "2117 collected" in r.stdout and "209 skipped" in r.stdout


def test_skip_growth_fails(tmp_path):
    """The load-bearing case: every way coverage silently vanishes shows up as more skips."""
    r = _run(_xml(tmp_path, 1800, 300))
    assert r.returncode == 1
    assert "300 tests skipped" in r.stderr


def test_collection_collapse_fails(tmp_path):
    """A directory ceasing to be collected leaves few tests and zero failures — otherwise
    indistinguishable from success."""
    r = _run(_xml(tmp_path, 50, 0))
    assert r.returncode == 1
    assert "50 tests collected" in r.stderr


def test_the_count_is_always_printed(tmp_path):
    """The original defect was invisibility, not failure: `pytest -q` twice suppressed the
    summary line, so the job never said how many tests ran."""
    r = _run(_xml(tmp_path, 1908, 209))
    assert "1908 passed" in r.stdout
    assert "skip reasons:" in r.stdout and "requires real measurement data" in r.stdout
