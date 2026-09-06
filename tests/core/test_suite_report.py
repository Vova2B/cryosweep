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


# ---- the pasteable verification block (--verify-block) --------------------------------

def _run_block(xml, *extra, max_skipped=215, min_total=2000):
    return subprocess.run([sys.executable, str(TOOL), str(xml),
                           "--max-skipped", str(max_skipped), "--min-total", str(min_total),
                           "--verify-block", *extra],
                          capture_output=True, text=True, cwd=TOOL.parent.parent)


def test_verify_block_is_emitted_and_traceable(tmp_path):
    """The block a contributor pastes into a PR: commit, counts, real-data line, and the
    junit file's digest — every line re-derivable by re-running the same command, which is
    what makes a pasted block falsifiable rather than testimony."""
    import hashlib
    xml = _xml(tmp_path, 1908, 209, reason="local-only measurement file for key 'hc'")
    r = _run_block(xml)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "### Verification" in out
    assert "1908 passed" in out and "209 skipped" in out
    assert "commit:" in out
    digest = hashlib.sha256(xml.read_bytes()).hexdigest()
    assert digest[:12] in out, "the block must carry the junit digest it was derived from"
    assert "gates: PASS" in out


def test_verify_block_says_real_data_did_not_run(tmp_path):
    xml = _xml(tmp_path, 1908, 209, reason="local-only measurement file for key 'hc'")
    out = _run_block(xml).stdout
    assert "real-data tests: NOT RUN" in out and "209 local-only skips" in out


def test_verify_block_says_real_data_ran(tmp_path):
    xml = _xml(tmp_path, 2160, 1, reason="flaky benchmark")
    out = _run_block(xml).stdout
    assert "real-data tests: RAN" in out and "0 local-only skips" in out


def test_require_real_data_is_a_gate_not_a_note(tmp_path):
    """--require-real-data turns the real-data line into a failure: the pre-push run in
    the data-bearing tree must not green-tick over ~200 silently skipped real-data tests
    (the documented worktree blind spot)."""
    xml = _xml(tmp_path, 1908, 209, reason="local-only measurement file for key 'hc'")
    r = _run_block(xml, "--require-real-data")
    assert r.returncode == 1
    assert "real-data" in r.stderr
    sub = tmp_path / "c"
    sub.mkdir()
    clean = _xml(sub, 2160, 1, reason="flaky benchmark")
    assert _run_block(clean, "--require-real-data").returncode == 0


def test_real_data_skip_markers_match_what_the_conftests_actually_emit():
    """The tool classifies skips by matching the conftests' skip-message spellings. That is
    one fact in three files — so pin the agreement, or the block would silently misreport
    real-data coverage after a conftest reword (the one-fact-many-spellings defect class)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("suite_report", TOOL)
    sr = importlib.util.module_from_spec(spec); spec.loader.exec_module(sr)
    root = TOOL.parent.parent
    core = (root / "tests/core/conftest.py").read_text()
    gui = (root / "tests/gui/conftest.py").read_text()
    for marker in sr.REAL_DATA_SKIP_MARKERS:
        assert marker in core or marker in gui, (
            f"marker {marker!r} matches nothing in either conftest")
    # every local-only skip the conftests can emit is classified by some marker
    assert any(m in "local-only measurement file for key 'x' is not available"
               for m in sr.REAL_DATA_SKIP_MARKERS)
    assert any(m in "gallery manifest is not available (it is not part of the published tree)"
               for m in sr.REAL_DATA_SKIP_MARKERS)
