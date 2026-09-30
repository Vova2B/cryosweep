"""The PR verification gate must itself be able to fail.

CONTRIBUTING asked for a pasted verification block and nothing checked for one, so the
convention decayed to optional: none of the six pull requests before this gate carried a
block. These pin both directions — it fires on a missing, stale, dirty, failing or
hand-shortened block, and stays quiet on the block `suite_report.py --verify-block` really
emits for the commit under review.
"""
import contextlib
import importlib.util
import io
import pathlib
import subprocess
import sys

TOOLS = pathlib.Path(__file__).resolve().parents[2] / "tools"
TOOL = TOOLS / "check_pr_verification.py"

HEAD = "a0cee30f4c1d2b3a4958677685746352413f2e1d"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _block(commit="a0cee30 (clean working tree)", failed=0, errored=0, total=2368,
           real="NOT RUN — 222 local-only skips (expected in a public checkout; run the "
                "suite in the data-bearing tree before release)", gates="PASS"):
    return ("### Verification — cryosweep test suite\n"
            f"- commit: {commit}\n"
            f"- suite: 2143 passed, 225 skipped, {failed} failed, {errored} errored "
            f"({total} collected)\n"
            f"- real-data tests: {real}\n"
            "- junit sha256: 70d684815cea\n"
            f"- gates: {gates}\n")


def _xml(tmp_path, total=2368):
    p = tmp_path / "r.xml"
    p.write_text("<testsuites><testsuite>" + '<testcase name="t"/>' * total
                 + "</testsuite></testsuites>")
    return p


def _run(tmp_path, body, *extra, head=HEAD):
    b = tmp_path / "body.md"
    b.write_text(body, encoding="utf-8")
    return subprocess.run([sys.executable, str(TOOL), str(b), "--head-sha", head, *extra],
                          capture_output=True, text=True)


def test_a_block_for_the_head_commit_passes(tmp_path):
    r = _run(tmp_path, "## What this changes\n\nA thing.\n\n" + _block())
    assert r.returncode == 0, r.stderr


def test_the_body_github_actually_sends_passes(tmp_path):
    """The web editor submits CRLF line endings; a parser that only splits on LF sees every
    value with a trailing carriage return and rejects a correct block."""
    r = _run(tmp_path, ("intro\n\n" + _block()).replace("\n", "\r\n"))
    assert r.returncode == 0, r.stderr


def test_no_block_fails(tmp_path):
    """The state every pull request was in before this gate."""
    r = _run(tmp_path, "## What this changes\n\nTrust me, I ran the tests.\n")
    assert r.returncode == 1
    assert "no verification block" in r.stderr


def test_an_empty_body_fails(tmp_path):
    r = _run(tmp_path, "")
    assert r.returncode == 1
    assert "no verification block" in r.stderr


def test_a_block_inside_an_html_comment_is_not_a_block(tmp_path):
    """Otherwise an example block in the PR template would satisfy the gate unedited."""
    r = _run(tmp_path, "<!-- paste it here, e.g.\n" + _block() + "-->\n")
    assert r.returncode == 1
    assert "no verification block" in r.stderr


def test_a_block_for_an_earlier_commit_fails(tmp_path):
    """The load-bearing case: a block pasted once and never refreshed after later pushes
    describes a tree that is no longer the one being merged."""
    r = _run(tmp_path, _block(commit="3c643f5 (clean working tree)"))
    assert r.returncode == 1
    assert "3c643f5" in r.stderr and HEAD[:7] in r.stderr


def test_a_fresh_block_beside_a_stale_one_passes(tmp_path):
    r = _run(tmp_path, _block(commit="3c643f5 (clean working tree)") + "\nupdated:\n\n"
             + _block())
    assert r.returncode == 0, r.stderr


def test_a_too_short_commit_prefix_fails(tmp_path):
    """`a0` is a prefix of the head SHA and of one commit in 256."""
    r = _run(tmp_path, _block(commit="a0 (clean working tree)"))
    assert r.returncode == 1


def test_a_dirty_tree_fails(tmp_path):
    """The counts then describe files that are not in the commit the block names."""
    r = _run(tmp_path, _block(commit="a0cee30 (dirty working tree)"))
    assert r.returncode == 1
    assert "dirty" in r.stderr


def test_failures_in_the_pasted_run_fail(tmp_path):
    assert _run(tmp_path, _block(failed=2)).returncode == 1
    assert _run(tmp_path, _block(errored=1)).returncode == 1
    assert _run(tmp_path, _block(gates="FAIL (details on stderr)")).returncode == 1


def test_a_block_with_a_line_removed_fails(tmp_path):
    """Deleting the inconvenient line must not read as that line being fine."""
    body = "\n".join(ln for ln in _block().splitlines() if "gates" not in ln)
    r = _run(tmp_path, body)
    assert r.returncode == 1
    assert "gates" in r.stderr


def test_collected_count_must_match_this_run(tmp_path):
    """The one line CI can re-derive: the same commit collects the same number of tests
    with or without measurement data (they skip, they do not vanish)."""
    ok = _run(tmp_path, _block(), "--junit", str(_xml(tmp_path, 2368)))
    assert ok.returncode == 0, ok.stderr
    bad = _run(tmp_path, _block(total=2300), "--junit", str(_xml(tmp_path, 2368)))
    assert bad.returncode == 1
    assert "2300" in bad.stderr and "2368" in bad.stderr


def test_a_release_branch_needs_the_real_data_run(tmp_path):
    """CI has no measurement files, so for a release the pasted block is the only evidence
    that the real-data tests ran at all."""
    public = _run(tmp_path, _block(), "--head-ref", "release-0.8.0")
    assert public.returncode == 1
    assert "real-data" in public.stderr
    ran = _run(tmp_path, _block(real="RAN — 0 local-only skips"),
               "--head-ref", "release-0.8.0")
    assert ran.returncode == 0, ran.stderr


def test_an_ordinary_branch_does_not_need_real_data(tmp_path):
    """An outside contributor cannot run those tests; demanding it would block every one."""
    r = _run(tmp_path, _block(), "--head-ref", "fix/some-typo")
    assert r.returncode == 0, r.stderr


def test_it_reads_what_suite_report_actually_emits(tmp_path):
    """Two tools, one format: if the emitter's wording drifts, this fails rather than the
    gate rejecting every honest block in CI."""
    report, check = _load("suite_report"), _load("check_pr_verification")
    xml = _xml(tmp_path, 12)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        report.emit_verify_block(report.parse_junit(xml), xml, gates_ok=True)
    blocks = check.parse_blocks(out.getvalue())
    assert len(blocks) == 1
    b = blocks[0]
    assert set(check.REQUIRED) <= set(b)
    assert check.suite_counts(b["suite"]) == {"failed": 0, "errored": 0, "collected": 12}
    assert b["gates"] == "PASS" and b["real-data tests"].startswith("RAN")
    assert check.commit_state(b["commit"])[1] in ("clean", "dirty")


def test_every_copy_of_the_command_uses_the_ceiling_ci_enforces():
    """CONTRIBUTING told contributors `--max-skipped 215` while CI enforced 231 and a public
    checkout really skips 225: the documented command failed for anyone who ran it."""
    root = TOOLS.parent
    ci = (root / ".github/workflows/ci.yml").read_text()
    enforced = set(__import__("re").findall(r"--max-skipped (\d+)", ci))
    assert len(enforced) == 1, enforced
    check = _load("check_pr_verification")
    for name, text in (("CONTRIBUTING.md", (root / "CONTRIBUTING.md").read_text()),
                       ("PULL_REQUEST_TEMPLATE.md",
                        (root / ".github/PULL_REQUEST_TEMPLATE.md").read_text()),
                       ("check_pr_verification.HOW", check.HOW)):
        found = set(__import__("re").findall(r"--max-skipped (\d+)", text))
        assert found == enforced, f"{name} says {found}, CI enforces {enforced}"
