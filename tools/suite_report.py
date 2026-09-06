"""Print what the test run actually covered, and fail when it silently shrinks.

Why this exists: CI ran `pytest -q` while `pytest.ini` already set `addopts = -q`, and
doubled `-q` suppresses pytest's summary line. So the job printed progress dots and a green
tick and never said how many tests ran. Measured on the 0.4.0 release commit, CI was running
1908 passed / 209 skipped while the same commit ran 2115 passed / 1 skipped locally — 208
tests, 9.9% of the suite, absent with no indication. Every "CI is green" in this project's
history meant green over roughly 90% of the tests.

Those 209 skips are legitimate: a public checkout has no real measurement data and no
reference gallery, so the tests needing either skip by design. The defect was never that they
skip. It was that the number was invisible and unpinned, so drift in either direction — a
fixture quietly degrading to a skip, a marker misapplied, a whole directory ceasing to be
collected — would look exactly like success.

`--max-skipped` is therefore the load-bearing gate: every way this fails shows up as MORE
skips. `--min-total` is the coarse backstop for collection collapse; it is a floor, so a
growing suite never needs it bumped.

`--verify-block` additionally emits a markdown block to paste into a PR: commit, counts,
whether the real-data tests actually ran, and the junit file's digest. Every line is
re-derivable by re-running the same command against the same commit, which is what makes a
pasted block falsifiable rather than testimony. `--require-real-data` turns the real-data
line into a gate for the pre-push run in the data-bearing tree (the documented blind spot:
a green worktree run silently skips ~200 real-data tests).
"""
import argparse
import collections
import hashlib
import subprocess
import sys
import xml.etree.ElementTree as ET

#: Skip-message spellings the test-suite conftests emit for tests that need local-only
#: development data (real measurement files, the reference gallery). This tuple is the
#: single place the classification lives; tests/core/test_suite_report.py pins that these
#: markers still match what tests/core/conftest.py and tests/gui/conftest.py actually emit.
REAL_DATA_SKIP_MARKERS = (
    "local-only measurement file",
    "gallery manifest is not available",
)


def parse_junit(path):
    """One parse for both the report and the verify block (one fact, one parser)."""
    root = ET.parse(path).getroot()
    total = skipped = failed = errored = 0
    skip_reasons = collections.Counter()
    real_data_skips = 0
    for c in root.iter("testcase"):
        total += 1
        if c.find("skipped") is not None:
            skipped += 1
            msg = (c.find("skipped").get("message") or "").strip()
            skip_reasons[msg.split("\n")[0][:70] or "(no reason given)"] += 1
            if any(m in msg for m in REAL_DATA_SKIP_MARKERS):
                real_data_skips += 1
        elif c.find("failure") is not None:
            failed += 1
        elif c.find("error") is not None:
            errored += 1
    return {"total": total, "passed": total - skipped - failed - errored,
            "skipped": skipped, "failed": failed, "errored": errored,
            "skip_reasons": skip_reasons, "real_data_skips": real_data_skips}


def _git_line():
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return f"{sha} ({'dirty working tree' if dirty else 'clean working tree'})"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "(not a git checkout)"


def emit_verify_block(stats, xml_path, gates_ok):
    digest = hashlib.sha256(open(xml_path, "rb").read()).hexdigest()
    s = stats
    rd = s["real_data_skips"]
    real = (f"RAN — {rd} local-only skips" if rd == 0 else
            f"NOT RUN — {rd} local-only skips (expected in a public checkout; "
            f"run the suite in the data-bearing tree before release)")
    print("### Verification — cryosweep test suite")
    print(f"- commit: {_git_line()}")
    print(f"- suite: {s['passed']} passed, {s['skipped']} skipped, {s['failed']} failed, "
          f"{s['errored']} errored ({s['total']} collected)")
    print(f"- real-data tests: {real}")
    print(f"- junit sha256: {digest[:12]}")
    print(f"- gates: {'PASS' if gates_ok else 'FAIL (details on stderr)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("junit_xml")
    ap.add_argument("--max-skipped", type=int, required=True,
                    help="fail if more tests skipped than this (the real gate)")
    ap.add_argument("--min-total", type=int, required=True,
                    help="fail if fewer tests collected than this (collection-collapse floor)")
    ap.add_argument("--verify-block", action="store_true",
                    help="emit the markdown verification block to paste into a PR")
    ap.add_argument("--require-real-data", action="store_true",
                    help="fail if any local-only real-data test skipped (pre-push, dev tree)")
    a = ap.parse_args()

    s = parse_junit(a.junit_xml)

    problems = []
    if s["skipped"] > a.max_skipped:
        problems.append(
            f"{s['skipped']} tests skipped, limit is {a.max_skipped}. Something stopped "
            f"running that used to run — this is the failure mode that looks identical to "
            f"success.")
    if s["total"] < a.min_total:
        problems.append(
            f"only {s['total']} tests collected, floor is {a.min_total}. Collection shrank; "
            f"a module or directory is probably not being collected at all.")
    if a.require_real_data and s["real_data_skips"] > 0:
        problems.append(
            f"{s['real_data_skips']} real-data tests skipped but --require-real-data was "
            f"given: this run proves nothing about the real-file behaviour. Run the suite "
            f"in the tree that has the local data map.")

    if a.verify_block:
        emit_verify_block(s, a.junit_xml, gates_ok=not problems)
    else:
        print(f"suite: {s['total']} collected — {s['passed']} passed, {s['skipped']} skipped, "
              f"{s['failed']} failed, {s['errored']} errored")
    if s["skip_reasons"]:
        print("skip reasons:")
        for reason, n in s["skip_reasons"].most_common(10):
            print(f"  {n:5d}  {reason}")

    for p in problems:
        print(f"FAIL: {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
