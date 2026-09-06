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
"""
import argparse
import collections
import sys
import xml.etree.ElementTree as ET


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("junit_xml")
    ap.add_argument("--max-skipped", type=int, required=True,
                    help="fail if more tests skipped than this (the real gate)")
    ap.add_argument("--min-total", type=int, required=True,
                    help="fail if fewer tests collected than this (collection-collapse floor)")
    a = ap.parse_args()

    root = ET.parse(a.junit_xml).getroot()
    cases = root.iter("testcase")
    total = skipped = failed = errored = 0
    skip_reasons = collections.Counter()
    for c in cases:
        total += 1
        if c.find("skipped") is not None:
            skipped += 1
            msg = (c.find("skipped").get("message") or "").strip()
            skip_reasons[msg.split("\n")[0][:70] or "(no reason given)"] += 1
        elif c.find("failure") is not None:
            failed += 1
        elif c.find("error") is not None:
            errored += 1
    passed = total - skipped - failed - errored

    print(f"suite: {total} collected — {passed} passed, {skipped} skipped, "
          f"{failed} failed, {errored} errored")
    if skip_reasons:
        print("skip reasons:")
        for reason, n in skip_reasons.most_common(10):
            print(f"  {n:5d}  {reason}")

    problems = []
    if skipped > a.max_skipped:
        problems.append(
            f"{skipped} tests skipped, limit is {a.max_skipped}. Something stopped running "
            f"that used to run — this is the failure mode that looks identical to success.")
    if total < a.min_total:
        problems.append(
            f"only {total} tests collected, floor is {a.min_total}. Collection shrank; a "
            f"module or directory is probably not being collected at all.")
    for p in problems:
        print(f"FAIL: {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
