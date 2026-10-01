"""Fail a pull request whose description does not carry a verification block for its head.

Why this exists: CONTRIBUTING asks for the block `suite_report.py --verify-block` prints to
be pasted into every pull request, and nothing checked that it was. A rule with no gate is
optional — none of the six pull requests merged before this check carried one, three of them
releases. The block matters most exactly where CI is blind: a public checkout has no
measurement files, so CI cannot run the real-data tests, and the pasted block from the
data-bearing tree is the only record that they ran.

What is checked, against the newest commit on the pull request:

* a block exists (outside HTML comments, so a template example does not count);
* its `commit` line names the head commit — a block pasted before later pushes describes a
  tree that is no longer the one being merged — and says the working tree was clean;
* its run had 0 failed, 0 errored and `gates: PASS`, with no line deleted;
* with `--junit`, its collected count equals this run's. The same commit collects the same
  tests with or without measurement data (they skip, they do not vanish), so this is the one
  line CI can re-derive for itself;
* on a `release*` branch, its real-data line says RAN.

What it cannot see: whether the block is true. It is pasted text. The commit and collected
lines are cross-checked; the pass count and the real-data line are not re-derivable without
the data, and a block typed by hand to match would pass. This gate stops the block being
forgotten or going stale. It does not stop it being forged.
"""
import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET

HEADING = "### Verification — cryosweep test suite"
REQUIRED = ("commit", "suite", "real-data tests", "junit sha256", "gates")
#: Shortest commit prefix accepted; `git rev-parse --short` never prints fewer.
MIN_SHA = 7
RELEASE_PREFIX = "release"

HOW = ("Run the suite on the commit you pushed and paste the block it prints into the "
       "pull request description:\n"
       "    pytest --junitxml=pytest-results.xml\n"
       "    python tools/suite_report.py pytest-results.xml --max-skipped 240 "
       "--min-total 2000 --verify-block")


def parse_blocks(body):
    """Every verification block in `body`, each as {key: value}, in document order."""
    text = re.sub(r"<!--.*?-->", "", body or "", flags=re.S)
    blocks, cur = [], None
    for raw in text.splitlines():
        line = raw.strip()
        if line == HEADING:
            cur = {}
            blocks.append(cur)
        elif cur is not None and line.startswith("- ") and ": " in line:
            key, value = line[2:].split(": ", 1)
            cur[key.strip()] = value.strip()
        elif cur is not None:
            cur = None
    return blocks


def commit_state(value):
    """('a0cee30', 'clean' | 'dirty' | None) from the block's commit line."""
    m = re.match(r"([0-9a-f]+) \((clean|dirty) working tree\)$", value)
    return (m.group(1), m.group(2)) if m else (value.split(" ")[0], None)


def suite_counts(value):
    m = re.search(r"(\d+) failed, (\d+) errored \((\d+) collected\)", value)
    return {"failed": int(m.group(1)), "errored": int(m.group(2)),
            "collected": int(m.group(3))} if m else None


def block_problems(block, require_real_data, junit_total):
    """Why this block, which names the head commit, is still not acceptable."""
    problems = [f"the block has no `{k}` line" for k in REQUIRED if k not in block]
    if problems:
        return problems
    if commit_state(block["commit"])[1] != "clean":
        problems.append(
            f"the block was produced from a dirty working tree ({block['commit']}): its "
            f"counts describe files that are not in the commit it names")
    counts = suite_counts(block["suite"])
    if counts is None:
        problems.append(f"the `suite` line is not in the emitted format: {block['suite']}")
    else:
        if counts["failed"] or counts["errored"]:
            problems.append(f"the pasted run did not pass: {block['suite']}")
        if junit_total is not None and counts["collected"] != junit_total:
            problems.append(
                f"the block says {counts['collected']} tests collected; this run collected "
                f"{junit_total} on the same change. Re-run after bringing the branch up to "
                f"date, and paste the block unedited")
    if block["gates"] != "PASS":
        problems.append(f"the pasted run failed its own gates: {block['gates']}")
    if require_real_data and not block["real-data tests"].startswith("RAN"):
        problems.append(
            f"a release needs the real-data tests, which CI cannot run: the block says "
            f"`{block['real-data tests'][:40]}…`. Run the suite in the tree that has the "
            f"measurement files, with --require-real-data")
    return problems


def check(body, head_sha, require_real_data=False, junit_total=None):
    """[] when some block in `body` verifies `head_sha`; otherwise the reasons."""
    blocks = parse_blocks(body)
    if not blocks:
        return [f"no verification block in the pull request description. {HOW}"]
    shas = [commit_state(b.get("commit", ""))[0] for b in blocks]
    for_head = [b for b, s in zip(blocks, shas)
                if len(s) >= MIN_SHA and head_sha.startswith(s)]
    if not for_head:
        return [f"the verification block is for {', '.join(s or '(no commit)' for s in shas)}"
                f", but the head of this pull request is {head_sha[:MIN_SHA]}. It describes "
                f"a tree that is no longer the one being merged. {HOW}"]
    results = [block_problems(b, require_real_data, junit_total) for b in for_head]
    return [] if any(not r for r in results) else results[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("body", nargs="?",
                    help="file holding the pull request description ('-' for stdin)")
    ap.add_argument("--body-env", metavar="NAME",
                    help="read the description from this environment variable instead")
    ap.add_argument("--head-sha", required=True, help="full SHA of the pull request head")
    ap.add_argument("--head-ref", default="",
                    help="head branch name; a release* branch requires the real-data run")
    ap.add_argument("--junit", help="this run's junit file, to cross-check the collected count")
    a = ap.parse_args()

    if a.body_env:
        body = os.environ.get(a.body_env, "")
    elif a.body in (None, "-"):
        body = sys.stdin.read()
    else:
        body = open(a.body, encoding="utf-8").read()
    total = (sum(1 for _ in ET.parse(a.junit).getroot().iter("testcase"))
             if a.junit else None)

    problems = check(body, a.head_sha, a.head_ref.startswith(RELEASE_PREFIX), total)
    for p in problems:
        print(f"FAIL: {p}", file=sys.stderr)
    if not problems:
        print(f"verification block matches {a.head_sha[:MIN_SHA]}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
