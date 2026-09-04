# Contributing to cryosweep

Bug reports, measurement files that break the loader, and pull requests are all welcome.

## Before a pull request: the CLA

cryosweep is dual-licensed — free for noncommercial use, separately licensed for commercial
use — and that requires one party to be able to grant both licences over the whole codebase.
So contributions need a [Contributor License Agreement](CLA.md). You keep your copyright; you
grant a licence broad enough to sublicense.

Signing takes a minute and happens once, on your first pull request: a bot comments with a
sentence to reply with. See [How to sign](CLA.md#how-to-sign).

Typo, link, and whitespace fixes are not copyrightable and are not blocked by the check.

## Reporting a bug

The most useful bug report for this project is **a file that misbehaves**. If you can share
the `.dat` file, that is worth more than any description of the symptom.

If you cannot share it — most measurement data cannot be shared — the next best thing is:

1. the header block (everything above `[Data]`), with the sample name and any comment fields
   removed;
2. the column-name line;
3. a handful of representative data rows;
4. what you expected, and what the app reported instead.

`cryosweep analyze <file>` output (JSON on stdout) is also useful: it carries the probe detection,
capabilities and warnings without carrying your data.

## Development setup

```bash
git clone <your fork>
cd cryosweep
pip install -e .
pip install pytest
QT_QPA_PLATFORM=offscreen python -m pytest -q
```

Run the tests from the repository root — some build fixture paths relative to the working
directory and launch the CLI as a subprocess.

A fresh clone is green with no data at all. Tests that need real measurement files, or the
maintainer's reference gallery, **skip** when those are absent. That is deliberate: an absent
local-only file is a skip, never a failure. Keep it that way in new tests.

## What a good pull request looks like

**Tests come with the change, and they must fail before it.** The convention here is to show
that: write the test, watch it fail for the stated reason, then fix. A test that passes both
before and after is not pinning anything.

**Physics claims carry their evidence.** If a change alters a fitted number, a threshold, or a
detection rule, say what you measured — the file, the before and after values, and why the new
one is right. Several of the rules in this codebase exist because a number that looked fine
turned out to be a search bound or an artifact of a window; comments record those measurements
so the next person does not re-derive them.

**Prefer declining to guessing.** Where a fit is not determined, the analyzers report that it
declined and why, rather than publishing a number. New analysis should follow that: a result
that cannot be trusted is more useful reported as untrustworthy than as a value.

**Don't commit measurement data.** `*.dat` is gitignored except for the committed fixtures and
examples. If you add an example derived from a real measurement, it must go through the
anonymization path in `tests/core/fixtures/_anonymize.py` — note that scrubbing the header is
not enough, because the `Time Stamp` column can decode to the acquisition date and the
`Comment` column can carry lab paths and calibration serials.

## Naming the project

`cryosweep` lowercase for the package, the command, the module and the repository;
**CryoSweep** in prose and in user-facing titles (the report header, the window title). Same
convention NumPy and SciPy use. Keep new code and docs on that split.

## Style

Match the surrounding code: it is plain Python with no formatter enforced, comments that
explain *why* rather than *what*, and module docstrings that carry the measured facts a reader
would otherwise have to rediscover.

Conventional commit prefixes: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`.

## Questions

Open an issue, or email **info@cryosweep.org**.
