"""Shared anonymisation for the real-derived `.dat` subsets (spec §2b).

`make_tto` and `make_acms` each grew a private copy of this, differing only in the
probe-specific sample-line rule. The VSM and heat-capacity examples are the third and fourth
users, so the common part lives here once and each generator passes its own rules in.

The contract every caller relies on: values are REPLACED in place, never deleted, so the
loader's parse path stays exercised on the shipped file, and `assert_no_identity_leak` is a
post-condition that fails the write loudly rather than shipping a leak.
"""
from __future__ import annotations
import pathlib
import re

# Neutral publication stamp: the shipped subsets carry the repo's sanitisation date, never the
# real acquisition date. The FIRST field (the QD time base) is zeroed too — it is the ONLY thing
# that ever tied the absolute "Time Stamp (sec)" column to a calendar date, so with it gone that
# column is a bare session counter and needs no rebasing.
NEUTRAL_TIME = "FILEOPENTIME,0.00,09/01/2026,12:00 am"


def anonymize_header(lines, title, sample_rules=()):
    """Neutralise the identifying header fields, keep everything load-bearing.

    TITLE and FILEOPENTIME are handled here because every QD format carries them.
    `sample_rules` is an iterable of (predicate, replacement) for the probe-specific free-text
    sample field — replacement may be a string or a callable taking the line. BYAPP and the
    geometry/INFO rows the analyzers read are load-bearing and are never touched.
    """
    out = []
    for ln in lines:
        if ln.startswith("TITLE,"):
            out.append(f"TITLE,{title}")
            continue
        if ln.startswith("FILEOPENTIME,"):
            out.append(NEUTRAL_TIME)
            continue
        for pred, repl in sample_rules:
            if pred(ln):
                out.append(repl(ln) if callable(repl) else repl)
                break
        else:
            out.append(ln)
    return out


def identity_values(src_lines, sample_line):
    """The SOURCE header's identity-bearing values: TITLE plus the free-text sample field."""
    vals = [ln.split(",", 1)[1] for ln in src_lines if ln.startswith("TITLE,")]
    vals += [v for v in (sample_line(ln) for ln in src_lines) if v and v.strip()]
    return vals


def assert_no_identity_leak(head, src_lines, sample_line):
    """Post-condition: no token of the source's identity survives anywhere in the header.

    `anonymize_header` rewrites a FIXED list of fields, so by construction it can only clean
    fields we already thought of. This is the net for the case it cannot cover — the same
    sample string duplicated into some OTHER line (a COMMENT, a second INFO, a free-text
    geometry note). It fails the regeneration loudly instead of shipping the leak.

    Tokens are >=4 alphanumeric characters, compared case-insensitively; shorter fragments
    ("U0", "AC", "He3") are too common to match on without false positives. Only the source's
    identity fields seed the tokens: seeding from the whole header would match the APPNAME and
    Quantum Design lines, which are format provenance and are deliberately kept.
    """
    toks = {t.lower()
            for v in identity_values(src_lines, sample_line)
            for t in re.split(r"[^A-Za-z0-9]+", v) if len(t) >= 4}
    blob = "\n".join(head).lower()
    leaked = sorted(t for t in toks if t in blob)
    if leaked:
        raise AssertionError(
            f"anonymisation leak: source identity token(s) {leaked} survive in the "
            f"header of the subset about to be written — extend the sample rules")


def split_at_data(src, encoding="latin-1"):
    """(header lines incl. the column row, non-empty body rows). latin-1: QD headers carry
    micro signs, and a bare read_text() raises UnicodeDecodeError 0xb5 on them."""
    lines = pathlib.Path(src).read_text(encoding=encoding).splitlines()
    di = next(i for i, ln in enumerate(lines) if ln.strip() == "[Data]")
    return lines[:di + 2], [ln for ln in lines[di + 2:] if ln.strip()]


def set_info(head, key, value, desc):
    """Set `INFO,<value>,<KEY>:<desc>`, replacing that key's row if present, else inserting it
    just before `[Data]`. The loader keys on the part before the ':' (io/header.py), so the
    description is free text and only the key and value are load-bearing.

    Used to publish NEUTRAL sample metadata: a real formula weight is a fingerprint, and a file
    with no MOLWGHT/MASS at all analyzes as `gated`, which is a poor first-run example.
    """
    row = f"INFO,{value},{key}:{desc}"
    out, seen = [], False
    for ln in head:
        parts = [p.strip() for p in ln.split(",")]
        is_key = (len(parts) >= 3 and parts[0].upper() == "INFO"
                  and parts[2].split(":", 1)[0].strip().upper() == key.upper())
        if is_key and not seen:
            out.append(row)
            seen = True
        elif not is_key:
            out.append(ln)
    if not seen:
        i = next(j for j, ln in enumerate(out) if ln.strip() == "[Data]")
        out.insert(i, row)
    return out


def write_subset(dst, head, body, step=1, encoding="latin-1"):
    """Write header + every `step`-th body row."""
    pathlib.Path(dst).write_text("\n".join(head + body[::step]) + "\n", encoding=encoding)


# A QD "Time Stamp (sec)" column is either a small session counter or an ABSOLUTE count of
# seconds since 1900-01-01. The TTO/ACMS subsets carry the former (~3-5e6, decodes to nothing);
# the VSM and heat-capacity sources carry the latter (~3.9e9) and decode to the real
# acquisition instant — 2025-01-11 12:43:14 and 2023-08-19 23:09:39. Anything past this
# threshold is a date and gets rebased to zero; intervals are preserved either way.
_ABSOLUTE_EPOCH_MIN = 1.0e9

# Comment cells are operator/instrument free text. Publishing uses an ALLOWLIST, not a
# blocklist: the real heat-capacity file carries
#   "CALFILE: C:\QDDYNA~1\...\Puck1659.cal|Addenda #51 measured on 8/16/2023 ..."
# — a lab filesystem path, a calibration-puck serial, an addenda number and a date, in a column
# no analyzer reads. Only benign instrument warnings (no path separators, no dates, no '#')
# survive; everything else is replaced. Guessing which patterns are identifying is exactly the
# mistake this avoids.
_BENIGN_COMMENT = re.compile(r"^Error: Warning: [A-Za-z0-9 .,;:()\-]+$")


def _col_index(cols, predicate):
    return next((i for i, c in enumerate(cols) if predicate(c)), None)


def scrub_body(header_cols, body, benign=_BENIGN_COMMENT):
    """Rebase an absolute Time Stamp column and neutralise identifying Comment cells.

    Returns (body, report). Neither column is canonicalized or read by any analyzer (verified
    by grep over cryosweep_core), so this changes no computed result — it only removes the two
    body-level identity channels that header anonymisation cannot reach.
    """
    cols = [c.strip() for c in header_cols]
    ti = _col_index(cols, lambda c: "time stamp" in c.lower())
    ci = _col_index(cols, lambda c: c.lower().startswith("comment"))
    rows = [ln.split(",") for ln in body]
    report = {"time_rebased": False, "comments_replaced": 0, "comments_kept": 0}

    if ti is not None:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[ti]) if ti < len(r) and r[ti].strip() else None)
            except ValueError:
                vals.append(None)
        first = next((v for v in vals if v is not None), None)
        if first is not None and first >= _ABSOLUTE_EPOCH_MIN:
            sample = next((r[ti] for r in rows if ti < len(r) and r[ti].strip()), "0")
            nd = len(sample.split(".")[1]) if "." in sample else 0
            for r, v in zip(rows, vals):
                if v is not None and ti < len(r):
                    r[ti] = f"{v - first:.{nd}f}"
            report["time_rebased"] = True

    if ci is not None:
        for r in rows:
            if ci < len(r) and r[ci].strip():
                if benign.match(r[ci].strip()):
                    report["comments_kept"] += 1
                else:
                    r[ci] = "anonymized"
                    report["comments_replaced"] += 1

    return [",".join(r) for r in rows], report
