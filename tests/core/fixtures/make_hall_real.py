"""Anonymized real Hall example — the file KNOWN-ISSUES 18/19/20 needed and never had.

The source is a real Hall-wired PPMS Resistivity-option measurement: nine field loops at
held temperatures (2–300 K, B ±90 kOe) interleaved with fixed-field temperature ramps
(B = 0 between loops, ±40 kOe over 2–25 K, ±90 kOe over 2–150 K). Its value is exactly
what synthetic files lack: temperature setpoints that drift across the old round(·,1)
bin edge (the 200 K loop spans 199.84–199.99 K), and temperatures covered by a single ±
field pair. Items 18–20 were found on this data and reproduce on no synthetic example.

Decimation is STRUCTURE-AWARE because naive row-stepping destroys the payload (measured:
a global every-3rd-row subset already loses the +40 kOe ramp to the sweep segmenter, and
every-6th loses everything but ±90 kOe): held-field temperature-ramp rows are identified
as contiguous runs of ≥ `_RAMP_MIN_ROWS` rows with |ΔB| ≤ 500 Oe between neighbours; the
short ±40 kOe ramps are kept WHOLE, the longer B=0 / ±90 kOe ramps are kept at every 2nd
row, and the field-loop rows (everything else) at every 4th — each stretch keeping its
first and last row, so field extremes and ramp endpoints survive.

Identity: the source header is already anonymous (`TITLE, default name`, empty sample
names) — this file's identity lives in its FILENAME and the file-open date. On top of
the shared header/body machinery, `_assert_no_filename_leak` extends the identity net to
the source-filename channel, which the header-seeded net cannot see.
"""
from __future__ import annotations
import csv
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _anonymize import (anonymize_header, assert_no_identity_leak,   # noqa: E402
                        scrub_body, split_at_data)

_SAMPLE_RULES = ()                      # sample-name INFO rows are empty in the source
_SAMPLE_LINE = (lambda ln: None)

_RAMP_MIN_ROWS = 20                     # a held-field T-ramp; loops step B every row
_RAMP_HOLD_OE = 500.0                   # |dB| below this = same held field
_SHORT_RAMP_KEEP_WHOLE_OE = (30000.0, 60000.0)   # the ±40 kOe ramps: segmenter-fragile


def _field_runs(B):
    runs, s = [], 0
    for i in range(1, len(B) + 1):
        if i == len(B) or abs(B[i] - B[i - 1]) > _RAMP_HOLD_OE:
            runs.append((s, i))
            s = i
    return runs


def _select_rows(body, loop_step=4, ramp_step=2):
    """Structure-aware row selection; returns sorted kept indices."""
    rows = [r for r in csv.reader(body)]
    B = np.array([float(r[4]) for r in rows])
    keep = np.zeros(len(B), bool)
    is_ramp = np.zeros(len(B), bool)
    for a, b in _field_runs(B):
        if b - a < _RAMP_MIN_ROWS:
            continue
        is_ramp[a:b] = True
        lo, hi = _SHORT_RAMP_KEEP_WHOLE_OE
        step = 1 if lo < abs(float(np.median(B[a:b]))) < hi else ramp_step
        keep[a:b:step] = True
        keep[a] = keep[b - 1] = True
    i = 0
    while i < len(B):                    # loop rows: contiguous non-ramp stretches
        if is_ramp[i]:
            i += 1
            continue
        j = i
        while j < len(B) and not is_ramp[j]:
            j += 1
        keep[i:j:loop_step] = True
        keep[i] = keep[j - 1] = True
        i = j
    return np.flatnonzero(keep)


def _assert_no_filename_leak(src, out_text):
    """No ≥4-char token of the source FILENAME may survive in the output. The shared
    header net seeds only from header identity fields; this source's identity is its
    name. Generic format words that legitimately appear in any such file are excluded."""
    benign = {"resistivity", "option", "hall"}
    toks = {t.lower() for t in re.split(r"[^A-Za-z0-9]+", pathlib.Path(src).stem)
            if len(t) >= 4} - benign
    blob = out_text.lower()
    leaked = sorted(t for t in toks if t in blob)
    if leaked:
        raise AssertionError(
            f"anonymisation leak: source-filename token(s) {leaked} survive in the "
            f"subset about to be written")


def write_real_example(src, dst, title="hall_mixed_sweeps.dat"):
    head, body = split_at_data(src)
    head = anonymize_header(head, title, _SAMPLE_RULES)
    # The source TITLE is the QD placeholder "default name" — format boilerplate, not an
    # identity. Seeding the leak net from it would flag the word "name" in the format's
    # own "Sample1 Name" rows. Only that exact placeholder is excluded; a real title in
    # any future source still seeds the net.
    src_head = [ln for ln in split_at_data(src)[0]
                if ln.strip().lower() != "title, default name"]
    assert_no_identity_leak(head, src_head, _SAMPLE_LINE)
    # Geometry stays UNSET (Cross Section = 1, Length = 1) on purpose: this ships the
    # first public reproducer of the geometry-unset warning path. No MOLWGHT/MASS —
    # the resistivity format needs neither.
    body, rep = scrub_body(head[-1].split(","), body)
    idx = _select_rows(body)
    kept = [body[i] for i in idx]

    # Post-conditions on the PAYLOAD, not just the identity: the file ships FOR its
    # bin-edge-straddling 200 K loop and its intact ±40 kOe ramps. Failing the write is
    # better than shipping a subset that no longer exercises anything.
    T = np.array([float(next(csv.reader([ln]))[3]) for ln in kept])
    tt = T[np.abs(T - 200.0) < 0.5]
    lo, hi = tt[tt < 199.95], tt[tt >= 199.95]
    assert len(lo) >= 5 and len(hi) >= 5 and np.median(hi) - np.median(lo) > 0.05, \
        "decimation lost the 200 K bin-edge straddle"

    out = "\n".join(head + kept) + "\n"
    _assert_no_filename_leak(src, out)
    pathlib.Path(dst).write_text(out, encoding="latin-1")
    return {"n_rows": len(kept), **rep}
