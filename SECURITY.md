# Security policy

## Supported versions

Only the latest release receives fixes.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Email
**info@cryosweep.org** with a description and, if possible, a file or
command that reproduces it. You will get an acknowledgement within 30 days.

## Scope worth knowing about

cryosweep parses instrument files (`.dat`) that may come from untrusted
sources. Parser crashes on malformed files are ordinary bugs; anything that
makes the parser execute code, write outside the chosen output paths, or
exfiltrate file contents is a security issue.

Measurement files can embed identifying metadata (sample names, operator
comments, instrument serial numbers, acquisition timestamps). cryosweep never
transmits your data anywhere — analysis is entirely local — but be aware of
that metadata before sharing raw files in bug reports; see CONTRIBUTING.md
for how to report a file without sharing it.
