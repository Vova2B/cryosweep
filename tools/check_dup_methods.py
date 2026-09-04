#!/usr/bin/env python3
"""Duplicate-method gate: Python silently keeps the LAST def of a repeated method name.

Walks every *.py under the given roots (default: cryosweep_core, cryosweep_gui, cli, tests relative
to the CWD), exits 1 printing `path:Class:method` for every class with duplicate direct
method names. Stdlib-only by design: runs on a bare CI python.
"""
from __future__ import annotations
import ast
import collections
import pathlib
import sys

DEFAULT_ROOTS = ("cryosweep_core", "cryosweep_gui", "cryosweep_cli", "tests")


def find_duplicates(source: str, path: str) -> list[str]:
    """Formatted `path:Class:method` lines for one module's source (sorted, deduped)."""
    tree = ast.parse(source, filename=path)
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # DIRECT children only: a nested class's methods belong to the nested class.
            names = [n.name for n in node.body
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            for name, count in sorted(collections.Counter(names).items()):
                if count > 1:
                    out.append(f"{path}:{node.name}:{name}")
    return out


def main(argv: list[str]) -> int:
    roots = [pathlib.Path(a) for a in argv] or [pathlib.Path(r) for r in DEFAULT_ROOTS]
    lines: list[str] = []
    for root in roots:
        for py in sorted(root.rglob("*.py")):
            if "__pycache__" in py.parts:
                continue
            try:
                src = py.read_text(encoding="utf-8", errors="replace")
                lines.extend(find_duplicates(src, str(py)))
            except SyntaxError as e:  # a file that doesn't parse is its own loud failure
                lines.append(f"{py}:<syntax error>:{e.lineno}")
    for line in lines:
        print(line)
    return 1 if lines else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
