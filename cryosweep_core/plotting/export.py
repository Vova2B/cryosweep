"""Figure export: the single enforcement point of the exact-mm sizing contract.

Every save path (GUI single-plot, GUI batch dialog, CLI plot) routes through
save_figure(); export_plots() is the batch loop over a PlotLayout.
"""
from __future__ import annotations

import pathlib

import matplotlib
import matplotlib.pyplot as plt

from .render import render_kind
from .spec import GlobalStyle, PlotLayout, PlotSpec

_FORMATS = ("png", "pdf", "svg")

# Volatile metadata pinned per backend so exporting twice is byte-identical.
_METADATA = {"png": None,
             "pdf": {"CreationDate": None},
             "svg": {"Date": None}}


def save_figure(fig, path, style: GlobalStyle, spec: PlotSpec | None = None,
                fmt: str | None = None, tight: bool = False) -> pathlib.Path:
    """Save `fig` honouring the exact-mm contract: the file measures
    (spec.width_mm or style.width_mm) x (spec.height_mm or style.height_mm)
    at style.dpi (PNG), unless tight=True re-crops (off by default)."""
    path = pathlib.Path(path)
    fmt = (fmt or path.suffix.lstrip(".")).lower()
    if fmt not in _FORMATS:
        raise ValueError(f"unsupported format {fmt!r}: allowed {'/'.join(_FORMATS)}")
    if path.suffix.lstrip(".").lower() != fmt:
        path = path.with_suffix(f".{fmt}")
    # an explicitly named output directory is created, same as export_plots does
    path.parent.mkdir(parents=True, exist_ok=True)

    w_mm = spec.width_mm if spec is not None and spec.width_mm is not None else style.width_mm
    h_mm = spec.height_mm if spec is not None and spec.height_mm is not None else style.height_mm
    # A relocated dense legend grew the canvas at render time; keep that extra
    # width at export or the axes get squeezed back to a sliver.
    w_in = w_mm / 25.4 + getattr(fig, "_cryosweep_legend_extra_in", 0.0)
    h_in = h_mm / 25.4
    if fmt == "png" and not tight:
        # Snap to whole pixels: mpl truncates w_in*dpi, which would undershoot the
        # mm target by up to one pixel. The +1e-6 px keeps the product strictly
        # above the integer so Agg's int() truncation can't drop it back down.
        # Vector formats keep the true mm size.
        w_in = (round(w_in * style.dpi) + 1e-6) / style.dpi
        h_in = (round(h_in * style.dpi) + 1e-6) / style.dpi

    kwargs = {"facecolor": "white", "edgecolor": "none"}
    if _METADATA[fmt] is not None:
        kwargs["metadata"] = _METADATA[fmt]
    if tight:
        kwargs["bbox_inches"] = "tight"

    old_size = fig.get_size_inches().copy()
    # Journal text: keep PDF/SVG text as editable text, not paths.
    # svg.hashsalt pins the otherwise-randomized SVG element ids (determinism).
    with matplotlib.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none",
                                "svg.hashsalt": "cryosweep"}):
        fig.set_size_inches(w_in, h_in)
        try:
            fig.savefig(path, dpi=style.dpi, **kwargs)
        finally:
            fig.set_size_inches(*old_size)
    return path


def export_plots(result, layout: PlotLayout, style: GlobalStyle, out_dir, prefix: str,
                 kinds=None, formats=("png",), tight: bool = False) -> list[pathlib.Path]:
    """Render each layout entry (optionally filtered to `kinds`) and save it in
    each format as {prefix}_{kind}.{ext}. Unrenderable kinds are skipped, not
    fatal. Returns written paths in layout order x formats order."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[pathlib.Path] = []
    for entry in layout.plots:
        if kinds is not None and entry.kind not in kinds:
            continue
        try:
            fig = render_kind(result, entry.kind, entry.spec, style)
        except (ValueError, KeyError):
            continue        # gated / unknown-for-probe kind: skip, keep going
        try:
            for f in formats:
                written.append(save_figure(fig, out_dir / f"{prefix}_{entry.kind}",
                                           style, spec=entry.spec, fmt=f, tight=tight))
        finally:
            plt.close(fig)
    return written
