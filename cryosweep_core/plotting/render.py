from __future__ import annotations
import dataclasses
import math
import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.lines import Line2D
from matplotlib.ticker import (FixedLocator, FuncFormatter, MaxNLocator, ScalarFormatter)
from cryosweep_core.fitting.heat_capacity import _LOWT_FUNCS
from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle
from cryosweep_core.plotting.catalog import (BUILTIN_PLOTKINDS, select_series, series_label,
                                        _field_scale, _held, _acms_label,
                                        tto_field_ls_map, tto_field_ls_label)
from cryosweep_core.robust import robust_range
from cryosweep_core.fitting.heat_capacity import MU_B_OVER_KB
from cryosweep_core.fitting.entropy import dulong_petit_limit

_MM = 1.0 / 25.4
_KIND = {k.key: k for k in BUILTIN_PLOTKINDS}

_CONNECT_KINDS = {
    "inverse_chi", "vsm_moment_t", "vsm_chi_t", "vsm_chi_t_product",
    "resistivity_rho_t", "resistivity_mr_pct_t", "cp_over_t", "hc_c_over_t_linear",
    "hall_rh_t", "hall_mobility_t", "hall_n_t", "hall_r2_t",
    "hall_tdep_RH_T", "hall_tdep_n_T", "hall_tdep_mobility_T", "hall_tdep_J_T",
    "tto_summary_t", "tto_kappa_t", "tto_seebeck_t", "tto_zt_t", "tto_wf_t", "tto_lorenz_t",
}

# Artists that are DRAWN CONCLUSIONS, not measurements. Every "which lines are the data?"
# filter must exclude all of them, so the set lives in ONE place: adding a new overlay gid
# here updates the robust view, the legend-marker snapshots and the tests at once. (When
# "fit-extrap" was added at 7 literal call sites instead, 7 tests silently reclassified the
# extrapolation as DATA -- the same shape as every other "check that no longer sees what it
# checks" bug in this file's history.)
NON_DATA_GIDS = ("refline", "fit", "fit-extrap")

#: The frameless stats box every renderer draws (theta/C, gamma/beta/theta_D, rho0/RRR/Tc,
#: S_total, E_a). One source of truth: it used to be identified by its pinned literal
#: position `(0.02, 0.98)`, which stopped being an identity the moment `_place_annotation`
#: earned the right to move it.
ANNOTATION_GID = "annotation"

LEGEND_INSIDE_MAX = 11   # <= this many entries -> legend stays inside, byte-identical to today
LEGEND_MAX_ROWS   = 14   # rows per column when relocated (caps legend height)
LEGEND_MAX_COLS   = 4    # max columns when relocated (caps legend width -> axes can't collapse)
LEGEND_DENSE_PT   = 7    # font CAP for a relocated (dense) legend -> widens the plot at narrow sizes

class NothingToPlot(ValueError):
    """This file simply has no data for the requested plot kind (e.g. an MR panel on a file
    with no field sweep). A ValueError subclass so every existing `except ValueError` caller
    keeps working; the GUI shows it as a calm "not applicable" note instead of an error."""


def _legend_ncol(n):
    return min(LEGEND_MAX_COLS, math.ceil(n / LEGEND_MAX_ROWS))

def _draw_legend(ax, legend_prop, style, spec, handles=None, force_loc=None):
    """legend_on toggle, frame toggle, and placement. Placement states:
      - an explicit matplotlib position ("upper left", ...) -> verbatim, the user's call;
      - "inside" -> the occupancy chooser's clearest inside position, never relocated;
      - "best" (default) -> occupancy chooser when <=LEGEND_INSIDE_MAX entries; relocated
        outside-right when the figure has no clear inside spot (or is genuinely dense);
      - "outside" -> forced outside-right.
    `force_loc` overrides the spec/style loc resolution (composite renderers)."""
    legend_on = spec.legend_on if spec.legend_on is not None else style.legend_on
    if not legend_on:
        return
    n = len(handles) if handles is not None else len(ax.get_legend_handles_labels()[1])
    if n == 0:
        return
    if force_loc is not None:
        loc = force_loc
    else:
        loc = spec.legend_loc if spec.legend_loc is not None else style.legend_loc
    base = {"prop": legend_prop, "frameon": style.legend_frame}
    if handles is not None:
        base["handles"] = handles
    if loc in _LEGEND_INSIDE_LOCS:               # explicit position: pass through verbatim
        ax.legend(**base, loc=loc)
        return
    if loc == "inside" or (loc == "best" and n <= LEGEND_INSIDE_MAX):
        pick, clear = _occupancy_legend_loc(ax, handles, legend_prop, style)
        if clear or loc == "inside":             # "inside" always stays in (least-bad spot)
            ax.legend(**base, loc=pick)
            return
        # "best" with no clear inside spot -> the outside-right relocation below
    dense_prop = dict(legend_prop)                       # cap font only for the relocated dense legend
    dense_prop["size"] = min(legend_prop.get("size", LEGEND_DENSE_PT), LEGEND_DENSE_PT)
    base["prop"] = dense_prop
    anchor_x = 1.02
    if len(ax.get_figure().axes) > 1:
        # Twin/offset composites: the right margin at x>1.0 already holds the twin axis'
        # tick numbers + rotated ylabel (and an offset axis' spine at e.g. 1.18). Anchoring
        # at the host's 1.02 overprints them, so push the anchor past ALL axes' right-side
        # y-decorations instead.
        anchor_x = max(anchor_x, _right_decor_frac(ax) + 0.02)
    base.update(loc="center left", bbox_to_anchor=(anchor_x, 0.5), ncol=_legend_ncol(n))
    leg = ax.legend(**base)
    _grow_canvas_for_legend(ax, leg, gap_frac=anchor_x - 1.02)

def _right_decor_frac(ax_host):
    """Rightmost extent of every axis' y-axis decorations (tick labels + axis label), in
    ax_host axes-fraction x. Draws the canvas first so tick/label extents are realized.
    Never less than 1.0 (the axes' own right edge)."""
    fig = ax_host.get_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax_host.transAxes.inverted()
    xmax = 1.0
    for ax in fig.axes:
        bb = ax.yaxis.get_tightbbox(renderer)
        if bb is not None:
            xmax = max(xmax, inv.transform((bb.x1, bb.y0))[0])
    return xmax

def _grow_canvas_for_legend(ax, leg, gap_frac=0.0):
    """An outside-right legend at fixed canvas width squeezes the axes (constrained
    layout takes the legend's space from the axes). Grow the canvas by the legend's
    measured width instead so the axes keep their intended size. Once per figure.
    `gap_frac` = extra anchor offset beyond the standard 1.02 (axes-fraction of the host),
    reserved additionally so a decoration-cleared anchor doesn't re-squeeze the axes."""
    fig = ax.get_figure()
    if getattr(fig, "_cryosweep_legend_grown", False):
        return
    fig._cryosweep_legend_grown = True
    fig.canvas.draw()                                    # realize layout to measure the legend
    leg_in = leg.get_window_extent().width / fig.dpi
    if gap_frac > 0:
        leg_in += gap_frac * ax.get_window_extent().width / fig.dpi
    fig._cryosweep_legend_extra_in = leg_in + 0.1             # honoured by GUI export resize too
    w, h = fig.get_size_inches()
    fig.set_size_inches(w + fig._cryosweep_legend_extra_in, h)

def _new_fig(style):
    fig = Figure(figsize=(style.width_mm * _MM, style.height_mm * _MM),
                 dpi=style.dpi, layout="constrained")
    FigureCanvasAgg(fig)
    return fig

def _as_list(results):
    return results if isinstance(results, list) else [results]

def _cmap_colors(name, n, reverse):
    try:
        cmap = matplotlib.colormaps[name]
    except KeyError:
        return None                      # unknown name -> caller falls back to default cycle
    if reverse:
        cmap = cmap.reversed()
    return [cmap(0.5)] if n == 1 else [cmap(t) for t in np.linspace(0.0, 1.0, n)]

def _file_colours(overlay, n, style):
    """One concrete colour per file: explicit override > colormap sample > palette > default cycle.
    Contract: n == len(overlay) (the overlay branch zips results+overlay, so it always holds)."""
    cyc = matplotlib.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    cmap = _cmap_colors(style.colormap, n, style.colormap_reverse) if style.colormap else None
    out = []
    for i, of in enumerate(overlay):
        if of.colour is not None:
            out.append(of.colour)
        elif cmap is not None:
            out.append(cmap[i])
        elif style.palette:
            out.append(style.palette[i % len(style.palette)])
        else:
            out.append(cyc[i % len(cyc)])
    return out

def _errorbar_series(ax, s, kw):
    """Draw a Series that carries per-point error bars.
    Solid points (open_mask[i] False or open_mask absent) are drawn first and lock
    the color.  Hollow points (open_mask[i] True) are drawn in the SAME color with
    markerfacecolor='none' and without a duplicate legend label.
    The existing ax.plot path is NOT used; kw is only consulted for color/label."""
    x = np.asarray(s.x, float)
    y = np.asarray(s.y, float)
    ye = np.asarray(s.yerr, float)
    ebkw = {k: v for k, v in kw.items() if k in ("color", "label")}
    ebkw.update(fmt="o", capsize=3, linestyle="none")
    mask = np.asarray(s.open_mask, bool) if s.open_mask is not None else np.zeros(x.size, bool)
    color = ebkw.get("color")
    has_solid = (~mask).any()
    if has_solid:
        cont = ax.errorbar(x[~mask], y[~mask], yerr=ye[~mask], **ebkw)
        if color is None:                       # lock cycled colour so hollow points match
            color = cont.lines[0].get_color()
    if mask.any():                              # flagged / non-identifiable -> hollow, same colour
        ekw = dict(ebkw)
        if has_solid:
            ekw.pop("label", None)              # solid call already owns the legend entry
        # if all points are masked (no solid), keep label so series appears in the legend once
        ekw["markerfacecolor"] = "none"
        ekw["color"] = color
        ax.errorbar(x[mask], y[mask], yerr=ye[mask], **ekw)


def _connect_sort(x, y):
    """Order (x, y) by x for a connected line. If x carries NaN break markers (VSM ramp-split
    concatenates same-(field,direction) blocks and inserts a NaN between them, per catalog.py),
    sort WITHIN each finite segment but preserve one NaN separator between segments so the line
    breaks at the data gap instead of bridging it. Plain argsort would sink the NaN to the end.
    No NaN in x -> plain argsort (byte-identical to prior behaviour for every existing kind)."""
    finite = np.isfinite(x)
    if finite.all():
        order = np.argsort(x)
        return x[order], y[order]
    out_x, out_y = [], []
    i, n, first = 0, len(x), True
    while i < n:
        if not finite[i]:
            i += 1
            continue
        j = i
        while j < n and finite[j]:
            j += 1
        o = np.argsort(x[i:j])
        if not first:
            out_x.append(np.nan); out_y.append(np.nan)
        out_x.extend(x[i:j][o].tolist()); out_y.extend(y[i:j][o].tolist())
        first = False
        i = j
    return np.asarray(out_x, float), np.asarray(out_y, float)


def _plot_data(ax, results, kind, spec, style, overlay=None):
    """Plot the selected data series (markers only). Raises ValueError if nothing selected."""
    if overlay is None:
        plotted = []
        for r in results:
            for s in select_series(kind.series(r, field_unit=style.field_unit), spec):
                plotted.append((r, s))
        if not plotted:
            raise NothingToPlot(f"no series selected for kind {kind.key}")
        n = len(plotted)
        cmap_colors = _cmap_colors(style.colormap, n, style.colormap_reverse) if style.colormap else None
        connect = (spec.connect_lines if spec.connect_lines is not None else style.connect_lines) \
            and kind.key in _CONNECT_KINDS
        # PQ-3 Task 4: VSM ramp split. Only when a selected series carries a `linestyle`
        # (set exclusively by the VSM ramp-split builders) do we colour-by-group + apply the
        # per-series linestyle. Every existing kind leaves linestyle None -> byte-identical.
        ramp_mode = any(getattr(s, "linestyle", None) for _, s in plotted)
        ramp_gcolor = None
        if ramp_mode:
            _rgroups = []
            for _, s in plotted:
                if s.group not in _rgroups:
                    _rgroups.append(s.group)
            ramp_gcolor = _group_color_map(_rgroups, style)
        for i, (r, s) in enumerate(plotted):
            x, y = np.asarray(s.x, float), np.asarray(s.y, float)
            if connect and x.size:
                x, y = _connect_sort(x, y)
            label = series_label(r, s)
            if getattr(s, "label_suffix", "") and spec.direction_arrows:
                label = label + s.label_suffix
            kw = dict(marker=style.marker, ls=("-" if connect else "none"),
                      ms=style.marker_size, label=label)
            if getattr(s, "marker", None) is not None and spec.channel_markers:
                kw["marker"] = s.marker
            if connect:
                kw["lw"] = style.line_width
            if ramp_mode:
                kw["color"] = ramp_gcolor[s.group]              # shared colour per quantity
                if connect:
                    kw["ls"] = getattr(s, "linestyle", None) or "-"
            elif n == 1 and style.color:
                kw["color"] = style.color
            elif cmap_colors is not None:
                kw["color"] = cmap_colors[i]
            elif style.palette:
                kw["color"] = style.palette[i % len(style.palette)]
            if style.edge_color is not None:
                kw["markeredgecolor"] = style.edge_color
            if style.edge_width is not None:
                kw["markeredgewidth"] = style.edge_width
            if getattr(s, "yerr", None) is not None:
                _errorbar_series(ax, s, kw)
            else:
                ax.plot(x, y, **kw)
        return plotted
    # ---- overlay branch: file-qualified select, colour-by-file, within-file markers ----
    want = None if spec.curves is None else set(spec.curves)
    file_colours = _file_colours(overlay, len(results), style)
    markers = (style.marker, "s", "^", "D", "v", "P", "X")
    plotted = []
    for fi, (r, of) in enumerate(zip(results, overlay)):
        j = 0
        for s in kind.series(r, field_unit=style.field_unit):
            eff = f"{of.file_id}::{s.key}"
            if (want is None and s.default_on) or (want is not None and eff in want):
                s_label = s.label + (s.label_suffix if (getattr(s, "label_suffix", "")
                                                        and spec.direction_arrows) else "")
                kw = dict(marker=markers[j % len(markers)], ls="none", ms=style.marker_size,
                          color=file_colours[fi], label=f"{of.label} · {s_label}")
                if style.edge_color is not None:
                    kw["markeredgecolor"] = style.edge_color
                if style.edge_width is not None:
                    kw["markeredgewidth"] = style.edge_width
                ax.plot(s.x, s.y, **kw)
                plotted.append((r, s)); j += 1
    if not plotted:
        raise NothingToPlot(f"no series selected for kind {kind.key}")
    return plotted

def _group_color_map(groups, style):
    """One colour per group (e.g. temperature): explicit style.color wins only for a single
    group (lets a single-T/single-group figure honour GlobalStyle.color); otherwise colormap
    > palette > default cycle, matched by position to `groups`. Extracted from
    _plot_data_grouped so panelled renderers (PQ-2 Task 2 hall_tdep_stages) can share one
    group->colour map across multiple axes on the same figure."""
    ng = len(groups)
    if ng == 1 and style.color:
        gcolors = [style.color]
    else:
        cmap_colors = _cmap_colors(style.colormap, ng, style.colormap_reverse) if style.colormap else None
        if cmap_colors is not None:
            gcolors = cmap_colors
        elif style.palette:
            gcolors = [style.palette[i % len(style.palette)] for i in range(ng)]
        else:
            cyc = matplotlib.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
            gcolors = [cyc[i % len(cyc)] for i in range(ng)]
    return {g: gcolors[i] for i, g in enumerate(groups)}


def _plot_data_grouped(ax, results, kind, spec, style, role_colors=None):
    """Colour-by-group rendering for group_colored kinds (non-overlay only). One colour per
    group (e.g. temperature), one marker per within-group role; legend = one proxy per group
    + a per-role key. Returns (plotted, handles). Raises ValueError if nothing selected.

    role_colors: optional {role: colour} override, applied only when exactly one group is
    plotted (e.g. a single temperature) — lets a renderer force a classic role-based
    colouring (blue/red branches) for the single-T case while multi-T keeps colour-by-group.
    Kind-agnostic: no-op unless a caller passes it and ng == 1."""
    plotted = []
    for r in results:
        for s in select_series(kind.series(r, field_unit=style.field_unit), spec):
            plotted.append((r, s))
    if not plotted:
        raise NothingToPlot(f"no series selected for kind {kind.key}")
    # ordered distinct groups + roles (first-appearance order; catalog emits T-sorted)
    groups, roles = [], []
    for _, s in plotted:
        if s.group not in groups:
            groups.append(s.group)
        if s.role is not None and s.role not in roles:
            roles.append(s.role)
    ng = len(groups)
    gcolor = _group_color_map(groups, style)
    use_role_colors = role_colors is not None and ng == 1
    marker_chars = (style.marker, "s", "^", "D", "v", "P", "X")
    rmarker = {rl: marker_chars[i % len(marker_chars)] for i, rl in enumerate(roles)}
    def _color_for(s):
        if use_role_colors and s.role in role_colors:
            return role_colors[s.role]
        return gcolor[s.group]
    for r, s in plotted:
        if s.role == "fit":
            # fit curves are dense computed lines, not markers -> draw connected, no marker
            kw = dict(marker="none", ls=(style.fit_linestyle or "-"),
                      lw=style.line_width, color=_color_for(s), label="_nolegend_")
            ax.plot(s.x, s.y, **kw)
            continue
        kw = dict(marker=rmarker.get(s.role, style.marker), ls="none",
                  ms=style.marker_size, color=_color_for(s), label="_nolegend_")
        if style.edge_color is not None:
            kw["markeredgecolor"] = style.edge_color
        if style.edge_width is not None:
            kw["markeredgewidth"] = style.edge_width
        if getattr(s, "yerr", None) is not None:
            _errorbar_series(ax, s, kw)
        else:
            ax.plot(s.x, s.y, **kw)
    handles = [Line2D([], [], ls="none", marker="o", color=gcolor[g], label=g) for g in groups]
    if use_role_colors:
        handles += [Line2D([], [], ls="none", marker=rmarker[rl],
                            color=role_colors.get(rl, "0.35"), label=rl) for rl in roles]
    else:
        handles += [Line2D([], [], ls="none", marker=rmarker[rl], color="0.35", label=rl) for rl in roles]
    return plotted, handles


_ROBUST_NOOP_EPS = 0.05    # only narrow when robust hi < data_max·(1-eps) (or lo > data_min·(1+eps))
_ROBUST_PAD = 0.05         # fractional padding added around the robust [lo, hi] view range
# Garbage-line discriminator on robust SPANS (hi - lo). Span, NOT center/disjointness:
# garbage Hall noise is typically SYMMETRIC ABOUT ZERO, so its robust range always overlaps
# a zero-centered pooled bulk — a disjointness prong can never fire on it. And NOT a plain
# span-vs-median cut either: a legit multi-field M(T) family mixes flat low-field curves
# (span ~2e-4) with swinging high-field ones (span ~0.9, ~1000× the median) — real VSM_N
# example. What separates the populations is the SHAPE of the sorted-span sequence: a legit
# same-quantity family forms a quasi-continuous ladder whose CONSECUTIVE members are within
# ≲2 decades of each other (VSM_N max gap ~60×, MPMS 500 Oe→40000 Oe ~85×), while a garbage
# channel (non-Hall segment folded into an antisymmetrized R_xy view, instrument sentinels)
# sits ISOLATED, orders beyond the whole ladder (real Resistivity_example ch1 300 K line:
# ~3400× above its nearest neighbour, ~8000× the median). So: cut at the first consecutive
# sorted-span gap ≥ K whose upper side is also > K× the median span, and drop everything
# above it. K=100 (~2 decades) sits in the empty gap between the two populations.
_ROBUST_UNION_SPAN_K = 100.0

DEGENERATE_REL_SPAN = 1e-12


def _pad_degenerate_axis(ax, which="y", rel_floor=DEGENERATE_REL_SPAN):
    """Give an axis a usable span when its data is constant to machine precision.

    Without this the axis spans ~nothing, AutoLocator places ticks separated by float
    noise and ScalarFormatter prints them at full 17-significant-digit precision -- e.g.
    the R_H axis on hall_tdep_summary / hall_tdep_rh_n_twin (rel-span 3.31e-15) rendered
    a tick reading '-3.0000000000000004'. The data really is constant there, so no
    information is lost by giving the axis a readable scale.

    NOT a global tick-formatter override: that would risk every figure to fix two, and
    the formatter is not the cause. No-op on healthy axes (measured: 2 of 41 entries
    are degenerate; every other axis sits at rel-span >= 1e-1). matplotlib's own
    `nonsingular` does not help -- it expands only below ~1e-15 relative, and these sit
    just above that tolerance.
    """
    getter, setter = ((ax.get_ylim, ax.set_ylim) if which == "y" else (ax.get_xlim, ax.set_xlim))
    lo, hi = sorted(getter())
    mag = max(abs(lo), abs(hi), 1e-300)
    if (hi - lo) / mag >= rel_floor:
        return
    pad = abs(mag) * 1e-3
    mid = 0.5 * (lo + hi)
    setter(mid - pad, mid + pad)


def _apply_robust_view(ax, spec, style):
    """Robust y-view, then a degenerate-span guard.

    The guard lives here rather than in `_finish` because the hall tdep renderers
    (`render_hall_tdep_summary`, `_render_hall_rh_n_twin`) hand-roll their finish
    sequence and never call `_finish` on the non-overlay path -- but every one of them
    does come through here. It runs on all exit paths, including the early returns.
    """
    _apply_robust_view_core(ax, spec, style)
    if ax.get_yscale() == "linear":
        _pad_degenerate_axis(ax, "y")


def _apply_robust_view_core(ax, spec, style):
    """Default-on robust y-view. Genuine no-op on clean data: only sets ylim when a heavy tail
    is actually present. Bypassed on log-y, explicit spec ymin/ymax."""
    use = spec.robust_view if spec.robust_view is not None else style.robust_view
    if not use:
        return
    if ax.get_yscale() != "linear":
        return
    if spec.ymin is not None or spec.ymax is not None:
        return
    # Union of PER-LINE robust ranges: a multi-magnitude family (e.g. 500 Oe vs 40000 Oe
    # M(T)) must show every series' bulk. Pooling all y before robust_range let the low-
    # magnitude majority set an envelope that hid/clipped the high-magnitude series. Each
    # line contributes its own median±k·MAD envelope; the view is their union so every
    # series' bulk is inside it. Single series -> union of one -> byte-identical to before.
    # Fit overlays (gid=="fit") are already envelope-clipped to their series' data and must
    # not steer the view; reference lines carry transform-space coords -> both excluded.
    per_line = []
    for ln in ax.lines:
        if ln.get_gid() in NON_DATA_GIDS:
            continue
        a = np.asarray(ln.get_ydata(), float)
        a = a[np.isfinite(a)]
        if a.size:
            per_line.append(a)
    if not per_line:
        return
    ranges = []
    for a in per_line:
        lo_i, hi_i = robust_range(a, k=style.robust_k)
        if np.isfinite(lo_i) and np.isfinite(hi_i) and hi_i > lo_i:
            ranges.append((lo_i, hi_i))
        else:                                    # degenerate line: keep its full extent visible
            ranges.append((float(a.min()), float(a.max())))
    # Garbage-line discriminator (only meaningful with a family of ≥3 lines, so a stable
    # median line-span exists and "outlier vs family" is well-defined). With <3 lines the
    # union is taken as-is, so single-series stays a byte-identical union-of-one.
    keep = list(range(len(per_line)))
    if len(per_line) >= 3:
        spans = [hi_i - lo_i for lo_i, hi_i in ranges]
        med_span = float(np.median(spans))
        if med_span > 0:
            s_sorted = sorted(spans)
            cut = None                           # smallest span considered garbage
            for lower, upper in zip(s_sorted, s_sorted[1:]):
                if (upper > _ROBUST_UNION_SPAN_K * med_span
                        and lower > 0 and upper / lower >= _ROBUST_UNION_SPAN_K):
                    cut = upper
                    break
            if cut is not None:
                kept = [i for i, s in enumerate(spans) if s < cut]
                if kept:                         # never exclude every line
                    keep = kept
    # The union [lo, hi] comes from the KEPT lines; the tail/no-op test below runs against
    # ALL displayed data (excluded garbage included): an excluded line still drives matplotlib
    # autoscale, so once anything was excluded the view must be set, never left to autoscale.
    ranges = [ranges[i] for i in keep]
    ys = np.concatenate(per_line)
    if ys.size < 8:
        return
    lo = min(r[0] for r in ranges); hi = max(r[1] for r in ranges)
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return
    dmin, dmax = float(ys.min()), float(ys.max())
    data_span = dmax - dmin
    if data_span <= 0:
        return
    # narrow only when the robust range is materially tighter than the data on either end.
    # span-relative (not a signed multiplicative factor) so it is correct for negative data too.
    tail_hi = (dmax - hi) > _ROBUST_NOOP_EPS * data_span
    tail_lo = (lo - dmin) > _ROBUST_NOOP_EPS * data_span
    if not (tail_hi or tail_lo):
        return
    view_span = hi - lo
    ax.set_ylim(bottom=lo - _ROBUST_PAD * view_span, top=hi + _ROBUST_PAD * view_span)


def _axis_maxabs(lines, which):
    """Return max |value| in x or y data of all lines, excluding 'refline' gids."""
    vals = []
    for ln in lines:
        if ln.get_gid() == "refline":      # reference lines carry transform-space coords; exclude
            continue
        d = ln.get_xdata() if which == "x" else ln.get_ydata()
        a = np.abs(np.asarray(d, float))
        vals.append(a[np.isfinite(a)])
    if not vals:
        return 0.0
    a = np.concatenate(vals)
    return float(a.max()) if a.size else 0.0


def _apply_thousands(ax):
    """Comma-group ticks only on a linear axis whose data max|value| >= 1000; else no-op."""
    fmt = FuncFormatter(lambda v, pos: f"{v:,.0f}")
    if ax.get_xscale() == "linear" and _axis_maxabs(ax.lines, "x") >= 1000:
        ax.xaxis.set_major_formatter(fmt)
    if ax.get_yscale() == "linear" and _axis_maxabs(ax.lines, "y") >= 1000:
        ax.yaxis.set_major_formatter(fmt)


def _apply_frame(ax, style, spec):
    """Journal frame: spines (width), inward major+minor ticks on all 4 sides, grid toggle."""
    if style.spine_width is not None:
        for sp in ax.spines.values():
            sp.set_linewidth(style.spine_width)
    if style.minor_ticks:
        ax.minorticks_on()
    ax.tick_params(which="both", direction=style.tick_direction,
                   top=style.ticks_top, right=style.ticks_right)
    grid_on = spec.grid if spec.grid is not None else style.grid
    if grid_on:
        ax.grid(True, linestyle=style.grid_style, alpha=style.grid_alpha)
    else:
        ax.grid(False)
    if style.thousands_sep:
        _apply_thousands(ax)


def _draw_reference_lines(ax, spec):
    for rl in (spec.reference_lines or []):
        if rl.axis == "h":
            ax.axhline(rl.value, color=rl.color, linestyle=rl.linestyle,
                       linewidth=rl.linewidth, gid="refline")
            if rl.label:
                ax.text(0.02, rl.value, rl.label, transform=ax.get_yaxis_transform(),
                        va="bottom", ha="left", fontsize="small", gid="refline-label:h")
        else:
            ax.axvline(rl.value, color=rl.color, linestyle=rl.linestyle,
                       linewidth=rl.linewidth, gid="refline")
            if rl.label:
                ax.text(rl.value, 0.98, rl.label, transform=ax.get_xaxis_transform(),
                        va="top", ha="left", fontsize="small", gid="refline-label:v")


def refresh_legend(ax, style, spec):
    """Re-draw *ax*'s legend with the same props `_finish` used, for a caller that added or
    removed a labelled artist on an already-rendered figure (the GUI's live "model (manual)"
    overlay). Placement/toggle rules are `_draw_legend`'s, so the legend stays consistent
    with the original render."""
    spec = spec or PlotSpec()
    legend_sz = style.legend_size if style.legend_size is not None else style.font_pt - 1
    legend_prop = {"size": legend_sz}
    if style.font_family:
        legend_prop["family"] = style.font_family
    _draw_legend(ax, legend_prop, style, spec)
    # a re-drawn (taller) legend can land under a low-T inset the original cleared;
    # re-apply the same nudge the renderer used (no-op on inset-free figures)
    iax = next((a for a in ax.get_figure().axes if a.get_label() == "inset"), None)
    _legend_clear_of_inset(ax, iax)


def _finish(ax, kind, spec, style, xlabel, ylabel, legend_handles=None, draw_legend=True):
    ax.set_xscale(spec.xscale if spec.xscale is not None else kind.default_xscale)
    ax.set_yscale(spec.yscale if spec.yscale is not None else kind.default_yscale)
    if spec.xmin is not None or spec.xmax is not None:
        ax.set_xlim(left=spec.xmin, right=spec.xmax)
    if spec.ymin is not None or spec.ymax is not None:
        ax.set_ylim(bottom=spec.ymin, top=spec.ymax)
    fam = {"fontfamily": style.font_family} if style.font_family else {}
    label_sz = style.label_size if style.label_size is not None else style.font_pt
    title_sz = style.title_size if style.title_size is not None else style.font_pt
    legend_sz = style.legend_size if style.legend_size is not None else style.font_pt - 1
    ax.set_xlabel(xlabel, fontsize=label_sz, **fam)
    ax.set_ylabel(ylabel, fontsize=label_sz, **fam)
    # A view that is degenerate here would render 17-significant-digit tick labels.
    # Conditional, so healthy axes are untouched (measured: 2 of 41 entries qualify).
    if ax.get_yscale() == "linear":
        _pad_degenerate_axis(ax, "y")
    if ax.get_xscale() == "linear":
        _pad_degenerate_axis(ax, "x")
    _draw_reference_lines(ax, spec)
    legend_prop = {"size": legend_sz}
    if style.font_family:
        legend_prop["family"] = style.font_family
    if style.tick_size is not None:
        ax.tick_params(labelsize=style.tick_size)
    if spec.title:
        ax.set_title(spec.title, fontsize=title_sz, **fam)
    _apply_robust_view(ax, spec, style)
    _apply_frame(ax, style, spec)
    # Reference-line labels slide along their lines only now (final limits settled),
    # and BEFORE the legend, which treats text as an obstacle: inset -> labels -> legend.
    _place_refline_labels(ax, style)
    # Legend LAST, after the robust view has settled the final axis limits: the occupancy
    # chooser measures where the data sits in the realized view, so placing it against
    # pre-robust limits would score a stale geometry.
    if draw_legend:                     # callers that place the legend themselves pass False
        _draw_legend(ax, legend_prop, style, spec, legend_handles)


def _setup(results, kind_key, spec, style):
    spec = spec or PlotSpec(); style = style or GlobalStyle()
    fig = _new_fig(style); ax = fig.add_subplot(111)
    return _as_list(results), _KIND[kind_key], spec, style, fig, ax


def _setup_panels(results, kind_key, spec, style, n):
    """Same contract as `_setup` but returns a list of n side-by-side axes on one `_new_fig`
    figure, for composite/panelled renderers (PQ-2 Task 2 hall_tdep_stages; reused by Task 3's
    composites)."""
    spec = spec or PlotSpec(); style = style or GlobalStyle()
    fig = _new_fig(style)
    axes = fig.subplots(1, n)
    axes = [axes] if n == 1 else list(axes)
    return _as_list(results), _KIND[kind_key], spec, style, fig, axes

# ---- PQ-2 Task 3: shared multi-axes helpers (reused by PQ-3/4/5 composites) ----

def _twin_axis(ax, style, color):
    """ax.twinx() with journal-frame-consistent ticks; spine/label/tick colour-matched to
    `color` so a right-hand series axis reads as belonging to that series. Grid stays off
    (the host axis already draws it) to avoid double-drawn gridlines on composites."""
    tax = ax.twinx()
    if style.spine_width is not None:
        tax.spines["right"].set_linewidth(style.spine_width)
    if style.minor_ticks:
        tax.minorticks_on()
    tax.tick_params(which="both", direction=style.tick_direction, colors=color)
    tax.spines["right"].set_color(color)
    tax.yaxis.label.set_color(color)
    return tax

def _offset_axis(ax, style, color, pos=1.18):
    """Like `_twin_axis` but the right spine is pushed outward to `pos` (axes fraction) for a
    third y-axis; patch made invisible so it doesn't occlude the host plot."""
    oax = _twin_axis(ax, style, color)
    oax.spines["right"].set_position(("axes", pos))
    oax.set_frame_on(True)
    oax.patch.set_visible(False)
    return oax

_LEGEND_INSIDE_CORNERS = ("upper right", "upper left", "lower right", "lower left")
#: All nine matplotlib inside positions, corners first (matplotlib-best default bias), then
#: edge-centres, then centre — the tie-break preference order of the occupancy chooser.
_LEGEND_INSIDE_LOCS = _LEGEND_INSIDE_CORNERS + (
    "upper center", "lower center", "center left", "center right", "center")
_LEGEND_MULTIAXIS_OVERLAP_MAX = 0.02   # >2% of visible points under every position -> go outside

def _axes_points_in_host_frac(ax_host):
    """Every plotted finite point across ALL of the figure's axes (host + twin/offset), expressed
    in ax_host axes-fraction coords and clipped to the visible [0,1]² view. Twin axes share the
    display frame, so transData->display->host.transAxes⁻¹ puts them on one comparable grid. The
    figure must have been drawn (transforms realized) before calling."""
    pts = []
    for ax in ax_host.get_figure().axes:
        if not ax.get_visible():                 # e.g. an inset hidden by the small-canvas dodge
            continue
        inv = ax_host.transAxes.inverted()
        for ln in ax.get_lines():
            if ln.get_gid() == "refline":
                continue
            xy = ln.get_xydata()
            if xy is None or len(xy) == 0:
                continue
            xy = np.asarray(xy, float)
            xy = xy[np.isfinite(xy).all(axis=1)]
            if xy.size == 0:
                continue
            frac = inv.transform(ax.transData.transform(xy))
            pts.append(frac)
    if not pts:
        return np.empty((0, 2))
    p = np.vstack(pts)
    return p[(p[:, 0] >= 0) & (p[:, 0] <= 1) & (p[:, 1] >= 0) & (p[:, 1] <= 1)]

def _candidate_boxes(w, h, pad=0.02):
    """Axes-fraction (x0, y0, x1, y1) rects a legend of fractional size w×h would occupy at
    each of the nine matplotlib inside positions."""
    cx0, cx1 = (1 - w) / 2, (1 + w) / 2
    cy0, cy1 = (1 - h) / 2, (1 + h) / 2
    return {
        "upper right":  (1 - pad - w, 1 - pad - h, 1 - pad, 1 - pad),
        "upper left":   (pad, 1 - pad - h, pad + w, 1 - pad),
        "lower right":  (1 - pad - w, pad, 1 - pad, pad + h),
        "lower left":   (pad, pad, pad + w, pad + h),
        "upper center": (cx0, 1 - pad - h, cx1, 1 - pad),
        "lower center": (cx0, pad, cx1, pad + h),
        "center left":  (pad, cy0, pad + w, cy1),
        "center right": (1 - pad - w, cy0, 1 - pad, cy1),
        "center":       (cx0, cy0, cx1, cy1),
    }


def _obstacle_boxes_in_host_frac(ax_host):
    """Bboxes (host axes-fraction) of everything a legend must never sit on that matplotlib's
    own placement cannot see: visible free-text artists (annotations, the Dulong-Petit label,
    Rln labels — on ANY axis) and inset axes. Twin/offset axes are NOT obstacles: they share
    the host frame (their bbox ~ the whole axes) — only an axis properly inside the host frame
    (an inset) counts. The figure must have been drawn (transforms realized)."""
    fig = ax_host.get_figure()
    rend = fig.canvas.get_renderer()
    inv = ax_host.transAxes.inverted()

    def _frac(bb):
        (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    boxes = []
    host_bb = ax_host.get_window_extent(rend)
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        for t in ax.texts:
            if not t.get_visible() or not str(t.get_text()).strip():
                continue
            boxes.append(_frac(t.get_window_extent(rend)))
        if ax is ax_host:
            continue
        bb = ax.get_window_extent(rend)
        ix0, iy0 = max(bb.x0, host_bb.x0), max(bb.y0, host_bb.y0)
        ix1, iy1 = min(bb.x1, host_bb.x1), min(bb.y1, host_bb.y1)
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        host_area = max(host_bb.width * host_bb.height, 1e-9)
        if 0.0 < inter / host_area < 0.95:           # properly inside -> an inset, not a twin
            boxes.append(_frac(bb))
    return boxes


def _rects_overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _segment_hits_rect(pts, rect):
    """True when ANY segment of the polyline `pts` intersects the axis-aligned `rect`
    (x0, y0, x1, y1), both in the same coordinate system. Liang-Barsky clipping.

    VERTEX CONTAINMENT IS NOT ENOUGH, and that is the whole reason this exists. An
    `axvline` is a two-vertex line whose vertices sit at the TOP and BOTTOM of the axes;
    against a text box in the middle of the panel both vertices are outside while the
    segment between them runs straight through it. A point-in-box scan over the same
    figures reports a clean zero — a check that cannot see the thing it checks.
    `tests/core/test_refline_text_crossing.py` pins that exact case."""
    x0, y0, x1, y1 = rect
    for (ax_, ay), (bx, by) in zip(pts[:-1], pts[1:]):
        dx, dy = bx - ax_, by - ay
        t0, t1, ok = 0.0, 1.0, True
        for p, q in ((-dx, ax_ - x0), (dx, x1 - ax_), (-dy, ay - y0), (dy, y1 - ay)):
            if p == 0:                      # parallel to this edge: outside it -> no hit
                if q < 0:
                    ok = False
                    break
            else:
                r = q / p
                if p < 0:
                    if r > t1:
                        ok = False
                        break
                    t0 = max(t0, r)
                else:
                    if r < t0:
                        ok = False
                        break
                    t1 = min(t1, r)
        if ok and t0 <= t1:
            return True
    return False


def _refline_polylines_in_host_frac(ax_host):
    """Every visible reference LINE (gid 'refline', on any of the figure's axes) as a
    polyline in ax_host axes-fraction coords.

    Uses `ln.get_transform()`, NEVER `ax.transData`: `axvline`/`axhline` store one
    coordinate in AXES fraction on a blended transform, so pushing their xydata through
    transData yields nonsense (measured: y = -107 on the rho(T) reproducer). The figure
    must have been drawn (transforms realized) before calling."""
    inv = ax_host.transAxes.inverted()
    out = []
    for ax in ax_host.get_figure().axes:
        if not ax.get_visible():
            continue
        for ln in ax.get_lines():
            if ln.get_gid() != "refline" or not ln.get_visible():
                continue
            xy = np.asarray(ln.get_xydata(), float)
            if xy.size == 0 or len(xy) < 2:
                continue
            out.append(inv.transform(ln.get_transform().transform(xy)))
    return out


#: Annotation corner preference: UPPER LEFT FIRST — the position every frameless stats box
#: has shipped at — so a figure whose corner is already clear keeps it byte-identical.
#: Stability is the tiebreak, not re-optimization. Corners only: a multi-line stats box at
#: an edge-centre or dead centre is not a journal idiom (unlike a legend, which considers
#: all nine matplotlib positions).
_ANNOTATION_CORNERS = ("upper left", "upper right", "lower left", "lower right")


def _annotation_corner_boxes(w, h, pad=0.02):
    """Axes-fraction (x0, y0, x1, y1) rects a w x h annotation occupies at each corner."""
    return {
        "upper left":  (pad, 1 - pad - h, pad + w, 1 - pad),
        "upper right": (1 - pad - w, 1 - pad - h, 1 - pad, 1 - pad),
        "lower left":  (pad, pad, pad + w, pad + h),
        "lower right": (1 - pad - w, pad, 1 - pad, pad + h),
    }


#: (x, y, va, ha) for `Text.set_position`/`set_va`/`set_ha` at each corner, pad=0.02.
_ANNOTATION_ANCHORS = {
    "upper left":  (0.02, 0.98, "top", "left"),
    "upper right": (0.98, 0.98, "top", "right"),
    "lower left":  (0.02, 0.02, "bottom", "left"),
    "lower right": (0.98, 0.02, "bottom", "right"),
}


def _effective_yscale(spec, kind):
    """The y-scale the figure will actually end up with: the spec override, else the kind's
    default. `_finish` resolves it the same way, but only at the END of a render — so any
    helper that must know the scale BEFORE then has to resolve it itself."""
    if spec is not None and spec.yscale is not None:
        return spec.yscale
    return getattr(kind, "default_yscale", "linear") or "linear"


def _realize_layout(fig):
    """Realize the figure's transforms (final axes positions + a renderer) WITHOUT drawing,
    then put the layout state back exactly as it was.

    Measuring costs a layout pass, and a layout pass is not free: running
    ConstrainedLayoutEngine an extra time moves the axes and shifts marker antialiasing by
    1/255 on a handful of pixels — measured, sub-visible, and still a byte difference on
    figures that have no defect to fix. Snapshotting each axes' original+active position and
    restoring it afterwards makes the extra pass invisible: verified byte-identical across
    all 101 example x kind renders. Returns the renderer, or None if this canvas cannot
    measure text (a vector canvas mid-save) — callers then leave placement alone."""
    eng = fig.get_layout_engine()
    saved = [(ax, ax.get_position(original=True).frozen(), ax.get_position().frozen(),
              ax.get_in_layout()) for ax in fig.axes] if eng is not None else []
    if eng is not None:
        eng.execute(fig)
    try:
        return fig.canvas.get_renderer()
    except Exception:
        return None
    finally:
        for ax, orig, active, in_layout in saved:
            ax.set_position(orig, which="original")
            ax.set_position(active, which="active")
            # set_position evicts the axes from the layout engine ("called externally to
            # the library"), so the FINAL layout pass would then leave it at the position
            # we just restored. Measured: every annotated figure shifted ~86 px down and
            # ~31 px left. Restoring the flag is what makes the measurement a read.
            ax.set_in_layout(in_layout)


def _place_annotation(txt, spec=None, style=None, kind=None):
    """Move a frameless stats box off whatever it is sitting on, or leave it exactly where
    it is (the same root cause already fixed for the legend, the low-T inset and the
    reference-line labels: placement decided without measuring).

    Every one of these boxes shipped pinned at `ax.text(0.02, 0.98, ...)`. On
    `resistivity_superconductor.dat` / `resistivity_rho_t` that put the wide
    `RRR = 86.7 / T_c = 8.03 K (onset 8.80, zero 7.49)` box across the vertical dashed T_c
    guide (measured: box x 180.5-761.5 px against a line at x = 224.7 px spanning the whole
    panel).

    Scored with the legend/inset chooser's machinery — plotted points, text and inset
    obstacle boxes — PLUS reference lines, which those choosers deliberately ignore (they
    exist to dodge DATA) and which nothing measured until now. A refline is scored by SEGMENT
    intersection, never vertex containment: see `_segment_hits_rect`.

    Returns the chosen corner. `upper left` is tried first and returned whenever it is clear,
    so every figure without the defect renders byte-identical; only a box with nowhere clear
    at upper left ever moves. When NO corner is clear the least-bad one is taken, ordered
    (refline crossed, obstacle hit, share of data covered) — upper left winning ties by
    preference order, so a genuinely hopeless panel still does not churn.

    Given `spec`/`style`/`kind` the robust y-view is settled FIRST (idempotent — it derives
    ylim from the lines, not from the current limits, and `_finish` re-applies it to the
    identical result). The low-T inset learned this the hard way: a box scored against the
    creation-time autoscale is stale, because the robust view moves ylim afterwards and takes
    the data out from under the score. Measured here on the rho(T) reproducer: the raw
    autoscale puts the Ch1 curve under an upper-right box, the robust view (ylim -> 43.9-178.3)
    does not, and the stale score sent the box to the lower-right corner — pushing the legend
    and the inset to new corners in turn, for a figure whose upper right was empty all along.

    Settled only when the kind's EFFECTIVE y-scale is linear. On a log-y kind the scale is set
    in `_finish`, i.e. after this runs, so applying the robust view here would fix a
    linear-scale ylim that the later log switch inherits (measured on `resistivity_arrhenius`:
    ylim 0.0074-8.32 became -0.007-0.372). `_apply_robust_view` self-guards on the CURRENT
    scale, which is still linear at this point — hence the explicit `kind`.

    `tests/core/test_refline_text_crossing.py` scans the FINISHED figure, so a placement that
    a stale score still got wrong fails a gate rather than shipping."""
    ax = txt.axes
    if ax is None:
        return None
    fig = ax.get_figure()
    if spec is not None and style is not None and _effective_yscale(spec, kind) == "linear":
        _apply_robust_view(ax, spec, style)
    rend = _realize_layout(fig)
    if rend is None:
        return None
    inv = ax.transAxes.inverted()
    bb = txt.get_window_extent(rend)
    (fx0, fy0), (fx1, fy1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    w, h = abs(fx1 - fx0), abs(fy1 - fy0)
    if not (0.0 < w < 1.0 and 0.0 < h < 1.0):     # a box wider or taller than the panel:
        return None                              # no corner helps, keep the shipped spot

    txt.set_visible(False)                       # never let the box veto itself
    try:
        obstacles = _obstacle_boxes_in_host_frac(ax)
    finally:
        txt.set_visible(True)
    reflines = _refline_polylines_in_host_frac(ax)
    pts = _axes_points_in_host_frac(ax)
    total = max(len(pts), 1)
    boxes = _annotation_corner_boxes(w, h)

    def _score(box):
        crossed = any(_segment_hits_rect(pl, box) for pl in reflines)
        blocked = any(_rects_overlap(box, ob) for ob in obstacles)
        covered = int(((pts[:, 0] >= box[0]) & (pts[:, 0] <= box[2]) &
                       (pts[:, 1] >= box[1]) & (pts[:, 1] <= box[3])).sum()) if len(pts) else 0
        # the clear-standard the legend and the inset already ship with
        on_data = not (covered < 3 or covered / total <= _LEGEND_MULTIAXIS_OVERLAP_MAX)
        return crossed, blocked, on_data, covered / total

    best, best_key = None, None
    for loc in _ANNOTATION_CORNERS:
        crossed, blocked, on_data, frac = _score(boxes[loc])
        if not crossed and not blocked and not on_data:
            _apply_annotation_corner(txt, loc)
            return loc
        key = (crossed, blocked, frac)
        if best_key is None or key < best_key:
            best, best_key = loc, key
    _apply_annotation_corner(txt, best)
    return best


def _apply_annotation_corner(txt, loc):
    x, y, va, ha = _ANNOTATION_ANCHORS[loc]
    txt.set_position((x, y))
    txt.set_va(va)
    txt.set_ha(ha)


def _data_line_endpoints_in_host_frac(ax_host):
    """First and last finite point of every visible DATA line (reflines and fit overlays
    excluded), in host axes-fraction coords. Used by the inset fallback: a box hiding a
    line's terminal point makes the curve appear to stop there."""
    ends = []
    inv = ax_host.transAxes.inverted()
    for ax in ax_host.get_figure().axes:
        if not ax.get_visible():
            continue
        for ln in ax.get_lines():
            if ln.get_gid() in NON_DATA_GIDS:
                continue
            xy = np.asarray(ln.get_xydata(), float)
            if xy.size == 0:
                continue
            xy = xy[np.isfinite(xy).all(axis=1)]
            if not len(xy):
                continue
            for pnt in (xy[0], xy[-1]):
                fx, fy = inv.transform(ax.transData.transform(pnt))
                ends.append((float(fx), float(fy)))
    return ends


def _place_refline_labels(ax, style):
    """Slide every reference-line label along ITS OWN LINE to the clearest stretch (a 1-D
    search — the line is the constraint, position along it the free parameter; a label
    that leaves its line loses the association with the line it names). Labels are found
    by gid `refline-label:h` / `refline-label:v` and processed in creation order.

    The CURRENT position is always the first candidate, so a label that is already clear
    keeps its exact position and no golden image moves. A candidate is clear by the
    shipped standard (no obstacle-box intersection; covered points < 3 or <= 2%); when no
    stretch is clear the minimum-coverage candidate wins (ties -> earliest). A label is
    never dropped: unlike the inset it is the only thing tying the line to its meaning.

    Runs inside `_finish` AFTER `_apply_robust_view` (final limits — scoring stale
    pre-robust geometry misplaced both the legend and the inset during their fixes) and
    BEFORE `_draw_legend`, giving the pinned order inset -> labels -> legend. The one
    late-created label (tto_lorenz_t's WF label, drawn after `_finish` because its
    in-view guard needs the settled ylim) calls this again itself; the already-drawn
    legend is then treated as an obstacle, so nothing sits on anything either way."""
    labels = [t for t in ax.texts if (t.get_gid() or "").startswith("refline-label:")]
    if not labels:
        return
    fig = ax.get_figure()
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()

    def _frac(bb):
        (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    pts = _axes_points_in_host_frac(ax)
    total = max(len(pts), 1)
    leg = ax.get_legend()

    for t in labels:
        horizontal = t.get_gid().endswith(":h")
        t.set_visible(False)                       # a label is not its own obstacle
        try:
            obstacles = _obstacle_boxes_in_host_frac(ax)
        finally:
            t.set_visible(True)
        if leg is not None and leg.get_visible():
            obstacles.append(_frac(leg.get_window_extent(rend)))
        box = _frac(t.get_window_extent(rend))
        w, h = box[2] - box[0], box[3] - box[1]
        px, py = t.get_position()                  # blended: one coord frac, one data
        free0 = box[0] if horizontal else box[1]
        extent = w if horizontal else h
        anchor_off = (px - box[0]) if horizontal else (py - box[1])

        def _cand_box(lo):
            return (lo, box[1], lo + w, box[3]) if horizontal else \
                   (box[0], lo, box[2], lo + h)

        def _coverage(b):
            if not len(pts):
                return 0
            return int(((pts[:, 0] >= b[0]) & (pts[:, 0] <= b[2]) &
                        (pts[:, 1] >= b[1]) & (pts[:, 1] <= b[3])).sum())

        hi = 1.0 - extent - 0.02
        cands = [free0] + [float(c) for c in np.arange(0.02, hi + 1e-9, 0.04)
                           if abs(c - free0) > 1e-9]
        placed, best, best_n = None, None, None
        for c in cands:
            b = _cand_box(c)
            if any(_rects_overlap(b, ob) for ob in obstacles):
                continue
            n = _coverage(b)
            if n < 3 or n / total <= _LEGEND_MULTIAXIS_OVERLAP_MAX:
                placed = c
                break
            if best_n is None or n < best_n:
                best, best_n = c, n
        if placed is None:
            placed = best if best is not None else free0
        if abs(placed - free0) > 1e-9:
            if horizontal:
                t.set_position((placed + anchor_off, py))
            else:
                t.set_position((px, placed + anchor_off))


#: Inset corner preference: lower right FIRST — the shipped journal default — so every
#: figure whose corner was already genuinely clear keeps its position; stability is the
#: tiebreak, not re-optimization. Corners only: a 42%x40% panel at an edge-centre is not a
#: journal idiom (unlike a legend, which considers all nine positions).
_INSET_CORNERS = ("lower right", "upper right", "upper left", "lower left")


def _inset_spot(ax, w=0.42, h=0.40, pad=0.05):
    """Measured position for a w x h (axes-fraction) low-T inset (KNOWN-ISSUES 1). The fixed
    lower-right corner hid 35% of the reproducer's curve — the curve appeared to STOP where
    the inset started. Reuses the legend chooser's machinery: plotted points (union over
    axes, reflines excluded) and text-annotation obstacle boxes. A position is clear by the
    same standard the legend ships: no text intersection, and <=2% of all points or <3
    points under the box.

    Two candidate tiers. First the four corners in `_INSET_CORNERS` order (lower right = the
    shipped journal default, so figures whose corner was already genuinely clear keep it
    byte-similar) — a clear corner returns its loc STRING. When no corner is clear, a coarse
    anchor grid (columns right-to-left, rows bottom-to-top) is scanned and the first clear
    (x0, y0) TUPLE is returned — on the SC reproducer the only clear region is the right
    mid-band between the curve top and the wide Tc annotation, which no rigid corner box can
    reach (measured: the annotation clips the upper-right box by ~2% of the axes). Third, a
    LEAST-BAD corner fallback: the lowest-coverage non-text-vetoed corner is accepted when it
    covers <=10% of the points AND hides NO data line's terminal point — a grazed midsection
    reads as a curve passing behind a panel (the shipped look on the ACT fixture, measured
    5.25%), while a hidden endpoint is exactly the "curve stops here" illusion of the
    original defect, so endpoint containment is a hard veto that a bare coverage percentage
    cannot express. None means genuinely nowhere: the caller then DROPS the inset (with an
    on-figure note) — a supplement must never hide the primary data.

    Placed BEFORE the legend by construction (renderer body vs _finish), and the legend
    chooser treats the resulting inset bbox as a hard obstacle — so ordering is acyclic and
    deterministic. Draws the canvas to realize transforms."""
    fig = ax.get_figure()
    fig.canvas.draw()
    pts = _axes_points_in_host_frac(ax)
    total = max(len(pts), 1)
    obstacles = _obstacle_boxes_in_host_frac(ax)

    def _clear(box):
        if any(_rects_overlap(box, ob) for ob in obstacles):
            return False
        covered = int(((pts[:, 0] >= box[0]) & (pts[:, 0] <= box[2]) &
                       (pts[:, 1] >= box[1]) & (pts[:, 1] <= box[3])).sum()) if len(pts) else 0
        return covered < 3 or covered / total <= _LEGEND_MULTIAXIS_OVERLAP_MAX

    cx0, cx1 = 1 - pad - w, 1 - pad
    boxes = {
        "lower right": (cx0, pad, cx1, pad + h),
        "upper right": (cx0, 1 - pad - h, cx1, 1 - pad),
        "upper left":  (pad, 1 - pad - h, pad + w, 1 - pad),
        "lower left":  (pad, pad, pad + w, pad + h),
    }
    for loc in _INSET_CORNERS:
        if _clear(boxes[loc]):
            return loc
    step = 0.05
    xs = np.arange(cx0, pad - 1e-9, -step)           # columns: right to left
    ys = np.arange(pad, 1 - pad - h + 1e-9, step)    # rows: bottom to top
    for x0 in xs:
        for y0 in ys:
            if _clear((x0, y0, x0 + w, y0 + h)):
                return (float(x0), float(y0))
    # least-bad corner fallback: midsection grazing only, never a hidden endpoint.
    # The veto box is padded by ~a marker width: matplotlib's 5% axis margins put a
    # full-range curve's terminal point at frac 0.9545, a hair OUTSIDE the pad=0.05
    # chooser box (right edge 0.95) — unpadded, the veto is geometrically dead for
    # exactly the full-range sweep it was written for, while the curve's run-in still
    # vanishes under the drawn panel with one lone marker peeking past its edge.
    eps = 0.02
    ends = _data_line_endpoints_in_host_frac(ax)

    def _coverage(box):
        if not len(pts):
            return 0.0
        return float(((pts[:, 0] >= box[0]) & (pts[:, 0] <= box[2]) &
                      (pts[:, 1] >= box[1]) & (pts[:, 1] <= box[3])).sum()) / total

    best, best_cov = None, None
    for loc in _INSET_CORNERS:
        box = boxes[loc]
        if any(_rects_overlap(box, ob) for ob in obstacles):
            continue
        if any(box[0] - eps <= ex <= box[2] + eps and
               box[1] - eps <= ey <= box[3] + eps for ex, ey in ends):
            continue                                 # would hide where a curve STOPS
        cov = _coverage(box)
        if cov <= 0.10 and (best_cov is None or cov < best_cov - 1e-12):
            best, best_cov = loc, cov
    return best


def _lowt_inset_axes(ax, spec, style):
    """Create the shared 42%x40% low-T inset at the measured corner (KNOWN-ISSUES 1), or —
    when no corner is clear — draw a small grey note and return None: dropping must be said
    on the figure, never silent, so a reader does not mistake a missing inset for missing
    data. Callers run their own data guards first, so the note only ever appears where an
    inset was actually warranted.

    The view is settled FIRST: insets are created in the renderer body, before `_finish`
    applies the robust view / explicit spec limits, and on the SC reproducer the robust view
    moves ylim from (-4, 107) to (44, 178) — a corner scored against the creation-time
    autoscale is stale (measured; the same staleness class the legend chooser fixed by
    drawing after `_apply_robust_view`). Both calls are idempotent — `_finish` re-applies
    them to the identical result."""
    if spec.xscale is not None:
        ax.set_xscale(spec.xscale)
    if spec.yscale is not None:                      # BEFORE the robust view: it must see the
        ax.set_yscale(spec.yscale)                   # final scale (it skips log-y by contract)
    if spec.ymin is not None or spec.ymax is not None:
        ax.set_ylim(bottom=spec.ymin, top=spec.ymax)
    if spec.xmin is not None or spec.xmax is not None:
        ax.set_xlim(left=spec.xmin, right=spec.xmax)
    _apply_robust_view(ax, spec, style)
    spot = _inset_spot(ax)
    if spot is None:
        ax.text(0.99, 1.01, "low-T inset omitted (no clear corner)", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=max(style.font_pt - 3, 5), color="0.45",
                gid="inset_note")
        return None
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes as _inset_axes
    if isinstance(spot, str):                        # a clear corner: the shipped geometry
        iax = _inset_axes(ax, width="42%", height="40%", loc=spot, borderpad=1.4)
    else:                                            # grid anchor: exact fractional box
        x0, y0 = spot
        iax = _inset_axes(ax, width="100%", height="100%", loc="lower left",
                          bbox_to_anchor=(x0, y0, 0.42, 0.40),
                          bbox_transform=ax.transAxes, borderpad=0)
    iax.set_label("inset")   # identifies the inset to the visual gate; never drawn
    return iax


def _occupancy_legend_loc(ax_host, handles, legend_prop, style):
    """Choose an inside legend position from what is actually drawn (KNOWN-ISSUES 4/11).

    Scores the nine matplotlib inside positions for THIS legend's measured bbox against the
    figure's ink: the fraction of plotted points covered (union over all axes, reflines
    excluded) plus hard vetoes for intersecting any text annotation or inset — both invisible
    to matplotlib's own 'best'. Returns (loc, clear): loc is the least-covered candidate in
    preference order (corners, edge-centres, centre); clear is False when even that candidate
    covers >2% of the points or sits on a text/inset — the 'best' caller then relocates
    outside-right, which is the honest answer when the figure has no room.

    A render-time decision by design: font size and canvas size (the owner's screen-size
    point) enter through the measured legend extent, and the choice never shifts under the
    user without a re-render."""
    fig = ax_host.get_figure()
    kw = {"prop": legend_prop, "frameon": style.legend_frame, "loc": "upper right"}
    if handles is not None:
        kw["handles"] = handles
    tmp = ax_host.legend(**kw)
    fig.canvas.draw()                                  # realize transforms + legend extent
    bb = tmp.get_window_extent()
    inv = ax_host.transAxes.inverted()
    (fx0, fy0), (fx1, fy1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    w, h = abs(fx1 - fx0), abs(fy1 - fy0)
    tmp.remove()
    pts = _axes_points_in_host_frac(ax_host)
    total = max(len(pts), 1)
    obstacles = _obstacle_boxes_in_host_frac(ax_host)
    boxes = _candidate_boxes(w, h)
    best, best_key = None, None
    for loc in _LEGEND_INSIDE_LOCS:
        x0, y0, x1, y1 = boxes[loc]
        vetoed = any(_rects_overlap((x0, y0, x1, y1), ob) for ob in obstacles)
        frac = (((pts[:, 0] >= x0) & (pts[:, 0] <= x1) &
                 (pts[:, 1] >= y0) & (pts[:, 1] <= y1)).sum() / total) if len(pts) else 0.0
        key = (vetoed, frac)                           # clear-of-text first, then least data ink
        if best_key is None or key < (best_key[0], best_key[1] - 1e-12):
            best, best_key = loc, key
    if best_key is None:
        return "upper right", False
    vetoed, frac = best_key
    area = max(w * h, 1e-9)
    covered = frac * (len(pts) if len(pts) else 0)
    # Occupied = EITHER the absolute >2%-of-all-points rule (with a 3-point floor: on a
    # sparse synthetic curve one marker is already >2% and would exile every legend) OR a
    # DENSITY rule — a small legend over uniformly dense data covers few of the total points
    # yet sits fully on ink, which the absolute rule cannot see. Same legend size at every
    # candidate, so the frac ordering above already minimizes both.
    absolute = frac > _LEGEND_MULTIAXIS_OVERLAP_MAX and covered >= 3
    dense = (frac / area) > 0.30 and covered >= 5
    clear = bool(not vetoed and not absolute and not dense and w < 1.0 and h < 1.0)
    return best, clear

def _merged_legend(ax_host, handles, labels, style, spec):
    """One legend on ax_host combining handles gathered from twin/offset axes. Assigns labels
    onto the handles (matplotlib legend reads Artist.get_label()) and reuses `_draw_legend`
    for on/off toggle, placement, and the >11-entry relocation rule (PQ-1).

    For a "best"-placed legend on a multi-axes composite (twin/offset), matplotlib's own "best"
    only dodges the HOST axis' artists — so the merged legend can land squarely on a twin curve.
    Here we evaluate the four inside corners against the union of ALL axes' visible data and pin
    the clearest one; if the plot leaves no clear corner, we reuse the outside-right relocation.
    Explicit "inside"/"outside" overrides pass straight through unchanged."""
    for h, l in zip(handles, labels):
        h.set_label(l)
    legend_sz = style.legend_size if style.legend_size is not None else style.font_pt - 1
    legend_prop = {"size": legend_sz}
    if style.font_family:
        legend_prop["family"] = style.font_family
    # _draw_legend's occupancy chooser already scores the union of ALL axes' data (twin/offset
    # included) and falls back to the outside-right relocation when nothing inside is clear —
    # the multiaxis special case this function used to carry is now the general path.
    _draw_legend(ax_host, legend_prop, style, spec, handles)


from cryosweep_core.fitting.transport import (POWER_LAW_DECLINE_FLAGS,
                                              NO_FIT_LINE_FLAGS)

_RHO_T2_FITS = ("linear", "power_law")

_LOWT_FIT_KEYS = ("debye_t3", "debye_t3_t5", "spin_fluct_noninteracting", "spin_fluct_weak")
_LOWT_FIT_STYLE = {                                        # distinct colour + linestyle per model
    "debye_t3": ("#d62728", "-"),
    "debye_t3_t5": ("#2ca02c", "--"),
    "spin_fluct_noninteracting": ("#9467bd", "-."),
    "spin_fluct_weak": ("#ff7f0e", ":"),
}

def _fit_lines_wanted(spec, supported):
    """Which fit-line names to draw. None -> all supported; subset -> intersection; () -> none."""
    if spec.fit_lines is None:
        return set(supported)
    return set(spec.fit_lines) & set(supported)

_FIT_ENVELOPE_PAD = 0.10      # +-10% of the data span padded around the y envelope
_FIT_MIN_SUBSPAN_FRAC = 0.05  # skip if the clipped in-envelope B sub-span is < 5% of the full span

def _clip_fit_to_envelope(Bspan, b0, slope, y_data, robust_k=8.0):
    """Clip a fit line (y = b0 + slope*B) drawn over Bspan=[B0,B1] to the sub-span that falls
    inside the paired series' own data envelope (padded +-10% of the span). Returns (xs, ys) --
    the two endpoints of the clipped segment -- or None if the in-envelope sub-span is empty or
    < 5% of the full B span (display-only clip; fixes the full-height stripe artifact from a
    pathological Stage-B fit whose line is off-scale for its own data).

    The envelope uses the same robust range as the robust y-view (`cryosweep_core.robust.robust_range`,
    median +- k*MAD) rather than literal min/max: on real data a pathological Stage-B fit can
    itself poison a couple of antisymmetrized points near B~0 (division by a near-zero
    denominator) to values orders of magnitude past the bulk of that same series -- using literal
    min/max as the envelope would then always contain the equally-extreme fit line and never clip
    it. Falls back to literal min/max when the robust estimate degenerates (e.g. too few points,
    zero MAD)."""
    y_data = np.asarray(y_data, float)
    y_data = y_data[np.isfinite(y_data)]
    if y_data.size == 0:
        return None
    ylo, yhi = robust_range(y_data, k=robust_k)
    if not (np.isfinite(ylo) and np.isfinite(yhi)) or yhi <= ylo:
        ylo, yhi = float(y_data.min()), float(y_data.max())
    span = yhi - ylo
    pad = _FIT_ENVELOPE_PAD * span if span > 0 else _FIT_ENVELOPE_PAD * max(abs(yhi), 1e-30)
    ylo -= pad; yhi += pad
    B0, B1 = float(Bspan[0]), float(Bspan[1])
    if B1 <= B0:
        return None
    Bfull = B1 - B0
    if slope == 0:
        return (B0, B1) if ylo <= b0 <= yhi else None
    # solve b0 + slope*B in [ylo, yhi] for B (swap bounds if slope < 0)
    Ba, Bb = (ylo - b0) / slope, (yhi - b0) / slope
    Blo, Bhi = (Ba, Bb) if Ba <= Bb else (Bb, Ba)
    lo, hi = max(B0, Blo), min(B1, Bhi)
    if hi <= lo or (hi - lo) < _FIT_MIN_SUBSPAN_FRAC * Bfull:
        return None
    return (lo, hi)


def _draw_asym_fit_lines(ax, results, spec, style, plotted):
    """Overlay the exact Stage-B regression line (intercept + slope*B) on each asym
    series, colour-matched to that T's markers. Non-overlay only: plotted is 1:1 with
    the marker lines (reflines/fit lines excluded below). No label -> excluded from the
    legend."""
    if not spec.fit_line:
        return
    # marker snapshot: exclude reflines (e.g. the H=0 axhline drawn before the data) and
    # any already-appended fit lines so plotted[i] stays 1:1 with markers[i]
    markers = [ln for ln in ax.lines if ln.get_gid() not in NON_DATA_GIDS]
    for i, (r, s) in enumerate(plotted):
        if not s.key.startswith("asym:"):
            continue
        pmap = {f"{p['temperature']}K": p for p in (r.data or {}).get("points", [])}
        p = pmap.get(s.group)
        if not p:
            continue
        slope = p.get("slope_ohm_per_T"); b0 = p.get("asym_intercept_ohm")
        B = p.get("field_asym_T") or []
        if slope is None or b0 is None or len(B) < 2:
            continue
        Bspan = (min(B), max(B))
        clip = _clip_fit_to_envelope(Bspan, b0, slope, s.y, style.robust_k)
        if clip is None:
            continue
        xs = np.array(clip, float)
        _fit_plot(ax, xs, b0 + slope * xs, style, series_color=markers[i].get_color())

def _draw_branch_fit_lines(ax, results, spec, style, plotted):
    """Port of Step_2_Hall_fit_temp_dep.py's per-branch dashed fit (b--/r--): overlay the
    Stage-B regression line (asym_intercept_ohm + slope_ohm_per_T * B) on each ±branch series
    (rawpos:/rawneg:), colour-matched to that branch's own markers, evaluated over that
    branch's |B| span. Always dashed regardless of style.fit_linestyle (matches the reference
    figure). Non-overlay only: plotted is 1:1 with the marker lines (reflines/fit lines
    excluded below)."""
    if not spec.fit_line:
        return
    # marker snapshot: exclude reflines (the H=0 axhline precedes the data at ax.lines[0])
    # and any fit lines so plotted[i] stays 1:1 with markers[i]
    markers = [ln for ln in ax.lines if ln.get_gid() not in NON_DATA_GIDS]
    for i, (r, s) in enumerate(plotted):
        if not (s.key.startswith("rawpos:") or s.key.startswith("rawneg:")):
            continue
        pmap = {f"{p['temperature']}K": p for p in (r.data or {}).get("points", [])}
        p = pmap.get(s.group)
        if not p:
            continue
        slope = p.get("slope_ohm_per_T"); b0 = p.get("asym_intercept_ohm")
        if slope is None or b0 is None or len(s.x) < 2:
            continue
        Bspan = (min(s.x), max(s.x))
        clip = _clip_fit_to_envelope(Bspan, b0, slope, s.y, style.robust_k)
        if clip is None:
            continue
        xs = np.array(clip, float)
        _fit_plot(ax, xs, b0 + slope * xs, style,
                  series_color=markers[i].get_color(), linestyle="--")

def _draw_rxy_mirror_fit_lines(ax, results, spec, style, plotted):
    """Sub-feature A: on hall_rxy_vs_B (raw R_xy vs SIGNED B), overlay each temperature's
    antisymmetrized Stage-B slope (slope_ohm_per_T) as two MIRROR fit lines — one over its
    +B raw branch, one over its -B raw branch. Both branches share the same slope
    (Step_2-faithful mirror), but each line is anchored to that branch's OWN centroid
    (median B, median R_xy of that T's raw points on that side) so it overlays the raw data,
    which carries the even-in-B R_xx offset. Colour-matched to that T's raw markers, dashed,
    gid='fit'. Drawn only where the point is antisymmetrized with a finite slope and the
    branch has >=1 raw point. Not envelope-clipped: the centroid anchor keeps it on-scale."""
    if not spec.fit_line:
        return
    markers = [ln for ln in ax.lines if ln.get_gid() not in NON_DATA_GIDS]
    for i, (r, s) in enumerate(plotted):
        if not s.key.startswith("raw:"):
            continue
        pmap = {f"{p['temperature']}K": p for p in (r.data or {}).get("points", [])}
        p = pmap.get(s.group)
        if not p or not p.get("antisymmetrized"):
            continue
        slope = p.get("slope_ohm_per_T")
        if slope is None or not np.isfinite(slope):
            continue
        B = np.asarray(p.get("field_raw_T") or [], float)
        Rxy = np.asarray(p.get("R_xy_raw") or [], float)
        if B.size == 0 or B.size != Rxy.size:
            continue
        for mask in (B > 0, B < 0):
            bb, rr = B[mask], Rxy[mask]
            if bb.size < 1:
                continue
            b0 = float(np.median(bb)); y0 = float(np.median(rr))
            intercept = y0 - slope * b0            # centroid anchor as an (intercept, slope) line
            # Clip to THIS branch's own raw envelope: a good Hall slope stays on the band and
            # draws fully; a blown-up antisym fit (e.g. garbage sparse-300K slope ~1e6 Ω/T)
            # exits the envelope immediately -> None -> skipped (no near-vertical artifact).
            clip = _clip_fit_to_envelope((float(bb.min()), float(bb.max())),
                                         intercept, slope, rr, style.robust_k)
            if clip is None:
                continue
            xs = np.array(clip, float)
            _fit_plot(ax, xs, intercept + slope * xs, style,
                      series_color=markers[i].get_color(), linestyle="--")

def _fit_plot(ax, x, y, style, series_color=None, label=None, linestyle="-"):
    """Draw a fit line: colour = fit_color override else the paired series colour else default;
    style = fit_linestyle (override via `linestyle` arg for a secondary model). Tagged gid='fit'."""
    color = style.fit_color if style.fit_color is not None else series_color
    ls = linestyle if linestyle != "-" else style.fit_linestyle
    kw = dict(lw=style.line_width, gid="fit")
    if color is not None:
        kw["color"] = color
    if label is not None:
        kw["label"] = label
    return ax.plot(x, y, ls, **kw)[0]

def _extrap_plot(ax, x, y, style, color, y_ref):
    """Extrapolated continuation of a fit line down to its 0-intercept — a claim about
    behaviour OUTSIDE the fitted window, so it must never read as fit: dotted, thinner
    (0.75x), half-alpha, the SAME color as its fit line, no legend entry, gid="fit-extrap"
    (excluded from the robust view and the legend-marker snapshots like gid="fit").

    Insurance against pathological parameter sets (a modified-CW pole, a runaway power
    law): points with |y| > 3*max|y_ref| (the fitted segment's own values) are dropped, so
    an extrapolation can never blow up the axis — y-limits stay driven by the data and the
    finite intercept the figure exists to show (gamma, rho0, -theta/C)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ref = np.asarray(y_ref, float)
    cap = 3.0 * float(np.nanmax(np.abs(ref[np.isfinite(ref)])))
    m = np.isfinite(x) & np.isfinite(y) & (np.abs(y) <= cap)
    kw = dict(lw=0.75 * style.line_width, ls=":", alpha=0.5, gid="fit-extrap",
              label="_nolegend_", color=color)
    return ax.plot(x[m], y[m], **kw)[0]


# ---- VSM renderers ----
_INVCHI_FITS = ("cw", "cw_modified")      # per-fit toggle set for inverse_chi (_fit_lines_wanted)


def _vsm_is_si(results):
    """Unit system inferred from the analyzer's exported inv_chi_unit ('mol/m^3' -> SI).
    The renderer never sees cfg, so labels/data selection are sourced from the result."""
    for r in results:
        u = (r.data or {}).get("inv_chi_unit")
        if u:
            return u == "mol/m^3"
    return False


def _chi_labels(is_si):
    """Unit-true axis labels for χ and 1/χ (PQ-3 M1 — no hardcoded CGS)."""
    if is_si:
        return "χ (m³/mol)", "1/χ (mol/m³)"
    return "χ (emu/(mol·Oe))", "1/χ (mol·Oe/emu)"


def _cw_annotation(ax, fit, fitmod, drew_mod, style, spec=None, kind=None):
    """Frameless θ/C[/χ₀] text box, fontsize font_pt-1, placed by `_place_annotation`. Units come
    from FitResult.units (never hardcoded). χ₀ line only when the modified line was drawn."""
    p = (fit or {}).get("params") or {}
    if "C" not in p or "theta" not in p:
        return
    u = (fit or {}).get("units") or {}
    lines = [f"θ = {p['theta']:.3g} K, C = {p['C']:.3g} {u.get('C', '')}".rstrip()]
    if drew_mod:
        pm = (fitmod or {}).get("params") or {}
        um = (fitmod or {}).get("units") or {}
        if "chi0" in pm:
            lines.append(f"χ₀ = {pm['chi0']:.3g} {um.get('chi0', '')}".rstrip())
    fam = {"fontfamily": style.font_family} if style.font_family else {}
    t = ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
                fontsize=style.font_pt - 1, gid=ANNOTATION_GID, **fam)
    _place_annotation(t, spec, style, kind)
    return t


def render_inverse_chi(results, spec=None, style=None, overlay=None):
    results, kind, spec, style, fig, ax = _setup(results, "inverse_chi", spec, style)
    _plot_data(ax, results, kind, spec, style, overlay)
    _, inv_lbl = _chi_labels(_vsm_is_si(results))
    ann = None                               # the upper-left CW annotation, when one is drawn
    if overlay is None and spec.fit_line:
        want = _fit_lines_wanted(spec, _INVCHI_FITS)
        annotated = False
        for r in results:
            d = r.data or {}
            fit = d.get("fit") or {}
            p = fit.get("params") or {}
            fitmod = d.get("fit_modified") or {}
            pm = fitmod.get("params") or {}
            T = np.asarray(d.get("temperature") or [], float)
            if not (T.size and "C" in p and "theta" in p):
                continue
            # extrapolation target: through T = 0 to the theta crossing (1/chi = 0 at
            # T = theta) — the intercept the CW annotation already claims in text
            t_lo = float(np.nanmin(T))
            x_end = min(0.0, float(p["theta"]))
            if "cw" in want:
                ln = _fit_plot(ax, T, (T - p["theta"]) / p["C"], style, label="Curie-Weiss fit")
                if t_lo > x_end:
                    Tx = np.linspace(x_end, t_lo, 60)
                    _extrap_plot(ax, Tx, (Tx - p["theta"]) / p["C"], style,
                                 ln.get_color(), (T - p["theta"]) / p["C"])
            drew_mod = False
            if "cw_modified" in want and {"C", "theta", "chi0"} <= set(pm):
                with np.errstate(divide="ignore", invalid="ignore"):
                    y = 1.0 / (pm["chi0"] + pm["C"] / (T - pm["theta"]))
                # dashed grey ("0.45") second model; gid='fit' -> excluded from robust view
                lnm = _fit_plot(ax, T, y, style, series_color="0.45", label="modified CW", linestyle="--")
                # the modified-CW curve has a pole below theta (where chi0 + C/(T-theta)
                # crosses 0): never extend past theta itself, where 1/chi -> 0 from above
                xm_end = max(x_end, float(pm["theta"]) + 1e-9)
                if t_lo > xm_end:
                    Tx = np.linspace(xm_end, t_lo, 60)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ym = 1.0 / (pm["chi0"] + pm["C"] / (Tx - pm["theta"]))
                    _extrap_plot(ax, Tx, ym, style, lnm.get_color(), y)
                drew_mod = True
            if not annotated:                    # one box (first fitted result) — avoid stacking
                ann = _cw_annotation(ax, fit, fitmod, drew_mod, style, spec, kind)
                annotated = True
    _finish(ax, kind, spec, style, "Temperature (K)", inv_lbl)
    # The CW annotation is pinned at axes upper-left with ax.text, and matplotlib's legend
    # placement scores DATA artists only — text is invisible to it. On a real multi-field M(T)
    # (one legend entry per held field plus the two fit lines) "best" lands the legend straight
    # on top of the annotation, leaving both unreadable. Re-place it only when the two text
    # blocks ACTUALLY overlap, the same rule as the rho(T) dodge: every figure whose legend was
    # already clear — including the pinned single-curve oracles — stays byte-identical.
    if ann is not None:
        _install_legend_dodge(fig, ax, ann)
    return fig

def render_vsm_moment_t(results, spec=None, style=None, overlay=None):
    results, kind, spec, style, fig, ax = _setup(results, "vsm_moment_t", spec, style)
    _plot_data(ax, results, kind, spec, style, overlay)
    _finish(ax, kind, spec, style, "Temperature (K)", "Moment (μ_B/f.u.)")
    return fig

def _plot_marker_series(ax, r, s, style, spec, kind_key, color, linestyle=None):
    """Draw one marker(+optional connect) series on `ax` with a forced colour; returns the
    Line2D. Mirrors _plot_data's per-series styling for the twin-axis composite. `linestyle`
    (PQ-3 Task 4 ramp split) overrides the connected line style when connect is on."""
    x, y = np.asarray(s.x, float), np.asarray(s.y, float)
    connect = (spec.connect_lines if spec.connect_lines is not None else style.connect_lines) \
        and kind_key in _CONNECT_KINDS
    if connect and x.size:
        x, y = _connect_sort(x, y)
    ls = "-" if connect else "none"
    if connect and linestyle:
        ls = linestyle
    kw = dict(marker=style.marker, ls=ls,
              ms=style.marker_size, color=color, label=series_label(r, s))
    if connect:
        kw["lw"] = style.line_width
    if style.edge_color is not None:
        kw["markeredgecolor"] = style.edge_color
    if style.edge_width is not None:
        kw["markeredgewidth"] = style.edge_width
    return ax.plot(x, y, **kw)[0]


def render_vsm_chi_t(results, spec=None, style=None, overlay=None):
    """χ (left, C0) + χ⁻¹ (right twin, C3) vs T — PQ-3 Item 2. Twin axis only when both the χ
    (key "curve") and the "inv_chi" series are selected; deselecting inv_chi ⇒ plain single-axes
    χ exactly as before (no dead right spine). Overlay (file comparison) ⇒ single-axes fallback
    drawing only the χ series (PQ-2 convention)."""
    results, kind, spec, style, fig, ax = _setup(results, "vsm_chi_t", spec, style)
    is_si = _vsm_is_si(results)
    chi_lbl, inv_lbl = _chi_labels(is_si)
    if overlay is not None:
        # single-axes fallback: draw only χ (drop inv_chi via a filtered view of the kind)
        chi_kind = dataclasses.replace(
            kind, series=lambda r, _k=kind, field_unit="Oe": [
                s for s in _k.series(r, field_unit=field_unit) if s.role != "inv_chi"])
        _plot_data(ax, results, chi_kind, spec, style, overlay)
        _finish(ax, kind, spec, style, "Temperature (K)", chi_lbl)
        return fig
    selected = [(r, s) for r in results for s in select_series(kind.series(r, field_unit=style.field_unit), spec)]
    if not selected:
        raise NothingToPlot(f"no series selected for kind {kind.key}")
    chi_items = [(r, s) for (r, s) in selected if s.role != "inv_chi"]
    inv_items = [(r, s) for (r, s) in selected if s.role == "inv_chi"]
    if not (chi_items and inv_items):
        # single axes: whichever set is selected, drawn exactly as the legacy path
        _plot_data(ax, results, kind, spec, style, overlay)
        _finish(ax, kind, spec, style, "Temperature (K)",
                chi_lbl if chi_items else inv_lbl)
        return fig
    # --- twin axes: χ left (C0), χ⁻¹ right (C3) ---
    fam = {"fontfamily": style.font_family} if style.font_family else {}
    label_sz = style.label_size if style.label_size is not None else style.font_pt
    handles, labels = [], []
    # χ (left) all C0, χ⁻¹ (right) all C3 — one colour per quantity even when ramp-split
    # into ↑/↓ series (PQ-3 Task 4); the ramp is encoded by linestyle, not colour.
    for r, s in chi_items:
        ln = _plot_marker_series(ax, r, s, style, spec, kind.key, "C0",
                                 linestyle=getattr(s, "linestyle", None))
        handles.append(ln); labels.append(series_label(r, s))
    ax.set_ylabel(chi_lbl, color="C0", fontsize=label_sz, **fam)
    ax.tick_params(axis="y", colors="C0")
    ax.spines["left"].set_color("C0")
    tax = _twin_axis(ax, style, "C3")
    for r, s in inv_items:
        ln = _plot_marker_series(tax, r, s, style, spec, kind.key, "C3",
                                 linestyle=getattr(s, "linestyle", None))
        handles.append(ln); labels.append(series_label(r, s))
    tax.set_ylabel(inv_lbl, color="C3", fontsize=label_sz, **fam)
    ax.set_xscale(spec.xscale if spec.xscale is not None else kind.default_xscale)
    if spec.xmin is not None or spec.xmax is not None:
        ax.set_xlim(left=spec.xmin, right=spec.xmax)
    ax.set_xlabel("Temperature (K)", fontsize=label_sz, **fam)
    _draw_reference_lines(ax, spec)
    if style.tick_size is not None:
        ax.tick_params(labelsize=style.tick_size)
    if spec.title:
        title_sz = style.title_size if style.title_size is not None else style.font_pt
        ax.set_title(spec.title, fontsize=title_sz, **fam)
    _apply_robust_view(ax, spec, style)      # per-axis robust view (PQ-2 composite rule)
    _apply_robust_view(tax, spec, style)
    _apply_frame(ax, style, spec)
    _merged_legend(ax, handles, labels, style, spec)
    return fig

def render_vsm_chi_t_product(results, spec=None, style=None, overlay=None):
    results, kind, spec, style, fig, ax = _setup(results, "vsm_chi_t_product", spec, style)
    _plot_data(ax, results, kind, spec, style, overlay)
    _finish(ax, kind, spec, style, "Temperature (K)", "χT (emu·K/mol·Oe)")
    return fig


def _field_axis_label(prefix, style):
    """'{prefix} (Oe)' vs '{prefix} (T)' from the display unit. style may be None -> Oe."""
    unit = getattr(style, "field_unit", "Oe") if style is not None else "Oe"
    return f"{prefix} ({unit})"


def _mh_xlabel(style):
    return _field_axis_label("Magnetic Field", style)


_MH_YLABEL = "Moment (μ_B/f.u.)"         # moment uses moment_per_fu convention
_MH_ZOOM_FRAC = 0.10                      # low-field zoom half-width = 10% of max|field|


def _mh_group_label(group):
    """'{T}K' group tag -> a '{T:.1f} K' legend label; falls back to the raw tag if unparseable."""
    try:
        return f"{float(group[:-1]):.1f} K"
    except (ValueError, IndexError):
        return group


def render_vsm_mh(results, spec=None, style=None, overlay=None):
    """M(H) hysteresis: full-range main panel + a low-field zoom companion panel
    (PQ-3 Task 2). Connected line+markers, categorical colour per T group (shared across
    both panels), H=0/M=0 reference lines on both panels, one legend of T labels. spec
    xmin/xmax apply to the MAIN panel only; the zoom panel is always ±10% of max|field|.
    Overlay (file comparison) falls back to a single main-panel axes (PQ-2 convention)."""
    if overlay is not None:
        results, kind, spec, style, fig, ax = _setup(results, "vsm_mh", spec, style)
        ax.axvline(0, color="black", lw=0.8, gid="refline")
        ax.axhline(0, color="black", lw=0.8, gid="refline")
        _plot_data(ax, results, kind, spec, style, overlay)
        _finish(ax, kind, spec, style, _mh_xlabel(style), _MH_YLABEL)
        return fig

    spec = spec or PlotSpec(); style = style or GlobalStyle()
    kind = _KIND["vsm_mh"]
    results = _as_list(results)

    plotted = []
    for r in results:
        for s in select_series(kind.series(r, field_unit=style.field_unit), spec):
            plotted.append((r, s))
    if not plotted:
        raise NothingToPlot(f"no series selected for kind {kind.key}")

    # distinct T groups (first-appearance order) -> one colour + one legend entry each
    groups = []
    for _, s in plotted:
        if s.group not in groups:
            groups.append(s.group)
    gcolor = _group_color_map(groups, style)

    connect = spec.connect_lines if spec.connect_lines is not None else style.connect_lines

    # zoom half-width from the selected loops' max |field| (constant 10% of it)
    maxabs = 0.0
    for _, s in plotted:
        xa = np.abs(np.asarray(s.x, float))
        xa = xa[np.isfinite(xa)]
        if xa.size:
            maxabs = max(maxabs, float(xa.max()))
    zoom_half = _MH_ZOOM_FRAC * maxabs

    _, _, spec, style, fig, axes = _setup_panels(results, "vsm_mh", spec, style, 2)
    main_ax, zoom_ax = axes

    fam = {"fontfamily": style.font_family} if style.font_family else {}
    label_sz = style.label_size if style.label_size is not None else style.font_pt
    title_sz = style.title_size if style.title_size is not None else style.font_pt

    for ax in axes:
        ax.axvline(0, color="black", lw=0.8, gid="refline")   # H = 0
        ax.axhline(0, color="black", lw=0.8, gid="refline")   # M = 0
        for _, s in plotted:
            kw = dict(marker=style.marker, ms=style.marker_size,
                      color=gcolor[s.group], label="_nolegend_")
            if connect:
                kw["ls"] = "-"; kw["lw"] = style.line_width   # row-order preserved (no x-sort)
            else:
                kw["ls"] = "none"
            if style.edge_color is not None:
                kw["markeredgecolor"] = style.edge_color
            if style.edge_width is not None:
                kw["markeredgewidth"] = style.edge_width
            ax.plot(s.x, s.y, **kw)
        ax.set_xscale(spec.xscale if spec.xscale is not None else kind.default_xscale)
        ax.set_yscale(spec.yscale if spec.yscale is not None else kind.default_yscale)
        ax.set_xlabel(_mh_xlabel(style), fontsize=label_sz, **fam)
        _draw_reference_lines(ax, spec)
        if style.tick_size is not None:
            ax.tick_params(labelsize=style.tick_size)
        if ax is main_ax:
            if spec.xmin is not None or spec.xmax is not None:
                ax.set_xlim(left=spec.xmin, right=spec.xmax)
            if spec.ymin is not None or spec.ymax is not None:
                ax.set_ylim(bottom=spec.ymin, top=spec.ymax)
            # Cap x-tick count so wide 6-digit field labels (e.g. 50000 / 100000) don't collide
            # on the narrow half-width main panel. The panel is only ~1.24 in wide (two panels in
            # a 90 mm figure), which fits ~2 six-digit labels; nbins=5 (spec's first suggestion)
            # and even 3 overlap on the real VSM_N ±136 kOe range (measured), so a tighter cap is
            # the working equivalent. nice steps keep round ticks; composes with the thousands
            # FuncFormatter set by _apply_frame (locator + formatter are independent). Zoom panel
            # keeps its own auto locator (its ±10% window is already sparse).
            ax.xaxis.set_major_locator(MaxNLocator(nbins=2, steps=[1, 2, 2.5, 5, 10]))
        else:
            ax.set_title("low field", fontsize=title_sz, **fam)
            if zoom_half > 0:
                ax.set_xlim(-zoom_half, zoom_half)
        _apply_robust_view(ax, spec, style)
        _apply_frame(ax, style, spec)

    main_ax.set_ylabel(_MH_YLABEL, fontsize=label_sz, **fam)

    handles = [Line2D([], [], ls=("-" if connect else "none"), marker=style.marker,
                      color=gcolor[g], label=_mh_group_label(g)) for g in groups]
    legend_sz = style.legend_size if style.legend_size is not None else style.font_pt - 1
    legend_prop = {"size": legend_sz}
    if style.font_family:
        legend_prop["family"] = style.font_family
    _draw_legend(main_ax, legend_prop, style, spec, handles)
    return fig

# ---- Heat capacity renderers ----
_SPIN_FLUCT_KEYS = ("spin_fluct_noninteracting", "spin_fluct_weak")


def _hc_lowt_annotation(ax, d, style, spec=None, kind=None):
    """Frameless γ/β/θ_D text box (corner chosen by `_place_annotation`), matching _cw_annotation
    pattern. θ_D renders as 'n/a' when non-finite or the chosen model is spin-fluctuation
    (β is not a lattice property there, so θ_D is undefined)."""
    fit = d.get("fit") or {}
    p = fit.get("params") or {}
    gamma = p.get("gamma")
    beta = p.get("beta")
    theta_D = p.get("theta_D")
    if gamma is None or beta is None:
        return
    # A flagged value carries its verdict onto the figure — the artifact that travels into
    # a talk without the status bar. The value stays (it IS what was measured; this is not
    # the decline case), the tag rides its own line, Tc "(low confidence)" idiom.
    gamma_tag = " (unphysical)" if "gamma_negative" in (fit.get("quality_flags") or []) else ""
    lines = [f"γ = {gamma:.1e} J/mol·K²{gamma_tag}", f"β = {beta:.1e} J/mol·K⁴"]
    spin = d.get("model") in _SPIN_FLUCT_KEYS
    if theta_D is None or not np.isfinite(theta_D) or spin:
        lines.append("θ_D = n/a")
    else:
        lines.append(f"θ_D = {theta_D:.3g} K")
    fam = {"fontfamily": style.font_family} if style.font_family else {}
    _place_annotation(
        ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
                fontsize=style.font_pt - 1, gid=ANNOTATION_GID, **fam), spec, style, kind)


def _hc_fit_window_shade(ax, values):
    """Light vertical shade over [min, max] of the windowed x-values (T² for cp_over_t, T for
    the linear kind), marking the fitted low-T region. gid='refline' so it is excluded from the
    robust-view autoscale union."""
    v = np.asarray(values or [], float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return
    lo, hi = float(v.min()), float(v.max())
    if hi > lo:
        ax.axvspan(lo, hi, alpha=0.1, color="0.5", gid="refline", zorder=0)


def render_cp_over_t(results, spec=None, style=None, overlay=None):
    results, kind, spec, style, fig, ax = _setup(results, "cp_over_t", spec, style)
    _plot_data(ax, results, kind, spec, style, overlay)
    if overlay is None and spec.fit_line:
        want = _fit_lines_wanted(spec, _LOWT_FIT_KEYS)
        for r in results:
            for mfit in (r.data or {}).get("lowt_fits") or []:
                if not mfit.get("ok") or mfit.get("key") not in want:
                    continue
                x = np.asarray(mfit.get("t2_grid") or [], float)
                y = np.asarray(mfit.get("cp_over_t_fit") or [], float)
                if x.size and y.size == x.size:
                    color, ls = _LOWT_FIT_STYLE.get(mfit["key"], (None, "-"))
                    r2 = mfit.get("r2")
                    label = f'{mfit.get("label", mfit["key"])} (R²={r2:.3f})' if r2 is not None else mfit.get("label")
                    ln = _fit_plot(ax, x, y, style, series_color=color, label=label, linestyle=ls)
                    # continue the model to T² = 0: the intercept IS gamma, the number the
                    # annotation prints (incl. a negative/unphysical one — the reader must
                    # see the crossing the flag talks about, not just be told of it)
                    fn = _LOWT_FUNCS.get(mfit["key"]); prm = mfit.get("params") or {}
                    x_lo = float(x.min())
                    if fn is not None and prm and x_lo > 0.0:
                        xg = np.linspace(0.0, x_lo, 40)
                        _extrap_plot(ax, xg, fn(xg, prm), style, ln.get_color(), y)
    if overlay is None and results:
        d0 = results[0].data or {}
        if spec.fit_window_shade:      # opt-in (owner: useful, but OFF by default)
            _hc_fit_window_shade(ax, d0.get("t_squared"))
        _hc_lowt_annotation(ax, d0, style, spec, kind)
    _finish(ax, kind, spec, style, "T² (K²)", "Cp/T (J/mol·K²)")
    return fig

def render_cp_vs_t(results, spec=None, style=None, overlay=None):
    results, kind, spec, style, fig, ax = _setup(results, "cp_vs_t", spec, style)
    _plot_data(ax, results, kind, spec, style, overlay)
    if overlay is None and spec.fit_line:
        for r in results:
            ff = (r.data or {}).get("full_fit") or {}
            if not ff.get("ok"):
                continue
            x = np.asarray(ff.get("t_grid") or [], float)
            y = np.asarray(ff.get("cp_fit") or [], float)
            if x.size and y.size == x.size:
                r2 = ff.get("r2")
                label = f"Debye-Einstein (R²={r2:.3f})" if r2 is not None else "Debye-Einstein"
                _fit_plot(ax, x, y, style, label=label)
    _finish(ax, kind, spec, style, "Temperature (K)", "Cp (J/mol·K)")
    return fig

_FULL_CP_FIT_COLOR = "#d62728"       # solid-red Debye-Einstein overlay (journal target)


def _add_lowt_inset(ax, d, spec, style):
    """Low-T (0→~10 K) inset: the ≤10 K Cp(T) subset as open squares + the chosen low-T fit
    line evaluated over that range. Self-contained — draws on its own `ax.inset_axes` panel
    (excluded from constrained layout, so the global frame/layout is untouched). Returns the
    inset axis, or None when there is no low-T data."""
    T = np.asarray(d.get("temperature") or [], float)
    cp = np.asarray(d.get("cp") or [], float)
    m = np.isfinite(T) & np.isfinite(cp)
    T, cp = T[m], cp[m]
    if T.size == 0:
        return None
    iax = _lowt_inset_axes(ax, spec, style)    # measured corner, or dropped-with-note (None)
    if iax is None:
        return None
    iax.plot(T, cp, marker="s", markerfacecolor="none", ls="none",
             ms=max(style.marker_size - 1.0, 2.0), color="0.2",
             markeredgewidth=(style.edge_width if style.edge_width is not None else 0.8))
    # chosen low-T fit (cp = (Cp/T)·T), reconstructed from its Cp/T-vs-T² grid
    model = d.get("model")
    lf = next((f for f in (d.get("lowt_fits") or [])
               if f.get("key") == model and f.get("ok")), None)
    if lf and lf.get("t2_grid") and lf.get("cp_over_t_fit"):
        t2 = np.asarray(lf["t2_grid"], float)
        cot = np.asarray(lf["cp_over_t_fit"], float)
        good = np.isfinite(t2) & (t2 >= 0) & np.isfinite(cot)
        Tg = np.sqrt(t2[good])
        iax.plot(Tg, cot[good] * Tg, ls="-", color=_FULL_CP_FIT_COLOR, lw=style.line_width)
    iax.tick_params(labelsize=5, length=2, width=0.5)
    iax.set_xlabel("T (K)", fontsize=5, labelpad=1)
    iax.set_ylabel("Cp (J/mol·K)", fontsize=5, labelpad=1)
    for sp in iax.spines.values():
        sp.set_linewidth(0.5)
    return iax


def render_hc_full_cp_t(results, spec=None, style=None, overlay=None):
    """PQ-5 Task 5 headline journal figure. Full Cp(T) vs T: open-square data (colour-by-field
    viridis when >1 field group), a solid-red Debye-Einstein fit overlay (when full_fit ok), a
    labelled dashed Dulong–Petit line at 3nR (only when n_atoms is known), and a low-T inset
    (≤10 K data + chosen low-T fit). Overlay/file-comparison mode falls back to the generic
    marker plot."""
    results, kind, spec, style, fig, ax = _setup(results, "hc_full_cp_t", spec, style)
    if overlay is not None:
        _plot_data(ax, results, kind, spec, style, overlay)
        _finish(ax, kind, spec, style, "Temperature (K)", "Cp (J/mol·K)")
        return fig

    plotted = []
    for r in results:
        for s in select_series(kind.series(r, field_unit=style.field_unit), spec):
            plotted.append((r, s))
    if not plotted:
        raise NothingToPlot(f"no series selected for kind {kind.key}")
    data_items = [(r, s) for (r, s) in plotted if s.role != "fit"]
    fit_items = [(r, s) for (r, s) in plotted if s.role == "fit"]

    # colour-by-field (viridis) when >1 data group; else a single colour
    groups = []
    for _, s in data_items:
        if s.group not in groups:
            groups.append(s.group)
    multi = len(groups) > 1
    if multi:
        cmap_name = style.colormap or "viridis"
        cols = _cmap_colors(cmap_name, len(groups), style.colormap_reverse) or []
        gcolor = {g: cols[i] for i, g in enumerate(groups)} if cols else {}
    else:
        gcolor = {}
    for r, s in data_items:
        color = gcolor.get(s.group) if multi else style.color
        kw = dict(marker="s", markerfacecolor="none", ls="none",
                  ms=style.marker_size, label=series_label(r, s))
        if color is not None:
            kw["color"] = color
        if style.edge_width is not None:
            kw["markeredgewidth"] = style.edge_width
        ax.plot(np.asarray(s.x, float), np.asarray(s.y, float), **kw)
    # solid-red Debye-Einstein overlay (role='fit' -> gid='fit', excluded from robust view)
    for r, s in fit_items:
        ax.plot(np.asarray(s.x, float), np.asarray(s.y, float), ls="-",
                color=_FULL_CP_FIT_COLOR, lw=style.line_width, gid="fit",
                label=series_label(r, s))

    # Dulong–Petit line at 3nR (only when n_atoms known)
    d0 = (results[0].data or {}) if results else {}
    if d0.get("n_atoms_available"):
        dp = dulong_petit_limit(d0.get("n_atoms"))
        if dp is not None:
            ax.axhline(dp, ls="--", color="0.4", lw=1.0, gid="refline")
            # Label at the LEFT edge (small x-fraction) near the DP line rather than in the
            # legend, so it can't collide with the low-T inset that sits at the lower-right.
            ax.text(0.02, dp, "Dulong–Petit", transform=ax.get_yaxis_transform(),
                    va="top", ha="left", fontsize="small", gid="refline-label:h")

    _hc_iax = _add_lowt_inset(ax, d0, spec, style)
    _finish(ax, kind, spec, style, "Temperature (K)", "Cp (J/mol·K)")
    # loc='best' cannot see the inset (a separate Axes); at 12pt+ the legend lands under
    # it. Conditional on real overlap, so the 9 pt gallery render is untouched.
    _legend_clear_of_inset(ax, _hc_iax)
    return fig

def render_hc_entropy_vs_t(results, spec=None, style=None, overlay=None):
    """PQ-5 Task 4 (+ task-9 twin-axis vizfix): entropy S(T). Total S (solid) on the LEFT axis,
    full-range ylim so the cumulative headline curve is never clipped. When magnetic entropy is
    MEANINGFULLY resolved (see gate below), the magnetic S (dashed) + the Rln(2J+1) reference
    line move onto a right-hand TWIN axis auto-scaled to the magnetic data, so the approach to
    the Rln plateau is readable even on lattice-dominated samples (where S_magnetic ~ 0 sits
    thousands of times below S_total and would otherwise collapse into an invisible bottom band).

    Meaningful-magnetic gate: `entropy_magnetic` is not None AND has >=1 finite entry AND
    max(finite magnetic) > 0.05 * Rln. When NOT meaningful (magnetic None, or trivially ~0),
    fall back to the current single-axis total-only behaviour: NO twin, NO Rln line (this also
    preserves the HC_N bbox-blowup fix — an off-scale Rln label would blow up bbox_inches).

    Reuses the shared _twin_axis / _merged_legend machinery (unchanged) from the Hall twin plots.
    Overlay/file-comparison mode falls back to the generic marker plot."""
    results, kind, spec, style, fig, ax = _setup(results, "hc_entropy_vs_t", spec, style)
    if overlay is not None:
        _plot_data(ax, results, kind, spec, style, overlay)
        _finish(ax, kind, spec, style, "Temperature (K)", "S (J/mol·K)")
        return fig

    plotted = []
    for r in results:
        for s in select_series(kind.series(r, field_unit=style.field_unit), spec):
            plotted.append((r, s))
    if not plotted:
        raise NothingToPlot(f"no series selected for kind {kind.key}")

    # Rln(2J+1) suggestion (from the first result carrying one)
    sug = None
    for r in results:
        s0 = (r.data or {}).get("entropy_rln_suggestion")
        if s0 and s0.get("value") is not None:
            sug = s0; break
    rln_val = float(sug["value"]) if (sug is not None and sug.get("value")) else None
    rln_lbl = sug.get("label", "Rln") if sug is not None else "Rln"

    # meaningful-magnetic gate on the primary result's entropy_magnetic
    d0 = (results[0].data or {}) if results else {}
    mag0 = d0.get("entropy_magnetic")
    mag_finite = ([float(v) for v in mag0 if v is not None and np.isfinite(float(v))]
                  if mag0 is not None else [])
    meaningful = bool(rln_val is not None and mag_finite
                      and max(mag_finite) > 0.05 * rln_val)

    magnetic_roles = {"magnetic", "magnetic_field"}

    # colour by group: total/magnetic (group None) share one colour in the single-axis case;
    # on the twin, the main total is pinned C0 (left) and the main magnetic C3 (right) so each
    # axis reads as belonging to its series. Per-field overlays keep their own group colour.
    groups = []
    for _, s in plotted:
        if s.group not in groups:
            groups.append(s.group)
    gcolor = _group_color_map(groups, style)

    def _color(s):
        if meaningful and s.group is None:
            return "C3" if s.role in magnetic_roles else "C0"
        return gcolor[s.group]

    fam = {"fontfamily": style.font_family} if style.font_family else {}
    label_sz = style.label_size if style.label_size is not None else style.font_pt

    if meaningful:
        # ---- TWIN-AXIS path: total (left) + magnetic/Rln (right) ------------------------------
        tax = _twin_axis(ax, style, "C3")
        handles, labels = [], []
        left_y, right_y = [], []
        for r, s in plotted:
            x = np.asarray(s.x, float); y = np.asarray(s.y, float)
            host = tax if s.role in magnetic_roles else ax
            ln, = host.plot(x, y, marker="none", ls=(getattr(s, "linestyle", None) or "-"),
                            lw=style.line_width, color=_color(s), label=series_label(r, s))
            handles.append(ln); labels.append(series_label(r, s))
            (right_y if host is tax else left_y).append(y[np.isfinite(y)])

        ax.set_ylabel("S (J/mol·K)", color="C0", fontsize=label_sz, **fam)
        ax.tick_params(axis="y", colors="C0")
        ax.spines["left"].set_color("C0")
        tax.set_ylabel("S magnetic (J/mol·K)", fontsize=label_sz, **fam)

        # Rln reference line on the RIGHT (magnetic) axis
        tax.axhline(rln_val, color="0.4", linestyle="--", linewidth=1.0, gid="refline")

        # x-axis / frame (mirror the Hall-twin manual finish; no _finish so our explicit
        # per-axis ylims survive)
        ax.set_xscale(spec.xscale if spec.xscale is not None else kind.default_xscale)
        if spec.xmin is not None or spec.xmax is not None:
            ax.set_xlim(left=spec.xmin, right=spec.xmax)
        ax.set_xlabel("Temperature (K)", fontsize=label_sz, **fam)
        _draw_reference_lines(ax, spec)
        if style.tick_size is not None:
            ax.tick_params(labelsize=style.tick_size)
        if spec.title:
            title_sz = style.title_size if style.title_size is not None else style.font_pt
            ax.set_title(spec.title, fontsize=title_sz, **fam)
        _apply_frame(ax, style, spec)

        # LEFT (total) ylim: full-range, never clip the cumulative curve
        lfin = [a for a in left_y if a.size]
        if spec.ymin is not None or spec.ymax is not None:
            ax.set_ylim(bottom=spec.ymin, top=spec.ymax)
        elif lfin:
            allcat = np.concatenate(lfin)
            lo = min(0.0, float(allcat.min())); hi = float(allcat.max())
            if hi > lo:
                ax.set_ylim(bottom=lo, top=hi * 1.05)

        # RIGHT (magnetic) ylim: span [min(0, mag_min), max(mag_max, Rln)] * small pad so both
        # the magnetic curve AND the Rln line are on-scale and the plateau approach is readable
        rfin = [a for a in right_y if a.size]
        rcat = np.concatenate(rfin) if rfin else np.array([0.0])
        rmin = min(0.0, float(rcat.min()))
        rmax = max(float(rcat.max()), rln_val)
        pad = 0.05 * (rmax - rmin) if rmax > rmin else max(abs(rmax), 1.0) * 0.05
        tax.set_ylim(bottom=rmin - pad, top=rmax + pad)

        # Rln label: place in axes-fraction (clamped) so it can NEVER sit outside the right axes
        # and blow up savefig(bbox_inches="tight"). Anchored right (near its own axis) so it
        # stays clear of the top-left saturation annotation.
        if rln_lbl:
            ylo, yhi = tax.get_ylim()
            frac = (rln_val - ylo) / (yhi - ylo) if yhi > ylo else 0.5
            frac = min(max(frac, 0.02), 0.97)
            va = "top" if frac > 0.9 else "bottom"
            tax.text(0.98, frac, rln_lbl, transform=tax.transAxes,
                     va=va, ha="right", fontsize="small", color="0.4")

        # The saturation annotation sits top-left; default the (2-entry) merged legend to the
        # lower-right so they never collide. An explicit user legend_loc still passes through.
        legspec = spec
        if spec.legend_loc in (None, "best"):
            legspec = spec.model_copy(update={"legend_loc": "lower right"})
        _merged_legend(ax, handles, labels, style, legspec)
        _entropy_saturation_annotation(ax, d0, rln_val, rln_lbl, style, spec, kind)
        return fig

    # ---- SINGLE-AXIS path (non-meaningful magnetic): total-only, no twin, no Rln --------------
    # KNOWN-ISSUES 12: this branch is taken exactly when magnetic entropy is NOT meaningfully
    # resolved, so the headline "S magnetic" series is a flat ~0 line crushed invisibly on the
    # x-axis — yet it used to be drawn and legended, sending the reader hunting for a curve
    # that is not visible. Skip it here (display-only: entropy_magnetic still reaches the CSV
    # and JSON). Per-field magnetic overlays (role "magnetic_field") are default-off explicit
    # user selections and stay drawable.
    plotted = [(r, s) for (r, s) in plotted if s.role != "magnetic"]
    ally = []
    for r, s in plotted:
        x = np.asarray(s.x, float); y = np.asarray(s.y, float)
        kw = dict(marker="none", ls=(getattr(s, "linestyle", None) or "-"),
                  lw=style.line_width, color=_color(s), label=series_label(r, s))
        ax.plot(x, y, **kw)
        ally.append(y[np.isfinite(y)])

    # full-range y-view (mirrors the override applied after _finish)
    y_bottom = y_top = None
    if spec.ymin is not None or spec.ymax is not None:
        y_bottom, y_top = spec.ymin, spec.ymax
    else:
        finite = [a for a in ally if a.size]
        if finite:
            allcat = np.concatenate(finite)
            lo = min(0.0, float(allcat.min())); hi = float(allcat.max())
            if hi > lo:
                y_bottom, y_top = lo, hi * 1.05

    _entropy_saturation_annotation(ax, d0, rln_val, rln_lbl, style, spec, kind)
    _finish(ax, kind, spec, style, "Temperature (K)", "S (J/mol·K)")
    # Override AFTER _finish so full-range wins over _apply_robust_view (robust clipping is wrong
    # for a cumulative monotone S_total). Respect an explicit user ymin/ymax.
    if spec.ymin is None and spec.ymax is None and y_bottom is not None and y_top is not None:
        ax.set_ylim(bottom=y_bottom, top=y_top)
    return fig


def _entropy_saturation_annotation(ax, d0, rln_val, rln_lbl, style, spec=None, kind=None):
    """Saturation annotation for hc_entropy_vs_t (on the LEFT axis, corner chosen by
    `_place_annotation`): TOTAL saturation is
    the headline (always); the magnetic saturation and its % of the Rln plateau are shown ONLY
    when magnetic entropy was actually resolved (last finite S_magnetic > 5% of the Rln value).
    On a fixture where the fitted lattice ~ total Cp, S_magnetic ~ 0 (or slightly negative), so a
    "% Rln" comparison would be meaningless."""
    tot = d0.get("entropy_total") or []
    mag = d0.get("entropy_magnetic")
    sat_tot = float(tot[-1]) if tot else None
    sat_mag = None
    if mag is not None:
        mfin = [float(v) for v in mag if v is not None and np.isfinite(v)]
        if mfin:
            sat_mag = mfin[-1]
    lines_txt = []
    if sat_tot is not None:
        lines_txt.append(f"S_total(T_max) = {sat_tot:.1f} J/mol·K")
    if sat_mag is not None and rln_val is not None and sat_mag > 0.05 * rln_val:
        lines_txt.append(
            f"S_mag(T_max) = {sat_mag:.2f} J/mol·K ({100.0 * sat_mag / rln_val:.0f}% {rln_lbl})")
    if lines_txt:
        fam = {"fontfamily": style.font_family} if style.font_family else {}
        _place_annotation(
            ax.text(0.02, 0.98, "\n".join(lines_txt), transform=ax.transAxes, va="top",
                    ha="left", fontsize=style.font_pt - 1, gid=ANNOTATION_GID, **fam), spec, style, kind)


def render_hc_c_over_t_linear(results, spec=None, style=None, overlay=None):
    results, kind, spec, style, fig, ax = _setup(results, "hc_c_over_t_linear", spec, style)
    _plot_data(ax, results, kind, spec, style, overlay)
    if overlay is None and results:
        d0 = results[0].data or {}
        if spec.fit_window_shade:      # opt-in (owner: useful, but OFF by default)
            _hc_fit_window_shade(ax, d0.get("temperature"))
        _hc_lowt_annotation(ax, d0, style, spec, kind)
    _finish(ax, kind, spec, style, "Temperature (K)", "Cp/T (J/mol·K²)")
    return fig

def render_hc_lowt_multifield(results, spec=None, style=None, overlay=None):
    results, kind, spec, style, fig, ax = _setup(results, "hc_lowt_multifield", spec, style)
    if overlay is None and kind.group_colored:
        plotted, handles = _plot_data_grouped(ax, results, kind, spec, style)
    else:
        plotted = _plot_data(ax, results, kind, spec, style, overlay); handles = None
    if overlay is None and spec.fit_line:                  # N3: master fit-line toggle gates all lines
        want = spec.fit_lines                              # None => all per-(model,field) lines
        for r in results:
            for g in (r.data or {}).get("field_groups", []):
                if g["status"] != "ok":
                    continue
                for f in g["fits"]:
                    if not f.get("ok"):
                        continue
                    lkey = f"{f['key']}@{g['field_oe']:g}"
                    if want is not None and lkey not in want:
                        continue
                    xg = np.asarray(f["t2_grid"], float)
                    yg = np.asarray(f["cp_over_t_fit"], float)
                    ln = _fit_plot(ax, xg, yg, style, label=lkey)
                    fn = _LOWT_FUNCS.get(f["key"]); prm = f.get("params") or {}
                    if fn is not None and prm and xg.size and float(xg.min()) > 0.0:
                        xe = np.linspace(0.0, float(xg.min()), 40)
                        _extrap_plot(ax, xe, fn(xe, prm), style, ln.get_color(), yg)
    _finish(ax, kind, spec, style, "T² (K²)", "Cp/T (J/mol·K²)", legend_handles=handles)
    return fig

def _render_param_vs_field(results, kind_key, ylabel, spec=None, style=None, overlay=None):
    results, kind, spec, style, fig, ax = _setup(results, kind_key, spec, style)
    _plot_data(ax, results, kind, spec, style, overlay)
    _finish(ax, kind, spec, style, _field_axis_label("Field", style), ylabel)
    return fig

def render_hc_gamma_vs_field(results, spec=None, style=None, overlay=None):
    return _render_param_vs_field(results, "hc_gamma_vs_field", "γ (J/mol·K²)", spec, style, overlay)

def render_hc_thetaD_vs_field(results, spec=None, style=None, overlay=None):
    return _render_param_vs_field(results, "hc_thetaD_vs_field", "θ_D (K)", spec, style, overlay)

def render_hc_A_vs_field(results, spec=None, style=None, overlay=None):
    return _render_param_vs_field(results, "hc_A_vs_field", "A (J/mol·K⁴)", spec, style, overlay)

def render_hc_T0_vs_field(results, spec=None, style=None, overlay=None):
    return _render_param_vs_field(results, "hc_T0_vs_field", "T₀ (K)", spec, style, overlay)

# ---- HC Schottky plot-kind renderers (HC slice 3 / Task 7) ----

def render_hc_f_vs_field(results, spec=None, style=None, overlay=None):
    return _render_param_vs_field(results, "hc_f_vs_field", "f (TLS / f.u.)", spec, style, overlay)

def render_hc_alphaN_vs_field(results, spec=None, style=None, overlay=None):
    return _render_param_vs_field(results, "hc_alphaN_vs_field", "αN (J·K/mol)", spec, style, overlay)

def render_hc_delta_vs_field(results, spec=None, style=None, overlay=None):
    results, kind, spec, style, fig, ax = _setup(results, "hc_delta_vs_field", spec, style)
    _plot_data(ax, results, kind, spec, style, overlay)
    if overlay is None:                                    # optional Zeeman/ZFS overlay line
        for r in results:  # results is already a list after _setup
            ov = (r.data or {}).get("schottky_overlay")
            if not (ov and ov.get("ok")):
                continue
            fields = [g["field_oe"] for g in (r.data or {}).get("field_groups", [])
                      if g.get("status") == "ok"
                      and g.get("schottky", {}).get("delta_determined")]
            if not fields:
                continue
            B = np.linspace(min(fields), max(fields), 50) / 1e4   # Oe -> T
            g_ = ov["g_factor"]
            D0 = ov.get("Delta0") or 0.0
            if ov["model"] == "zeeman":
                y = g_ * MU_B_OVER_KB * B + D0
            else:
                y = np.sqrt(D0 ** 2 + (g_ * MU_B_OVER_KB * B) ** 2)
            disp = _field_scale(getattr(style, "field_unit", "Oe"))   # Oe*1 or Oe*1e-4
            _fit_plot(ax, (B * 1e4 * disp).tolist(), y.tolist(), style,
                      label=f"{ov['model']} g={g_:.2g}")
    _finish(ax, kind, spec, style, _field_axis_label("Field", style), "Δ (K)")
    return fig

def render_hc_schottky_multifield(results, spec=None, style=None, overlay=None):
    results, kind, spec, style, fig, ax = _setup(results, "hc_schottky_multifield", spec, style)
    if overlay is None and kind.group_colored:
        plotted, handles = _plot_data_grouped(ax, results, kind, spec, style)
    else:
        plotted = _plot_data(ax, results, kind, spec, style, overlay)
        handles = None
    _finish(ax, kind, spec, style, "T (K)", "Cp (J/mol·K)", legend_handles=handles)
    return fig

# ---- HC transition-search plot-kind renderers (HC slice 4 / Task 10) ----

def render_hc_tc_vs_field(results, spec=None, style=None, overlay=None):
    return _render_param_vs_field(results, "hc_tc_vs_field", "T_c (K)", spec, style, overlay)

def render_hc_transition_multifield(results, spec=None, style=None, overlay=None):
    results, kind, spec, style, fig, ax = _setup(results, "hc_transition_multifield", spec, style)
    if overlay is None and kind.group_colored:
        plotted, handles = _plot_data_grouped(ax, results, kind, spec, style)
    else:
        plotted = _plot_data(ax, results, kind, spec, style, overlay); handles = None
    _finish(ax, kind, spec, style, "T (K)", "Cp (J/mol·K)", legend_handles=handles)
    return fig

def render_hc_transition_signal(results, spec=None, style=None, overlay=None):
    results, kind, spec, style, fig, ax = _setup(results, "hc_transition_signal", spec, style)
    if overlay is None and kind.group_colored:
        plotted, handles = _plot_data_grouped(ax, results, kind, spec, style)
    else:
        plotted = _plot_data(ax, results, kind, spec, style, overlay); handles = None
    _finish(ax, kind, spec, style, "T (K)", "Cp − background (J/mol·K)", legend_handles=handles)
    return fig

# ---- Resistivity renderers ----
_RHO_UNIT_LADDER = ((1e-3, 1e6, "µΩ·cm"), (1.0, 1e3, "mΩ·cm"))   # (median cut, factor, label)


def _rho_axis_autoscale(ax):
    """Auto engineering prefix for the ρ axis (spec D3): pick µΩ·cm / mΩ·cm / Ω·cm from the
    median |ρ| of the plotted DATA lines (gid None) and multiply every non-refline line's
    y-data by the factor (fit lines scale with the data). Call after all data+fit lines are
    drawn and BEFORE reflines/annotations/inset/_finish, so robust-view sees scaled data and
    axvline reflines (y-data in axes coords) are never touched. Returns (factor, unit_label)."""
    ys = [np.asarray(ln.get_ydata(), float) for ln in ax.lines if ln.get_gid() is None]
    ys = np.concatenate(ys) if ys else np.array([])
    ys = ys[np.isfinite(ys)]
    factor, unit = 1.0, "Ω·cm"
    if ys.size:
        med = float(np.median(np.abs(ys)))
        for cut, f, u in _RHO_UNIT_LADDER:
            if med < cut:
                factor, unit = f, u
                break
    if factor != 1.0:
        for ln in ax.lines:
            if ln.get_gid() != "refline":
                ln.set_ydata(np.asarray(ln.get_ydata(), float) * factor)
        ax.relim()
        ax.autoscale_view()
    return factor, unit


_RHO_T_FITS = ("power_law",)


def _best_tc_curve(b):
    """Widest rho(T) ramp of bridge dict `b` that carries a detected Tc, else None."""
    for c in sorted(b.get("rho_t_curves", []), key=lambda c: -(c.get("n_points") or 0)):
        if c.get("tc_mid_k") is not None:
            return c
    return None


def _rho_bridge_color(ax, b):
    """Colour of the bridge's plotted data lines. Multi-channel series labels start 'Ch{n} '
    (catalog _chan_prefix); single-channel plots carry one bridge, so the sole data line wins.
    Returns None when no line matches the prefix (multi-channel: never paint one bridge's fit/
    marker in another bridge's colour — let the style layer pick a colour instead)."""
    pre = f"Ch{b.get('channel')} "
    data = [ln for ln in ax.lines if ln.get_gid() is None]
    for ln in data:
        if str(ln.get_label()).startswith(pre):
            return ln.get_color()
    if len(data) == 1 and not str(data[0].get_label()).startswith("Ch"):
        return data[0].get_color()                   # single-channel plot: unprefixed sole line
    return None


def _rho_tc_markers(ax, d, spec, style):
    """Vertical dashed T_c line + rotated label per bridge with a detected transition, in that
    bridge's series colour. gid='refline' keeps them out of robust-view/prefix scaling.
    A low-confidence detection (tc_low_confidence — e.g. a noisy plateau that bypasses the
    narrowness gate) renders visually WEAKER (dotted, thinner, semi-transparent) so the doubt
    is visible on the figure, not only flagged in the data (whole-branch review N1)."""
    if not spec.tc_marker:
        return
    for b in d.get("bridges", []):
        c = _best_tc_curve(b)
        if c is None:
            continue
        col = _rho_bridge_color(ax, b) or "black"
        if c.get("tc_low_confidence"):
            ax.axvline(c["tc_mid_k"], color=col, ls=":", lw=0.6, alpha=0.5, gid="refline")
            ax.text(c["tc_mid_k"], 0.02, f" $T_c$ = {c['tc_mid_k']:.2f} K?",
                    transform=ax.get_xaxis_transform(), rotation=90, va="bottom", ha="left",
                    fontsize=style.font_pt - 2, color=col, alpha=0.6, gid="refline-label:v")
        else:
            ax.axvline(c["tc_mid_k"], color=col, ls="--", lw=0.8, gid="refline")
            ax.text(c["tc_mid_k"], 0.02, f" $T_c$ = {c['tc_mid_k']:.2f} K",
                    transform=ax.get_xaxis_transform(), rotation=90, va="bottom", ha="left",
                    fontsize=style.font_pt - 2, color=col, gid="refline-label:v")


def _rho_annotation(ax, d, style, factor=1e6, unit="µΩ·cm", spec=None, kind=None):
    """Frameless rho0/n/RRR (+ Tc onset/mid/zero) box — _hc_lowt_annotation pattern, with
    the corner chosen by `_place_annotation` (this box is the one that shipped across the
    T_c guide line). ρ₀/n/RRR from the first bridge carrying any of them (box stays compact); the Tc
    line is taken from whichever bridge actually has a transition, so a Tc on bridge 2 (which
    _rho_tc_markers already marks) is never dropped just because bridge 1 supplied the scalars.
    Omitted when there is nothing to say. ρ₀ is shown in the SAME engineering unit the y-axis was
    autoscaled to (`factor`/`unit` from _rho_axis_autoscale), so the annotation never disagrees
    with the axis; `.3g` keeps the number non-scientific whenever the scaled value is compact."""
    lines = []
    bridges = d.get("bridges", [])
    for b in bridges:                                # scalars: first bridge that carries any
        _pl = b.get("power_law") or {}
        # A declined fit publishes no number: a search bound or an unresolved exponent is not
        # a measurement (POWER_LAW_DECLINE_FLAGS — the TTO kappa_ph rule applied here).
        p = ({} if set(_pl.get("quality_flags") or []) & POWER_LAW_DECLINE_FLAGS
             else (_pl.get("params") or {}))
        seg = []
        if b.get("residual_rho") is not None:
            seg.append(f"ρ₀ = {b['residual_rho'] * factor:.3g} {unit}")
        if p.get("n") is not None:
            seg.append(f"n = {p['n']:.2f}")
        if b.get("rrr") is not None:
            seg.append(f"RRR = {b['rrr']:.3g}")
        if seg:
            lines.extend(seg)
            break
    for b in bridges:                                # Tc: first bridge (any) with a transition
        c = _best_tc_curve(b)
        if c is not None:
            t = f"$T_c$ = {c['tc_mid_k']:.2f} K"
            if c.get("tc_onset_k") is not None and c.get("tc_zero_k") is not None:
                t += f" (onset {c['tc_onset_k']:.2f}, zero {c['tc_zero_k']:.2f})"
            if c.get("tc_low_confidence"):           # N1: doubt visible on the figure, not
                t += " (low confidence)"             # only in the data/CSV
            lines.append(t)
            break
    if not lines:
        return None
    fam = {"fontfamily": style.font_family} if style.font_family else {}
    t = ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
                fontsize=style.font_pt - 1, gid=ANNOTATION_GID, **fam)
    _place_annotation(t, spec, style, kind)
    return t


def _ann_hits_legend_text(leg, ann, renderer):
    """True when the annotation's text block overlaps any of the legend's TEXT glyph boxes.
    Deliberately ignores the legend frame and handles: the shipped gallery figure sits with the
    annotation tail inside the legend's padding by design, and only text-through-text is the
    render defect (D4)."""
    ab = ann.get_window_extent(renderer)
    for t in leg.get_texts():
        b = t.get_window_extent(renderer)
        if b.x0 <= ab.x1 and ab.x0 <= b.x1 and b.y0 <= ab.y1 and ab.y0 <= b.y1:
            return True
    return False


def _install_legend_dodge(fig, ax, ann, inset_ax=None):
    """Keep an annotated panel readable at ANY canvas size.

    Written for the rho(T) headline; the 1/chi panel reuses it unchanged (inset_ax=None)
    for the identical defect on a different plot kind.

    Placement is decided once at the creation size (90x70 mm), where annotation, legend and
    the low-T inset all clear each other — but a GUI card canvas is smaller, fonts keep their
    point size, and the annotation's longest line then runs THROUGH the legend text
    (matplotlib re-anchors the legend on resize but is blind to text artists), while the
    42%x40% inset covers the very curves it supplements. So re-check at every realized draw;
    a genuine text-on-text hit is the "this canvas is too small for the full journal layout"
    signal: hide the inset (Focus mode / export sizes re-render it fresh) and move the legend
    to whichever lower corner covers fewer data points (upper left IS the annotation). A draw
    with no collision changes nothing, keeping every fixed-size export byte-identical."""
    state = {"busy": False}

    def _leg_area_frac():
        leg = ax.get_legend()
        if leg is None:
            return None
        lb = leg.get_window_extent()
        ab = ax.get_window_extent()
        area = ab.width * ab.height
        return (lb.width * lb.height) / area if area > 0 else None

    # Baseline: the legend's area fraction of the axes AT RENDER SIZE — the as-designed
    # layout (the occupancy chooser accepted it, or _legend_avoiding pinned it inside as the
    # owner's least-bad call). The second trigger below is pure geometry: fonts keep their
    # point size while a realized canvas shrinks, so the legend's area fraction GROWS in
    # exact proportion to the lost axes area; >1.25× the baseline (~12% linear shrink) is
    # the "this canvas is too small for the journal layout" signal in a form that cannot
    # fire at the creation size (ratio 1.0) and needs no absolute magic fraction. Measured
    # now, while the figure still has its render geometry — a GUI canvas realizes its first
    # draw at the card size, which must not become the baseline.
    try:
        fig.canvas.draw()
        state["base_area"] = _leg_area_frac()
    except Exception:
        state["base_area"] = None

    def _on_draw(event):
        if state["busy"]:
            return
        leg = ax.get_legend()
        if leg is None or not ann.get_visible():
            return
        if getattr(leg, "_cryosweep_dodged", False):
            return
        # the event's own renderer first; vector canvases (PDF/SVG) have no get_renderer()
        renderer = getattr(event, "renderer", None) or \
            getattr(fig.canvas, "get_renderer", lambda: None)()
        try:
            hit = renderer is not None and _ann_hits_legend_text(leg, ann, renderer)
            if not hit and renderer is not None and state["base_area"]:
                now = _leg_area_frac()
                hit = now is not None and now > 1.25 * state["base_area"]
        except Exception:      # a renderer that cannot measure text extents -> leave layout alone
            return
        if not hit:
            return
        state["busy"] = True
        try:
            if inset_ax is not None and inset_ax.get_visible():
                inset_ax.set_visible(False)
            # legend size in axes fraction -> candidate lower-corner boxes -> fewest points
            bb = leg.get_window_extent(renderer)
            inv = ax.transAxes.inverted()
            (fx0, fy0), (fx1, fy1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
            w, h = abs(fx1 - fx0), abs(fy1 - fy0)
            pts = _axes_points_in_host_frac(ax)
            pad = 0.02
            boxes = {"lower right": (1 - pad - w, pad, 1 - pad, pad + h),
                     "lower left":  (pad, pad, pad + w, pad + h)}

            def _count(box):
                if len(pts) == 0:
                    return 0
                x0, y0, x1, y1 = box
                return int(((pts[:, 0] >= x0) & (pts[:, 0] <= x1)
                            & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)).sum())
            leg.set_loc(min(boxes, key=lambda k: _count(boxes[k])))
            # on a canvas this crowded no corner is guaranteed data-free: back the legend
            # with a soft white patch so its text stays readable over markers (the journal
            # frameless default is for full-size figures, which never reach this branch)
            leg.set_frame_on(True)
            fr = leg.get_frame()
            fr.set_facecolor("white"); fr.set_alpha(0.85); fr.set_edgecolor("none")
            leg._cryosweep_dodged = True
            fig.canvas.draw_idle()
        finally:
            state["busy"] = False

    fig.canvas.mpl_connect("draw_event", _on_draw)


def _rho_lowt_inset(ax, d, spec, style, factor, unit):
    """Self-contained low-T inset (PQ-5 _add_lowt_inset pattern): widest rho(T) ramp of the
    first bridge as open squares + dashed power-law fit + Tc marker, over
    [0, max(30 K, fit-window top, Tc+5 K)]. Suppressed when spec.lowt_inset is False, data
    starts above 30 K, or fewer than 5 points fall in the window. y shares the main axis'
    engineering prefix (`factor`/`unit` from _rho_axis_autoscale)."""
    if not spec.lowt_inset:
        return None
    b = next((bb for bb in d.get("bridges", []) if bb.get("rho_t_curves")), None)
    if b is None:
        return None
    curve = max(b["rho_t_curves"], key=lambda c: len(c.get("temperature") or []))
    T = np.asarray(curve.get("temperature") or [], float)
    R = np.asarray(curve.get("rho") or [], float) * factor
    m = np.isfinite(T) & np.isfinite(R)
    T, R = T[m], R[m]
    if T.size == 0 or float(T.min()) > 30.0:
        return None
    hi = 30.0
    pl = b.get("power_law") or {}
    if pl.get("fit_range"):
        hi = max(hi, float(pl["fit_range"][1]))
    tc = _best_tc_curve(b)
    if tc is not None:
        hi = max(hi, float(tc["tc_mid_k"]) + 5.0)
    w = (T >= 0) & (T <= hi)
    if int(np.count_nonzero(w)) < 5:
        return None
    iax = _lowt_inset_axes(ax, spec, style)    # measured corner, or dropped-with-note (None)
    if iax is None:
        return None
    iax.plot(T[w], R[w], marker="s", markerfacecolor="none", ls="none",
             ms=max(style.marker_size - 1.0, 2.0), color="0.2",
             markeredgewidth=(style.edge_width if style.edge_width is not None else 0.8))
    if pl.get("params") and pl.get("fit_range") and \
            not (set(pl.get("quality_flags") or []) & NO_FIT_LINE_FLAGS):
        lo_f, hi_f = pl["fit_range"]
        p = pl["params"]
        Tg = np.linspace(lo_f, hi_f, 50)
        iax.plot(Tg, (p["rho0"] + p["A"] * np.power(Tg, p["n"])) * factor, "--",
                 color="0.2", lw=style.line_width)
    if tc is not None:
        iax.axvline(tc["tc_mid_k"], color="0.2", ls="--", lw=0.6)
    iax.tick_params(labelsize=5, length=2, width=0.5)
    # No x-axis label: the inset sits at lower-right, so an "T (K)" xlabel drops into the host's
    # x-tick/label band and reads as clipped. The tick numbers already denote temperature and the
    # host axis is labelled "Temperature (K)", so the inset x-label is redundant. (Journal inset
    # craft — matches the compact PQ-5 lower-right inset; the y-label stays, it clears everything.)
    iax.set_ylabel(f"ρ ({unit})", fontsize=5, labelpad=1)
    for sp in iax.spines.values():
        sp.set_linewidth(0.5)
    return iax


_RHO_SPIKE_MULT = 8.0    # strip only y > this multiple of the line's p90 — real contact-glitch
_RHO_SPIKE_Q = 90.0      # spikes run ~200-340× the bulk; the strongest legitimate rho(T) rise on
                         # file (dense-low-T monotone ramps incl.) keeps max/p90 ≲ 1.2, so a whole
                         # decade+ of headroom separates the two populations.


def _rho_insulating_labels(results, kind, spec, style):
    """Legend labels of plotted rho(T) series whose underlying curve is classified 'insulating'.
    Curve identity travels on the Series.key (`b{ch}:T:{field}:{dir}`, catalog); we map each
    plotted key to its curve's classification straight from the analyzer result. A label shared
    by >1 curve is protected whenever ANY of them is insulating (conservative — worst case skips
    a glitch, never deletes a genuine low-T divergence)."""
    protected = set()
    for r in results:
        classif = {}
        for b in (r.data or {}).get("bridges", []):
            ch = b.get("channel")
            for c in b.get("rho_t_curves", []):
                key = f"b{ch}:T:{_held(c.get('held_field_oe'), '%.0f')}:{c.get('direction', 0)}"
                classif[key] = c.get("classification")
        for s in select_series(kind.series(r, field_unit=style.field_unit), spec):
            if classif.get(s.key) == "insulating":
                protected.add(series_label(r, s))
    return protected


def _rho_strip_spikes(ax, spec, style, protected=frozenset()):
    """Display-only: break each rho(T) data line at extreme UPPER spikes (contact glitches /
    instrument sentinels — real files carry points hundreds of × the bulk clustered at the ramp
    ends). A point is a spike only when it exceeds _RHO_SPIKE_MULT × the line's own upper
    quantile (p90): quantile-relative, NOT median±MAD — a median/MAD envelope mis-fires on a
    smooth monotone ramp whose sampling is dense at low T (the low-T plateau owns the median and
    the tiny MAD makes the ramp's legitimate upper half read as a "tail"; regression caught by
    the controller re-gate on rho_sc_synth ch2).

    A ≥8× excursion past p90 CAN come from a smooth curve's shape — a densely-sampled-at-high-T /
    sparse-at-low-T Arrhenius insulator (ρ ∝ e^{Ea/T}) piles its p90 near the bulk and its genuine
    diverging low-T points sit far above it. Those points are real, so curves the analyzer
    classified 'insulating' (`protected` labels) are skipped entirely — the multiple only strips
    off-scale glitches on metallic/non-monotonic ramps where the p90 headroom argument holds.
    Spikes are set to NaN so (a) the connect-line no longer draws a full-height vertical stripe and
    (b) autoscale/robust-view frame the bulk. Upper side only — a superconducting drop to ~0 is
    never removed. gid!=None (fit/refline) untouched; analyzer data + CSV unchanged. Gated on
    robust_view (off -> raw)."""
    use = spec.robust_view if spec.robust_view is not None else style.robust_view
    if not use or ax.get_yscale() != "linear":
        return
    for ln in ax.lines:
        if ln.get_gid() is not None:                 # data lines only
            continue
        if str(ln.get_label()) in protected:         # insulating curve -> real low-T divergence
            continue
        y = np.asarray(ln.get_ydata(), float)
        finite = np.isfinite(y)
        if int(finite.sum()) < 8:
            continue
        hi = float(np.percentile(y[finite], _RHO_SPIKE_Q))
        if not (np.isfinite(hi) and hi > 0):         # non-positive scale: no meaningful multiple
            continue
        spike = finite & (y > _RHO_SPIKE_MULT * hi)
        if not spike.any():
            continue
        y2 = y.copy(); y2[spike] = np.nan
        ln.set_ydata(y2)


def _legend_avoiding(ax, spec, style, inset_present=False, annotation_present=False):
    """Place a legend that must avoid a corner-pinned inset and/or annotation (PQ-4 fix wave).

    Named for rho(T), where it was first needed, but nothing in it is rho-specific — the 1/chi
    panel reuses it for the same defect: a Curie-Weiss annotation is pinned at upper-left and
    matplotlib's "best" cannot see text, so on a multi-curve file it drops the legend straight
    on top of the annotation. De-duplicates repeated fit labels (each
    channel's power-law overlay shares the one 'power-law fit' entry) and, in auto mode, chooses
    the inside corner that avoids BOTH the lower-right inset and the upper-left annotation box —
    neither of which is a line artist, so matplotlib's own 'best' cannot see them and would drop
    the legend on top of / behind them (and, when wide, off the right figure edge). Falls back to
    the shared outside-right relocation when no inside corner is clear (grows the canvas -> never
    clipped). Explicit user legend_loc still wins."""
    legend_on = spec.legend_on if spec.legend_on is not None else style.legend_on
    if not legend_on:
        return
    handles, labels = ax.get_legend_handles_labels()
    seen, H, L = set(), [], []
    for h, l in zip(handles, labels):
        if not l or l.startswith("_") or l in seen:
            continue
        seen.add(l); H.append(h); L.append(l)
    if not H:
        return
    legend_sz = style.legend_size if style.legend_size is not None else style.font_pt - 1
    prop = {"size": legend_sz}
    if style.font_family:
        prop["family"] = style.font_family
    # Placement delegates to _draw_legend's occupancy chooser, which MEASURES the inset and
    # annotation bboxes (they are obstacle rects) instead of trusting these two booleans —
    # kept in the signature so call sites don't churn, no longer read.
    #
    # "best" is pinned to INSIDE here (owner decision, PQ-4): the ρ(T) headline stays inside
    # even when crowded — a frameless legend over the ramp tail is journal-fine, an outside
    # relocation grows the export off its exact-mm size, and the small-canvas draw-time dodge
    # (_install_legend_dodge: hide the inset, back the legend with white) only works on an
    # inside legend. The chooser still picks the measured least-bad spot. A genuinely dense
    # legend (>LEGEND_INSIDE_MAX) and an explicit "outside" still relocate.
    loc = spec.legend_loc if spec.legend_loc is not None else style.legend_loc
    if loc == "best" and len(H) <= LEGEND_INSIDE_MAX:
        _draw_legend(ax, prop, style, spec, H, force_loc="inside")
    else:
        _draw_legend(ax, prop, style, spec, H)


def render_resistivity(results, spec=None, style=None, overlay=None):
    results, kind, spec, style, fig, ax = _setup(results, "resistivity_rho_t", spec, style)
    _plot_data(ax, results, kind, spec, style, overlay)
    # Effective y-scale is resolved in _finish; compute it here so spike-strip (which runs before
    # _finish) matches robust-view's "linear only" contract -- on a log request neither fires.
    eff_yscale = spec.yscale if spec.yscale is not None else kind.default_yscale
    if overlay is None and eff_yscale == "linear":
        _rho_strip_spikes(ax, spec, style, _rho_insulating_labels(results, kind, spec, style))
    if overlay is None and spec.fit_line:
        want = _fit_lines_wanted(spec, _RHO_T_FITS)
        for r in results:
            for b in (r.data or {}).get("bridges", []):
                f = b.get("power_law")
                if "power_law" in want and f and \
                        not (set(f.get("quality_flags") or []) & NO_FIT_LINE_FLAGS):
                    lo, hi = f["fit_range"]
                    p = f["params"]
                    T = np.linspace(lo, hi, 50)
                    yfit = p["rho0"] + p["A"] * np.power(T, p["n"])
                    ln = _fit_plot(ax, T, yfit, style,
                                   series_color=_rho_bridge_color(ax, b),
                                   label="power-law fit", linestyle="--")
                    if lo > 0.0:
                        # continuation to T = 0: the intercept is rho0, the residual
                        # resistivity the annotation reports
                        Te = np.linspace(0.0, lo, 40)
                        _extrap_plot(ax, Te, p["rho0"] + p["A"] * np.power(Te, p["n"]),
                                     style, ln.get_color(), yfit)
                    if spec.fit_window_shade:
                        _hc_fit_window_shade(ax, [lo, hi])
    _f, unit = _rho_axis_autoscale(ax)
    inset_present = False
    annotation_present = False
    ann_text = None
    _iax = None
    if overlay is None and results:
        d0 = results[0].data or {}
        _rho_tc_markers(ax, d0, spec, style)
        if spec.annotation:
            ann_text = _rho_annotation(ax, d0, style, _f, unit, spec, kind)
            annotation_present = ann_text is not None
        _iax = _rho_lowt_inset(ax, d0, spec, style, _f, unit)
        inset_present = _iax is not None
    # Draw the legend ourselves (draw_legend=False) so it can dodge the lower-right inset and the
    # upper-left annotation, which matplotlib's own placement is blind to (D2/D3 fix wave).
    _finish(ax, kind, spec, style, "Temperature (K)", f"ρ ({unit})", draw_legend=(overlay is not None))
    if overlay is None:
        _legend_avoiding(ax, spec, style, inset_present, annotation_present)
        loc = spec.legend_loc if spec.legend_loc is not None else style.legend_loc
        if annotation_present and loc == "best":       # auto placement only; a pinned loc is the user's call
            _install_legend_dodge(fig, ax, ann_text, _iax)
    return fig

def _mr_journal_style(style):
    """MR default craft (PQ-4 Task 8): T-ordered colormap dark->light + black marker edges.
    Applied only when the user has set neither colormap nor palette, so either knob opts out."""
    if style.colormap is not None or style.palette is not None:
        return style
    return style.model_copy(update={
        "colormap": "viridis",
        "edge_color": style.edge_color if style.edge_color is not None else "black",
        "edge_width": style.edge_width if style.edge_width is not None else 0.5,
    })

def render_resistivity_mr(results, spec=None, style=None, overlay=None):
    results, kind, spec, style, fig, ax = _setup(results, "resistivity_mr", spec, style)
    _plot_data(ax, results, kind, spec, _mr_journal_style(style), overlay)
    _finish(ax, kind, spec, style, _field_axis_label("Field", style), "ρ (Ohm·cm)")
    return fig

def render_resistivity_mr_pct(results, spec=None, style=None, overlay=None):
    results, kind, spec, style, fig, ax = _setup(results, "resistivity_mr_pct", spec, style)
    _plot_data(ax, results, kind, spec, _mr_journal_style(style), overlay)
    _finish(ax, kind, spec, style, _field_axis_label("Field", style), "MR (%)")
    return fig

_KB_MEV_PER_K = 8.617333262e-2   # Boltzmann, meV/K (matches fitting.transport.KB_MEV_PER_K)


def render_resistivity_arrhenius(results, spec=None, style=None, overlay=None):
    """log rho vs 1000/T with the Arrhenius fit line and the E_a annotation. The gap line
    carries its assumption ON the figure ("only if intrinsic" — the factor-of-two trap);
    a declined fit (insufficient_rho_span / ea_unresolved) draws NO fit line and states the
    reason instead, the transport decline discipline."""
    results, kind, spec, style, fig, ax = _setup(results, "resistivity_arrhenius", spec, style)
    _plot_data(ax, results, kind, spec, style, overlay)
    if overlay is None:
        lines = []
        for r in results:
            for b in (r.data or {}).get("bridges", []):
                f = b.get("arrhenius")
                if not f:
                    continue
                flags = set(f.get("quality_flags") or [])
                declined = flags & {"insufficient_rho_span", "ea_unresolved"}
                if declined:
                    # two lines: the flag names are long and a single line clips at the
                    # right edge of the default 90 mm canvas (seen, not guessed)
                    lines.append("Arrhenius fit declined —")
                    lines.append("  " + "; ".join(sorted(declined)))
                    continue                              # a non-measurement gets no line
                p_ = f["params"]; ea = p_["e_a_mev"]
                if spec.fit_line:
                    lo, hi = f["fit_range"]
                    Tg = np.linspace(lo, hi, 200)
                    rho_fit = np.exp(p_["ln_rho0"] + p_["e_a_mev"] / (_KB_MEV_PER_K * Tg))
                    _fit_plot(ax, 1000.0 / Tg, rho_fit, style, label="Arrhenius fit")
                sig = (f.get("sigma") or {}).get("e_a_mev")
                sig_txt = f" ± {sig:.2g}" if sig is not None else ""
                lines.append(f"E$_a$ = {ea:.1f}{sig_txt} meV")
                lines.append(f"E$_g$ = 2·E$_a$ = {2 * ea:.1f} meV (only if intrinsic)")
                if "window_sensitive" in flags:
                    spread = b.get("arrhenius_ea_spread_mev")
                    if spread is not None:
                        # wrapped like the decline note: one line clips at 90 mm (seen)
                        lines.append("WINDOW-SENSITIVE:")
                        lines.append(f"  E$_a$ moves {spread:.1f} meV across windows")
        if lines:
            fam = {"fontfamily": style.font_family} if style.font_family else {}
            _place_annotation(
                ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, va="top",
                        ha="left", fontsize=style.font_pt - 1, gid=ANNOTATION_GID, **fam), spec, style, kind)
    _finish(ax, kind, spec, style, "1000/T (1/K)", "ρ (Ω·cm)")
    return fig


def render_resistivity_mr_pct_t(results, spec=None, style=None, overlay=None):
    results, kind, spec, style, fig, ax = _setup(results, "resistivity_mr_pct_t", spec, style)
    _plot_data(ax, results, kind, spec, style, overlay)
    ax.axhline(0, color="black", lw=0.8, gid="refline")   # MR = 0 reference
    _finish(ax, kind, spec, style, "Temperature (K)", "MR at H$_{max}$ (%)")
    return fig

def render_resistivity_rho_t2(results, spec=None, style=None, overlay=None):
    results, kind, spec, style, fig, ax = _setup(results, "resistivity_rho_t2", spec, style)
    _plot_data(ax, results, kind, spec, style, overlay)
    if overlay is None:
        want = _fit_lines_wanted(spec, _RHO_T2_FITS)
        for r in results:
            for b in (r.data or {}).get("bridges", []):
                if "linear" in want:
                    f = b.get("rho_t2_linear")
                    if f and "rho0_unresolved" not in (f.get("quality_flags") or []):
                        lo, hi = f["fit_range"]; p = f["params"]
                        T = np.linspace(lo, hi, 50)
                        yfit = p["rho0"] + p["beta"] * T * T
                        ln = _fit_plot(ax, T * T, yfit, style, label="ρ=ρ₀+βT² fit")
                        if lo > 0.0:      # continue to T² = 0: intercept = rho0
                            Te = np.linspace(0.0, lo, 40)
                            _extrap_plot(ax, Te * Te, p["rho0"] + p["beta"] * Te * Te,
                                         style, ln.get_color(), yfit)
                if "power_law" in want:
                    f = b.get("power_law")
                    if f and not (set(f.get("quality_flags") or [])
                                  & NO_FIT_LINE_FLAGS):
                        lo, hi = f["fit_range"]; p = f["params"]
                        T = np.linspace(lo, hi, 50)
                        yfit = p["rho0"] + p["A"] * np.power(T, p["n"])
                        ln = _fit_plot(ax, T * T, yfit, style,
                                       label="power-law fit", linestyle="--")
                        if lo > 0.0:
                            Te = np.linspace(0.0, lo, 40)
                            _extrap_plot(ax, Te * Te, p["rho0"] + p["A"] * np.power(Te, p["n"]),
                                         style, ln.get_color(), yfit)
    _finish(ax, kind, spec, style, "T² (K²)", "ρ (Ohm·cm)")
    return fig

# ---- Hall renderers ----
def render_hall(results, spec=None, style=None, overlay=None):
    results, kind, spec, style, fig, ax = _setup(results, "hall_rh_t", spec, style)
    _plot_data(ax, results, kind, spec, style, overlay)
    _finish(ax, kind, spec, style, "Temperature (K)", "R_H (m³/C)")
    return fig

def render_hall_mobility_t(results, spec=None, style=None, overlay=None):
    results, kind, spec, style, fig, ax = _setup(results, "hall_mobility_t", spec, style)
    _plot_data(ax, results, kind, spec, style, overlay)
    _finish(ax, kind, spec, style, "Temperature (K)", "μ (m²/V·s)")
    return fig

def render_hall_n_t(results, spec=None, style=None, overlay=None):
    results, kind, spec, style, fig, ax = _setup(results, "hall_n_t", spec, style)
    _plot_data(ax, results, kind, spec, style, overlay)
    _finish(ax, kind, spec, style, "Temperature (K)", "n (1/m³)")
    return fig

def render_hall_r2_t(results, spec=None, style=None, overlay=None):
    results, kind, spec, style, fig, ax = _setup(results, "hall_r2_t", spec, style)
    _plot_data(ax, results, kind, spec, style, overlay)
    _finish(ax, kind, spec, style, "Temperature (K)", "R² (Stage-B fit)")
    return fig

def render_hall_rxy_vs_b(results, spec=None, style=None, overlay=None):
    results, kind, spec, style, fig, ax = _setup(results, "hall_rxy_vs_B", spec, style)
    plotted = _plot_data(ax, results, kind, spec, style, overlay)
    if overlay is None:
        _draw_rxy_mirror_fit_lines(ax, results, spec, style, plotted)
    _finish(ax, kind, spec, style, "Field B (T)", "R_xy (Ω)")
    return fig

def render_hall_asym_vs_b(results, spec=None, style=None, overlay=None):
    results, kind, spec, style, fig, ax = _setup(results, "hall_asym_vs_B", spec, style)
    plotted = _plot_data(ax, results, kind, spec, style, overlay)
    if overlay is None:
        _draw_asym_fit_lines(ax, results, spec, style, plotted)
    _finish(ax, kind, spec, style, "|B| (T)", "R_asym (Ω)")
    return fig

_HALL_BRANCH_ROLE_COLORS = {"R_xy(+B)": "C0", "R_xy(−B)": "C3", "R_asym": "0.2"}

def render_hall_raw_vs_asym(results, spec=None, style=None, overlay=None):
    results, kind, spec, style, fig, ax = _setup(results, "hall_raw_vs_asym", spec, style)
    ax.axhline(0, color="black", lw=0.8, gid="refline")   # H=0 reference, under the markers
    if overlay is None and kind.group_colored:
        plotted, handles = _plot_data_grouped(ax, results, kind, spec, style,
                                               role_colors=_HALL_BRANCH_ROLE_COLORS)
    else:
        plotted = _plot_data(ax, results, kind, spec, style, overlay); handles = None
    if overlay is None:
        _draw_branch_fit_lines(ax, results, spec, style, plotted)
        _draw_asym_fit_lines(ax, results, spec, style, plotted)
    _finish(ax, kind, spec, style, "|B| (T)", "R_xy / R_asym (Ω)", legend_handles=handles)
    return fig

# ---- PQ-2 Task 3: composite kinds ----

def render_hall_two_panel(results, spec=None, style=None, overlay=None):
    """Side-by-side R_xy(B) raw sweeps (left, colour-by-T) + longitudinal (right); port of
    Step3_Hall_fit_field_dep_Mobility_sept.py:89-143. When a per-T longitudinal field sweep is
    present (HallTempPoint.R_xx_raw), the right panel plots the literal zero-field-subtracted
    R_xx(B) color-by-T, mirroring the left R_xy(B) panel (matches Step3's adjusted longitudinal
    R vs B). Falls back to the scalar rho_xx(T) plot when no per-T longitudinal sweep exists
    (e.g. a separate longitudinal file supplies only scalar rho_xx(T))."""
    if overlay is not None:
        results, kind, spec, style, fig, ax = _setup(results, "hall_two_panel", spec, style)
        _plot_data(ax, results, kind, spec, style, overlay)
        _finish(ax, kind, spec, style, "Field (T) / Temperature (K)", "R_xy (Ω) / ρ_xx (Ω·m)")
        return fig

    results, kind, spec, style, fig, axes = _setup_panels(results, "hall_two_panel", spec, style, 2)
    ax_left, ax_right = axes
    plotted = [(r, s) for r in results for s in select_series(kind.series(r, field_unit=style.field_unit), spec)]
    if not plotted:
        raise NothingToPlot(f"no series selected for kind {kind.key}")

    left_items = [(r, s) for r, s in plotted if s.key.startswith("rxy:")]

    groups = []
    for _, s in left_items:
        if s.group not in groups:
            groups.append(s.group)
    gcolor = _group_color_map(groups, style)
    for _, s in left_items:
        ax_left.plot(s.x, s.y, marker=style.marker, ls="none", ms=style.marker_size,
                     color=gcolor[s.group], label=s.label)
    ax_left.set_title("Hall")
    _finish(ax_left, kind, spec, style, "Field B (T)", "R_xy (Ω)")

    rxxb_items = [(r, s) for r, s in plotted if s.key.startswith("rxxb:")]
    rho_items = [(r, s) for r, s in plotted if s.key == "rhoxx"]

    if rxxb_items:
        rgroups = []
        for _, s in rxxb_items:
            if s.group not in rgroups:
                rgroups.append(s.group)
        rcolor = _group_color_map(rgroups, style)
        for _, s in rxxb_items:
            x = np.asarray(s.x, float); y = np.asarray(s.y, float)
            order = np.argsort(x)
            y0 = y - float(np.interp(0.0, x[order], y[order]))  # zero-subtract at B=0
            ax_right.plot(x, y0, marker=style.marker, ls="none", ms=style.marker_size,
                          color=rcolor[s.group], label=s.label)
        ax_right.set_title("Longitudinal")
        _finish(ax_right, kind, spec, style, "Field B (T)", "R_xx − R_xx(0) (Ω)")
    else:
        for _, s in rho_items:
            ax_right.plot(s.x, s.y, marker=style.marker, ls="-", lw=style.line_width,
                          ms=style.marker_size, color=style.color or "C0", label=s.label)
        ax_right.set_title("Longitudinal")
        _finish(ax_right, kind, spec, style, "Temperature (K)", "ρ_xx (Ω·m)")
    return fig


def render_hall_tdep_summary(results, spec=None, style=None, overlay=None):
    """R_H (left) / mobility (first twin) / J (offset second twin, when present) vs a shared T
    axis -- port of Step3_Hall_fit_field_dep_Mobility_sept.py:554-614's triple-axis summary
    figure. The J axis appears only when >=1 point carries current_density_J (populated since
    KNOWN-ISSUES 21: J = I/(w*t), gated on width AND thickness both supplied); absent J
    degrades cleanly to a two-axis figure (no dead third spine). Each axis carries exactly one series, so per-axis robust view (`_apply_robust_view`)
    is well-defined and applied to the host + every twin/offset axis (real data: a single
    pathological R_H point at one T can crush the whole left-axis view otherwise)."""
    if overlay is not None:
        results, kind, spec, style, fig, ax = _setup(results, "hall_tdep_summary", spec, style)
        _plot_data(ax, results, kind, spec, style, overlay)
        _finish(ax, kind, spec, style, "Temperature (K)", "R_H / μ / J")
        return fig

    results, kind, spec, style, fig, ax = _setup(results, "hall_tdep_summary", spec, style)
    plotted = {s.key: s for r in results for s in select_series(kind.series(r, field_unit=style.field_unit), spec)}
    if not plotted:
        raise NothingToPlot(f"no series selected for kind {kind.key}")

    fam = {"fontfamily": style.font_family} if style.font_family else {}
    label_sz = style.label_size if style.label_size is not None else style.font_pt
    handles, labels = [], []

    if "rh" in plotted:
        s = plotted["rh"]
        ln, = ax.plot(s.x, s.y, marker="o", ls="none", ms=style.marker_size, color="C0")
        handles.append(ln); labels.append(s.label)
        ax.set_ylabel("R_H (m³/C)", color="C0", fontsize=label_sz, **fam)
        ax.tick_params(axis="y", colors="C0")
        ax.spines["left"].set_color("C0")

    tax = oax = None
    if "mu" in plotted:
        s = plotted["mu"]
        tax = _twin_axis(ax, style, "C3")
        ln, = tax.plot(s.x, s.y, marker="s", ls="none", ms=style.marker_size, color="C3")
        handles.append(ln); labels.append(s.label)
        tax.set_ylabel("μ (m²/V·s)", color="C3", fontsize=label_sz, **fam)

    if "j" in plotted:
        s = plotted["j"]
        oax = _offset_axis(ax, style, "C2")
        ln, = oax.plot(s.x, s.y, marker="^", ls="none", ms=style.marker_size, color="C2")
        handles.append(ln); labels.append(s.label)
        oax.set_ylabel("J (A/m²)", color="C2", fontsize=label_sz, **fam)

    ax.set_xscale(spec.xscale if spec.xscale is not None else kind.default_xscale)
    if spec.xmin is not None or spec.xmax is not None:
        ax.set_xlim(left=spec.xmin, right=spec.xmax)
    ax.set_xlabel("Temperature (K)", fontsize=label_sz, **fam)
    _draw_reference_lines(ax, spec)
    if style.tick_size is not None:
        ax.tick_params(labelsize=style.tick_size)
    if spec.title:
        title_sz = style.title_size if style.title_size is not None else style.font_pt
        ax.set_title(spec.title, fontsize=title_sz, **fam)
    _apply_robust_view(ax, spec, style)
    if tax is not None:
        _apply_robust_view(tax, spec, style)
    if oax is not None:
        _apply_robust_view(oax, spec, style)
    _apply_frame(ax, style, spec)
    # First real exercise of the third axis (KNOWN-ISSUES 21) showed _offset_axis's fixed
    # pos=1.18 colliding with the mu axis' realized tick+label extents at >=14 pt: measure
    # the first twin's tight bbox and push the J spine past it — placement by measurement,
    # the same rule as the legend/inset/label choosers. No-op at small fonts (max with the
    # shipped 1.18 keeps those figures byte-identical when the labels already fit).
    if tax is not None and oax is not None:
        # Convergence loop, not a one-shot: constrained_layout re-flows after every spine
        # move, shifting the pixel geometry the measurement was taken in. Three passes
        # settle on every case measured (9/14 pt); the loop exits early once the J spine
        # clears the mu label's realized right edge.
        for _ in range(15):
            fig.canvas.draw()
            rend = fig.canvas.get_renderer()
            mu_x1 = tax.yaxis.label.get_window_extent(rend).x1
            sp_x = oax.spines["right"].get_window_extent(rend).x0
            if sp_x >= mu_x1 + 4:
                break
            axbb = ax.get_window_extent(rend)
            # 2x overshoot on the measured shortfall: every spine move consumes right
            # margin, constrained_layout shrinks the axes and the mu label chases the
            # spine, so a same-geometry step undershoots in the re-flowed one (measured:
            # 3 exact steps left 23 px short; 8 at 1.5x still 4.5 px short at 14 pt).
            cur = float(oax.spines["right"].get_position()[1])
            shortfall_frac = (mu_x1 + 8 - sp_x) / max(axbb.width, 1e-9)
            oax.spines["right"].set_position(("axes", cur + 2.0 * shortfall_frac))
    _merged_legend(ax, handles, labels, style, spec)
    return fig


def _render_hall_rh_n_twin(results, kind_key, marker, spec=None, style=None, overlay=None):
    """R_H (left) + carrier n (twinx, log-y) vs T; port of 37_Hall.py:101-130. Shared by both
    the field-dep `hall_rh_n_twin` (marker 's') and temp-dep `hall_tdep_rh_n_twin` (marker 'o')
    kinds -- same series structure, different point provenance. Each axis carries exactly one
    series, so per-axis robust view is applied to both the R_H host axis and the carrier-n
    twin axis (the latter is log-scale, where `_apply_robust_view` is already a no-op) --
    real data showed a single pathological R_H point (e.g. -2.9e-4 against a ~1e-9 envelope)
    otherwise crushes the whole left-axis view."""
    if overlay is not None:
        results, kind, spec, style, fig, ax = _setup(results, kind_key, spec, style)
        _plot_data(ax, results, kind, spec, style, overlay)
        _finish(ax, kind, spec, style, "Temperature (K)", "R_H (m³/C)")
        return fig

    results, kind, spec, style, fig, ax = _setup(results, kind_key, spec, style)
    plotted = {s.key: s for r in results for s in select_series(kind.series(r, field_unit=style.field_unit), spec)}
    if not plotted:
        raise NothingToPlot(f"no series selected for kind {kind.key}")

    fam = {"fontfamily": style.font_family} if style.font_family else {}
    label_sz = style.label_size if style.label_size is not None else style.font_pt
    handles, labels = [], []

    if "rh" in plotted:
        s = plotted["rh"]
        ln, = ax.plot(s.x, s.y, marker=marker, ls="none", ms=style.marker_size, color="C0")
        handles.append(ln); labels.append(s.label)
        ax.set_ylabel("R_H (m³/C)", color="C0", fontsize=label_sz, **fam)
        ax.tick_params(axis="y", colors="C0")
        ax.spines["left"].set_color("C0")

    tax = None
    if "n" in plotted:
        s = plotted["n"]
        tax = _twin_axis(ax, style, "C3")
        tax.set_yscale("log")
        ln, = tax.plot(s.x, s.y, marker=marker, ls="--", lw=style.line_width,
                       ms=style.marker_size, color="C3")
        handles.append(ln); labels.append(s.label)
        tax.set_ylabel("n (1/m³)", color="C3", fontsize=label_sz, **fam)

    ax.set_xscale(spec.xscale if spec.xscale is not None else kind.default_xscale)
    if spec.xmin is not None or spec.xmax is not None:
        ax.set_xlim(left=spec.xmin, right=spec.xmax)
    ax.set_xlabel("Temperature (K)", fontsize=label_sz, **fam)
    _draw_reference_lines(ax, spec)
    if style.tick_size is not None:
        ax.tick_params(labelsize=style.tick_size)
    if spec.title:
        title_sz = style.title_size if style.title_size is not None else style.font_pt
        ax.set_title(spec.title, fontsize=title_sz, **fam)
    _apply_robust_view(ax, spec, style)
    if tax is not None:
        _apply_robust_view(tax, spec, style)     # log-scale -> no-op via _apply_robust_view's scale guard
    _apply_frame(ax, style, spec)
    _merged_legend(ax, handles, labels, style, spec)
    return fig

def render_hall_rh_n_twin(results, spec=None, style=None, overlay=None):
    return _render_hall_rh_n_twin(results, "hall_rh_n_twin", "s", spec, style, overlay)

def render_hall_tdep_rh_n_twin(results, spec=None, style=None, overlay=None):
    return _render_hall_rh_n_twin(results, "hall_tdep_rh_n_twin", "o", spec, style, overlay)

# ---- Hall TempDep renderers ----
def render_hall_tdep_rh_t(results, spec=None, style=None, overlay=None):
    results, kind, spec, style, fig, ax = _setup(results, "hall_tdep_RH_T", spec, style)
    _plot_data(ax, results, kind, spec, style, overlay)
    _finish(ax, kind, spec, style, "Temperature (K)", "R_H (m³/C)")
    return fig

def render_hall_tdep_n_t(results, spec=None, style=None, overlay=None):
    results, kind, spec, style, fig, ax = _setup(results, "hall_tdep_n_T", spec, style)
    _plot_data(ax, results, kind, spec, style, overlay)
    _finish(ax, kind, spec, style, "Temperature (K)", "n (1/m³)")
    return fig

def render_hall_tdep_mobility_t(results, spec=None, style=None, overlay=None):
    results, kind, spec, style, fig, ax = _setup(results, "hall_tdep_mobility_T", spec, style)
    _plot_data(ax, results, kind, spec, style, overlay)
    _finish(ax, kind, spec, style, "Temperature (K)", "μ (m²/V·s)")
    return fig

def render_hall_tdep_asym_vs_b(results, spec=None, style=None, overlay=None):
    results, kind, spec, style, fig, ax = _setup(results, "hall_tdep_asym_vs_B", spec, style)
    _plot_data(ax, results, kind, spec, style, overlay)
    _finish(ax, kind, spec, style, "Magnetic Field (T)", "R_asym (Ω)")
    return fig

def render_hall_tdep_interp_rt(results, spec=None, style=None, overlay=None):
    results, kind, spec, style, fig, ax = _setup(results, "hall_tdep_interp_RT", spec, style)
    _plot_data(ax, results, kind, spec, style, overlay)
    _finish(ax, kind, spec, style, "Temperature (K)", "R_xy (Ω)")
    return fig

_STAGE_PANELS = (("raw:", "Raw"), ("zsub:", "Zero-sub"), ("asym:", "Antisym."))

def render_hall_tdep_stages(results, spec=None, style=None, overlay=None):
    """PQ-2 Task 2: staged raw -> zero-subtracted -> antisymmetrized diagnostic panels
    (port of Step_2_Hall_fit_temp_dep.py's per-stage plots), one panel per stage that has
    selected data, side by side, sharing one colour-by-temperature map and one legend.
    Overlay mode (file comparison) keeps the original single-axes rendering -- panels are a
    single-file diagnostic view, not a comparison view."""
    if overlay is not None:
        results, kind, spec, style, fig, ax = _setup(results, "hall_tdep_stages", spec, style)
        _plot_data(ax, results, kind, spec, style, overlay)
        _finish(ax, kind, spec, style, "Magnetic Field (T)", "R (Ω)")
        return fig

    spec = spec or PlotSpec(); style = style or GlobalStyle()
    kind = _KIND["hall_tdep_stages"]
    results = _as_list(results)

    plotted = []
    for r in results:
        for s in select_series(kind.series(r, field_unit=style.field_unit), spec):
            plotted.append((r, s))
    if not plotted:
        raise NothingToPlot(f"no series selected for kind {kind.key}")

    by_prefix = {prefix: [(r, s) for r, s in plotted if s.key.startswith(prefix)]
                 for prefix, _title in _STAGE_PANELS}

    # Zero-subtracted panel is skipped when it would be visually identical to Raw for every
    # plotted temperature (current analyzer output: zero-field subtraction is an identity,
    # R_zero_sub == R_raw) -- avoids two redundant panels. Any T where Raw isn't also plotted
    # (or the values differ) counts as "differs", so the panel is shown.
    raw_by_group = {s.group: s for _, s in by_prefix["raw:"]}
    zsub_differs = any(
        s.group not in raw_by_group or list(raw_by_group[s.group].y) != list(s.y)
        for _, s in by_prefix["zsub:"]
    )

    panels = []
    for prefix, title in _STAGE_PANELS:
        items = by_prefix[prefix]
        if not items:
            continue
        if prefix == "zsub:" and not zsub_differs:
            continue
        panels.append((title, items))
    if not panels:
        raise NothingToPlot(f"no series selected for kind {kind.key}")

    # one colour per temperature, shared across every panel (first-appearance order)
    groups = []
    for _, items in panels:
        for _, s in items:
            if s.group not in groups:
                groups.append(s.group)
    gcolor = _group_color_map(groups, style)

    _, _, spec, style, fig, axes = _setup_panels(results, "hall_tdep_stages", spec, style, len(panels))

    fam = {"fontfamily": style.font_family} if style.font_family else {}
    label_sz = style.label_size if style.label_size is not None else style.font_pt
    title_sz = style.title_size if style.title_size is not None else style.font_pt

    for ax, (title, items) in zip(axes, panels):
        for _, s in items:
            kw = dict(marker=style.marker, ls="none", ms=style.marker_size,
                      color=gcolor[s.group], label="_nolegend_")
            if style.edge_color is not None:
                kw["markeredgecolor"] = style.edge_color
            if style.edge_width is not None:
                kw["markeredgewidth"] = style.edge_width
            ax.plot(s.x, s.y, **kw)
        ax.set_title(title, fontsize=title_sz, **fam)
        ax.set_xscale(spec.xscale if spec.xscale is not None else kind.default_xscale)
        ax.set_yscale(spec.yscale if spec.yscale is not None else kind.default_yscale)
        if spec.xmin is not None or spec.xmax is not None:
            ax.set_xlim(left=spec.xmin, right=spec.xmax)
        if spec.ymin is not None or spec.ymax is not None:
            ax.set_ylim(bottom=spec.ymin, top=spec.ymax)
        ax.set_xlabel("|B| (T)", fontsize=label_sz, **fam)
        _draw_reference_lines(ax, spec)
        if style.tick_size is not None:
            ax.tick_params(labelsize=style.tick_size)
        _apply_robust_view(ax, spec, style)
        _apply_frame(ax, style, spec)

    axes[0].set_ylabel("R (Ω)", fontsize=label_sz, **fam)  # leftmost panel only

    handles = [Line2D([], [], ls="none", marker="o", color=gcolor[g], label=g) for g in groups]
    legend_sz = style.legend_size if style.legend_size is not None else style.font_pt - 1
    legend_prop = {"size": legend_sz}
    if style.font_family:
        legend_prop["family"] = style.font_family
    _draw_legend(axes[-1], legend_prop, style, spec, handles)   # one legend for the whole figure

    return fig

def render_hall_tdep_j_t(results, spec=None, style=None, overlay=None):
    results, kind, spec, style, fig, ax = _setup(results, "hall_tdep_J_T", spec, style)
    _plot_data(ax, results, kind, spec, style, overlay)
    _finish(ax, kind, spec, style, "Temperature (K)", "J (A/m²)")
    return fig

def _acms_axis_view(ax, spec, style):
    """Bimodal-safe y-view for AC-susceptibility panels, applied AFTER _finish (overrides the
    shared robust view for these axes only). A SC χ′(T) curve is two-level — diamagnetic low-T
    level + normal-state plateau — and `_apply_robust_view`'s per-line median±k·MAD envelope
    treats whichever level holds fewer points as a heavy tail, clipping it off the axis (visual
    gate: SC synth plateau cut at the panel edge). Instead take the union of per-line 1–99%
    quantile envelopes: both physical levels of a bimodal curve sit inside it (each holds ≫1%
    of the points), while a lone measurement glitch (≪1% of points) still cannot blow up the
    view. Same bypass guards as `_apply_robust_view`: explicit spec limits, log-y, or
    robust_view off -> no-op (matplotlib autoscale then shows everything)."""
    use = spec.robust_view if spec.robust_view is not None else style.robust_view
    if not use or ax.get_yscale() != "linear" or spec.ymin is not None or spec.ymax is not None:
        return
    los, his = [], []
    for ln in ax.lines:
        if ln.get_gid() in NON_DATA_GIDS:
            continue
        a = np.asarray(ln.get_ydata(), float)
        a = a[np.isfinite(a)]
        if not a.size:
            continue
        lo_i, hi_i = (np.quantile(a, [0.01, 0.99]) if a.size >= 8
                      else (float(a.min()), float(a.max())))
        los.append(float(lo_i)); his.append(float(hi_i))
    if not los:
        return
    lo, hi = min(los), max(his)
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return
    pad = _ROBUST_PAD * (hi - lo)
    ax.set_ylim(bottom=lo - pad, top=hi + pad)


def _acms_units(result):
    d = result.data or {}
    molar = any((c.get("chi_prime_molar")) for c in (d.get("curves") or []))
    return ("χ′ (emu·mol⁻¹·Oe⁻¹)", "χ″ (emu·mol⁻¹·Oe⁻¹)") if molar else \
           ("χ′ (emu/Oe)", "χ″ (emu/Oe)")


def render_acms_chi_t(results, spec=None, style=None, overlay=None):
    """Headline: vertically-stacked shared-T panels — χ′(T) top, χ″(T) bottom. Net-new layout
    (fig.subplots(2,1,sharex=True)); existing panelled renderers are side-by-side. Curves of one
    (freq,amp,field) group share a colour; ramp direction encodes the marker (up ○ / down △).
    A Tc/T_f marker (from the aggregated sc_transition) is drawn on the χ′ panel — dotted +
    weaker when low-confidence, per PQ-4."""
    spec = spec or PlotSpec(); style = style or GlobalStyle()
    results = _as_list(results); kind = _KIND["acms_chi_t"]
    fig = _new_fig(style)
    ax_top, ax_bot = fig.subplots(2, 1, sharex=True)
    plotted = [(r, s) for r in results
               for s in select_series(kind.series(r, field_unit=style.field_unit), spec)]
    if not plotted:
        raise NothingToPlot(f"no series selected for kind {kind.key}")
    top = [(r, s) for r, s in plotted if s.key.startswith("chip:")]
    bot = [(r, s) for r, s in plotted if s.key.startswith("chipp:")]
    groups = []
    for _, s in plotted:
        if s.group not in groups:
            groups.append(s.group)
    gcolor = _group_color_map(groups, style)

    def _draw(ax, items):
        for _, s in items:
            kw = dict(marker=s.marker or style.marker, ls="none", ms=style.marker_size,
                      color=gcolor[s.group], label="_nolegend_")
            if style.edge_color is not None:
                kw["markeredgecolor"] = style.edge_color
            if style.edge_width is not None:
                kw["markeredgewidth"] = style.edge_width
            ax.plot(s.x, s.y, **kw)

    _draw(ax_top, top); _draw(ax_bot, bot)
    # Folded legend: one proxy per (freq,amp,field) group (colour), so up/down ramps of one
    # group never mint two identical entries; a direction key (○ up / △ down) is appended only
    # when both directions are actually present, since marker encodes ramp direction.
    handles = [Line2D([], [], ls="none", marker="o", color=gcolor[g], label=g) for g in groups]
    markers_seen = {s.marker for _, s in plotted if s.marker}
    dir_names = {"o": "↑ warming", "^": "↓ cooling", "s": "mixed"}
    if len(markers_seen) > 1:
        handles += [Line2D([], [], ls="none", marker=m, color="0.35", label=dir_names.get(m, m))
                    for m in ("o", "^", "s") if m in markers_seen]
    yl_top, yl_bot = _acms_units(results[0])
    # scientific offset (chi ~1e-12) must coexist with any thousands-sep (PQ-1 build note).
    for ax in (ax_top, ax_bot):
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useOffset=True)
    _finish(ax_top, kind, spec, style, "", yl_top, draw_legend=False)
    _finish(ax_bot, kind, spec, style, "Temperature (K)", yl_bot, draw_legend=False)
    # Bimodal-safe y-view (overrides the shared robust view _finish applied): a SC χ′ curve's
    # two levels must BOTH stay inside the panel — see _acms_axis_view.
    _acms_axis_view(ax_top, spec, style)
    _acms_axis_view(ax_bot, spec, style)
    # Tc / T_f marker from the aggregated result (dotted + weaker when low-confidence, per PQ-4).
    sc = (results[0].data or {}).get("sc_transition")
    if sc and sc.get("tc_mid_k") is not None:
        low = sc.get("low_confidence")
        ls = ":" if low else "--"
        ax_top.axvline(sc["tc_mid_k"], color="0.35", linestyle=ls, lw=style.line_width,
                       gid="refline")
        if low:                                      # doubt visible on the figure, not only in
            ax_top.text(sc["tc_mid_k"], 0.02, " (low confidence)",   # the data/CSV (PQ-4 conv.)
                        transform=ax_top.get_xaxis_transform(), rotation=90, va="bottom",
                        ha="left", fontsize=style.font_pt - 2, color="0.35")
    # Legend: an AC file's data fan often fills the χ′ panel edge-to-edge, in which case the
    # occupancy chooser relocates outside-right exactly as the old unconditional rule did —
    # but it now MEASURES that (KNOWN-ISSUES 5) instead of assuming it, so a file with a clear
    # inside spot keeps its canvas width. Explicit legend_loc wins as before.
    legend_sz = style.legend_size if style.legend_size is not None else style.font_pt - 1
    legend_prop = {"size": legend_sz}
    if style.font_family:
        legend_prop["family"] = style.font_family
    _draw_legend(ax_top, legend_prop, style, spec, handles)
    return fig


def _acms_single(results, kind_key, ylabel, spec, style, overlay):
    """Single-panel AC-susceptibility renderer (χ′, χ″, or M-DC vs T). Kinds stay
    group_colored=True and route through _plot_data_grouped (precedent: other group_colored
    renderers) so the legend folds to one proxy per (freq,amp,field) group + a per-direction
    marker key — the plain _plot_data path would emit duplicate group labels for up/down ramps.
    Scientific offset (χ ~1e-12) is set before _finish; it coexists with the PQ-1 thousands-sep
    formatter, which only force-sets a comma FuncFormatter when max|value| >= 1000 (never true
    for these small-magnitude axes), so useOffset survives."""
    results, kind, spec, style, fig, ax = _setup(results, kind_key, spec, style)
    if overlay is None and kind.group_colored:
        plotted, handles = _plot_data_grouped(ax, results, kind, spec, style)
        # Lone-direction key suppression (same rule as the acms_chi_t headline): the direction
        # role key only disambiguates when BOTH ramp directions are present; on a single-
        # direction file a lone 'up' legend entry is noise, so drop role proxies then.
        roles = {s.role for _, s in plotted if s.role is not None}
        if len(roles) <= 1:
            handles = [h for h in handles if h.get_label() not in roles]
    else:
        handles = None
        _plot_data(ax, results, kind, spec, style, overlay)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useOffset=True)
    _finish(ax, kind, spec, style, "Temperature (K)", ylabel, legend_handles=handles)
    _acms_axis_view(ax, spec, style)
    return fig


def render_acms_chi_prime_t(results, spec=None, style=None, overlay=None):
    yl = _acms_units(_as_list(results)[0])[0]
    return _acms_single(results, "acms_chi_prime_t", yl, spec, style, overlay)


def render_acms_chi_dprime_t(results, spec=None, style=None, overlay=None):
    results = _as_list(results)
    yl = _acms_units(results[0])[1]
    fig = _acms_single(results, "acms_chi_dprime_t", yl, spec, style, overlay)
    spec = spec or PlotSpec(); style = style or GlobalStyle()
    # Per-curve T_f markers: one vertical line per curve with a detected χ″ peak, in ITS
    # group's colour (matches the point colour beside it); dotted + reduced alpha when that
    # peak is low-confidence (Tc-marker convention). spec.tc_marker switches them all off.
    # Non-overlay only — overlay recolours by file, so group colours would not match.
    if overlay is not None or not spec.tc_marker:
        return fig
    ax = fig.axes[0]
    kind = _KIND["acms_chi_dprime_t"]
    plotted = [s for r in results
               for s in select_series(kind.series(r, field_unit=style.field_unit), spec)]
    groups = []
    for s in plotted:
        if s.group not in groups:
            groups.append(s.group)
    gcolor = _group_color_map(groups, style)      # same map _plot_data_grouped built
    for c in (results[0].data or {}).get("curves") or []:
        p = c.get("peak")
        if not p or p.get("t_f_k") is None:
            continue
        col = gcolor.get(_acms_label(c))
        if col is None:                            # curve deselected via spec.curves
            continue
        low = p.get("low_confidence")
        ax.axvline(p["t_f_k"], color=col, linestyle=(":" if low else "--"),
                   alpha=(0.55 if low else 1.0), lw=style.line_width, gid="refline")
        if low:                                      # doubt visible on the figure (PQ-4 conv.)
            ax.text(p["t_f_k"], 0.02, " (low confidence)",
                    transform=ax.get_xaxis_transform(), rotation=90, va="bottom", ha="left",
                    fontsize=style.font_pt - 2, color=col, alpha=0.6)
    return fig


def render_acms_mdc_t(results, spec=None, style=None, overlay=None):
    return _acms_single(results, "acms_mdc_t", "M-DC (emu)", spec, style, overlay)


# ---- TTO (thermal transport) renderers ----
_TTO_DIR_NAMES = {"o": "↑ warming", "^": "↓ cooling", "s": "mixed"}


def _tto_handles(plotted, spec, style, gcolor=None):
    """(group->colour map, folded legend handles). One proxy per colour group, plus a
    direction key (marker encodes ramp direction) only when more than one direction is
    actually present — a lone 'cooling' marker key would be noise.

    The group proxy carries the marker THAT GROUP IS ACTUALLY DRAWN WITH (same expression as
    `_tto_draw`), not a hard-coded 'o': the real gate file is a cooling ramp drawn with '^',
    so a fixed 'o' key matched nothing on the canvas.
    `gcolor` overrides the group->colour map (overlay mode colours by FILE)."""
    groups, gmarker = [], {}
    for _, s in plotted:
        if s.group not in groups:
            groups.append(s.group)
            gmarker[s.group] = s.marker if (s.marker and spec.channel_markers) else style.marker
    gcolor = gcolor or _group_color_map(groups, style)
    handles = [Line2D([], [], ls="none", marker=gmarker[g], color=gcolor[g], label=g)
               for g in groups]
    markers_seen = {s.marker for _, s in plotted if s.marker}
    if len(markers_seen) > 1:
        handles += [Line2D([], [], ls="none", marker=m, color="0.35",
                           label=_TTO_DIR_NAMES.get(m, m))
                    for m in ("o", "^", "s") if m in markers_seen]
    handles += _tto_field_handles(plotted, style)
    return gcolor, handles


def _tto_field_handles(plotted, style):
    """Grey linestyle proxies naming the FIELD, added only when the linestyle actually encodes
    it — i.e. `tto_field_ls_map` fired (multi-field `tto_wf_t`), which shows up here as a 1:1
    linestyle<->field mapping over the plotted series. Single-field wf maps three linestyles
    onto ONE field, and the other three kinds carry no linestyle at all, so both stay
    unchanged. Field is read back from the pinned series key `<prefix>:<field %g>:<dir>`."""
    ls_field = {}
    for _, s in plotted:
        if not s.linestyle:
            continue
        try:
            f = float(s.key.split(":")[1])
        except (IndexError, ValueError):
            return []
        ls_field.setdefault(s.linestyle, set()).add(f)
    if len(ls_field) < 2 or any(len(v) != 1 for v in ls_field.values()):
        return []
    pairs = [(ls, next(iter(v))) for ls, v in ls_field.items()]
    if len({f for _, f in pairs} ) != len(pairs):
        return []
    return [Line2D([], [], ls=ls, color="0.35",
                   label=tto_field_ls_label(f, style.field_unit))
            for ls, f in sorted(pairs, key=lambda p: p[1])]


# Bands are an explicit allow-list because `_tto_draw` is SHARED (I2): without it `tto_wf_t`
# would get a band on its kappa component -- built by the same `_tto_curve_series` with a
# `kappa_std` array available -- and none on kappa_e/kappa_ph, silently meaning "only one of
# these three curves has an error estimate". The derived quantities have no propagated sigma;
# that work is deferred.
_TTO_BAND_KINDS = {"tto_kappa_t", "tto_seebeck_t", "tto_zt_t", "tto_summary_t"}
# Kinds carrying the opt-in fitted-window shade (PlotSpec.fit_window_shade, default OFF)
_SHADE_KINDS = {"cp_over_t", "hc_c_over_t_linear", "resistivity_rho_t"}
_TTO_BAND_ALPHA = 0.20
_TTO_BAND_ZORDER = 1.5      # under the lines' default zorder 2 -> paint order is immaterial


_TTO_BAND_MARKER_SCALE = 0.55   # marker shrink applied ONLY while a band is drawn (I3)


def _tto_marker_size(kind, spec, style):
    """Marker size for a TTO series, shrunk while the error band is on (final review I3).

    The band was measurably invisible: rendering each kind twice at IDENTICAL y-limits and
    diffing pixel-by-pixel, turning the band on changed 14 px on `tto_seebeck_t` (0.014 % of
    the figure), 297 on `tto_zt_t`, 747 on `tto_kappa_t` and 1255 on `tto_summary_t`. The
    cause is footprint, not opacity: 976 FILLED markers at the default size cover essentially
    the whole ribbon wherever it is narrow, and they are the same colour as the band.

    The lever is the marker, NOT alpha/zorder: `_TTO_BAND_ALPHA == 0.20` and
    `_TTO_BAND_ZORDER == 1.5` are a pinned artist contract, and raising either would paint the
    ribbon OVER the data it is supposed to qualify.

    Applies only when a band is actually drawn (`spec.error_band` AND a band-carrying kind), so
    every default render -- and every non-band kind at any setting -- is byte-identical. On
    `tto_seebeck_t` the band is honestly sub-pixel (sigma_S is ~0.1 % of S) and nothing here
    can or should change that."""
    ms = style.marker_size
    if spec.error_band and kind.key in _TTO_BAND_KINDS:
        return max(ms * _TTO_BAND_MARKER_SCALE, 1.5)
    return ms


def _tto_draw_bands(ax, plotted, kind, spec, style, gcolor, yscale=1.0):
    """±1σ shaded bands for TTO series, opt-in via `spec.error_band` (E2, default OFF).

    `fill_between`, not `errorbar` (E6): 976 dense points would make per-point caps an
    unreadable hairball. Returns the finite (lo, hi) y-extent of every band drawn on `ax`, or
    None -- the y-view helpers read `ax.lines` only, so without this the band is cropped
    exactly where the uncertainty is largest (I5).

    Factored out of `_tto_draw` so `render_tto_summary_t` can DEFER the rho panel's band until
    after `_rho_axis_autoscale` has applied its engineering-prefix factor (C2)."""
    if not spec.error_band or kind.key not in _TTO_BAND_KINDS:
        return None
    connect = (spec.connect_lines if spec.connect_lines is not None else style.connect_lines) \
        and kind.key in _CONNECT_KINDS
    lo = hi = None
    for _, s in plotted:
        if getattr(s, "yerr", None) is None:
            continue
        x = np.asarray(s.x, float)
        y = np.asarray(s.y, float) * yscale
        # The `* yscale` is load-bearing and COVERED: the only caller with yscale != 1 is the
        # summary rho panel (yscale = 100.0 * the engineering-prefix factor, C2), where an
        # unscaled sigma is off by up to 1e8. Pinned by the ratio assertions in
        # test_rho_panel_band_brackets_the_rho_line_on_the_RENDERED_axis and by
        # test_draw_bands_scales_sigma_by_yscale (which drives this helper directly).
        e = np.asarray(s.yerr, float) * yscale
        # Zero a bad sigma BEFORE _connect_sort (M2): that helper INSERTS NaN separators
        # between finite x segments, and a NaN->0 substitution applied afterwards would turn
        # those separators into real zero-width band points, bridging exactly the gaps the
        # line deliberately breaks. Zeroing (not dropping) shrinks the band locally instead
        # of blanking the whole polygon.
        e = np.where(np.isfinite(e), e, 0.0)
        if connect and x.size:
            x_raw = x
            x, y = _connect_sort(x_raw, y)
            # `_connect_sort` returns sorted ARRAYS, not a permutation, so the aligned band
            # comes from a second call on the SAME pre-sort x. Do not re-implement it, and do
            # not modify it -- every existing kind depends on its byte-identical behaviour.
            _, e = _connect_sort(x_raw, e)
        ax.fill_between(x, y - e, y + e, color=gcolor[s.group], alpha=_TTO_BAND_ALPHA,
                        linewidth=0, zorder=_TTO_BAND_ZORDER, gid="errband")
        edge = np.concatenate([y - e, y + e])
        edge = edge[np.isfinite(edge)]
        if edge.size:
            lo = float(edge.min()) if lo is None else min(lo, float(edge.min()))
            hi = float(edge.max()) if hi is None else max(hi, float(edge.max()))
    if lo is None:
        return None
    prev = getattr(ax, "_tto_band_extent", None)
    ax._tto_band_extent = ((lo, hi) if prev is None
                           else (min(prev[0], lo), max(prev[1], hi)))
    return lo, hi


def _tto_expand_ylim_for_bands(ax, spec, style):
    """Grow this axis's ylim to contain any recorded band extent (I5). EXPANSION ONLY, never
    contraction; skipped entirely when explicit spec limits are set (the same bypass
    `_apply_robust_view` uses). No band -> no recorded extent -> no call effect, so an
    `error_band=False` render is byte-identical. Idempotent, so it is safe to call again after
    a second view setter (`render_tto_zt_t` does, because `_tto_full_view` reads lines only).

    `style` is unused and kept deliberately: the signature mirrors `_apply_robust_view` /
    `_tto_full_view` and is a documented interface in this plan. Do not "clean it up".

    THIS BODY IS LIVE, NOT DEFENSIVE. It is inert only at the DEFAULT `robust_k=8` on the
    current fixtures, where autoscale (which does see the fill_between collection) already
    covers the ribbon. It BITES as soon as `_apply_robust_view` does: that helper runs INSIDE
    `_finish`, after the band is drawn, and it reads `ax.lines` ONLY — so it re-clips the view
    to a line-derived envelope that knows nothing about the band. `robust_k` is a user-settable
    `GlobalStyle` knob exposed in the GUI, so this is a reachable configuration, not a
    hypothetical. Measured on tto_real_subset.dat / tto_summary_t / rho panel at
    `GlobalStyle(robust_k=1.5)`: with this body live the ylim is (220.514, 374.039) — exactly
    the band extent; with it stubbed the ylim is (246.602, 365.656) while the band still spans
    (220.514, 374.039), i.e. CLIPPED AT BOTH ENDS, including the low-T point where rho carries
    its largest uncertainty. Pinned by test_rho_band_survives_a_biting_robust_view (real render
    path, robust_k=1.5) and, on a constructed extent, by
    test_expand_ylim_grows_the_view_for_a_band_wider_than_the_data (plus the three bypasses)."""
    extent = getattr(ax, "_tto_band_extent", None)
    # The non-linear-y bypass is NOT optional (I2): `y - e` is by construction below `y`, and
    # kappa_ph/S/ZT cross or approach zero, so on a user-selected log y (`PlotSpec.yscale`,
    # exposed in the GUI's AxisStrip) `set_ylim(bottom<=0)` is silently IGNORED with a
    # UserWarning and the view is left half-expanded. Same bypasses `_apply_robust_view`
    # (render.py:369-372) and `_tto_full_view` (render.py:2740-2742) already carry.
    if (extent is None or ax.get_yscale() != "linear"
            or spec.ymin is not None or spec.ymax is not None):
        return
    lo, hi = extent
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom=min(bottom, lo), top=max(top, hi))


def _tto_draw(ax, plotted, kind, spec, style, gcolor, yscale=1.0, draw_bands=True):
    """Draw TTO series with group colours, direction markers and (default-on) connecting
    lines. Bespoke on purpose: `_plot_data` colours by series index, so its colours would not
    match the folded per-group legend, and `_plot_data_grouped` never connects — a 976-point
    smooth curve is unreadable as scatter. `yscale` lets the stacked headline convert the rho
    panel from Ohm*m to Ohm*cm before `_rho_axis_autoscale` picks an engineering prefix.

    `draw_bands=False` DEFERS the band to the caller — used by `render_tto_summary_t` for the
    rho panel only, whose band must be drawn after `_rho_axis_autoscale`'s factor (C2).
    Returns the band's (lo, hi) y-extent, or None."""
    extent = _tto_draw_bands(ax, plotted, kind, spec, style, gcolor, yscale) if draw_bands \
        else None
    connect = (spec.connect_lines if spec.connect_lines is not None else style.connect_lines) \
        and kind.key in _CONNECT_KINDS
    for _, s in plotted:
        x = np.asarray(s.x, float)
        y = np.asarray(s.y, float) * yscale
        if connect and x.size:
            x, y = _connect_sort(x, y)
        marker = s.marker if (s.marker and spec.channel_markers) else style.marker
        kw = dict(marker=marker, ls="none", ms=_tto_marker_size(kind, spec, style),
                  color=gcolor[s.group], label="_nolegend_")
        if connect:
            kw["ls"] = (s.linestyle or "-")
            kw["lw"] = style.line_width
        if style.edge_color is not None:
            kw["markeredgecolor"] = style.edge_color
        if style.edge_width is not None:
            kw["markeredgewidth"] = style.edge_width
        ax.plot(x, y, **kw)
    return extent


def _tto_legend(ax, handles, spec, style):
    """Folded legend, always drawn (the gate file is a single curve, so a '>1 series'
    condition would leave the figure with no field/direction annotation at all).

    Placement is the occupancy chooser's (KNOWN-ISSUES 5): the old unconditional
    outside-right relocation spent ~20-25% of canvas width on a two-entry legend even when
    the upper-right quadrant was empty. A file whose data fan really fills the panel still
    relocates — the chooser measures it instead of assuming it. Explicit legend_loc wins."""
    legend_sz = style.legend_size if style.legend_size is not None else style.font_pt - 1
    legend_prop = {"size": legend_sz}
    if style.font_family:
        legend_prop["family"] = style.font_family
    _draw_legend(ax, legend_prop, style, spec, handles)


def _tto_lowt_inset(ax, results, spec, style):
    """Low-T kappa inset: [0, 30] K, 42%/40% lower-right, data only (no fit line), per the
    UO2 reference figure. This is a RE-IMPLEMENTATION of the `_rho_lowt_inset` recipe, not a
    call into it — that helper reads resistivity-shaped data (d['bridges'][i]['rho_t_curves']).
    Suppressed when spec.lowt_inset is False, the data starts above 30 K, or fewer than 5
    points fall inside the window (same guards as _rho_lowt_inset)."""
    if not spec.lowt_inset:
        return None
    curves = (results[0].data or {}).get("curves") or []
    if not curves:
        return None
    curve = max(curves, key=lambda c: len(c.get("t") or []))
    T = np.asarray(curve.get("t") or [], float)
    K = np.asarray(curve.get("kappa") or [], float)
    m = np.isfinite(T) & np.isfinite(K)
    T, K = T[m], K[m]
    if T.size == 0 or float(T.min()) > 30.0:
        return None
    w = (T >= 0) & (T <= 30.0)
    if int(np.count_nonzero(w)) < 5:
        return None
    iax = _lowt_inset_axes(ax, spec, style)    # measured corner, or dropped-with-note (None)
    if iax is None:
        return None
    # The inset's markers shrink with the band for the same reason the host's do (I3): at
    # 976 points they are dense enough to bury the ribbon entirely (looked at: an unshrunk
    # inset is a solid black worm). error_band off -> unchanged, so gallery renders are
    # byte-identical.
    ims = max(style.marker_size - 1.0, 2.0)
    if spec.error_band:
        ims = max(ims * _TTO_BAND_MARKER_SCALE, 1.0)
    iax.plot(T[w], K[w], marker="s", markerfacecolor="none", ls="none",
             ms=ims, color="0.2",
             markeredgewidth=(style.edge_width if style.edge_width is not None else 0.8))
    # I1 (final review): the inset MUST carry the band too. It magnifies exactly the region
    # where kappa's relative sigma is largest (measured on the gate file: max 8.60 % at
    # 4.033 K, median 1.20 % over this window, against ~0.1 % at room T), so an unbanded inset
    # beside a banded main axis shows the low-T data as the EXACT data -- the same "only one of
    # these curves has an error estimate" defect the _TTO_BAND_KINDS allow-list exists to
    # prevent, applied to two panels of one figure.
    #
    # Drawn HERE rather than by `_tto_draw_bands(iax, plotted, ...)`: the inset plots ONE
    # curve (the longest), not the `plotted` series list, so feeding it every series would
    # band curves whose markers are not in the inset at all. Same alpha/zorder/gid contract.
    #
    # No `_tto_expand_ylim_for_bands` here either, and that is NOT an oversight: the inset has
    # no view override (`_apply_robust_view`/`_tto_full_view` run on the host axes only), so
    # its limits come from matplotlib's own autoscale, which DOES see the fill_between
    # collection. Calling the expander would `set_ylim` from a get_ylim() that has not been
    # autoscaled yet -- freezing the inset at the default (0, 1).
    std = curve.get("kappa_std") or []
    if spec.error_band and len(std) == len(curve.get("t") or []):
        S = np.asarray([np.nan if v is None else v for v in std], float)[m][w]
        S = np.where(np.isfinite(S), S, 0.0)
        iax.fill_between(T[w], K[w] - S, K[w] + S, color="0.2", alpha=_TTO_BAND_ALPHA,
                         linewidth=0, zorder=_TTO_BAND_ZORDER, gid="errband")
    # Inset text scales WITH the figure font instead of being pinned at 5 pt (the literal
    # `_rho_lowt_inset` carries): at the GUI's font_pt=14 a fixed 5 pt inset is a near-
    # illegible thumbnail. `max(5, 0.55*label)` reproduces exactly 5 pt at the 9 pt gallery
    # default (0.55*9 = 4.95 -> floored to 5), so gallery renders stay byte-identical.
    insz = max(5.0, 0.55 * (style.label_size if style.label_size is not None else style.font_pt))
    iax.tick_params(labelsize=insz, length=2, width=0.5)
    # No x-label: the inset sits at lower-right and an x-label drops into the host's tick band
    # (same craft note as _rho_lowt_inset). The host axis already reads "Temperature (K)".
    iax.set_ylabel("κ (W K⁻¹ m⁻¹)", fontsize=insz, labelpad=1)
    for sp in iax.spines.values():
        sp.set_linewidth(0.5)
    return iax


def _tto_full_view(ax, spec, style):
    """Full-extent y-view, applied AFTER `_finish` so it overrides `_apply_robust_view` for
    this axis only. Used by `tto_zt_t` alone.

    WHY: ZT is heavy-tailed BY PHYSICS (ZT = S²T/(ρκ) and S crosses zero, so the real gate
    file spans 2.2166e-10 .. 3.9232e-4 over 976 points) and the PEAK is the headline number.
    `_apply_robust_view`'s per-line median±8·MAD envelope tops out at 3.3457e-5; the tail test
    `dmax - hi > 0.05*(dmax-dmin)` clears, so it fires and sets top ≈ 3.5130e-5 — 11x below
    the peak (all measured on the real file). `_acms_axis_view` is NOT an alternative: its
    1-99% envelope tops out at q99 = 2.755e-4 + 5% pad ≈ 2.89e-4, still below the peak.

    Same bypass guards as `_apply_robust_view`, so this stays a targeted correction rather
    than a clamp: explicit spec limits, non-linear y, or robust_view switched off -> no-op
    (matplotlib autoscale already shows everything in that last case).

    ADDITIVE: called from `render_tto_zt_t` only. No existing kind reaches it."""
    use = spec.robust_view if spec.robust_view is not None else style.robust_view
    if not use or ax.get_yscale() != "linear" or spec.ymin is not None or spec.ymax is not None:
        return
    vals = [np.asarray(ln.get_ydata(), float) for ln in ax.lines
            if ln.get_gid() not in NON_DATA_GIDS]
    a = np.concatenate(vals) if vals else np.array([])
    a = a[np.isfinite(a)]
    if a.size == 0:
        return
    lo, hi = float(a.min()), float(a.max())
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return
    pad = 0.05 * (hi - lo)
    ax.set_ylim(bottom=lo - pad, top=hi + pad)


_TTO_KIND_CAP = {"tto_kappa_t": "thermal_conductivity", "tto_seebeck_t": "seebeck",
                 "tto_zt_t": "figure_of_merit", "tto_wf_t": "wiedemann_franz",
                 # L/L0 = kappa*rho/(L0*T) needs rho, so a rho-less file reuses the
                 # wiedemann_franz decline reason ("requires finite ρ > 0") verbatim rather
                 # than the generic "no curves selected".
                 "tto_lorenz_t": "wiedemann_franz"}


def _tto_empty_note(ax, results, kind_key, style):
    """Centred note on an empty TTO figure, so it reads as 'this file has no such data'
    rather than a broken plot (blank 0-1 axes were what the reviewer saw for tto_seebeck_t on
    the gap file and tto_wf_t on the no-rho file). The text is the ANALYZER'S OWN capability
    `reason` — no new copy invented here; when the capability IS applicable the emptiness came
    from the curve selection instead, so say that. Only ever drawn when nothing was plotted."""
    reason = None
    caps = ((results[0].data or {}).get("capabilities") or []) if results else []
    want = _TTO_KIND_CAP.get(kind_key)
    for c in caps:
        if c.get("name") == want and not c.get("applicable"):
            reason = c.get("reason") or None
            break
    ax.text(0.5, 0.5, reason or "no curves selected", transform=ax.transAxes,
            ha="center", va="center", color="0.35", gid=_TTO_NOTE_GID, clip_on=True,
            fontsize=(style.label_size if style.label_size is not None else style.font_pt),
            **({"fontfamily": style.font_family} if style.font_family else {}))


_TTO_NOTE_GID = "tto_note"
_TTO_NOTE_MIN_PT = 6.0          # floor: below this the explanation stops being readable at all
_TTO_NOTE_FIT_FRAC = 0.92       # keep a margin inside the axes box
_TTO_NOTE_FIT_PASSES = 4


def _fit_tto_notes(fig, floor=_TTO_NOTE_MIN_PT, frac=_TTO_NOTE_FIT_FRAC,
                   passes=_TTO_NOTE_FIT_PASSES):
    """Cap each empty-panel note's font so the CENTRED text fits inside its own axes width.

    Same remedy (and same shape) as `_fit_ylabels_to_axes`, which caps a ylabel to its axes
    HEIGHT. Without it the note is drawn at the full label size with no width fitting, so at a
    GUI card width of ≤60 mm and font_pt=14 the sentence runs past both spines and prints
    straight through the rotated y-axis label — an illegible garble on the very figures the
    note exists to make legible (measured on `tto_norho_synth.dat`: inside the axes at 90/70
    mm, outside and overprinting the ylabel at 60/50 mm). `clip_on=True` on the Text is the
    backstop for the floor case. No-op when the note already fits, so the 9 pt gallery renders
    are untouched. MUST be called after `_finish` — the axes box is only final once the
    layout engine has run with the real labels in place."""
    notes = [t for ax in fig.axes for t in ax.texts if t.get_gid() == _TTO_NOTE_GID]
    if not notes:
        return
    for _ in range(passes):
        fig.draw_without_rendering()        # run the layout engine: extents must be final
        rend = fig.canvas.get_renderer()
        changed = False
        for t in notes:
            ax_w = t.axes.get_window_extent(rend).width * frac
            t_w = t.get_window_extent(rend).width
            size = t.get_fontsize()
            if ax_w > 0 and t_w > ax_w and size > floor:
                t.set_fontsize(max(floor, size * ax_w / t_w))
                changed = True
        if not changed:
            break


def _tto_blank_yaxis(ax):
    """Strip the y ticks of an empty (note-only) panel: matplotlib's default 0-1 ticks under a
    real unit label read as a populated axis whose points are hiding somewhere (visual gate).
    MUST be called AFTER `_finish` — its `ax.set_yscale(...)` reinstalls the scale's default
    locator and would undo this."""
    ax.set_yticks([])


def _tto_single(results, kind_key, ylabel, spec, style, overlay):
    """Single-panel TTO renderer (kappa, S, ZT or the kappa decomposition vs T).

    EMPTY-SERIES TOLERANCE (spec §6): unlike `_plot_data`, an empty selection does NOT raise
    here — it produces an empty-axes figure. Required so `tto_seebeck_t` still renders on a
    file whose Seebeck column is absent (the gap fixture) instead of blowing up."""
    results, kind, spec, style, fig, ax = _setup(results, kind_key, spec, style)
    empty = False
    if overlay is not None:
        handles = None
        # The tolerance has to hold in OVERLAY mode too: `_plot_data`'s overlay branch raises
        # on an empty selection, so "Add to compare…" on a Seebeck-less file used to blow up
        # exactly where the single-file view renders an explanatory figure. The note reads the
        # FIRST result's capabilities (the same convention as the non-overlay path).
        try:
            _plot_data(ax, results, kind, spec, style, overlay)
        except ValueError:
            empty = True
            _tto_empty_note(ax, results, kind_key, style)
    else:
        plotted = [(r, s) for r in results
                   for s in select_series(kind.series(r, field_unit=style.field_unit), spec)]
        if plotted:
            gcolor, handles = _tto_handles(plotted, spec, style)
            _tto_draw(ax, plotted, kind, spec, style, gcolor)
        else:
            handles = None
            empty = True
            _tto_empty_note(ax, results, kind_key, style)
    # Overlay mode: `_plot_data` already labelled each line by file, so let `_finish` draw the
    # standard per-file legend — the `_acms_single` convention (render.py:2523 passes
    # legend_handles=None and leaves draw_legend at its default True, so `_draw_legend` falls
    # back to ax.get_legend_handles_labels()). Only the bespoke FOLDED per-group legend is
    # placed by hand, below. Hard-coding draw_legend=False here would render N unlabelled
    # curves in "Add to compare…" mode.
    _finish(ax, kind, spec, style, "Temperature (K)", ylabel,
            draw_legend=(overlay is not None and not empty))
    # ZT runs ~1e-4, which plain ticks render as "0.0000 … 0.0004" (the reader counts zeros);
    # a mathtext ScalarFormatter lifts the exponent into a "x10^-4" offset instead. TTO-only:
    # every other kind keeps matplotlib's default formatter. MUST come after `_finish` —
    # its `ax.set_yscale(...)` reinstalls the scale's default formatter and would wipe this
    # (which is exactly why the earlier `ticklabel_format` call had no visible effect).
    yfmt = ScalarFormatter(useMathText=True)
    yfmt.set_powerlimits((-2, 2))       # also supersedes the PQ-1 comma formatter on this
    ax.yaxis.set_major_formatter(yfmt)  # axis: an offset beats ",.0f" rounding for µV/K
    if handles is not None:
        _tto_legend(ax, handles, spec, style)
    if empty:
        _tto_blank_yaxis(ax)            # note-only figure: no y scale to read
        _fit_tto_notes(fig)             # after _finish: the axes box must be final
    _tto_expand_ylim_for_bands(ax, spec, style)     # I5: the y-view helpers read ax.lines only
    return fig


def _legend_clear_of_inset(ax, iax):
    """Nudge an INSIDE legend off a low-T inset. Shared by the TTO and heat-capacity
    inset-bearing kinds. `_draw_legend`'s inside route uses
    loc='best', whose overlap search sees only the host axes' own artists — the inset is a
    separate Axes, so at font_pt=14 (the GUI's on-screen size) 'best' put the lone 'cooling'
    entry exactly under the inset and it vanished. The gallery renders at 9 pt, where 'best'
    picks upper-left on its own, so `pq_compare` cannot see this (measured both sizes).

    ADDITIVE and conditional: only fires when the two boxes actually overlap, so the 9 pt
    gallery render is untouched. The inset is pinned lower-right, hence upper-left."""
    if iax is None:
        return
    leg = ax.get_legend()
    if leg is None:
        return
    fig = ax.get_figure()
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    if not leg.get_window_extent(rend).overlaps(iax.get_window_extent(rend)):
        return
    leg.set_loc("upper left")


def render_tto_kappa_t(results, spec=None, style=None, overlay=None):
    spec = spec or PlotSpec(); style = style or GlobalStyle()
    fig = _tto_single(results, "tto_kappa_t", "κ (W K⁻¹ m⁻¹)", spec, style, overlay)
    if overlay is None:
        iax = _tto_lowt_inset(fig.axes[0], _as_list(results), spec, style)
        _legend_clear_of_inset(fig.axes[0], iax)
    return fig


def render_tto_seebeck_t(results, spec=None, style=None, overlay=None):
    spec = spec or PlotSpec(); style = style or GlobalStyle()
    fig = _tto_single(results, "tto_seebeck_t", "S (µV/K)", spec, style, overlay)
    # Hard zero line: S crosses zero on the real file. Drawn AFTER _finish so the robust view
    # is already set; gid="refline" keeps it out of any later view computation.
    fig.axes[0].axhline(0, color="black", lw=0.8, gid="refline")
    return fig


def render_tto_zt_t(results, spec=None, style=None, overlay=None):
    spec = spec or PlotSpec(); style = style or GlobalStyle()
    fig = _tto_single(results, "tto_zt_t", "ZT", spec, style, overlay)
    # THE ONLY kind that overrides the robust view: it clips the ZT peak (measured top
    # 3.5130e-05 vs peak 3.9232e-04 on the real file). See _tto_full_view's docstring.
    _tto_full_view(fig.axes[0], spec, style)
    # _tto_full_view re-derives the view from ax.lines, so re-apply the band expansion after it
    _tto_expand_ylim_for_bands(fig.axes[0], spec, style)
    return fig


def render_tto_wf_t(results, spec=None, style=None, overlay=None):
    return _tto_single(results, "tto_wf_t", "κ (W K⁻¹ m⁻¹)", spec, style, overlay)


def render_tto_lorenz_t(results, spec=None, style=None, overlay=None):
    """L/L0 vs T with a hard reference line at the Sommerfeld value 1.0.

    L/L0 ~ 1 is elastic/electron-dominated transport, > 1 a phonon contribution, < 1 inelastic
    scattering -- so the line at 1.0 is the whole point of the figure, not decoration. Drawn
    AFTER _tto_single (i.e. after `_finish`) so the y-view is already set, with gid="refline"
    keeping it out of every later view computation, exactly as tto_seebeck_t's zero line is.

    NO ERROR BAND (deliberately absent from `_TTO_BAND_KINDS`): ZT is banded because its sigma
    is MEASURED (the file's `Merit Std.Dev.` column), whereas L/L0 -- like kappa_e and
    kappa_ph -- would need its sigma PROPAGATED through kappa*rho/(L0*T). There is no
    `lorenz_ratio_std` on the Series, so listing the kind here would either draw nothing or
    invite an ad-hoc band."""
    spec = spec or PlotSpec(); style = style or GlobalStyle()
    fig = _tto_single(results, "tto_lorenz_t", "L/L₀", spec, style, overlay)
    ax = fig.axes[0]
    ax.axhline(1.0, color="black", lw=0.8, gid="refline")
    # LABEL the line, same idiom as hc_full_cp_t's Dulong–Petit line: an unlabelled thin black
    # rule reads as a gridline, and the gallery reference this entry cites is characterised by a
    # *labeled* horizontal reference line. Zero (tto_seebeck_t) is self-evident; L/L₀ = 1 is not.
    # Guarded on the line being IN VIEW -- the robust view is derived from the data, so on a
    # curve that never approaches 1 (the synth fixtures sit at 1.013-1.030) 1.0 falls outside
    # ax.get_ylim() and a label would be drawn on empty canvas outside the axes.
    # ...and on there BEING data: an empty/gated selection keeps the default 0-1 view, which
    # brackets 1.0, so the guard alone would label the explanatory note's blank canvas.
    lo, hi = ax.get_ylim()
    if any(ln.get_gid() is None for ln in ax.lines) and lo <= 1.0 <= hi:
        # font_pt-1 + font_family, i.e. the repo's ANNOTATION idiom (1157/1504/1740), not the
        # literal fontsize="small" of the two older reference-line labels: rcParams is never
        # touched here, so "small" is a fixed absolute size and visibly under-scales at 14 pt.
        fam = {"fontfamily": style.font_family} if style.font_family else {}
        ax.text(0.02, 1.0, "Wiedemann–Franz (L = L₀)", transform=ax.get_yaxis_transform(),
                va="bottom", ha="left", fontsize=style.font_pt - 1, gid="refline-label:h",
                **fam)
        # created AFTER _finish (the in-view guard needs the settled ylim), so the
        # canonical inset->labels->legend order cannot apply: place it now, with the
        # already-drawn legend as an obstacle.
        _place_refline_labels(ax, style)
    return fig


_TTO_PANEL_PREFIX = ("kappa:", "seebeck:", "rho:")
# Panel -> the Task-6 single kind whose capability explains an empty panel, so `_tto_empty_note`
# can reuse the ANALYZER's own `reason` text (no new copy). rho borrows wiedemann_franz, whose
# reason is literally "requires finite ρ > 0".
_TTO_PANEL_NOTE_KIND = {"kappa:": "tto_kappa_t", "seebeck:": "tto_seebeck_t",
                        "rho:": "tto_wf_t"}
_TTO_PANEL_TAGS = ("(a)", "(b)", "(c)")
_TTO_EDGE_TICK_FRAC = 0.08      # ~half a tick-label height at the default panel height
YLABEL_MIN_PT = 6.0             # floor for the stacked-headline ylabel autofit
YLABEL_FIT_FRAC = 0.97          # leave a hair of slack so a rounded glyph box can't touch
YLABEL_FIT_PASSES = 3           # shrinking a label re-runs the layout -> re-measure


def _fit_ylabels_to_axes(fig, floor=YLABEL_MIN_PT, frac=YLABEL_FIT_FRAC,
                         passes=YLABEL_FIT_PASSES):
    """Cap each panel's ylabel font size so the ROTATED label fits inside its own axes height.

    Constrained layout does NOT grow the canvas for a ylabel taller than its axes: it lets the
    label overflow, so on the stacked headline (three panels sharing a 70 mm figure) the top
    label ran off the figure edge — clipped to `W K⁻¹ m` at 14 pt, losing its closing `)` even
    at the 9 pt default — and the middle one overprinted its neighbours. Same remedy as
    `LEGEND_DENSE_PT`: cap the font, keep the geometry, font-size-invariant. No-op when the
    label already fits, so single-panel kinds are untouched."""
    for _ in range(passes):
        fig.draw_without_rendering()        # run the layout engine: extents must be final
        rend = fig.canvas.get_renderer()
        changed = False
        for ax in fig.axes:
            lab = ax.yaxis.label
            if not lab.get_text():
                continue
            lab_h = lab.get_window_extent(rend).height
            ax_h = ax.get_window_extent(rend).height * frac
            size = lab.get_fontsize()
            if ax_h > 0 and lab_h > ax_h and size > floor:
                lab.set_fontsize(max(floor, size * ax_h / lab_h))
                changed = True
        if not changed:
            break
    # One common size across the stack: three panel labels at three different sizes reads as a
    # mistake in a journal figure. Shrinking further can only help containment, so this is safe.
    labs = [ax.yaxis.label for ax in fig.axes if ax.yaxis.label.get_text()]
    if labs:
        smallest = min(lab.get_fontsize() for lab in labs)
        for lab in labs:
            lab.set_fontsize(smallest)


def _tto_trim_edge_ticks(ax, low, high, frac=_TTO_EDGE_TICK_FRAC):
    """Drop the y ticks that sit against a SHARED panel edge of the stacked headline.

    Their labels straddle the boundary — they collide with the neighbouring panel's, and
    because constrained layout packs TIGHT bboxes (not axes boxes) they also re-open exactly
    the gutter `hspace=0` just closed, so the panels stop abutting. Density is first raised
    (nbins=6, AutoLocator's own steps) so trimming both edges cannot leave a lone tick, then
    the survivors are pinned with a FixedLocator — stable across the exporter's
    `set_size_inches`, since the ticks are data-space. Linear scales only (a log panel's
    ticks are not linearly spaced and are the user's explicit choice)."""
    if ax.get_yscale() != "linear":
        return
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    lo, hi = ax.get_ylim()
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        return
    keep = [t for t in ax.get_yticks() if lo <= t <= hi
            and not (low and (t - lo) < frac * span)
            and not (high and (hi - t) < frac * span)]
    ax.yaxis.set_major_locator(FixedLocator(keep))


def _tto_summary_plotted(results, kind, spec, style, overlay):
    """(plotted, gcolor_override) for the stacked headline.

    Overlay ("Add to compare…") mode: the GUI checklist commits FILE-QUALIFIED keys
    (`{file_id}::{raw_key}`, see `overlay_series`), so selection must match on the EFFECTIVE
    key — matching raw keys empties the selection and the figure raises "no series selected".
    The Series keeps its RAW key (the panel router reads the `kappa:`/`seebeck:`/`rho:`
    prefix) and is re-grouped onto the FILE label, so each file gets its own colour and one
    legend entry instead of every file sharing one curve-label colour (file identity lost).
    Non-overlay mode is unchanged and returns no colour override."""
    if overlay is None:
        plotted = [(r, s) for r in results
                   for s in select_series(kind.series(r, field_unit=style.field_unit), spec)]
        return plotted, None
    want = None if spec.curves is None else set(spec.curves)
    colours = _file_colours(overlay, len(results), style)
    plotted, gcolor = [], {}
    for fi, (r, of) in enumerate(zip(results, overlay)):
        for s in kind.series(r, field_unit=style.field_unit):
            eff = f"{of.file_id}::{s.key}"
            if (want is None and s.default_on) or (want is not None and eff in want):
                plotted.append((r, dataclasses.replace(s, group=of.label)))
                gcolor[of.label] = colours[fi]
    return plotted, gcolor


def render_tto_summary_t(results, spec=None, style=None, overlay=None):
    """HEADLINE: vertically-stacked shared-T panels — kappa(T) top, S(T) middle, rho(T)
    bottom (render_acms_chi_t layout). Panels are routed by the Series KEY PREFIX. The rho
    panel converts Ohm*m -> Ohm*cm (x100) before _rho_axis_autoscale picks an engineering
    prefix; that call must happen after all data lines are drawn and before any refline or
    _finish, so the robust view sees scaled data and axvline reflines are never rescaled.
    Panels ABUT (hspace=0) and carry (a)/(b)/(c) tags, per stacked shared-x journal figures."""
    spec = spec or PlotSpec(); style = style or GlobalStyle()
    results = _as_list(results); kind = _KIND["tto_summary_t"]
    fig = _new_fig(style)
    ax_k, ax_s, ax_r = fig.subplots(3, 1, sharex=True)
    # Shared x -> the panels abut (no inter-panel gutter). hspace alone is not enough under
    # constrained layout: h_pad still reserves padding around every axes, so both are zeroed.
    fig.get_layout_engine().set(hspace=0.0, h_pad=0.0)
    plotted, gcolor_override = _tto_summary_plotted(results, kind, spec, style, overlay)
    if not plotted:
        raise NothingToPlot(f"no series selected for kind {kind.key}")
    gcolor, handles = _tto_handles(plotted, spec, style, gcolor=gcolor_override)
    panels = dict(zip(_TTO_PANEL_PREFIX, (ax_k, ax_s, ax_r)))
    blank = []
    rho_items = []
    for prefix, ax in panels.items():
        items = [(r, s) for r, s in plotted if s.key.startswith(prefix)]
        if not items:
            # Degenerate file (no Seebeck / no rho): say WHY, don't leave blank 0-1 axes.
            _tto_empty_note(ax, results, _TTO_PANEL_NOTE_KIND[prefix], style)
            blank.append(ax)
            continue
        if prefix == "rho:":
            rho_items = items
        # Ohm*m -> Ohm*cm for the rho panel only; _rho_axis_autoscale then adds the prefix.
        # The rho panel's BAND is deferred to after that call (C2).
        _tto_draw(ax, items, kind, spec, style, gcolor,
                  yscale=(100.0 if prefix == "rho:" else 1.0),
                  draw_bands=(prefix != "rho:"))
    factor, rho_unit = _rho_axis_autoscale(ax_r)
    # C2: `_rho_axis_autoscale` multiplies `for ln in ax.lines` ONLY, and a fill_between
    # polygon is a PolyCollection -- it would never receive the factor, `ax.relim()` ignores
    # collections, and nothing would betray a band drawn 1e6x too small on the probe's HEADLINE
    # figure (the real file lands on the 1e6 / µΩ·cm rung). So the rho band is drawn HERE,
    # after the factor is known, at yscale = 100.0 * factor. Do NOT teach _rho_axis_autoscale
    # to walk collections: rescaling a PolyCollection's paths in place is a second way to get
    # the same number wrong, and its median is computed from gid-None LINES, so drawing the
    # band later cannot perturb the factor it picks.
    if rho_items:
        _tto_draw_bands(ax_r, rho_items, kind, spec, style, gcolor, yscale=100.0 * factor)
    # Zero reference line on the S panel, drawn only when the data actually crosses zero.
    sy = [v for _, s in plotted if s.key.startswith("seebeck:") for v in s.y]
    sy = np.asarray(sy, float)
    sy = sy[np.isfinite(sy)]
    if sy.size and float(sy.min()) < 0.0 < float(sy.max()):
        ax_s.axhline(0, color="black", lw=0.8, gid="refline")
    # No y-formatter override here: all three panels land O(1)-O(100) after the rho engineering
    # prefix, so matplotlib's default ticks are right. (The former ticklabel_format loop was
    # inert anyway — _finish's set_yscale reinstalls the scale's default formatter.)
    _finish(ax_k, kind, spec, style, "", "κ (W K⁻¹ m⁻¹)", draw_legend=False)
    _finish(ax_s, kind, spec, style, "", "S (µV/K)", draw_legend=False)
    _finish(ax_r, kind, spec, style, "Temperature (K)", f"ρ ({rho_unit})", draw_legend=False)
    for ax in (ax_k, ax_s, ax_r):
        _tto_expand_ylim_for_bands(ax, spec, style)
    for ax in blank:                    # after _finish: set_yscale reinstalls the locator
        _tto_blank_yaxis(ax)
    # Abutting panels, part 2: drop the y ticks that sit against a SHARED edge (top panel:
    # bottom edge only; middle: both; bottom panel: top edge only).
    for ax, edges in ((ax_k, (True, False)), (ax_s, (True, True)), (ax_r, (False, True))):
        if ax not in blank:
            _tto_trim_edge_ticks(ax, *edges)
    label_sz = style.label_size if style.label_size is not None else style.font_pt
    fam = {"fontfamily": style.font_family} if style.font_family else {}
    for tag, ax in zip(_TTO_PANEL_TAGS, (ax_k, ax_s, ax_r)):
        ax.text(0.012, 0.94, tag, transform=ax.transAxes, va="top", ha="left",
                fontsize=label_sz, **fam)
    _fit_ylabels_to_axes(fig)           # 3 panels in one figure height -> labels must be capped
    _fit_tto_notes(fig)                 # empty-panel notes must not run over their ylabels
    _tto_legend(ax_k, handles, spec, style)
    return fig


# ---- dispatch ----
_RENDERERS = {
    "inverse_chi": render_inverse_chi,
    "vsm_moment_t": render_vsm_moment_t,
    "vsm_chi_t": render_vsm_chi_t,
    "vsm_chi_t_product": render_vsm_chi_t_product,
    "vsm_mh": render_vsm_mh,
    "cp_over_t": render_cp_over_t,
    "cp_vs_t": render_cp_vs_t,
    "hc_entropy_vs_t": render_hc_entropy_vs_t,
    "hc_full_cp_t": render_hc_full_cp_t,
    "hc_c_over_t_linear": render_hc_c_over_t_linear,
    "hc_gamma_vs_field": render_hc_gamma_vs_field,
    "hc_thetaD_vs_field": render_hc_thetaD_vs_field,
    "hc_A_vs_field": render_hc_A_vs_field,
    "hc_T0_vs_field": render_hc_T0_vs_field,
    "hc_lowt_multifield": render_hc_lowt_multifield,
    "hc_delta_vs_field": render_hc_delta_vs_field,
    "hc_f_vs_field": render_hc_f_vs_field,
    "hc_alphaN_vs_field": render_hc_alphaN_vs_field,
    "hc_schottky_multifield": render_hc_schottky_multifield,
    "hc_tc_vs_field": render_hc_tc_vs_field,
    "hc_transition_multifield": render_hc_transition_multifield,
    "hc_transition_signal": render_hc_transition_signal,
    "resistivity_rho_t": render_resistivity,
    "resistivity_mr": render_resistivity_mr,
    "resistivity_mr_pct": render_resistivity_mr_pct,
    "resistivity_mr_pct_t": render_resistivity_mr_pct_t,
    "resistivity_arrhenius": render_resistivity_arrhenius,
    "resistivity_rho_t2": render_resistivity_rho_t2,
    "hall_rh_t": render_hall,
    "hall_mobility_t": render_hall_mobility_t,
    "hall_n_t": render_hall_n_t,
    "hall_r2_t": render_hall_r2_t,
    "hall_rxy_vs_B": render_hall_rxy_vs_b,
    "hall_asym_vs_B": render_hall_asym_vs_b,
    "hall_raw_vs_asym": render_hall_raw_vs_asym,
    "hall_two_panel": render_hall_two_panel,
    "hall_rh_n_twin": render_hall_rh_n_twin,
    "hall_tdep_RH_T": render_hall_tdep_rh_t,
    "hall_tdep_n_T": render_hall_tdep_n_t,
    "hall_tdep_mobility_T": render_hall_tdep_mobility_t,
    "hall_tdep_asym_vs_B": render_hall_tdep_asym_vs_b,
    "hall_tdep_interp_RT": render_hall_tdep_interp_rt,
    "hall_tdep_stages": render_hall_tdep_stages,
    "hall_tdep_J_T": render_hall_tdep_j_t,
    "hall_tdep_summary": render_hall_tdep_summary,
    "hall_tdep_rh_n_twin": render_hall_tdep_rh_n_twin,
    "acms_chi_t": render_acms_chi_t,
    "acms_chi_prime_t": render_acms_chi_prime_t,
    "acms_chi_dprime_t": render_acms_chi_dprime_t,
    "acms_mdc_t": render_acms_mdc_t,
    "tto_summary_t": render_tto_summary_t,
    "tto_kappa_t": render_tto_kappa_t,
    "tto_seebeck_t": render_tto_seebeck_t,
    "tto_zt_t": render_tto_zt_t,
    "tto_wf_t": render_tto_wf_t,
    "tto_lorenz_t": render_tto_lorenz_t,
}
_DEFAULT_KIND = {"vsm": "inverse_chi", "heatcapacity": "cp_over_t",
                 "resistivity": "resistivity_rho_t", "hall": "hall_rh_t",
                 "hall_tdep": "hall_tdep_RH_T", "acms": "acms_chi_t",
                 "tto": "tto_summary_t"}

def default_kind_for(probe):
    return _DEFAULT_KIND.get(probe, "inverse_chi")

def render_kind(results, kind_key, spec=None, style=None, overlay=None):
    fn = _RENDERERS.get(kind_key)
    if fn is None:
        raise KeyError(f"unknown plot kind: {kind_key}")
    return fn(results, spec, style, overlay)

def render_for(result, spec=None, style=None):
    if isinstance(result, list):
        raise TypeError("render_for takes a single Result; use render_kind for a list of results")
    probe = (result.data or {}).get("probe")
    return render_kind(result, default_kind_for(probe), spec, style)
