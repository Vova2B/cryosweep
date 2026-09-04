"""Render v2 plot kinds beside their committed reference images (the PQ visual gate).

Reads docs/superpowers/pq-reference-gallery/manifest.json. For each entry with a v2_kind,
renders it via the Qt-free core path, rasterizes the Figure, and tiles it beside the
reference image(s) into one PNG under /tmp/cryosweep_pq_compare/. Reference-only entries (v2_kind
null) render the reference alone with a 'no v2 kind yet' caption. `--check` asserts the
currently-assertable checklist items and exits non-zero on failure. Empty/gated kinds are
reported, never faked.

Run:  .venv/bin/python tools/pq_compare.py --all
      .venv/bin/python tools/pq_compare.py --id hall_raw_vs_asym
      .venv/bin/python tools/pq_compare.py --all --check
"""
from __future__ import annotations
import argparse, dataclasses, io, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # the app root, so cryosweep_core imports from any cwd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))      # tools/, for real_data
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry
from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle
from cryosweep_core.plotting.catalog import BUILTIN_PLOTKINDS
from cryosweep_core.plotting.render import render_kind

from real_data import repo_root          # sibling module; tools/ is on sys.path via the insert above

APP = pathlib.Path(__file__).resolve().parents[1]     # cryosweep — where cryosweep_core lives
ROOT = repo_root()                                    # the repo root — where docs/ and the data live
MANIFEST = ROOT / "docs/superpowers/pq-reference-gallery/manifest.json"
GALLERY = ROOT / "docs/superpowers/pq-reference-gallery"
OUT = pathlib.Path("/tmp/cryosweep_pq_compare"); OUT.mkdir(parents=True, exist_ok=True)
KINDS = {k.key: k for k in BUILTIN_PLOTKINDS}
STYLE = GlobalStyle(width_mm=160.0, height_mm=120.0, dpi=110)
LEGEND_CAP = 11  # matches render.py _draw_legend inside/outside threshold
DEGENERATE_REL_SPAN = 1e-12   # measured: 2/41 entries sit at 3.31e-15; healthy axes >= 1e-1
# Measured separation: hc_full_cp_t 27.9% @12pt / 62.3% @16pt; the clean population is 0%.
# 0.25 would leave only a 3-point margin under the 12pt case, so the threshold sits at 0.10
# -- still far above a clean 0%, and not trippable by a legend merely abutting an inset.
OCCLUSION_FRAC = 0.10


def _load_manifest():
    # The reference gallery is maintainer-local and deliberately NOT shipped (it indexes
    # real sample files and copyrighted journal figures), so for anyone outside this repo
    # the CLI half of this tool has nothing to compare against. Say so and exit, rather
    # than raising FileNotFoundError. The invariant helpers below (_check_fig,
    # _drawn_texts, _is_twin) do NOT need the manifest and are exercised by the suite.
    if not MANIFEST.exists():
        print(f"no reference gallery at {MANIFEST} — this comparison tool is "
              f"maintainer-only; its figure-invariant checks run as part of the test "
              f"suite and need nothing extra.", file=sys.stderr)
        sys.exit(3)
    return json.loads(MANIFEST.read_text())


def _runconfig(rc: dict | None):
    """Always return a 3-tuple (RunConfig, molar_mass|None, mass_mg|None)."""
    if not rc:
        return RunConfig.load(), None, None
    rc = dict(rc)
    molar = rc.pop("molar_mass", None); mass = rc.pop("mass_mg", None)  # patched onto header, not RunConfig
    return RunConfig.load(**rc), molar, mass


def _render_v2(entry):
    """Return (matplotlib Figure | None, status_str). Never raises on bad data.

    Returns the LIVE Figure (not a rasterized array) so _check_fig can inspect axes/legend
    before it is closed. Caller is responsible for plt.close(fig).
    """
    kind = entry.get("v2_kind")
    if not kind:
        return None, "reference-only (no v2 kind)"
    if kind not in KINDS:
        return None, f"unknown kind {kind!r}"
    # Manifest dat paths are repo-root-relative for real data, but the committed synthetic
    # fixtures moved under the app in the two-app split — try both roots.
    dat = ROOT / entry["dat"]
    if not dat.exists():
        dat = APP / entry["dat"]
    if not dat.exists():
        return None, f"missing dat {entry['dat']}"
    try:
        cfg, molar, mass = _runconfig(entry.get("runconfig"))
        rt = load_dat(str(dat))
        if molar is not None or mass is not None:
            rt = dataclasses.replace(rt, header=dataclasses.replace(rt.header, molar_mass=molar, mass_mg=mass))
        result = analyze_file(rt, cfg, build_default_registry())
        # F10 (final-review): the status expectation is PER ENTRY, not a widened global
        # accept-set. Closed O1 flips the three real VSM/MPMS entries ok -> low_confidence
        # (certificate only — params unchanged; all 41 gallery PNGs stayed byte-identical),
        # so those three carry `"expect_status": "low_confidence"` in the manifest. Simply
        # accepting `low_confidence` everywhere left `error` as the ONLY rejected status of
        # four, and would have blinded the harness to a FUTURE probe silently degrading to
        # low_confidence — which is precisely the event it caught here.
        st = getattr(result, "status", None)
        expect = entry.get("expect_status")
        if expect is not None:
            if st != expect:
                return None, f"status={st} (manifest expects {expect})"
        elif st not in ("ok", "gated", None):
            return None, f"status={st}"
        series = KINDS[kind].series(result)
        if not series:
            return None, "EMPTY (no series on this file)"
        fig = render_kind(result, kind, PlotSpec(), STYLE)
    except Exception as e:  # noqa: BLE001 — harness must never crash on one bad entry
        return None, f"{type(e).__name__}: {e}"
    return fig, f"ok series={len(series)}"


def _fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return mpimg.imread(buf)


def _drawn_texts(fig):
    """Text artists matplotlib will ACTUALLY draw.

    Ticks are harvested from the tick objects (label1/label2) and filtered to the axis
    view interval -- NEVER by zipping get_ticklocs() against get_ticklabels(), which
    misaligns with stale tick artists. matplotlib retains Text artists for ticks beyond
    the view limits, never draws them, but reports get_visible() True with window
    extents far off-canvas; counting those produced a ~100% false-positive rate.
    """
    out = list(fig.texts)
    for ax in fig.axes:
        out += [ax.title, ax.xaxis.label, ax.yaxis.label] + list(ax.texts)
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks():
                loc = tick.get_loc()
                if loc is None or not (lo - 1e-9 <= loc <= hi + 1e-9):
                    continue
                for lab in (tick.label1, tick.label2):
                    if lab is not None and lab.get_visible() and (lab.get_text() or "").strip():
                        out.append(lab)
    return [t for t in out if t is not None and t.get_visible() and (t.get_text() or "").strip()]


def _is_twin(ax, other):
    """True if the two Axes OVERLAY each other (twinx/twiny).

    Shared-axis membership alone is NOT sufficient and using it would be a false-green
    path: sharex-stacked panels (e.g. the tto_summary_t stacked headline) are equally
    "joined" on x but occupy DIFFERENT bboxes, and excluding them would blind the
    occlusion check across stacked panels. Twins are distinguished by sharing the host's
    position exactly. Measured (mpl 3.11.0): twinx -> joined=True, same bbox=True;
    sharex-stacked -> joined=True, same bbox=False.
    """
    if ax is other:
        return False
    shared = (ax.get_shared_x_axes().joined(ax, other)
              or ax.get_shared_y_axes().joined(ax, other))
    return shared and ax.get_position().bounds == other.get_position().bounds


def _check_fig(entry, fig, status) -> list[str]:
    """Assertable-now checklist subset. Returns failure strings ([] = pass).

    Runs while the Figure is still live (before close). Covers: series-non-empty where
    expected, no axes-collapse, legend entry-count cap OR relocated outside. Knob-driven
    items (fit-on-top, grid, etc.) are PQ-1 targets and are NOT asserted here.
    """
    fails = []
    if not entry.get("v2_kind"):
        return fails  # reference-only entries: nothing to assert yet
    if status.startswith(("EMPTY", "status=")):
        if entry.get("expect_nonempty", True):
            fails.append(f"{entry['id']}: expected non-empty series, got {status!r}")
        return fails
    if fig is None:
        fails.append(f"{entry['id']}: render failed ({status})")
        return fails
    for ax in fig.axes:
        bb = ax.get_position()
        if bb.width <= 0.01 or bb.height <= 0.01:
            fails.append(f"{entry['id']}: axes collapsed (w={bb.width:.3f} h={bb.height:.3f})")
        leg = ax.get_legend()
        if leg is not None:
            n = len(leg.get_texts())
            # get_bbox_to_anchor() is in DISPLAY pixels; convert back to axes-fraction so the
            # ">1.0 == relocated-outside" test (render.py uses bbox_to_anchor=(1.02,0.5)) is valid.
            anchor = leg.get_bbox_to_anchor()
            anchor_x = ax.transAxes.inverted().transform((anchor.x1, anchor.y1))[0] if anchor is not None else 0.0
            if n > LEGEND_CAP and anchor_x <= 1.0:
                fails.append(f"{entry['id']}: legend {n} entries > cap {LEGEND_CAP} and not relocated outside")

    # ---- I2: degenerate axis. An axis whose data is constant to 1e-12 relative carries no
    # scale; AutoLocator + ScalarFormatter then emit 17-significant-digit tick labels.
    # Measured population: 2 of 41 entries, both at rel-span 3.31e-15.
    for i, ax in enumerate(fig.axes):
        if not ax.get_visible():
            continue
        for nm, lim in (("x", ax.get_xlim()), ("y", ax.get_ylim())):
            lo, hi = sorted(lim)
            mag = max(abs(lo), abs(hi), 1e-300)
            rel = (hi - lo) / mag
            if rel < DEGENERATE_REL_SPAN:
                fails.append(f"{entry['id']}: DEGENERATE-AXIS ax{i} {nm} rel-span={rel:.2e}")

    # ---- I1: occlusion.
    r = fig.canvas.get_renderer()
    boxes = []
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is not None:
            boxes.append(("legend", ax, leg.get_window_extent(renderer=r)))
    for leg in getattr(fig, "legends", []):   # fig.legend() is invisible to ax.get_legend()
        boxes.append(("figlegend", None, leg.get_window_extent(renderer=r)))
    insets = [ax for ax in fig.axes if ax.get_label() == "inset"]

    # I1a: legend vs INSET BOX -- the check that catches the motivating defect. The
    # measured 28% (12pt) / 62% (16pt) on hc_full_cp_t is legend-box vs inset-BOX overlap
    # as a fraction of the LEGEND's area. A box-vs-text form does NOT detect it (the
    # inset's tick labels never intersect the legend) and would be inert.
    for kind, owner, bb in boxes:
        for iax in insets:
            if owner is not None and iax is owner:
                continue
            ib = iax.bbox
            ix0, iy0 = max(bb.x0, ib.x0), max(bb.y0, ib.y0)
            ix1, iy1 = min(bb.x1, ib.x1), min(bb.y1, ib.y1)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            frac = ((ix1 - ix0) * (iy1 - iy0)) / max(bb.width * bb.height, 1e-9)
            if frac > OCCLUSION_FRAC:
                fails.append(f"{entry['id']}: OCCLUSION {kind} {frac:.0%} covered by inset")

    # I1b: legend vs drawn TEXT belonging to some other Axes. No current entry trips this;
    # it is a real defect class kept as a guard. Harvest ONCE -- _drawn_texts walks every
    # axis and tick, so calling it inside the nested loop would be O(boxes x axes x texts).
    texts = _drawn_texts(fig)
    extents = {id(t): t.get_window_extent(renderer=r) for t in texts}
    for kind, owner, bb in boxes:
        for ax in fig.axes:
            if owner is not None and (ax is owner or _is_twin(owner, ax)):
                continue
            for t in texts:
                if t.axes is not ax:
                    continue
                tb = extents[id(t)]
                ix0, iy0 = max(bb.x0, tb.x0), max(bb.y0, tb.y0)
                ix1, iy1 = min(bb.x1, tb.x1), min(bb.y1, tb.y1)
                if ix1 <= ix0 or iy1 <= iy0:
                    continue
                frac = ((ix1 - ix0) * (iy1 - iy0)) / max(tb.width * tb.height, 1e-9)
                if frac > OCCLUSION_FRAC:
                    fails.append(
                        f"{entry['id']}: OCCLUSION {kind} covers {frac:.0%} of text "
                        f"{t.get_text()[:24]!r}")
    return fails


def _tile(entry, v2_img):
    refs = [GALLERY / p for p in entry.get("reference_images", [])]
    refs = [r for r in refs if r.exists()]
    panels = [("reference", mpimg.imread(str(r))) for r in refs]
    if v2_img is not None:
        panels.append((f"v2: {entry.get('v2_kind')}", v2_img))
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]
    for ax, (cap, img) in zip(axes, panels):
        ax.imshow(img); ax.set_title(cap, fontsize=9); ax.axis("off")
    fig.suptitle(entry["id"], fontsize=11)
    out = OUT / f"{entry['id']}.png"
    fig.savefig(str(out), dpi=110, bbox_inches="tight"); plt.close(fig)
    return out


def _run_entry(entry, do_check):
    fig, status = _render_v2(entry)
    # Capture FIRST -- byte-identity contract. The geometry checks draw the canvas, and a
    # pre-capture draw() changes first-save tight-bbox bytes on 10 of the 41 renderable
    # entries (vsm_chi_t, hc_cp_over_t, hall_raw_vs_asym, hall_rxy_vs_B, hall_mobility_t,
    # hall_tdep_asym_vs_B, hall_tdep_stages, acms_chi_t, acms_chi_t_sc, tto_wf_t).
    # Pinned by tests/core/test_pq_compare_order.py.
    v2_img = _fig_to_img(fig) if fig is not None else None
    fails = _check_fig(entry, fig, status) if do_check else []
    if fig is not None:
        plt.close(fig)
    out = _tile(entry, v2_img)
    print(f"{entry['id']:26s} {status:34s} -> {out if out else '(no panels)'}")
    return fails


SWEEP_DEFAULT = (9.0, 12.0, 16.0)


def _parse_font_pts(s):
    return tuple(float(x) for x in str(s).split(",") if str(x).strip())


def _with_font_pt(style, pt):
    """GlobalStyle is a pydantic BaseModel (cryosweep_core/plotting/spec.py), NOT a dataclass --
    dataclasses.replace would raise. model_copy also leaves the base style unmutated."""
    return style.model_copy(update={"font_pt": pt})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--id", default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--font-pt", default=None, dest="font_pt",
                    help=f"comma-separated font sizes to sweep, e.g. {','.join(str(int(s)) for s in SWEEP_DEFAULT)}")
    args = ap.parse_args()
    entries = _load_manifest()
    if args.id:
        entries = [e for e in entries if e["id"] == args.id]
        if not entries:
            print(f"no manifest entry id={args.id!r}"); sys.exit(2)
    elif not args.all:
        ap.error("pass --all or --id ID")
    all_fails = []
    global STYLE
    base_style = STYLE
    try:
        for pt in (_parse_font_pts(args.font_pt) if args.font_pt else (None,)):
            if pt is not None:
                STYLE = _with_font_pt(base_style, pt)
                print(f"--- font_pt = {pt}")
            for e in entries:
                all_fails += _run_entry(e, args.check)
    finally:
        STYLE = base_style          # never leave the module mutated
    print(f"=== DONE -> {OUT} ===")
    if args.check:
        if all_fails:
            print("CHECK FAILED:"); [print("  -", f) for f in all_fails]; sys.exit(1)
        print("CHECK PASSED"); sys.exit(0)


if __name__ == "__main__":
    main()
