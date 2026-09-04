"""PQ-1 2b: save_figure / export_plots — exact-mm contract, formats, determinism.
"""
import json
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import pytest
from PIL import Image

from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry
from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle, PlotLayout, PlotEntry
from cryosweep_core.plotting.render import render_kind
from cryosweep_core.plotting.export import save_figure, export_plots

FIX = pathlib.Path(__file__).parent / "fixtures"


def _res():
    return analyze_file(load_dat(str(FIX / "act_synth.dat")),
                        RunConfig.load(probe_override="resistivity"), build_default_registry())


def _fig(style=None, spec=None, res=None):
    return render_kind(res or _res(), "resistivity_rho_t", spec or PlotSpec(),
                       style or GlobalStyle())


def _px(mm, dpi):
    return round(mm / 25.4 * dpi)


# ---------- 1. exact-mm contract ----------

def test_png_pixel_dims_match_style_mm(tmp_path):
    style = GlobalStyle()  # 90 x 70 mm @ 300 dpi
    fig = _fig(style)
    p = save_figure(fig, tmp_path / "a.png", style)
    with Image.open(p) as im:
        assert im.size == (_px(90, 300), _px(70, 300))


def test_png_pixel_dims_match_nondefault_dpi(tmp_path):
    style = GlobalStyle(dpi=150)
    fig = _fig(style)
    p = save_figure(fig, tmp_path / "a.png", style)
    with Image.open(p) as im:
        assert im.size == (_px(90, 150), _px(70, 150))


@pytest.mark.parametrize("w_mm,h_mm,dpi", [
    (55.3, 44.1, 200),     # round(55.3/25.4*200)/200*200 = 434.9999... -> Agg int() truncated to 434
    (33.3, 21.7, 100),
    (172.0, 63.5, 96),
    (90.0, 70.0, 72),
])
def test_png_pixel_snap_survives_float_truncation(tmp_path, w_mm, h_mm, dpi):
    style = GlobalStyle(width_mm=w_mm, height_mm=h_mm, dpi=dpi)
    fig = _fig(style)
    p = save_figure(fig, tmp_path / "a.png", style)
    with Image.open(p) as im:
        assert im.size == (_px(w_mm, dpi), _px(h_mm, dpi))


# ---------- 2. spec override precedence ----------

def test_spec_mm_override_beats_globalstyle(tmp_path):
    style = GlobalStyle()
    spec = PlotSpec(width_mm=120.0, height_mm=80.0)
    fig = _fig(style, spec)
    p = save_figure(fig, tmp_path / "a.png", style, spec=spec)
    with Image.open(p) as im:
        assert im.size == (_px(120, 300), _px(80, 300))


def test_spec_none_mm_falls_back_to_style(tmp_path):
    style = GlobalStyle(width_mm=100.0)
    spec = PlotSpec()  # width_mm/height_mm None
    fig = _fig(style, spec)
    p = save_figure(fig, tmp_path / "a.png", style, spec=spec)
    with Image.open(p) as im:
        assert im.size == (_px(100, 300), _px(70, 300))


# ---------- 3. tight toggle ----------

def test_tight_changes_dims_and_is_off_by_default(tmp_path):
    style = GlobalStyle()
    fig = _fig(style)
    p_exact = save_figure(fig, tmp_path / "exact.png", style)
    fig2 = _fig(style)
    p_tight = save_figure(fig2, tmp_path / "tight.png", style, tight=True)
    with Image.open(p_exact) as a, Image.open(p_tight) as b:
        assert a.size == (_px(90, 300), _px(70, 300))
        assert b.size != a.size


# ---------- 4. formats / magic bytes ----------

def test_three_formats_written_with_magic_bytes(tmp_path):
    style = GlobalStyle()
    magics = {"png": b"\x89PNG", "pdf": b"%PDF"}
    for fmt, magic in magics.items():
        fig = _fig(style)
        p = save_figure(fig, tmp_path / f"f.{fmt}", style)
        head = p.read_bytes()[:8]
        assert head.startswith(magic), fmt
    fig = _fig(style)
    p = save_figure(fig, tmp_path / "f.svg", style)
    head = p.read_bytes()[:100].lstrip()
    assert head.startswith(b"<?xml") or head.startswith(b"<svg")


def test_fmt_arg_wins_and_appends_suffix(tmp_path):
    style = GlobalStyle()
    fig = _fig(style)
    p = save_figure(fig, tmp_path / "noext", style, fmt="pdf")
    assert p.suffix == ".pdf" and p.read_bytes()[:4] == b"%PDF"


# ---------- 5+10+11. batch naming, order, skip-on-failure, empty ----------

def test_export_plots_naming_and_order(tmp_path):
    res = _res()
    style = GlobalStyle()
    layout = PlotLayout(plots=[PlotEntry(kind="resistivity_rho_t")])
    paths = export_plots(res, layout, style, tmp_path, "sample", formats=("png", "pdf"))
    assert [p.name for p in paths] == ["sample_resistivity_rho_t.png", "sample_resistivity_rho_t.pdf"]
    assert all(p.exists() for p in paths)


def test_export_plots_empty_layout_returns_empty(tmp_path):
    paths = export_plots(_res(), PlotLayout(), GlobalStyle(), tmp_path, "x")
    assert paths == []


def test_export_plots_skips_failing_kind(tmp_path):
    res = _res()
    layout = PlotLayout(plots=[PlotEntry(kind="hc_cp_vs_T"),           # unknown for probe -> render fails
                               PlotEntry(kind="resistivity_rho_t")])
    paths = export_plots(res, layout, GlobalStyle(), tmp_path, "x")
    assert [p.name for p in paths] == ["x_resistivity_rho_t.png"]


def test_export_plots_kinds_filter(tmp_path):
    res = _res()
    layout = PlotLayout(plots=[PlotEntry(kind="resistivity_rho_t"),
                               PlotEntry(kind="resistivity_mr")])
    paths = export_plots(res, layout, GlobalStyle(), tmp_path, "x",
                         kinds=["resistivity_rho_t"])
    assert [p.name for p in paths] == ["x_resistivity_rho_t.png"]


def test_export_plots_creates_out_dir(tmp_path):
    res = _res()
    layout = PlotLayout(plots=[PlotEntry(kind="resistivity_rho_t")])
    out = tmp_path / "new" / "dir"
    paths = export_plots(res, layout, GlobalStyle(), out, "x")
    assert paths and paths[0].parent == out


# ---------- 6. white facecolor ----------

def test_png_background_is_white(tmp_path):
    style = GlobalStyle()
    fig = _fig(style)
    fig.patch.set_facecolor("black")   # sabotage: save must still force white
    p = save_figure(fig, tmp_path / "a.png", style)
    with Image.open(p) as im:
        assert im.convert("RGB").getpixel((1, 1)) == (255, 255, 255)


# ---------- 7. byte-identical double export ----------

@pytest.mark.parametrize("fmt", ["png", "pdf", "svg"])
def test_double_export_byte_identical(tmp_path, fmt):
    style = GlobalStyle()
    outs = []
    for i in range(2):
        fig = _fig(style)
        outs.append(save_figure(fig, tmp_path / f"r{i}.{fmt}", style).read_bytes())
        plt.close(fig)
    assert outs[0] == outs[1]


# ---------- 8. rcParams isolation ----------

def test_fonttype_rcparams_do_not_leak(tmp_path):
    before = (matplotlib.rcParams["pdf.fonttype"], matplotlib.rcParams["svg.fonttype"])
    fig = _fig()
    save_figure(fig, tmp_path / "a.pdf", GlobalStyle())
    after = (matplotlib.rcParams["pdf.fonttype"], matplotlib.rcParams["svg.fonttype"])
    assert before == after


def test_pdf_embeds_truetype_fonttype42(tmp_path):
    fig = _fig()
    p = save_figure(fig, tmp_path / "a.pdf", GlobalStyle())
    # fonttype 42 -> TrueType font program embedded; fonttype 3 -> Type3 dict
    data = p.read_bytes()
    assert b"/Type3" not in data


def test_svg_text_stays_text(tmp_path):
    fig = _fig()
    p = save_figure(fig, tmp_path / "a.svg", GlobalStyle())
    assert b"<text" in p.read_bytes()   # svg.fonttype 'none' keeps <text> elements


# ---------- 9. unknown format ----------

def test_unknown_format_raises_valueerror(tmp_path):
    fig = _fig()
    with pytest.raises(ValueError, match="png"):
        save_figure(fig, tmp_path / "a.tiff", GlobalStyle())
    with pytest.raises(ValueError, match="png"):
        save_figure(fig, tmp_path / "a", GlobalStyle(), fmt="eps")


# ---------- 12. legend-extra width honored ----------

def test_legend_extra_width_added(tmp_path):
    style = GlobalStyle()
    fig = _fig(style)
    fig._cryosweep_legend_extra_in = 1.0
    p = save_figure(fig, tmp_path / "a.png", style)
    with Image.open(p) as im:
        assert im.size[0] == round((90 / 25.4 + 1.0) * 300)
        assert im.size[1] == _px(70, 300)


# ---------- 13. PlotSpec new fields ----------

def test_plotspec_mm_fields_default_none_and_validate():
    s = PlotSpec()
    assert s.width_mm is None and s.height_mm is None
    s2 = PlotSpec(width_mm=85.5, height_mm=60.0)
    assert (s2.width_mm, s2.height_mm) == (85.5, 60.0)
    with pytest.raises(Exception):
        PlotSpec(width_mm=0)


def test_plotspec_old_json_without_mm_loads():
    s = PlotSpec.model_validate(json.loads('{"title": "t"}'))
    assert s.width_mm is None and s.height_mm is None


# ---------- 14. figure size restored ----------

def test_save_figure_restores_fig_size(tmp_path):
    fig = _fig()
    fig.set_size_inches(5.0, 4.0)
    save_figure(fig, tmp_path / "a.png", GlobalStyle())
    assert tuple(fig.get_size_inches()) == (5.0, 4.0)
