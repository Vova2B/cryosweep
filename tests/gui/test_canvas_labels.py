import pathlib
import matplotlib.image as mpimg
FIX = pathlib.Path(__file__).resolve().parents[1] / "core" / "fixtures"

def _win(qapp, tmp_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow(preset_path=tmp_path / "p.json"); win.resize(1180, 720)
    win.load_path(str(FIX / "vsm_synth.dat")); win.select_probe("vsm")
    tab = win.tabs.currentWidget()
    tab.panel.molar_mass_edit.setText("200"); tab.panel.mass_mg_edit.setText("5")
    win._reanalyze_active(); win.show(); qapp.processEvents()
    return win, tab

def test_embedded_canvas_labels_not_clipped(qapp, tmp_path):
    win, tab = _win(qapp, tmp_path)
    fig = tab.output._cards[0].figure
    assert fig.dpi == 100                                  # screen dpi, not the 300 export dpi
    fig.canvas.draw()
    fw, fh = fig.canvas.get_width_height(); ax = fig.axes[0]
    for art in (ax.xaxis.label, ax.yaxis.label):
        bb = art.get_window_extent()
        assert bb.x0 >= 0 and bb.y0 >= 0 and bb.x1 <= fw and bb.y1 <= fh, \
            f"clipped: ({bb.x0:.0f},{bb.y0:.0f},{bb.x1:.0f},{bb.y1:.0f}) fig=({fw}x{fh})"

def test_save_plot_exports_at_export_dpi(qapp, tmp_path):
    win, tab = _win(qapp, tmp_path)
    out = tmp_path / "e.png"
    tab._save_plot_to(str(out))
    assert out.exists()
    w = mpimg.imread(str(out)).shape[1]
    assert w > 1000          # exported at GlobalStyle.dpi (300), not the 100 screen dpi


def test_save_plot_keeps_dense_legend_canvas_growth(qapp, tmp_path):
    # A >11-entry legend relocates outside-right and grows the canvas at render time;
    # the export resize back to width_mm must keep that extra width or the axes squish again.
    from cryosweep_core.result import Result, Provenance
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow(preset_path=tmp_path / "p.json"); win.resize(1180, 720)
    win.select_probe("resistivity")
    tab = win.tabs.currentWidget()
    curves = [{"held_temp_k": 2.0 + 10 * k, "direction": 1,
               "field": [-1.0, 0.0, 1.0], "rho": [1.0, 0.9, 1.0],
               "rho_zero_field": 0.9} for k in range(18)]
    data = {"probe": "resistivity", "rho_source": "instrument_column",
            "bridges": [{"channel": 1, "rho_source": "instrument_column",
                         "classification": "metallic",
                         "rho_t_curves": [], "rho_h_curves": curves}],
            "capabilities": []}
    res = Result(status="ok", data=data,
                 provenance=Provenance(file="x", sha256="ab", app_version=None))
    tab.show_result(res)
    win.show(); qapp.processEvents()
    # export the MR card's figure (dense legend)
    dense = next(f for f in (c.figure for c in tab.output._cards)
                 if getattr(f, "_cryosweep_legend_grown", False))
    tab.output.last_figure = dense
    out = tmp_path / "dense.png"
    tab._save_plot_to(str(out))
    style = tab.controls.style
    w = mpimg.imread(str(out)).shape[1]
    assert w > 1.2 * style.width_mm / 25.4 * style.dpi   # canvas kept the legend's extra width
