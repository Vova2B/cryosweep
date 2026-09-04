"""
Task 1 — OutputPanel adaptive grid + Grid/Focus toggle
TDD: these tests are written BEFORE the implementation.
"""
import dataclasses
import pathlib

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry
from cryosweep_core.plotting.catalog import BUILTIN_PLOTKINDS, build_default_layout
from cryosweep_core.plotting.spec import PlotLayout, PlotEntry

FIX = pathlib.Path(__file__).resolve().parents[1] / "core" / "fixtures"


def _vsm():
    rt = load_dat(str(FIX / "vsm_synth.dat"))
    rt = dataclasses.replace(
        rt, header=dataclasses.replace(rt.header, molar_mass=200.0, mass_mg=5.0)
    )
    return analyze_file(rt, RunConfig.load(probe_override="vsm"), build_default_registry())


def _vsm_layout(res):
    return build_default_layout([k for k in BUILTIN_PLOTKINDS if k.probe == "vsm"], res)


_ALL_KIND_KEYS = [k.key for k in BUILTIN_PLOTKINDS]


def _make_layout(n: int) -> PlotLayout:
    """Return a PlotLayout with n entries (cycling all kind keys)."""
    keys = [_ALL_KIND_KEYS[i % len(_ALL_KIND_KEYS)] for i in range(n)]
    return PlotLayout(plots=[PlotEntry(kind=k) for k in keys])


def _card_pos(panel, card_idx: int):
    """Return (row, col) of _cards[card_idx] in panel._grid."""
    card = panel._cards[card_idx]
    item_idx = panel._grid.indexOf(card)
    row, col, _, _ = panel._grid.getItemPosition(item_idx)
    return row, col


# ── Grid arrangement ──────────────────────────────────────────────────────────

def test_grid_1_card_at_0_0(qapp):
    """1 card → 1×1: placed at (0, 0)."""
    from cryosweep_gui.output_panel import OutputPanel
    res = _vsm()
    p = OutputPanel()
    p.show_result(res, _make_layout(1))
    assert len(p._cards) == 1
    assert _card_pos(p, 0) == (0, 0)


def test_grid_2_cards_1x2(qapp):
    """2 cards → 1×2: both on row 0."""
    from cryosweep_gui.output_panel import OutputPanel
    res = _vsm()
    p = OutputPanel()
    p.show_result(res, _make_layout(2))
    assert len(p._cards) == 2
    assert _card_pos(p, 0) == (0, 0)
    assert _card_pos(p, 1) == (0, 1)


def test_grid_3_cards_2x2(qapp):
    """3 cards → 2×2: rows ∈ {0,1}, cols ∈ {0,1}."""
    from cryosweep_gui.output_panel import OutputPanel
    res = _vsm()
    p = OutputPanel()
    p.show_result(res, _make_layout(3))
    assert len(p._cards) == 3
    positions = {_card_pos(p, i) for i in range(3)}
    assert all(r in (0, 1) for r, _ in positions)
    assert all(c in (0, 1) for _, c in positions)


def test_grid_4_cards_full_2x2(qapp):
    """4 cards → 2×2: all four cells (0,0),(0,1),(1,0),(1,1) occupied."""
    from cryosweep_gui.output_panel import OutputPanel
    res = _vsm()
    p = OutputPanel()
    p.show_result(res, _make_layout(4))
    assert len(p._cards) == 4
    positions = {_card_pos(p, i) for i in range(4)}
    assert positions == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_grid_5_cards_2x3(qapp):
    """5 cards → 2×3: row max=1, col max=2."""
    from cryosweep_gui.output_panel import OutputPanel
    res = _vsm()
    p = OutputPanel()
    p.show_result(res, _make_layout(5))
    assert len(p._cards) == 5
    positions = [_card_pos(p, i) for i in range(5)]
    assert max(r for r, _ in positions) == 1
    assert max(c for _, c in positions) == 2


def test_grid_7_cards_2col(qapp):
    """>6 cards → 2-column: col max=1."""
    from cryosweep_gui.output_panel import OutputPanel
    res = _vsm()
    p = OutputPanel()
    p.show_result(res, _make_layout(7))
    assert len(p._cards) == 7
    positions = [_card_pos(p, i) for i in range(7)]
    assert max(c for _, c in positions) == 1


def test_grid_relayout_after_repeated_show(qapp):
    """Grid positions remain correct across repeated show_result calls."""
    from cryosweep_gui.output_panel import OutputPanel
    res = _vsm()
    p = OutputPanel()
    for _ in range(3):
        p.show_result(res, _make_layout(4))
    positions = {_card_pos(p, i) for i in range(4)}
    assert positions == {(0, 0), (0, 1), (1, 0), (1, 1)}


# ── Focus mode ────────────────────────────────────────────────────────────────

def test_focus_index_defaults_to_0(qapp):
    from cryosweep_gui.output_panel import OutputPanel
    p = OutputPanel()
    assert hasattr(p, "_focus_index")
    assert p._focus_index == 0


def test_focus_mode_hides_siblings(qapp):
    """Focus mode: only _focus_index card is NOT hidden; _cards and canvas count intact."""
    from cryosweep_gui.output_panel import OutputPanel
    res = _vsm()
    p = OutputPanel()
    p.show_result(res, _vsm_layout(res))          # 4 backed VSM cards
    n_cards = len(p._cards)
    n_canvases = len(p.findChildren(FigureCanvasQTAgg))

    p._focus_btn.click()                           # switch to Focus

    assert len(p._cards) == n_cards
    assert len(p.findChildren(FigureCanvasQTAgg)) == n_canvases
    for i, card in enumerate(p._cards):
        if i == p._focus_index:
            assert not card.isHidden(), f"card {i} (focused) should not be hidden"
        else:
            assert card.isHidden(), f"card {i} (sibling) should be hidden"


def test_focus_mode_canvas_count_unchanged(qapp):
    """Focus mode hides but never destroys canvases."""
    from cryosweep_gui.output_panel import OutputPanel
    res = _vsm()
    p = OutputPanel()
    p.show_result(res, _vsm_layout(res))
    before = len(p.findChildren(FigureCanvasQTAgg))
    p._focus_btn.click()
    after = len(p.findChildren(FigureCanvasQTAgg))
    assert before > 0
    assert before == after


def test_focus_navigation_to_nonzero_card(qapp):
    """Focus mode + prev/next must reach a non-zero card; counts unchanged."""
    from cryosweep_gui.output_panel import OutputPanel
    res = _vsm()
    p = OutputPanel()
    p.show_result(res, _vsm_layout(res))          # 4 backed VSM cards
    n_cards = len(p._cards)
    n_canvases = len(p.findChildren(FigureCanvasQTAgg))
    assert n_cards >= 3                            # need room to navigate

    p._focus_btn.click()                           # enter Focus (card 0)
    p._focus_next_btn.click()                      # → card 1
    p._focus_next_btn.click()                      # → card 2
    assert p._focus_index == 2

    # counts intact (hidden, not destroyed)
    assert len(p._cards) == n_cards
    assert len(p.findChildren(FigureCanvasQTAgg)) == n_canvases

    for i, card in enumerate(p._cards):
        if i == 2:
            assert not card.isHidden(), "focused card 2 should be visible"
        else:
            assert card.isHidden(), f"card {i} should be hidden"


def test_focus_prev_wraps_and_clamps(qapp):
    """prev from card 0 stays in valid range (no negative index)."""
    from cryosweep_gui.output_panel import OutputPanel
    res = _vsm()
    p = OutputPanel()
    p.show_result(res, _vsm_layout(res))
    p._focus_btn.click()
    p._focus_prev_btn.click()                      # from 0 → wrap to last
    assert 0 <= p._focus_index < len(p._cards)
    # exactly one card visible
    visible = [c for c in p._cards if not c.isHidden()]
    assert len(visible) == 1


def test_focus_index_clamped_on_fewer_cards(qapp):
    """If focus_index points past a smaller new layout, it is clamped."""
    from cryosweep_gui.output_panel import OutputPanel
    res = _vsm()
    p = OutputPanel()
    p.show_result(res, _make_layout(5))            # 5 cards
    p._focus_btn.click()
    p._focus_next_btn.click()
    p._focus_next_btn.click()
    p._focus_next_btn.click()
    p._focus_next_btn.click()                      # → card 4
    assert p._focus_index == 4
    p.show_result(res, _make_layout(2))            # now only 2 cards
    assert p._focus_index < len(p._cards)          # clamped to <2


def test_grid_mode_shows_all_cards(qapp):
    """Grid mode after Focus: no card should be explicitly hidden."""
    from cryosweep_gui.output_panel import OutputPanel
    res = _vsm()
    p = OutputPanel()
    p.show_result(res, _vsm_layout(res))
    p._focus_btn.click()   # focus
    p._grid_btn.click()    # back to grid
    for card in p._cards:
        assert not card.isHidden()


def test_grid_btn_and_focus_btn_exist(qapp):
    """OutputPanel must expose _grid_btn and _focus_btn."""
    from cryosweep_gui.output_panel import OutputPanel
    p = OutputPanel()
    assert hasattr(p, "_grid_btn")
    assert hasattr(p, "_focus_btn")


# ── _empty sentinel ───────────────────────────────────────────────────────────

def test_empty_visible_on_init(qapp):
    """Before any show_result, _empty must not be hidden."""
    from cryosweep_gui.output_panel import OutputPanel
    p = OutputPanel()
    assert not p._empty.isHidden()


def test_empty_hidden_when_plots_loaded(qapp):
    from cryosweep_gui.output_panel import OutputPanel
    res = _vsm()
    p = OutputPanel()
    p.show_result(res, _vsm_layout(res))
    assert p._empty.isHidden()


def test_empty_visible_with_empty_layout(qapp):
    from cryosweep_gui.output_panel import OutputPanel
    res = _vsm()
    p = OutputPanel()
    p.show_result(res, _vsm_layout(res))        # plots shown → hidden
    p.show_result(res, PlotLayout(plots=[]))    # empty layout → re-shown
    assert not p._empty.isHidden()


def test_empty_not_in_grid(qapp):
    """_empty must be a sibling of the grid, not inside it."""
    from cryosweep_gui.output_panel import OutputPanel
    p = OutputPanel()
    assert p._grid.indexOf(p._empty) == -1


# ── Preserved behaviour ───────────────────────────────────────────────────────

def test_no_canvas_accumulation_grid(qapp):
    """No canvas leak across repeated show_result in grid mode."""
    from cryosweep_gui.output_panel import OutputPanel
    res = _vsm()
    lay = _vsm_layout(res)
    p = OutputPanel()
    for _ in range(4):
        p.show_result(res, lay)
    assert len(p.findChildren(FigureCanvasQTAgg)) == 4


def test_layout_edited_signal_fires(qapp):
    """layout_edited signal still fires when a card re-renders via spec change."""
    from cryosweep_gui.output_panel import OutputPanel
    res = _vsm()
    p = OutputPanel()
    p.show_result(res, _vsm_layout(res))
    fired = []
    p.layout_edited.connect(lambda: fired.append(1))
    p._cards[0]._on_spec_changed()
    assert fired


# ── Canvas fills cell height (no 1×2 vertical gap) ────────────────────────────

def test_card_canvas_vertical_policy_expanding(qapp):
    """Rendered card's canvas must have Expanding vertical size policy + stretch>0,
    so it absorbs extra cell height (fixes 1×2 vertical gap)."""
    from PySide6.QtWidgets import QSizePolicy
    from cryosweep_gui.output_panel import OutputPanel
    res = _vsm()
    p = OutputPanel()
    p.show_result(res, _make_layout(2))
    card = p._cards[0]
    assert card.canvas is not None
    assert card.canvas.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Expanding
    # canvas added with stretch factor > 0 in the card's vertical layout
    idx = card._lay.indexOf(card.canvas)
    assert card._lay.stretch(idx) == 1


def test_relayout_2_cards_equal_row_col_stretch(qapp):
    """After relayout with 2 cards (1×2): 1 active row + 2 active cols, each stretch 1."""
    from cryosweep_gui.output_panel import OutputPanel
    res = _vsm()
    p = OutputPanel()
    p.show_result(res, _make_layout(2))
    assert p._grid.columnStretch(0) == 1
    assert p._grid.columnStretch(1) == 1
    assert p._grid.rowStretch(0) == 1


def test_relayout_clears_stale_stretch_on_shrink(qapp):
    """A larger grid's stretch must not linger when the next layout has fewer cards."""
    from cryosweep_gui.output_panel import OutputPanel
    res = _vsm()
    p = OutputPanel()
    p.show_result(res, _make_layout(7))     # 2 cols × 4 rows → rows 0..3 stretched
    p.show_result(res, _make_layout(2))     # 2 cols × 1 row
    # rows beyond the new active range must be reset to 0
    assert p._grid.rowStretch(0) == 1
    assert p._grid.rowStretch(1) == 0
    assert p._grid.rowStretch(2) == 0
    assert p._grid.rowStretch(3) == 0


def test_relayout_clears_stale_col_stretch_on_shrink(qapp):
    """Columns beyond the new active range must be reset to 0."""
    from cryosweep_gui.output_panel import OutputPanel
    res = _vsm()
    p = OutputPanel()
    p.show_result(res, _make_layout(5))     # 3 cols
    p.show_result(res, _make_layout(1))     # 1 col
    assert p._grid.columnStretch(0) == 1
    assert p._grid.columnStretch(1) == 0
    assert p._grid.columnStretch(2) == 0


def test_empty_layout_clears_stale_focus_nav(qapp):
    """After navigating in Focus mode, an empty layout must clear the focus
    label and disable the nav buttons (no stale 'k/n' over an empty view)."""
    from cryosweep_gui.output_panel import OutputPanel
    from cryosweep_core.plotting.spec import PlotLayout
    res = _vsm()
    p = OutputPanel()
    p.show_result(res, _vsm_layout(res))    # several cards
    p._focus_btn.click()
    p._focus_next_btn.click()               # label/nav now active ("2/N")
    assert p._focus_label.text() != ""
    assert p._focus_next_btn.isEnabled()

    p.show_result(res, PlotLayout(plots=[]))  # no plots → empty view
    assert p._focus_label.text() == ""
    assert not p._focus_prev_btn.isEnabled()
    assert not p._focus_next_btn.isEnabled()
