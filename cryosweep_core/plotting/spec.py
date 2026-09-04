from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

#: Legend placement vocabulary. Three modes plus the nine explicit matplotlib positions:
#:  - "best":    occupancy auto — clearest inside position, relocated outside when none is clear
#:  - "inside":  clearest inside position, never relocated
#:  - "outside": forced outside-right (canvas grows so the axes keep their size)
#:  - explicit positions pass to matplotlib verbatim — the user who can see the right spot wins
LegendLoc = Literal["best", "inside", "outside",
                    "upper right", "upper left", "lower right", "lower left",
                    "upper center", "lower center", "center left", "center right", "center"]

class GlobalStyle(BaseModel):
    """Global plot styling applied to all cards in a layout."""
    width_mm: float = Field(90.0, gt=0)
    height_mm: float = Field(70.0, gt=0)
    dpi: int = Field(300, gt=0)
    font_pt: float = Field(9.0, gt=0)
    font_family: str | None = None
    marker: str = "o"
    marker_size: float = Field(3.0, gt=0)
    line_width: float = Field(1.0, gt=0)
    palette: list[str] | None = None      # None -> matplotlib default colour cycle
    color: str | None = None              # applied iff exactly 1 series rendered
    # per-element font sizes; None -> fall back to font_pt / font_pt-1 (render.py)
    label_size: float | None = Field(None, gt=0)
    title_size: float | None = Field(None, gt=0)
    tick_size: float | None = Field(None, gt=0)
    legend_size: float | None = Field(None, gt=0)
    # marker edge; None -> matplotlib default
    edge_color: str | None = None
    edge_width: float | None = Field(None, gt=0)
    # colormap-based colour cycle; None -> palette/color/default
    colormap: str | None = None
    colormap_reverse: bool = False
    robust_view: bool = True               # default-on robust y-autoscale; no-op on clean data
    robust_k: float = Field(8.0, gt=0)     # global only (no per-plot k)
    # --- PQ-1 journal frame (defaults = journal look; all overridable) ---
    spine_width: float | None = Field(None, gt=0)
    tick_direction: Literal["in", "out", "inout"] = "in"
    minor_ticks: bool = True
    ticks_top: bool = True
    ticks_right: bool = True
    grid: bool = False
    grid_style: str = "--"
    grid_alpha: float = Field(0.4, ge=0, le=1)
    connect_lines: bool = True
    legend_on: bool = True
    legend_loc: LegendLoc = "best"
    legend_frame: bool = False
    fit_color: str | None = None
    fit_linestyle: str = "-"
    thousands_sep: bool = False
    # display unit for TOGGLE-GOVERNED field values/axes; storage stays Oe (append-only)
    field_unit: Literal["Oe", "T"] = "Oe"

class ReferenceLine(BaseModel):
    """A labeled reference line (axhline/axvline)."""
    axis: Literal["h", "v"]
    value: float
    label: str | None = None
    color: str = "black"
    linestyle: str = "-"
    linewidth: float = Field(0.8, gt=0)

class PlotSpec(BaseModel):
    """Per-plot settings: axis + curve selection + title."""
    xmin: float | None = None
    xmax: float | None = None
    ymin: float | None = None
    ymax: float | None = None
    xscale: Literal["linear", "log"] | None = None   # None -> PlotKind default
    yscale: Literal["linear", "log"] | None = None
    curves: list[str] | None = None       # None -> kind's default_on set; [] -> none; else exact keys
    fit_line: bool = True
    fit_lines: tuple[str, ...] | None = None   # per-fit toggle (ρ-T² only): None=all, ()=none, subset=those
    title: str | None = None
    robust_view: bool | None = None        # None -> use GlobalStyle.robust_view
    # --- PQ-1 per-plot overrides (None -> use GlobalStyle) ---
    grid: bool | None = None
    connect_lines: bool | None = None
    # --- PQ-1 2b per-plot export size (None -> GlobalStyle.width_mm/height_mm) ---
    width_mm: float | None = Field(None, gt=0)
    height_mm: float | None = Field(None, gt=0)
    legend_on: bool | None = None
    legend_loc: LegendLoc | None = None
    reference_lines: list[ReferenceLine] | None = None
    annotation: bool = True      # rho0/n/RRR/Tc text box on resistivity_rho_t (PQ-4)
    tc_marker: bool = True       # vertical dashed Tc line on resistivity_rho_t (PQ-4)
    lowt_inset: bool = True      # low-T inset on resistivity_rho_t (PQ-4)
    channel_markers: bool = True   # channel->marker shape on MR kinds (PQ-4)
    direction_arrows: bool = True  # legend up/down arrows on MR kinds (PQ-4)
    error_band: bool = False       # opt-in ±1σ shaded band on TTO kinds (default OFF keeps
                                   # every existing render byte-identical; E2)

class PlotEntry(BaseModel):
    """A single plot card: its kind key and per-plot overrides."""
    kind: str
    spec: PlotSpec = Field(default_factory=PlotSpec)

class PlotLayout(BaseModel):
    """Ordered list of plot cards for one probe result."""
    plots: list[PlotEntry] = Field(default_factory=list)
    # Kinds that were backed (series() non-empty) when this layout was saved. Lets reconcile
    # tell "unbacked at save" (append once it becomes backed, e.g. R_H(T) after thickness is
    # entered) from "user unchecked" (never resurrect). None = store predates this field ->
    # recovery semantics (treat plots as the known set).
    known: list[str] | None = None
