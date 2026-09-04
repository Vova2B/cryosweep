from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass(frozen=True)
class ChannelMeta:
    index: int
    name: str | None = None
    role: str | None = None
    cross_section: float | None = None
    length: float | None = None
    geometry_suspect: bool = False

@dataclass(frozen=True)
class HeaderMeta:
    app: str | None
    app_version: str | None
    title: str | None
    info: dict                      # KEY (or description) -> raw value string
    info_rows: tuple                # tuple of (key|None, value_str, description)
    channels: dict                  # int -> ChannelMeta
    molar_mass: float | None
    n_atoms: float | None
    mass_mg: float | None
    data_line: int                  # 0-based index of the "[Data]" marker line
    raw_lines: tuple
    bare_csv: bool = False

@dataclass
class RawTable:
    df: "object"   # pd.DataFrame; typed loosely to keep model.py import-light
    header: HeaderMeta
    path: str | None = None

@dataclass(frozen=True)
class ColumnMap:
    logical: dict     # logical name -> real column name
    unit: dict        # logical name -> unit string

@dataclass(frozen=True)
class Axis:
    name: str
    column: str
    unit: str

@dataclass
class Segment:
    swept: Axis
    direction: int               # +1 / -1 / 0
    branch: str | None
    fixed: dict
    tol: dict
    setpoint: dict
    idx: np.ndarray
    confidence: float
    x: np.ndarray = None         # swept-axis values for this segment (sorted by row order)
    data: dict = field(default_factory=dict)   # observable name -> ndarray aligned with idx
    normalized: set = field(default_factory=set)

class SegmentGrid:
    """Setpoint-indexed view over segments. No data copy."""
    def __init__(self, segments):
        self.segments = list(segments)

    def by_fixed(self, axis):
        out: dict = {}
        for s in self.segments:
            key = s.setpoint.get(axis)
            if key is None:        # Bug 5: swept-axis setpoint is None -> not a fixed bucket
                continue
            out.setdefault(key, []).append(s)
        return out

    def branches(self, **fixed):
        out = {}
        for s in self.segments:
            # Bug 5: skip segments whose fixed-axis setpoint is None (their swept axis);
            # np.isclose(None, v) would raise TypeError.
            sp = s.setpoint
            if any(sp.get(k) is None for k in fixed):
                continue
            if all(np.isclose(sp.get(k), v) for k, v in fixed.items()):
                out[s.branch] = s
        return out

    @staticmethod
    def _clean(x, y):
        order = np.argsort(x, kind="mergesort")
        x, y = np.asarray(x)[order], np.asarray(y)[order]
        ux = np.unique(x)
        if len(ux) == len(x):
            return x, y
        uy = np.array([y[x == v].mean() for v in ux])   # collapse dup-x by mean
        return ux, uy

    def on_common_axis(self, x_axis, columns, grid=None):
        cleaned = []
        for s in self.segments:
            for col in columns:
                cx, cy = self._clean(s.x, s.data[col])
                cleaned.append((s, col, cx, cy))
        lo = max(cx.min() for _, _, cx, _ in cleaned)   # no-extrapolation overlap
        hi = min(cx.max() for _, _, cx, _ in cleaned)
        if grid is None:
            n = max(len(cx) for _, _, cx, _ in cleaned)
            grid = np.linspace(lo, hi, n)
        else:
            grid = grid[(grid >= lo) & (grid <= hi)]
        out: dict = {}
        for s, col, cx, cy in cleaned:
            key = tuple(s.setpoint.values())
            out[key] = np.interp(grid, cx, cy)   # np.interp clamps; grid already within range
        return grid, out

@dataclass
class Measurement:
    df: object
    header: HeaderMeta
    probe: str
    segments: list
    grid: SegmentGrid
    columns: ColumnMap

@dataclass(frozen=True)
class ObservableData:
    key: str
    values: np.ndarray
    sigma: np.ndarray | None = None
    unit: str = ""
