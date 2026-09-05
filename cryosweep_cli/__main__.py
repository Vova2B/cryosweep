from __future__ import annotations
import argparse, json, sys, hashlib, pathlib
from cryosweep_core.io.loader import load_dat, expand_user_path
from cryosweep_core.config import RunConfig
from cryosweep_core.registry import build_default_registry
from cryosweep_core.detect.probe import detect_probe
from cryosweep_core.io.columns import canonicalize_columns
from cryosweep_core.result import EXIT_CODES, Result, Provenance
from cryosweep_core.io.export import export_result
from cryosweep_core.reports import build_report
from cryosweep_core.discovery import discover
from cryosweep_core.schema import get_schema, unknown_keys, SCHEMA_NAMES

def _sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:16]

def _emit(obj: dict) -> None:
    # deterministic: sorted keys, compact-but-stable separators.
    # allow_nan=False: reject bare NaN/Infinity (invalid RFC-8259 JSON that
    # strict agent parsers reject). Raises ValueError if a non-finite slips
    # through; the main() try/except degrades to an error envelope.
    sys.stdout.write(json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")

def _load(path, molar_mass=None, mass_mg=None):
    import dataclasses
    rt = load_dat(path)
    if molar_mass is not None or mass_mg is not None:
        h = rt.header
        rt = dataclasses.replace(rt, header=dataclasses.replace(
            h, molar_mass=molar_mass if molar_mass is not None else h.molar_mass,
            mass_mg=mass_mg if mass_mg is not None else h.mass_mg))
    return rt

def _analyze(rt, cfg):
    from cryosweep_core.analyzers.dispatch import analyze_file
    return analyze_file(rt, cfg, build_default_registry())

def main(argv=None):
    ap = argparse.ArgumentParser(prog="cryosweep")
    # --version fires during parsing, before the required `command` positional is checked,
    # so `cryosweep --version` works with no subcommand. Single-sourced from the package:
    # tests/core/test_version_consistency.py keeps that in step with pyproject/CITATION.
    from cryosweep_core import __version__ as _v
    ap.add_argument("--version", action="version", version=f"cryosweep {_v}")
    ap.add_argument("command", choices=["detect", "analyze", "fit", "export", "report", "plot",
                                        "probes", "fits", "plots", "observables", "schema", "run",
                                        "hall", "hall-tdep"])
    ap.add_argument("file", nargs="?", default=None)
    ap.add_argument("--out", default="cryosweep_out")
    # default None (not "CGS") so a --config file's unit_system is only overridden when the
    # flag is actually typed; RunConfig itself still defaults to CGS.
    ap.add_argument("--unit-system", default=None, choices=["CGS", "SI"])
    ap.add_argument("--config", default=None,
                    help="RunConfig JSON file (schema: `cryosweep schema config`); "
                         "explicit flags override it per key")
    ap.add_argument("--molar-mass", type=float, default=None)
    ap.add_argument("--mass-mg", type=float, default=None)
    ap.add_argument("--width-mm", type=float, default=None)
    ap.add_argument("--thickness-mm", type=float, default=None)
    ap.add_argument("--length-mm", type=float, default=None)
    ap.add_argument("--probe", default=None, help="override detected probe (e.g. hall)")
    ap.add_argument("--hall-channel", type=int, default=None)
    ap.add_argument("--long-channel", type=int, default=None)
    ap.add_argument("--long-file", default=None)
    ap.add_argument("--thickness", type=float, default=None)
    ap.add_argument("--thickness-unit", default="mm", choices=["mm", "um", "nm"])
    ap.add_argument("--geometry-sign", type=int, default=None, choices=[1, -1])
    ap.add_argument("--temp-interval", type=float, default=None)
    ap.add_argument("--plot-kind", default=None, help="plot kind key (default: probe's default kind)")
    ap.add_argument("--style-file", default=None, help="GlobalStyle JSON (deterministic styling)")
    ap.add_argument("--layout-file", default=None, help="PlotLayout JSON (per-plot specs; reconciled)")
    ap.add_argument("--format", default="png", help="comma list of plot formats: png,pdf,svg")
    ap.add_argument("--all", action="store_true",
                    help="plot: export every kind in the layout (--out = dir/prefix)")
    ap.add_argument("--tight", action="store_true",
                    help="plot: tight-crop bbox (overrides exact mm size)")
    ap.add_argument("--dpi", type=int, default=None, help="plot: override style dpi (PNG)")
    a = ap.parse_args(argv)
    # ~ expansion happens ONCE, at the argv boundary, for every path-valued arg
    # (the shell masks this interactively; agent/subprocess callers pass raw strings).
    # `schema` reuses the file slot for a schema name — never ~-leading, so a no-op.
    for _pathattr in ("file", "out", "long_file", "style_file", "layout_file", "config"):
        _v = getattr(a, _pathattr)
        if _v is not None:
            setattr(a, _pathattr, expand_user_path(_v))
    if a.all and a.plot_kind:
        ap.error("--all and --plot-kind are mutually exclusive")
    geom = {k: v for k, v in (("width_mm", a.width_mm),
                               ("thickness_mm", a.thickness_mm),
                               ("length_mm", a.length_mm)) if v is not None}
    _UNIT_MM = {"mm": 1.0, "um": 1e-3, "nm": 1e-6}
    hall = {}
    if a.hall_channel is not None: hall["hall_channel"] = a.hall_channel
    if a.long_channel is not None: hall["longitudinal_channel"] = a.long_channel
    if a.long_file is not None: hall["longitudinal_file"] = a.long_file
    if a.thickness is not None: hall["thickness_mm"] = a.thickness * _UNIT_MM[a.thickness_unit]
    if a.geometry_sign is not None: hall["geometry_sign"] = a.geometry_sign
    if a.temp_interval is not None: hall["temp_interval"] = a.temp_interval
    overrides = {}
    if a.unit_system is not None: overrides["unit_system"] = a.unit_system
    if geom: overrides["geometry"] = geom
    if hall: overrides["hall"] = hall
    probe_override = a.probe or ("hall" if a.command == "hall" else ("hall_tdep" if a.command == "hall-tdep" else None))
    if probe_override: overrides["probe_override"] = probe_override
    from pydantic import ValidationError
    cfg_warnings: list = []
    try:
        if a.config:
            base = json.loads(pathlib.Path(a.config).read_text())
            if not isinstance(base, dict):
                raise ValueError("--config must hold a JSON object (RunConfig; "
                                 "see `cryosweep schema config`)")
            _unk = unknown_keys(RunConfig, base)
            if _unk:
                _w = "--config: unknown key(s) ignored: " + ", ".join(_unk)
                cfg_warnings.append(_w)
                sys.stderr.write(_w + "\n")
            # explicit flags override the file, PER KEY inside the nested sub-configs, so a
            # config file's hall.temp_interval survives adding --hall-channel on the CLI.
            for k, v in overrides.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    base[k] = {**base[k], **v}
                else:
                    base[k] = v
            cfg = RunConfig(**base)
        else:
            cfg = RunConfig.load(**overrides)
    except (FileNotFoundError, OSError, UnicodeError, ValueError, ValidationError) as e:
        sys.stderr.write(f"error: --config: {e}\n")
        _emit(Result(status="error", errors=[f"--config: {e}"],
                     provenance=Provenance(file=str(a.file or ""), sha256="",
                                           app_version=None, config={})).model_dump(mode="json"))
        return 2

    # discovery / schema commands: no file needed
    if a.command in ("probes", "fits", "plots", "observables"):
        _emit(discover(build_default_registry())); return 0
    if a.command == "schema":
        if not a.file or a.file not in SCHEMA_NAMES:
            sys.stderr.write(f"usage: cryosweep schema <{'|'.join(SCHEMA_NAMES)}>\n"); return 3
        _emit(get_schema(a.file)); return 0
    if a.command == "run":
        if not a.file:
            sys.stderr.write("usage: cryosweep run <pipeline.json>\n"); return 2
        from cryosweep_core.pipeline import run_pipeline
        out = run_pipeline(a.file, cfg); _emit(out); return out.get("exit", 0)

    if not a.file:
        sys.stderr.write("error: this command requires a file\n"); return 2

    # Contract: EVERY file-requiring command emits exactly one JSON envelope.
    # Any load/analyze/serialize failure (missing/unreadable file, bad bytes,
    # validation error, non-finite JSON) degrades to a `status="error"` Result
    # with exit 2 instead of an uncaught traceback.
    from pydantic import ValidationError
    try:
        rt = _load(a.file, a.molar_mass, a.mass_mg)
        if a.command == "detect":
            df, cmap = canonicalize_columns(rt.df, rt.header)
            score, key = detect_probe(rt.header, set(df.columns), build_default_registry())
            prov = Provenance(file=str(a.file), sha256=_sha(a.file), app_version=rt.header.app_version, config=cfg.model_dump(mode="json"))
            res = Result(status="ok", confidence=score, data={"probe": key, "score": score}, provenance=prov)
        elif a.command in ("analyze", "fit", "hall", "hall-tdep"):
            res = _analyze(rt, cfg)
        elif a.command == "export":
            res = _analyze(rt, cfg)
            paths = export_result(res, a.out, fmt="csv")
            res = res.model_copy(update={"data": {**res.data, "exported": paths}})
        elif a.command == "report":
            res = _analyze(rt, cfg)
            rep = build_report(res)
            sys.stdout.write(rep["markdown"] + "\n"); return EXIT_CODES.get(res.status, 2)
        elif a.command == "plot":
            import pathlib as _pl
            from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle, PlotLayout, PlotEntry
            from cryosweep_core.plotting.render import render_kind, default_kind_for
            from cryosweep_core.plotting.presets import reconcile_layout
            from cryosweep_core.plotting.export import save_figure, export_plots, _FORMATS
            res = _analyze(rt, cfg)
            # Unknown keys in the two option files WARN instead of passing silently: pydantic
            # ignores extras, so a typo'd key or a spec boolean at the wrong nesting level
            # otherwise yields exit 0 and a figure without the requested feature. Warn, never
            # reject — presets/sidecars written against other versions must keep loading.
            _optfile_warnings = []
            if a.style_file:
                _style_raw = json.loads(_pl.Path(a.style_file).read_text())
                style = GlobalStyle.model_validate(_style_raw)
                _unk = unknown_keys(GlobalStyle, _style_raw)
                if _unk:
                    _optfile_warnings.append("--style-file: unknown key(s) ignored: "
                                             + ", ".join(_unk))
            else:
                style = GlobalStyle()
            if a.dpi is not None:
                style = style.model_copy(update={"dpi": a.dpi})
            formats = [f.strip().lower() for f in a.format.split(",") if f.strip()]
            bad = [f for f in formats if f not in _FORMATS]
            if bad or not formats:
                raise ValueError(f"--format: unsupported {bad or ['(empty)']}; allowed {'/'.join(_FORMATS)}")
            probe = (res.data or {}).get("probe")
            lay = None
            if a.layout_file:
                _lay_raw = json.loads(_pl.Path(a.layout_file).read_text())
                raw_lay = PlotLayout.model_validate(_lay_raw)
                _unk = unknown_keys(PlotLayout, _lay_raw)
                if _unk:
                    _optfile_warnings.append(
                        "--layout-file: unknown key(s) ignored: " + ", ".join(_unk)
                        + ' — expected shape {"plots": [{"kind": "...", "spec": {...}}]}')
                # a user-supplied layout file is deliberate (like a named preset): exact, no
                # newly-backed kinds appended
                lay = reconcile_layout(raw_lay, res, build_default_registry(), add_newly_backed=False)
                if raw_lay.plots and not lay.plots:
                    res = res.model_copy(update={"warnings": [*res.warnings,
                          f"--layout-file has no plots for probe '{probe}' (ignored)"]})
            if _optfile_warnings:
                for _w in _optfile_warnings:
                    sys.stderr.write(_w + "\n")
                res = res.model_copy(update={"warnings": [*res.warnings, *_optfile_warnings]})
            if a.all:
                if lay is None:      # no layout file: every kind known for this probe
                    lay = PlotLayout(plots=[PlotEntry(kind=k.key) for k in
                                            build_default_registry().plot_kinds_for(probe)])
                out = _pl.Path(a.out)
                paths = export_plots(res, lay, style, out.parent if out.parent != _pl.Path("") else _pl.Path("."),
                                     out.name, formats=formats, tight=a.tight)
                res = res.model_copy(update={"data": {**res.data, "plots": [str(p) for p in paths]}})
            else:
                kind = a.plot_kind or default_kind_for(probe)
                spec = (next((e.spec for e in lay.plots if e.kind == kind), PlotSpec())
                        if lay is not None else PlotSpec())
                stem = _pl.Path(a.out if not a.out.endswith(".png") else a.out[:-4])
                try:
                    fig = render_kind(res, kind, spec, style)
                    paths = [str(save_figure(fig, stem, style, spec=spec, fmt=f, tight=a.tight))
                             for f in formats]
                    res = res.model_copy(update={"data": {**res.data, "plot": paths[0],
                                                          "plots": paths}})
                except (ValueError, KeyError) as e:
                    res = res.model_copy(update={"data": {**res.data, "plot": None},
                                                 "warnings": [*res.warnings, f"plot kind '{kind}' unavailable: {e}"]})
        if cfg_warnings:
            res = res.model_copy(update={"warnings": [*res.warnings, *cfg_warnings]})
        _emit(res.model_dump(mode="json"))
        return EXIT_CODES.get(res.status, 2)
    except (FileNotFoundError, OSError, UnicodeError, ValueError, KeyError, ValidationError) as e:
        sys.stderr.write(f"error: {e}\n")
        res = Result(status="error", errors=[str(e)],
                     provenance=Provenance(file=str(a.file or ""), sha256="",
                                           app_version=None, config=cfg.model_dump(mode="json")))
        _emit(res.model_dump(mode="json"))
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
