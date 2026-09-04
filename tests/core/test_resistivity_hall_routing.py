"""Hall-channel routing in ResistivityAnalyzer (spec 2026-07-02).

A clear odd-in-B winner channel is Hall-wired, not longitudinal: its "resistivity"
is physically meaningless, so by default it is excluded from the bridge loop and
reported via a `hall_channel_excluded` capability + additive ResistivityData fields.
Ambiguous detection or exclude_hall_channel=False reproduce today's behavior exactly.
"""
import csv
import pathlib
import pytest
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.resistivity import ResistivityAnalyzer
from cryosweep_core.io.export import export_result
from cryosweep_core.reports import build_report
from cryosweep_core.plotting.catalog import BUILTIN_PLOTKINDS

FIX = pathlib.Path(__file__).parent / "fixtures"
KINDS = {k.key: k for k in BUILTIN_PLOTKINDS}


def _analyze(path, **res_cfg):
    cfg = RunConfig.load(resistivity=res_cfg) if res_cfg else RunConfig()
    return ResistivityAnalyzer().analyze(load_dat(str(path)), cfg)


def _cap(res, name="hall_channel_excluded"):
    return next((c for c in res.data["capabilities"] if c["name"] == name), None)


def test_config_defaults():
    cfg = RunConfig()
    assert cfg.resistivity.exclude_hall_channel is True
    assert cfg.resistivity.hall_channel_override is None


# ---- mixed synthetic: hall_synth has Ch1 = Hall (odd fraction ~0.81), Ch2 = longitudinal
def test_mixed_synth_excludes_hall_bridge(hall_synth_path):
    res = _analyze(hall_synth_path)
    assert [b["channel"] for b in res.data["bridges"]] == [2]
    assert res.data["excluded_hall_channel"] == 1
    assert res.data["excluded_hall_source"] == "detected"
    cap = _cap(res)
    assert cap is not None and cap["applicable"] is True
    assert "Ch1" in cap["reason"] and "0.81" in cap["reason"]


def test_mixed_synth_report_line_and_csv(tmp_path, hall_synth_path):
    res = _analyze(hall_synth_path)
    md = build_report(res)["markdown"]
    assert "Hall-channel routing" in md and "Ch1" in md
    out = export_result(res, str(tmp_path / "out"), fmt="csv")
    with open(out["derived"]) as f:
        assert {r["channel"] for r in csv.DictReader(f)} == {"2"}
    with open(out["curves"]) as f:
        assert {r["bridge"] for r in csv.DictReader(f)} == {"2"}


# ---- ambiguous detection: act_synth has two even channels -> byte-identical behavior
def test_ambiguous_synth_unchanged(act_synth_path):
    on = _analyze(act_synth_path)
    off = _analyze(act_synth_path, exclude_hall_channel=False)
    assert on.data["excluded_hall_channel"] is None
    assert on.data["excluded_hall_source"] == ""
    assert _cap(on) is None
    assert on.data == off.data
    assert build_report(on)["markdown"] == build_report(off)["markdown"]


# ---- feature off: today's behavior
def test_exclude_off_keeps_both_bridges(hall_synth_path):
    res = _analyze(hall_synth_path, exclude_hall_channel=False)
    assert sorted(b["channel"] for b in res.data["bridges"]) == [1, 2]
    assert res.data["excluded_hall_channel"] is None
    assert _cap(res) is None


# ---- override: force-excluded regardless of detection
def test_override_beats_detection(hall_synth_path):
    # detection picks Ch1; user says Ch2 -> Ch2 excluded, disagreement visible in reason
    res = _analyze(hall_synth_path, hall_channel_override=2)
    assert [b["channel"] for b in res.data["bridges"]] == [1]
    assert res.data["excluded_hall_channel"] == 2
    assert res.data["excluded_hall_source"] == "override"
    cap = _cap(res)
    assert "override" in cap["reason"] and "Ch1" in cap["reason"]


def test_override_works_when_detection_ambiguous(act_synth_path):
    res = _analyze(act_synth_path, hall_channel_override=2)
    assert [b["channel"] for b in res.data["bridges"]] == [1]
    assert res.data["excluded_hall_source"] == "override"
    assert "no clear winner" in _cap(res)["reason"]


# ---- rule 5: exclusion must never leave zero bridges
def _write_single_channel_hall(path):
    """One field loop, single populated bridge whose signal is strongly odd in B."""
    header = ("[Header]\nTITLE,single_hall\nBYAPP, Resistivity, 2.0, 1.0\n"
              "INFO, 1, Sample1 Cross Section\nINFO, 1, Sample1 Length\n[Data]\n"
              "Comment,Time Stamp (sec),Status (code),Temperature (K),Magnetic Field (Oe),"
              "Bridge 1 Resistivity (Ohm-m),Bridge 1 Resistance (Ohms)\n")
    rows = []
    n = 81
    for i in list(range(n)) + list(range(n - 1, -1, -1)):
        h = -90000.0 + i * (180000.0 / (n - 1))
        r = -5.0e-4 * (h / 10000.0) + 1.0e-3        # odd Hall + small even offset
        rows.append(f",0,,10.0000,{h:.4f},{r:.8e},{r:.8e}\n")
    path.write_text(header + "".join(rows))


def test_single_channel_hall_file_is_kept_with_warning(tmp_path):
    p = tmp_path / "single_hall.dat"
    _write_single_channel_hall(p)
    res = _analyze(p)
    assert [b["channel"] for b in res.data["bridges"]] == [1]   # never zero bridges
    assert res.data["excluded_hall_channel"] is None
    cap = _cap(res)
    assert cap is not None and cap["applicable"] is False
    assert "only populated bridge" in cap["reason"]


# ---- real Hall-wired file: Ch1 Hall-wired, Ch2 MR
def test_real_file_excludes_hall_channel_analyzes_the_other(hall_real_path):
    if not hall_real_path.exists():
        pytest.skip("real Hall measurement file not present (gitignored)")
    res = _analyze(hall_real_path)
    assert [b["channel"] for b in res.data["bridges"]] == [2]
    assert res.data["excluded_hall_channel"] == 1
    assert res.data["excluded_hall_source"] == "detected"
    off = _analyze(hall_real_path, exclude_hall_channel=False)
    assert sorted(b["channel"] for b in off.data["bridges"]) == [1, 2]
    # MR legend series count halves (one channel instead of two)
    for kind in ("resistivity_mr", "resistivity_mr_pct"):
        n_on = len(KINDS[kind].series(res))
        n_off = len(KINDS[kind].series(off))
        assert n_on * 2 == n_off, f"{kind}: {n_on} vs {n_off}"
