"""The resistivity power-law fit declines rather than publishing a non-measurement
(owner-approved 2026-09-01, follow-on to the SC low-T window gate).

Two defects, both measured on shipped files:

(1) The exponent was published whatever its uncertainty. On examples/resistivity_superconductor
    the SC gate narrows the window to 9-30 K of an almost flat normal state, giving
    n = 0.618 +/- 1.969 at r2 = 0.428 — sigma 3.2x the value — reported as a CLEAN fit with a
    rho0 = 92.2 uOhm-cm that had never been reported before.

(2) The ladder's fixed 10/15/20/30 K cutoffs are measured from 0 K, so once a transition gates
    the window from below they land inside the degenerate stub and pin at the n = 0.5 search
    bound. Rungs pinned at the SAME bound agree exactly, so n_spread read 0.118 while n truly
    ran 0.5 -> 0.99 as the upper limit went 15 -> 300 K: a bound-pinned ladder faked stability.

Rule: `n_at_bound`/`n_unresolved` (sigma_n >= |n|) decline — no n, no rho0, no fit line, blank
CSV cells, GUI says why. Rungs span the window ACTUALLY fitted when gated, and bound-pinned
rungs are excluded from the spread. `rho0_unresolved` is deliberately NOT a decline flag."""
import csv
import pathlib

from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.resistivity import ResistivityAnalyzer
from cryosweep_core.fitting.transport import POWER_LAW_DECLINE_FLAGS, NO_FIT_LINE_FLAGS
from cryosweep_core.io.export import export_result

ROOT = pathlib.Path(__file__).resolve().parents[2]
SC_EXAMPLE = str(ROOT / "examples/resistivity_superconductor.dat")


def _bridge1(path):
    r = ResistivityAnalyzer().analyze(load_dat(path), RunConfig())
    return r, r.data["bridges"][0]


def test_unresolved_exponent_declines_on_the_shipped_example():
    """(1) sigma_n >= |n| -> not a measurement: flagged, and rho0 withheld."""
    _, b = _bridge1(SC_EXAMPLE)
    pl = b["power_law"]
    assert pl["sigma"]["n"] >= abs(pl["params"]["n"])          # the condition itself
    assert "n_unresolved" in pl["quality_flags"]
    assert b["residual_rho"] is None, "a declined fit must not publish rho0"


def test_declined_fit_writes_blank_csv_cells_but_keeps_the_reason(tmp_path):
    """The CSV is the surface the owner publishes from: no bare number, reason retained."""
    r, _ = _bridge1(SC_EXAMPLE)
    out = export_result(r, str(tmp_path / "sc"), fmt="csv")
    rows = {row["channel"]: row for row in csv.DictReader(open(out["derived"]))}
    r1 = rows["1"]
    assert r1["power_law_n"] == "" and r1["power_law_A"] == "" and r1["power_law_r2"] == ""
    assert r1["residual_rho_ohm_cm"] == ""
    assert "n_unresolved" in r1["power_law_flags"].split(";")
    # ... while the channel whose fit IS resolved still publishes its number
    assert float(rows["2"]["power_law_n"]) > 0


def test_rho0_unresolved_is_not_a_decline_flag():
    """Guard against over-reach: rho0_unresolved withholds rho0 and the fit LINE, but n
    remains a real (window-sensitive) measurement and must still be published."""
    assert "rho0_unresolved" not in POWER_LAW_DECLINE_FLAGS
    assert "rho0_unresolved" in NO_FIT_LINE_FLAGS


def test_bound_pinned_rungs_do_not_fake_a_stable_window():
    """(2) Several rungs pinned at the same bound agree exactly; excluding them is what stops
    n_spread reading 'stable' for an exponent that is not determined at all."""
    _, b = _bridge1(SC_EXAMPLE)
    lad = b["power_law_ladder"]
    assert [e["at_bound"] for e in lad] == [True, True, True, False]
    # the three pinned rungs agree EXACTLY, because they are the same bound, not a measurement
    assert {round(e["n"], 6) for e in lad if e["at_bound"]} == {0.5}
    assert b["power_law_n_spread"] is None, "<2 resolved rungs -> no spread claim, never 0.0"
    assert "ladder_incomplete" in b["power_law"]["quality_flags"]
    assert lad, "the pinned rungs stay IN the ladder, visible with at_bound=True"


def test_gated_window_uses_relative_rungs_and_ungated_keeps_absolute():
    """The SC example is gated at its 8.80 K onset, so its rungs span the fitted window;
    its normal-metal channel is ungated and keeps the absolute 10/15/20/30 K cutoffs."""
    r, b1 = _bridge1(SC_EXAMPLE)
    cuts1 = [e["cutoff_k"] for e in b1["power_law_ladder"]]
    cuts2 = [e["cutoff_k"] for e in r.data["bridges"][1]["power_law_ladder"]]
    assert cuts2 == [10.0, 15.0, 20.0, 30.0]                   # byte-identical path preserved
    assert cuts1 != cuts2 and min(cuts1) > 9.0                 # rungs start above the onset
    assert max(cuts1) == 30.0                                  # ... and still end at the primary
