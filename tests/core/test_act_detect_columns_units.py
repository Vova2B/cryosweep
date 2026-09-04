import pytest

class _H:
    def __init__(self, app):
        self.app = app

def test_resistivity_detector_matches_qd_and_act():
    from cryosweep_core.detect.probe import ResistivityDetector
    # QD format (app "Resistivity" + bridge resistivity column)
    assert ResistivityDetector.matches(_H("Resistivity"), {"Bridge 1 Resistivity (Ohm-m)"}) == 1.0
    # ACT format (app "ACTRANSPORT" + Res. ch1 (ohm-cm) column)
    assert ResistivityDetector.matches(_H("ACTRANSPORT"), {"Res. ch1 (ohm-cm)"}) == 1.0
    # no mis-fire on other probes
    assert ResistivityDetector.matches(_H("VSM"), {"Moment (emu)"}) == 0.0
    assert ResistivityDetector.matches(_H("HeatCapacity"), {"Samp HC (mJ/mole-K)"}) == 0.0

def test_canonicalize_act_resistivity_unit():
    import pandas as pd
    from cryosweep_core.io.columns import canonicalize_columns, UNIT_OHM_CM, UNIT_OHM_M
    class _Hdr: pass
    df = pd.DataFrame({"Temperature (K)": [2.0], "Magnetic Field (Oe)": [0.0],
                       "Res. ch1 (ohm-cm)": [3e-4], "Res. ch2 (ohm-cm)": [1e-4]})
    _, cmap = canonicalize_columns(df, _Hdr())
    assert cmap.logical["resistivity_ch1"] == "Res. ch1 (ohm-cm)"
    assert cmap.logical["resistivity_ch2"] == "Res. ch2 (ohm-cm)"
    assert cmap.unit["resistivity_ch1"] == UNIT_OHM_CM
    assert cmap.unit["resistivity_ch2"] == UNIT_OHM_CM
    # QD resistivity rule still records Ohm-m
    df2 = pd.DataFrame({"Bridge 1 Resistivity (Ohm-m)": [1e-6]})
    _, cmap2 = canonicalize_columns(df2, _Hdr())
    assert cmap2.unit["resistivity_ch1"] == UNIT_OHM_M

def test_bridge_rho_unit_aware():
    import pandas as pd
    from cryosweep_core.analyzers.resistivity import _bridge_rho
    from cryosweep_core.model import ColumnMap
    from cryosweep_core.config import RunConfig
    cfg = RunConfig.load()                          # geometry incomplete -> instrument-column path
    df = pd.DataFrame({"R1": [1e-6, 1e-6], "R2": [3e-4, 3e-4]})
    # QD: Ohm-m -> *100
    cmap_qd = ColumnMap(logical={"resistivity_ch1": "R1"}, unit={"resistivity_ch1": "Ohm-m"})
    rho, src = _bridge_rho(df, cmap_qd, cfg, 1)
    assert src == "instrument_column"
    assert rho[0] == pytest.approx(1e-4)            # 1e-6 * 100
    # ACT: Ohm-cm -> *1
    cmap_act = ColumnMap(logical={"resistivity_ch2": "R2"}, unit={"resistivity_ch2": "Ohm-cm"})
    rho2, src2 = _bridge_rho(df, cmap_act, cfg, 2)
    assert src2 == "instrument_column"
    assert rho2[0] == pytest.approx(3e-4)           # 3e-4 * 1 (NOT *100)
