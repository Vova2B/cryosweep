import pytest
from cryosweep_core.io.loader import load_dat
from cryosweep_core.io.header import parse_header

def test_bare_csv_recognized_and_loads_all_rows(tmp_path):
    p = tmp_path / "bare.dat"
    p.write_text(
        "Time,Field (Oe),Temperature (K),Long Moment (emu),Long Scan Std Dev\n"
        "1,500,3.0,2.6e-4,1e-7\n"
        "1,500,5.0,2.7e-4,1e-7\n"
        "1,500,7.0,2.8e-4,1e-7\n"
    )
    h = parse_header(str(p))
    assert h.data_line == -1 and h.bare_csv is True
    rt = load_dat(str(p))
    assert len(rt.df) == 3                       # all rows kept (skip=0), not magic-27
    assert "Long Moment (emu)" in rt.df.columns

def test_qd_file_with_data_marker_not_bare(tmp_path):
    p = tmp_path / "qd.dat"
    p.write_text(
        "[Header]\nBYAPP,VSM,1.0,1.0\n[Data]\n"
        "Temperature (K),Moment (emu)\n2.0,1.0e-4\n3.0,1.1e-4\n"
    )
    h = parse_header(str(p))
    assert h.bare_csv is False and h.data_line >= 0
    rt = load_dat(str(p))
    assert len(rt.df) == 2 and "Moment (emu)" in rt.df.columns


class _H:
    def __init__(self, app):
        self.app = app

def test_vsm_detector_strong_fingerprint():
    from cryosweep_core.detect.probe import VSMDetector, HeatCapacityDetector, ResistivityDetector
    # QD VSM: app token + moment fingerprint -> unchanged 1.0
    assert VSMDetector.matches(_H("VSM"), {"Moment (emu)", "Magnetic Field (Oe)"}) == 1.0
    # MPMS: no app, but BOTH strong columns present (+0.6) plus 'moment (emu)' substring (+0.2) = 0.8
    s = VSMDetector.matches(_H(None), {"Long Moment (emu)", "Long Scan Std Dev",
                                       "Field (Oe)", "Temperature (K)"})
    assert s == pytest.approx(0.8) and s >= 0.5
    # only ONE strong column -> strong path does NOT fire (needs all) -> just the +0.2 substring
    assert VSMDetector.matches(_H(None), {"Long Moment (emu)"}) == pytest.approx(0.2)
    # non-VSM detectors must be unaffected by MPMS columns (empty strong_fingerprint no-ops)
    assert HeatCapacityDetector.matches(_H(None), {"Long Moment (emu)", "Long Scan Std Dev"}) == 0.0
    assert ResistivityDetector.matches(_H(None), {"Long Moment (emu)", "Long Scan Std Dev"}) == 0.0


def test_canonicalize_mpms_columns():
    import pandas as pd
    from cryosweep_core.io.columns import canonicalize_columns
    df = pd.DataFrame({"Temperature (K)": [3.0], "Field (Oe)": [500.0],
                       "Long Moment (emu)": [2.6e-4], "Long Scan Std Dev": [1e-7]})
    _, cmap = canonicalize_columns(df, None)
    assert cmap.logical["temperature"] == "Temperature (K)"
    assert cmap.logical["field"] == "Field (Oe)"
    assert cmap.logical["moment"] == "Long Moment (emu)"
    # QD VSM columns must still map (additive change)
    df2 = pd.DataFrame({"Magnetic Field (Oe)": [1000.0], "Moment (emu)": [1.0e-4]})
    _, cmap2 = canonicalize_columns(df2, None)
    assert cmap2.logical["field"] == "Magnetic Field (Oe)"
    assert cmap2.logical["moment"] == "Moment (emu)"
