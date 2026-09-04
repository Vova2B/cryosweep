import csv, pathlib
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.acms import ACMSAnalyzer
from cryosweep_core.io.export import export_result

FEATURE_HEADER = ["feature_type", "frequency_hz", "amplitude_oe", "field_oe", "direction",
                  "tc_onset_k", "tc_mid_k", "drop_emu_per_oe", "chi_dprime_peak_t_k",
                  "t_f_k", "prominence", "low_confidence", "reasons"]


def test_chi_and_features_csv_written(tmp_path, acms_real_path):
    r = ACMSAnalyzer().analyze(load_dat(acms_real_path), RunConfig())
    out = export_result(r, tmp_path / "acms")
    chi = pathlib.Path(out["chi"]); feat = pathlib.Path(out["features"])
    assert chi.exists() and feat.exists()
    with chi.open() as f:
        head = next(csv.reader(f))
        assert head[:7] == ["frequency_hz", "amplitude_oe", "field_oe", "direction",
                            "T", "chi_prime", "chi_dprime"]
    with feat.open() as f:
        assert next(csv.reader(f)) == FEATURE_HEADER      # empty-but-headered on the null file
