from __future__ import annotations
import re
import unicodedata
from cryosweep_core.model import ColumnMap

UNIT_OHM_M = "Ohm-m"
UNIT_OHM_CM = "Ohm-cm"
UNIT_MICROOHM_CM = "uOhm-cm"        # micro-ohm-cm (Origin "dc rho" exports); ->Ohm-cm is x1e-6

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s).strip().lower()

# (normalized real-name fragment) -> (logical, unit)
_TEMP = {"sample temp (kelvin)": "K", "temperature (k)": "K", "t (k)": "K",
         "sample temp. (k)": "K"}
_FIELD = {"field (oersted)": "Oe", "magnetic field (oe)": "Oe", "field (oe)": "Oe"}
_ANGLE = {"sample position (degrees)": "deg"}

def canonicalize_columns(df, header):
    norm = {_norm(c): c for c in df.columns}
    logical: dict = {}
    unit: dict = {}
    for key, real in norm.items():
        if key in _TEMP and "temperature" not in logical:
            logical["temperature"], unit["temperature"] = real, _TEMP[key]
        elif key in _FIELD and "field" not in logical:
            logical["field"], unit["field"] = real, _FIELD[key]
        elif key in _ANGLE and "angle" not in logical:
            logical["angle"], unit["angle"] = real, _ANGLE[key]
        m = re.match(r"bridge (\d+) resistivity", key)
        if m:
            ch = m.group(1)
            logical[f"resistivity_ch{ch}"] = real
            unit[f"resistivity_ch{ch}"] = UNIT_OHM_M
        ma = re.match(r"res\. ch(\d+) \(ohm-cm\)", key)
        if ma:
            ch = ma.group(1)
            logical[f"resistivity_ch{ch}"] = real
            unit[f"resistivity_ch{ch}"] = UNIT_OHM_CM
        # Generic bare-file resistivity column, e.g. Origin "dc rho" exports:
        #   "Resistivity He4+He3 (mikroOhm-cm)_H=0T_COOL" -> resistivity_ch1 (uOhm-cm)
        # No channel number in the name -> assign the next free channel (handles 1-2 columns;
        # wide-format multi-field exports are deferred). 'bridge N'/'res. chN' start with other
        # words, so the ^resistivity anchor keeps this from clobbering the QD/ACT paths above.
        mg = re.match(r"resistivity\b.*\((mikro|micro|u)?ohm-(cm|m)\)", key)
        if mg:
            scale, length = mg.group(1), mg.group(2)
            u = UNIT_OHM_M if length == "m" else (UNIT_MICROOHM_CM if scale else UNIT_OHM_CM)
            ch = 1
            while f"resistivity_ch{ch}" in logical:
                ch += 1
            logical[f"resistivity_ch{ch}"] = real
            unit[f"resistivity_ch{ch}"] = u
        mr = re.match(r"bridge (\d+) resistance", key)
        if mr:
            ch = mr.group(1)
            logical[f"resistance_ch{ch}"] = real
            unit[f"resistance_ch{ch}"] = "Ohm"
        # Instrument per-row std-dev columns (2026-08-10 spec §4): unit recorded so the
        # analyzer applies the SAME x100 (Ohm-m) / x1 (Ohm-cm) conversion as rho itself.
        ms = re.match(r"bridge (\d+) std\. dev\. \(ohm-m\)", key)
        if ms:
            ch = ms.group(1)
            logical[f"rho_std_bridge{ch}"] = real
            unit[f"rho_std_bridge{ch}"] = UNIT_OHM_M
        msa = re.match(r"res\. std\.dev\. ch(\d+)", key)
        if msa:
            ch = msa.group(1)
            logical[f"rho_std_ch{ch}"] = real
            unit[f"rho_std_ch{ch}"] = UNIT_OHM_CM
        if key == "samp hc (mj/mole-k)":
            logical["hc_sample"], unit["hc_sample"] = real, "mJ/mol-K"
        if key == "moment (emu)" or key == "long moment (emu)":
            logical["moment"], unit["moment"] = real, "emu"
        if key.startswith("m. std. err"):
            logical["moment_err"], unit["moment_err"] = real, "emu"
        if key == "m' (emu)":
            logical["m_prime"], unit["m_prime"] = real, "emu"
        if key == "m'' (emu)":
            logical["m_dprime"], unit["m_dprime"] = real, "emu"
        if key == "m-dc (emu)":
            logical["m_dc"], unit["m_dc"] = real, "emu"
        if key == "m-std.dev. (emu)":
            logical["m_stddev"], unit["m_stddev"] = real, "emu"
        if key == "frequency (hz)":
            logical["frequency"], unit["frequency"] = real, "Hz"
        if key == "amplitude (oe)":
            logical["amplitude"], unit["amplitude"] = real, "Oe"
        # ---- Thermal Transport Option (TTO). `Resistivity (Ohm-m)` deliberately ALSO keeps
        # its existing resistivity_ch1 mapping above; rho_tto is a second, additive logical
        # name for the same real column.
        if key == "conductivity (w/k-m)":
            logical["kappa"], unit["kappa"] = real, "W/K-m"
        if key == "cond. std.dev.":
            logical["kappa_std"], unit["kappa_std"] = real, "W/K-m"
        if key == "seebeck coef. (v/k)":        # _norm strips the micro sign
            logical["seebeck"], unit["seebeck"] = real, "uV/K"
        if key == "seebeck std.dev.":
            logical["seebeck_std"], unit["seebeck_std"] = real, "uV/K"
        if key == "resistivity (ohm-m)":
            logical["rho_tto"], unit["rho_tto"] = real, UNIT_OHM_M
        if key == "resist std.dev.":
            logical["rho_tto_std"], unit["rho_tto_std"] = real, UNIT_OHM_M
        if key == "figure of merit zt":
            logical["zt"], unit["zt"] = real, "1"
        if key == "merit std.dev.":
            logical["zt_std"], unit["zt_std"] = real, "1"
    return df, ColumnMap(logical=logical, unit=unit)
