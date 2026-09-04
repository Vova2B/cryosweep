from __future__ import annotations
import os
import pandas as pd
from pandas.errors import EmptyDataError, ParserError
from cryosweep_core.io.header import parse_header, _read_lines, _ENCODINGS
from cryosweep_core.model import RawTable

_LEGACY_NO_DATA_SKIP = 27   # legacy fallback for a QD file missing [Data]; currently unexercised


def _oneline(exc) -> str:
    """pandas ParserError messages carry embedded newlines; keep ours on one line."""
    return " ".join(str(exc).split())


def expand_user_path(path):
    """Expand a leading '~' in a user-supplied path string.

    The single boundary for ~ handling: applied where external path strings
    enter the program — CLI argv (cryosweep_cli.__main__) and pipeline step
    files (cryosweep_core.pipeline) — never re-applied downstream. A shell
    expands ~ interactively, but a path inside a pipeline JSON never meets a
    shell, so without this `{"file": "~/data/x.dat"}` fails with Errno 2.
    """
    return os.path.expanduser(path) if isinstance(path, str) else path

def load_dat(path) -> RawTable:
    header = parse_header(path)
    delim = ","
    if header.data_line >= 0:
        skip = header.data_line + 1          # QD path (unchanged)
    elif header.bare_csv:
        skip = 0                             # bare CSV: column header is line 0
        # Origin/"dc rho" exports are TAB-separated; QD/MPMS bare CSV is comma. Sniff the
        # column-header line so a tab file isn't read as a single unparseable column.
        if header.raw_lines and "\t" in header.raw_lines[0]:
            delim = "\t"
    else:
        skip = _LEGACY_NO_DATA_SKIP          # preserved last-resort fallback
    last = None
    for enc in _ENCODINGS:
        try:
            # index_col=False: QD .dat rows often carry trailing empty fields (more values than
            # header names); without this pandas auto-promotes the leading columns to an index and
            # shifts every named column (e.g. 'Sample Temp' would hold Samp-HC-Err values). When
            # counts match this is a no-op.
            df = pd.read_csv(path, skiprows=skip, delimiter=delim, encoding=enc, index_col=False)
            return RawTable(df=df, header=header, path=str(path))
        except UnicodeDecodeError as e:
            last = e
        # Not encoding problems — re-trying other encodings can't help, and the raw
        # pandas message ("Error tokenizing data...", "No columns to parse from file")
        # tells the user nothing actionable. Say what the file is not (a QD .dat) and
        # keep the pandas detail in the message: it lands in the envelope's errors[].
        except EmptyDataError as e:
            what = ("the file is empty" if os.path.getsize(path) == 0
                    else "no data rows found")
            raise ValueError(f"cannot read {path} as a Quantum Design .dat file: "
                             f"{what} (pandas: {_oneline(e)})") from e
        except ParserError as e:
            raise ValueError(f"cannot read {path} as a Quantum Design .dat file "
                             f"(pandas: {_oneline(e)})") from e
    raise ValueError(f"Could not read {path}: {last}")
