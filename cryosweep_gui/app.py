from __future__ import annotations

def build_window():
    """Construct (but do not exec) the main window. Importable in tests."""
    import sys
    from PySide6.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication(sys.argv)  # keep ref alive; must precede mpl QtAgg
    import matplotlib
    # switch_backend dereferences several matplotlib submodules (`backend_bases`, `backends`, ...);
    # when matplotlib was imported earlier in-process (e.g. v1 main.py/pyplot) those parent attrs can
    # be left unbound (mpl 3.11 + py3.14), and a cached re-import won't restore them, so `use` raises
    # AttributeError. Re-bind every already-imported top-level submodule from sys.modules first.
    import sys
    _pfx = "matplotlib."
    for _name, _mod in list(sys.modules.items()):
        if _mod is not None and _name.startswith(_pfx) and "." not in _name[len(_pfx):]:
            matplotlib.__dict__.setdefault(_name[len(_pfx):], _mod)
    matplotlib.use("QtAgg")             # set backend BEFORE any pyplot/canvas use; must run after QApp
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow()
    win._qapp = _app                    # anchor QApp to window lifetime so it is not GC'd
    return win

_NO_QT = (
    "cryosweep: the desktop GUI needs the optional Qt dependency, which is not installed.\n"
    "    pip install 'cryosweep[gui]'\n"
    "The command-line interface needs no Qt and is available now as `cryosweep`."
)


def run(argv=None):
    import sys
    # `pip install cryosweep` deliberately omits Qt, but console entry points are unconditional:
    # this script exists either way. Without the guard the user meets a ModuleNotFoundError
    # traceback instead of the one line that fixes it.
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:
        print(_NO_QT, file=sys.stderr)
        return 2
    app = QApplication.instance() or QApplication(argv or sys.argv)
    win = build_window()
    win.show()
    return app.exec()
