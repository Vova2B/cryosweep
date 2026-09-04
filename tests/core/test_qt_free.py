import subprocess, sys, textwrap

def test_core_imports_with_qt_blocked():
    code = textwrap.dedent('''
        import importlib, pkgutil, sys
        BANNED = ("PyQt6", "PyQt5", "PySide6", "PySide2", "utils", "magnetization")
        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BANNED:
                    raise ImportError(f"BANNED import: {name}")
                return None
        sys.meta_path.insert(0, Blocker())
        import cryosweep_core
        for m in pkgutil.walk_packages(cryosweep_core.__path__, cryosweep_core.__name__ + "."):
            importlib.import_module(m.name)
        # plotting + cli must also import clean under the Qt ban
        import cryosweep_core.plotting.render
        import cryosweep_cli.__main__
        for mod in ("PyQt6", "PyQt5", "PySide6"):
            assert mod not in sys.modules, f"{mod} leaked in"
        print("CORE_OK")
    ''')
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert "CORE_OK" in out.stdout, (out.stdout + out.stderr)
