"""The duplicate-method AST gate (spec C3) is itself under the suite."""
import subprocess, sys, textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from check_dup_methods import find_duplicates  # noqa: E402

DUP = textwrap.dedent("""\
    class A:
        def f(self): pass
        def g(self): pass
        def f(self): pass
    class B:
        def f(self): pass
""")

CLEAN = textwrap.dedent("""\
    class A:
        def f(self): pass
        class Inner:
            def f(self): pass   # same name in a NESTED class is not a duplicate
    async def f(): pass         # module-level function is not a method
""")


def test_detects_duplicate_method():
    assert find_duplicates(DUP, "x.py") == ["x.py:A:f"]


def test_clean_module_and_nested_class_are_silent():
    assert find_duplicates(CLEAN, "y.py") == []


def test_cli_exit_codes(tmp_path):
    (tmp_path / "bad.py").write_text(DUP)
    script = Path(__file__).resolve().parents[2] / "tools" / "check_dup_methods.py"
    r = subprocess.run([sys.executable, str(script), str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode == 1
    assert "bad.py:A:f" in r.stdout
    (tmp_path / "bad.py").write_text(CLEAN)
    r2 = subprocess.run([sys.executable, str(script), str(tmp_path)],
                        capture_output=True, text=True)
    assert r2.returncode == 0
    assert r2.stdout.strip() == ""


def test_real_tree_is_clean():
    """Spec C3's 'UNVERIFIED until the script exists' claim, made a permanent gate."""
    script = Path(__file__).resolve().parents[2] / "tools" / "check_dup_methods.py"
    r = subprocess.run([sys.executable, str(script)],
                       cwd=script.parents[1], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout
