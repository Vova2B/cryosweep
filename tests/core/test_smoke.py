import re


def test_import_cryosweep_core_is_clean():
    """The core imports on its own, and reports a plausible version.

    This used to assert a version LITERAL. That is precisely what made an earlier
    disagreement load-bearing and still green: the test pinned one of the version strings,
    so the suite agreed with itself while the package disagreed with pyproject. It also
    turns every release bump into a test edit, which is how a stale pin survives.

    Agreement between the four version strings is tests/core/test_version_consistency.py's
    job. Here we only check the shape, so this test has one reason to fail: the import.
    (Whether the import drags in Qt or matplotlib is checked properly, in a subprocess,
    by tests/core/test_qt_free.py -- asserting it here would be order-dependent, since
    another test may already have imported them into this process.)
    """
    import cryosweep_core

    assert re.fullmatch(r"\d+\.\d+\.\d+", cryosweep_core.__version__), (
        f"__version__ is {cryosweep_core.__version__!r}, not MAJOR.MINOR.PATCH"
    )
