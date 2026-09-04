"""The CLA check is only as good as the sentence it matches on.

The bot records a signature when a comment matches `custom-pr-sign-comment` in
`.github/workflows/cla.yml`. That string and the one CLA.md tells contributors to post are two
copies of the same sentence in two files, and nothing at the YAML level ties them together: a
later reword of either one leaves contributors posting a sentence the bot ignores, which reads
to them as "I signed and the check is still red". These pin the copies to each other, and pin
the links the documents make to each other.

Repo-metadata tests, not analysis tests -- they need no data and run in a bare clone.
"""
import pathlib
import re
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SIGN_SENTENCE = "I have read the cryosweep CLA and I hereby sign the Individual CLA."


def _read(rel):
    p = ROOT / rel
    assert p.exists(), f"{rel} must ship: it is linked from the contribution flow"
    return p.read_text(encoding="utf-8")


def test_workflow_sign_comment_is_the_sentence_cla_md_tells_people_to_post():
    wf = _read(".github/workflows/cla.yml")
    m = re.search(r"custom-pr-sign-comment:\s*'([^']*)'", wf)
    assert m, "the workflow must set custom-pr-sign-comment explicitly, not rely on the default"
    assert m.group(1) == SIGN_SENTENCE
    # CLA.md quotes it as a blockquote under "How to sign".
    assert f"> {SIGN_SENTENCE}" in _read("CLA.md")


def test_workflow_gate_would_admit_that_sentence():
    """The `if:` prefix guard must be a prefix OF the sentence, or no comment ever reaches the
    action and the check silently never records anything."""
    wf = _read(".github/workflows/cla.yml")
    m = re.search(r"contains\(github\.event\.comment\.body, '([^']*)'\)", wf)
    assert m, "the comment path must be guarded by a contains() on the signing text"
    assert SIGN_SENTENCE.startswith(m.group(1))


@pytest.mark.parametrize("rel", ["CLA.md", "CONTRIBUTING.md", "LICENSE", "COMMERCIAL.md",
                                 ".github/PULL_REQUEST_TEMPLATE.md"])
def test_contribution_documents_ship(rel):
    assert _read(rel).strip()


def test_relative_links_between_the_contribution_documents_resolve():
    """A dead relative link in CLA.md/CONTRIBUTING.md is invisible until a contributor clicks it."""
    broken = []
    for rel in ("CLA.md", "CONTRIBUTING.md", "README.md", "COMMERCIAL.md"):
        base = (ROOT / rel).parent
        for target in re.findall(r"\]\((?!https?:|mailto:|#)([^)#]+)", _read(rel)):
            if not (base / target).exists():
                broken.append(f"{rel} -> {target}")
    assert not broken, broken
