"""Asserts marqov.__version__ stays consistent with the distribution metadata.

The version is single-sourced from marqov/__init__.py (``__version__``) and read
by hatchling at build time (``[tool.hatch.version] path``), so pyproject.toml no
longer carries a static ``version`` field — it is ``dynamic``. This test therefore
compares the source ``__version__`` against the *installed distribution* metadata
(what hatchling produced), which is the PUG-recommended drift guard. (It previously
read ``pyproject["project"]["version"]``, which no longer exists now that the
version is dynamic.)
"""

import importlib.metadata

import marqov


def test_version_is_single_sourced():
    assert marqov.__version__ == importlib.metadata.version("marqov")


def test_security_md_lists_current_minor_as_supported():
    """SECURITY.md's supported-versions table must list the current MAJOR.MINOR as supported (marqov-sdk#167).

    Asserts only that the current minor has a ``Yes`` row; which older minors are
    listed is the maintainer's call and is deliberately not pinned.
    """
    import re
    from pathlib import Path

    major, minor = marqov.__version__.split(".")[:2]
    text = (Path(__file__).resolve().parent.parent / "SECURITY.md").read_text()
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in text.splitlines()
        if line.startswith("|") and not set(line) <= set("|-: ")
    ]
    supported = [r[0] for r in rows[1:] if len(r) >= 2 and r[1].lower().startswith("yes")]
    assert any(re.fullmatch(rf"{major}\.{minor}\.(x|\*)", v) for v in supported), (
        f"SECURITY.md supported-versions table does not list {major}.{minor}.x as supported "
        f"(rows: {rows})"
    )
