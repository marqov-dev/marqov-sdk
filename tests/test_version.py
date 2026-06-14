"""Asserts that marqov.__version__ matches pyproject.toml."""

import tomllib
import pathlib
import marqov


def test_version_matches_pyproject():
    pyproject = tomllib.loads(
        (pathlib.Path(__file__).parent.parent / "pyproject.toml").read_text()
    )
    assert marqov.__version__ == pyproject["project"]["version"]
