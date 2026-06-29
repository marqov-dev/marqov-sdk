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
