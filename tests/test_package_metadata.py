"""The published wheel and its metadata tell users and tools what the package offers."""

import subprocess
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]


def test_wheel_ships_the_py_typed_marker(tmp_path):
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path), str(ROOT)],
        check=True, capture_output=True, timeout=300,
    )
    (wheel,) = tmp_path.glob("marqov-*.whl")
    assert "marqov/py.typed" in zipfile.ZipFile(wheel).namelist()


def test_typed_classifier_is_declared():
    assert "Typing :: Typed" in PROJECT["classifiers"]


def test_documentation_url_points_at_the_docs_on_main():
    assert PROJECT["urls"]["Documentation"] == "https://github.com/marqov-dev/marqov-sdk/tree/main/docs"
    assert (ROOT / "docs").is_dir()

