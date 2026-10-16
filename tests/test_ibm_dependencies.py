"""Declared qiskit-ibm-runtime floor must cover the SamplerV2 import in IBMExecutor."""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"
SAMPLER_V2_FLOOR = Version("0.21.0")


def _declared_floors(extra: str) -> list[Version]:
    data = tomllib.loads(PYPROJECT.read_text())
    floors = []
    for spec in data["project"]["optional-dependencies"][extra]:
        req = Requirement(spec)
        if req.name != "qiskit-ibm-runtime":
            continue
        lowers = [Version(s.version) for s in req.specifier if s.operator == ">="]
        assert lowers, f"{extra}: qiskit-ibm-runtime has no >= floor: {spec}"
        floors.append(max(lowers))
    return floors


def test_ibm_extra_floor_includes_sampler_v2():
    floors = _declared_floors("ibm")
    assert floors, "[ibm] must declare qiskit-ibm-runtime"
    assert all(f >= SAMPLER_V2_FLOOR for f in floors)


def test_all_extra_floor_includes_sampler_v2():
    floors = _declared_floors("all")
    assert floors, "[all] must declare qiskit-ibm-runtime"
    assert all(f >= SAMPLER_V2_FLOOR for f in floors)


def test_sampler_v2_importable():
    # Deliberately not skipped: [all] promises this import (see IBMExecutor).
    from qiskit_ibm_runtime import SamplerV2

    assert callable(SamplerV2)
