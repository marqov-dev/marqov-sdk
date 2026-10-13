"""Guard the supported Python matrix and required CI gate against silent drift."""

import json
from pathlib import Path
import runpy

from importlib import import_module

yaml = import_module("yaml")

ROOT = Path(__file__).resolve().parents[1]


def test_qualified_profiles_are_required_and_restrictions_are_explicit() -> None:
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
    matrix = yaml.safe_load((ROOT / ".github/workflows/python-compatibility.yml").read_text())[
        "jobs"
    ]["qualify"]
    profiles = runpy.run_path(str(ROOT / "tools/qualify_python.py"))["PROFILES"]
    supported = set(matrix["strategy"]["matrix"]["profile"])
    unsupported = set(ci["python313-unsupported"]["strategy"]["matrix"]["profile"])
    assert supported.isdisjoint(unsupported)
    assert supported | unsupported == set(profiles)
    assert "python313-qualified" in ci["ci-success"]["needs"]
    assert ci["python313-qualified"]["uses"] == "./.github/workflows/python-compatibility.yml"
    assert not matrix.get("continue-on-error", False)
    assert not ci["python313-qualified"].get("continue-on-error", False)
    assert ci["python313-unsupported"]["continue-on-error"] is True
    assert "python313-unsupported" not in ci["ci-success"]["needs"]


def test_published_evidence_covers_every_profile_without_failed_supported_paths() -> None:
    record = json.loads((ROOT / "docs/qualification/python313.json").read_text())
    profiles = runpy.run_path(str(ROOT / "tools/qualify_python.py"))["PROFILES"]
    matrix = yaml.safe_load((ROOT / ".github/workflows/python-compatibility.yml").read_text())[
        "jobs"
    ]["qualify"]
    supported = set(matrix["strategy"]["matrix"]["profile"])
    rows = {r["results"][0]["profile"]: r["results"][0] for r in record["linux"]}
    assert set(rows) == set(profiles)
    for profile in supported:
        row = rows[profile]
        assert row["install"] == row["probe"] == row["dependency_check"] == "passed"
        assert row["tests"] in {"passed", "not_run"}
        assert not row.get("runner_error")
    assert record["live_provider_execution"] is False
