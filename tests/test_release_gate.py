"""Keep publication dependent on successful verification of built artifacts."""
import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_gate", ROOT / "tools/check_release_gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def workflow():
    return yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())


def test_release_gate():
    gate.check_gate(workflow())


@pytest.mark.parametrize("publisher", ["publish-testpypi", "publish-pypi"])
def test_missing_verification_blocks_lint(publisher):
    value = workflow()
    value["jobs"][publisher]["needs"] = ["build"]
    with pytest.raises(AssertionError, match="bypasses"):
        gate.check_gate(value)


@pytest.mark.parametrize("status", ["always", "failure", "cancelled"])
def test_status_override_blocks_lint(status):
    value = workflow()
    value["jobs"]["publish-pypi"]["if"] = f"{status}()"
    with pytest.raises(AssertionError, match="overrides"):
        gate.check_gate(value)
