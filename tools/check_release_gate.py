"""Regression check for release publication dependencies and safe conditions."""
from pathlib import Path

import yaml


def check_gate(workflow):
    jobs = workflow["jobs"]
    assert "verify" in jobs
    for name in ("publish-testpypi", "publish-pypi"):
        job = jobs[name]
        assert {"build", "verify"}.issubset(job["needs"]), f"{name} bypasses verification"
        condition = job.get("if", "").lower().replace(" ", "")
        assert not any(f"{status}(" in condition for status in ("always", "failure", "cancelled")), \
            f"{name} overrides the success dependency gate"
    print("gate OK")


if __name__ == "__main__":
    check_gate(yaml.safe_load(Path(".github/workflows/release.yml").read_text()))
