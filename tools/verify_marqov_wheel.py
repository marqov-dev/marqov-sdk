"""Scan the built marqov wheel's Requires-Dist for direct-URL deps and confirm
it depends on marqov-quantumflow. Reads the BUILT artifact's metadata (the exact
thing PyPI validates), using `packaging`. Run with the wheel installed."""
import importlib.metadata as md
import sys
from packaging.requirements import Requirement

dist = md.distribution("marqov")
assert dist.metadata["Name"] == "marqov", dist.metadata["Name"]
reqs = list(dist.requires or [])
bad = [r for r in reqs if Requirement(r).url is not None]
assert not bad, f"direct-URL deps present (PyPI will reject): {bad}"
assert any(Requirement(r).name == "marqov-quantumflow" for r in reqs), \
    f"marqov-quantumflow not in Requires-Dist: {reqs}"
print(f"OK: marqov {dist.version} has no URL deps; depends on marqov-quantumflow")
