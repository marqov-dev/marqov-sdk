"""Verify the SDK resolves against the fork: import quantumflow is the FORK,
and the SDK imports cleanly. Run from a NEUTRAL cwd (not the repo root) so the
installed package is checked, not the source tree."""
import importlib.metadata as md
import os
import sysconfig
import sys

# 1. The quantumflow import is provided by the marqov-quantumflow DISTRIBUTION.
qf_dist = md.distribution("marqov-quantumflow")
assert qf_dist.version == "1.0.0", f"expected marqov-quantumflow 1.0.0, got {qf_dist.version}"

# 2. Upstream 'quantumflow' distribution must NOT be installed (no collision).
try:
    up = md.version("quantumflow")
    sys.exit(f"FAIL: upstream 'quantumflow' distribution is also installed ({up})")
except md.PackageNotFoundError:
    pass

# 3. import quantumflow loads from site-packages (the installed fork), not source.
import quantumflow as qf  # noqa: E402
site = os.path.realpath(sysconfig.get_paths()["purelib"])
assert os.path.realpath(qf.__file__).startswith(site), f"quantumflow not from site-packages: {qf.__file__}"

# 4. The SDK imports and its QuantumFlow-coupled core works.
#    Real API (marqov/circuits.py): no-arg constructor, fluent chaining.
import marqov  # noqa: E402
from marqov.circuits import Circuit, bell_state  # noqa: E402
Circuit().h(0).cnot(0, 1)        # exercises the qf.* contract (H, CNot, Circuit)
bell_state()                     # the SDK's own helper

print(f"OK: SDK resolves against marqov-quantumflow {qf_dist.version}; "
      f"import quantumflow -> {qf.__file__}")
