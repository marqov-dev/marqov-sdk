"""Fail if pyproject.toml still has a git/URL dependency or the old quantumflow pin."""
import sys
import tomllib  # stdlib on 3.11+
from pathlib import Path

deps = tomllib.loads(Path("pyproject.toml").read_text())["project"]["dependencies"]
problems = []
for d in deps:
    if "git+" in d or "@ http" in d or "@ git" in d:
        problems.append(f"direct-URL dependency: {d}")
    if d.strip().startswith("quantumflow ") or d.strip() == "quantumflow":
        problems.append(f"still depends on upstream 'quantumflow' (should be 'marqov-quantumflow'): {d}")
if not any("marqov-quantumflow" in d for d in deps):
    problems.append("missing 'marqov-quantumflow' dependency")
if problems:
    print("FAIL:")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print("OK: no URL deps; depends on marqov-quantumflow")
