"""Fail if pyproject.toml has a git/URL dependency or the old quantumflow pin.

Scans BOTH `project.dependencies` and every `project.optional-dependencies`
group (e.g. [all], [qiskit], [quantinuum]). PyPI rejects direct-URL deps wherever
they appear, and hatchling emits optional-extra deps into Requires-Dist (with an
`; extra == "..."` marker), so a URL added to an extra must fail here too.
"""
import sys
import tomllib  # stdlib on 3.11+
from pathlib import Path

project = tomllib.loads(Path("pyproject.toml").read_text())["project"]

# (group_label, dependency_string) for the core deps + every optional-extra group
entries = [("dependencies", d) for d in project.get("dependencies", [])]
for group, group_deps in project.get("optional-dependencies", {}).items():
    entries += [(f"optional-dependencies.{group}", d) for d in group_deps]

problems = []
for group, d in entries:
    if "git+" in d or "@ http" in d or "@ git" in d:
        problems.append(f"direct-URL dependency in [{group}]: {d}")
    if d.strip().startswith("quantumflow ") or d.strip() == "quantumflow":
        problems.append(f"still depends on upstream 'quantumflow' (should be 'marqov-quantumflow') in [{group}]: {d}")

core = project.get("dependencies", [])
if not any("marqov-quantumflow" in d for d in core):
    problems.append("missing 'marqov-quantumflow' in project.dependencies")

if problems:
    print("FAIL:")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print("OK: no URL deps (core + all extras); depends on marqov-quantumflow")
