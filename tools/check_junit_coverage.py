"""Require a nonempty, entirely passing optional-provider verification leg."""
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
suites = list(root.iter("testsuite"))
assert sum(int(s.get("tests", 0)) for s in suites) > 0, "no tests ran"
for field in ("skipped", "failures", "errors"):
    assert sum(int(s.get(field, 0)) for s in suites) == 0, f"unexpected {field}"
print("optional-provider coverage OK: nonempty, no skips/failures/errors")
