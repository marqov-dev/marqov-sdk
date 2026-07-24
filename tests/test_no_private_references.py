"""Mechanical gate: the public SDK must not reference private-repo internals.

This is the same move as a CI guard that bans ``os.environ.copy()`` — a class of
mistake that has now reached the public repo from more than one source (a task
comment, and the platform-client subpackage's source-provenance citations). It
needs a gate, not vigilance.

What it bans (over ``marqov/``, ``docs/``, and ``tests/`` — the leak that started
this was a test comment, so tests are in scope too):
  - private repo names (``marqov-platform``, ``marqov-research``),
  - internal source paths (``platform/src``, ``app/api/``, ``route.ts``, ``.ts:<line>``),
  - DB migration / schema internals (``supabase/migrations``, ``*.sql``, ``job_runs``,
    ``selectCols``),
  - unqualified / private issue numbers (``#1234``, ``marqov-platform#1234``).

Explicit **public** cross-references — ``marqov-sdk#123`` — ARE allowed: public↔public
links are fine and encouraged over bare ``#123`` (which is ambiguous and banned so the
next contributor is forced to qualify it).

What is FINE and must stay: the observable **wire contract** — HTTP status codes,
error-code strings, query-param names, timeouts. Those are visible in real API
traffic. If this test fails, describe the behavior in your own words and delete the
private ``file:line`` / schema / issue citation.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SCAN_DIRS = ("marqov", "docs", "tests")
_SCAN_SUFFIXES = {".py", ".md", ".rst", ".txt"}
# This gate file legitimately contains the banned patterns (as regexes), so it must
# not scan itself.
_SELF = Path(__file__).name

# (label, pattern) — each matches one class of private-internal reference.
_BANNED: list[tuple[str, re.Pattern[str]]] = [
    ("private repo name", re.compile(r"marqov-platform|marqov-research")),
    ("internal source path", re.compile(r"platform/src|app/api/|route\.ts|\.ts:\d")),
    ("db migration / schema", re.compile(r"supabase/migrations|\.sql\b|\bjob_runs\b|selectCols")),
]

# Issue numbers are handled separately: strip the ALLOWED public cross-ref form
# (marqov-sdk#123) first, then any remaining #123 is an unqualified/private ref.
_ALLOWED_ISSUE_REF = re.compile(r"marqov-sdk#\d+")
_BARE_ISSUE_REF = re.compile(r"#\d{3,4}\b")


def _iter_files():
    # Directory trees (recursive).
    for d in _SCAN_DIRS:
        base = _ROOT / d
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in _SCAN_SUFFIXES and path.name != _SELF:
                yield path
    # Repo-root docs (NON-recursive): README.md, CHANGELOG.md, and the trio
    # (RELEASING/SECURITY/ARCHITECTURE) — all describe the SDK publicly and are
    # prime spots to leak platform internals, but live outside the dir trees above.
    for path in _ROOT.glob("*"):
        if path.is_file() and path.suffix in _SCAN_SUFFIXES and path.name != _SELF:
            yield path


def test_no_private_references_in_public_sdk():
    offenders: list[str] = []
    for path in _iter_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            rel = path.relative_to(_ROOT)
            for label, pattern in _BANNED:
                if pattern.search(line):
                    offenders.append(f"{rel}:{lineno} [{label}] {line.strip()[:100]}")
            # Issue-number check with the public-cross-ref allowance.
            if _BARE_ISSUE_REF.search(_ALLOWED_ISSUE_REF.sub("", line)):
                offenders.append(f"{rel}:{lineno} [unqualified issue number] {line.strip()[:100]}")
    assert not offenders, (
        f"{len(offenders)} private-internal reference(s) found in public SDK "
        "source/docs. Describe the wire contract in your own words (status codes, "
        "error strings, param names are fine); drop the private path/schema/issue "
        "citation:\n  " + "\n  ".join(offenders)
    )
