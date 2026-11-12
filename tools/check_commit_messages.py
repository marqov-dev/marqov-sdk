"""Fail if a commit message in a git range references private-repo internals.

Commit subjects and bodies are as public as tracked files and cannot be rewritten
after the fact, so the patterns the file-content guard enforces
(``tests/test_no_private_references.py``) are applied to them too. The patterns
are imported from that module, never copied: there is one list.

Usage::

    python tools/check_commit_messages.py <base>..<head>

Only commits after ``BASELINE_SHA`` are checked; earlier history is never
rewritten and is not scanned. If the range start is not a known commit (a new
branch, a force-push, an all-zero ``before`` SHA), only the tip commit is
checked. Exits 1 and names each offending commit and pattern on failure.

The one allowance beyond the file-content rules: GitHub's squash merge appends
the PR number to the subject line, ``(#123)``. That is this repository's own
public PR number, so a trailing ``(#<digits>)`` on the subject is ignored.
Bodies get no such allowance.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.test_no_private_references import (  # noqa: E402
    _ALLOWED_ISSUE_REF,
    _BANNED,
    _BARE_ISSUE_REF,
)

# Commits at or before this one predate the guard and are not checked. This is
# the main-branch tip when the check was introduced.
BASELINE_SHA = "4f85297f8b70a566530d55908c88d10cb9665254"

_SQUASH_PR_SUFFIX = re.compile(r"\s\(#\d+\)\s*$")
_FIELD_SEP = "\x1f"
_RECORD_SEP = "\x1e"


@dataclass(frozen=True)
class Violation:
    sha: str
    label: str
    line: str


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, capture_output=True, text=True,
    ).stdout


def _is_commit(repo: Path, rev: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", f"{rev}^{{commit}}"],
        capture_output=True,
    )
    return result.returncode == 0


def _scan_message(sha: str, message: str) -> list[Violation]:
    found: list[Violation] = []
    for index, line in enumerate(message.splitlines()):
        if index == 0:
            line = _SQUASH_PR_SUFFIX.sub("", line)
        for label, pattern in _BANNED:
            if pattern.search(line):
                found.append(Violation(sha, label, line.strip()[:100]))
        if _BARE_ISSUE_REF.search(_ALLOWED_ISSUE_REF.sub("", line)):
            found.append(Violation(sha, "unqualified issue number", line.strip()[:100]))
    return found


def check_range(
    repo: Path | str, rev_range: str, baseline: str | None = BASELINE_SHA
) -> list[Violation]:
    repo = Path(repo)
    if ".." in rev_range:
        start, _, end = rev_range.partition("..")
        if not start or not _is_commit(repo, start):
            # No usable range start: check only the tip.
            args = ["-1", end or "HEAD"]
        else:
            args = [rev_range]
    else:
        args = [rev_range]
    if baseline and _is_commit(repo, baseline):
        args.append(f"^{baseline}")
    log = _git(
        repo, "log", f"--format=%H{_FIELD_SEP}%B{_RECORD_SEP}", *args
    )
    violations: list[Violation] = []
    for record in log.split(_RECORD_SEP):
        if not record.strip():
            continue
        sha, _, message = record.strip().partition(_FIELD_SEP)
        violations.extend(_scan_message(sha, message))
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("range", help="git range, e.g. origin/main..HEAD")
    parser.add_argument("--repo", default=".", help="repository path (default: .)")
    args = parser.parse_args(argv)
    violations = check_range(args.repo, args.range)
    for v in violations:
        print(f"{v.sha} [{v.label}] {v.line}")
    if violations:
        print(
            f"{len(violations)} private-internal reference(s) in commit messages. "
            "Reword the commit (describe behavior in your own words; qualify "
            "public issues as marqov-sdk#N) before merging.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
