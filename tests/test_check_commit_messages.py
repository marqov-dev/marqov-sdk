"""The commit-message guard reuses the private-reference patterns on a git range.

Every repository here is built in ``tmp_path``; the real history is never touched.
Banned example strings are assembled from parts so this file does not trip the
file-content guard in ``test_no_private_references.py``.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "check_commit_messages", _ROOT / "tools" / "check_commit_messages.py"
)
assert _SPEC is not None and _SPEC.loader is not None
ccm = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = ccm  # dataclasses resolves the module by name
_SPEC.loader.exec_module(ccm)

PRIVATE_REPO = "marqov" + "-platform"
BARE_ISSUE = "#" + "1234"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _git(
        repo, "-c", "user.name=t", "-c", "user.email=t@example.com",
        "commit", "--allow-empty", "-q", "-m", message,
    )
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q", "-b", "main")
    return tmp_path


def test_clean_range_passes(repo: Path) -> None:
    base = _commit(repo, "chore: base")
    _commit(repo, "fix: tidy\n\nSee marqov-sdk#152 for context.")
    assert ccm.check_range(repo, f"{base}..HEAD") == []
    assert ccm.main([f"{base}..HEAD", "--repo", str(repo)]) == 0


def test_banned_pattern_in_body_names_commit_and_pattern(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = _commit(repo, "chore: base")
    bad = _commit(repo, f"fix: thing\n\nMirrors {PRIVATE_REPO} behaviour.")
    violations = ccm.check_range(repo, f"{base}..HEAD")
    assert [(v.sha, v.label) for v in violations] == [(bad, "private repo name")]
    assert ccm.main([f"{base}..HEAD", "--repo", str(repo)]) == 1
    out = capsys.readouterr().out
    assert bad in out and "private repo name" in out


def test_banned_pattern_in_subject_and_bare_issue_number(repo: Path) -> None:
    base = _commit(repo, "chore: base")
    ts = _commit(repo, "fix: cite handler" + "." + "ts directly")
    issue = _commit(repo, f"fix: closes {BARE_ISSUE}")
    found = {(v.sha, v.label) for v in ccm.check_range(repo, f"{base}..HEAD")}
    assert found == {(ts, "internal source path"), (issue, "unqualified issue number")}


def test_github_squash_pr_suffix_on_subject_is_allowed(repo: Path) -> None:
    base = _commit(repo, "chore: base")
    _commit(repo, "fix: something (#" + "320)")
    assert ccm.check_range(repo, f"{base}..HEAD") == []


def test_commits_at_or_before_baseline_are_ignored(repo: Path) -> None:
    old_bad = _commit(repo, f"old: {PRIVATE_REPO}")
    _commit(repo, "ok: after baseline")
    new_bad = _commit(repo, f"new: {PRIVATE_REPO}")
    # The range reaches the whole history; the baseline and its ancestors are skipped.
    found = ccm.check_range(repo, "HEAD", baseline=old_bad)
    assert [v.sha for v in found] == [new_bad]


def test_unknown_range_start_checks_only_the_tip(repo: Path) -> None:
    _commit(repo, "chore: base")
    tip = _commit(repo, f"fix: {PRIVATE_REPO}")
    found = ccm.check_range(repo, f"{'0' * 40}..{tip}")
    assert [v.sha for v in found] == [tip]
