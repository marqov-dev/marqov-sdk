"""Child-process environment builder for isolated task execution.

A host runner owns the allowlist — it provides ``MARQOV_SCRUB_ALLOWLIST``
(comma-separated) in the worker process, and this helper reads it at call
time.  If unset (SDK used standalone, tests), a documented minimal-safe
fallback is used — a small, secret-free set of variables only.

The only way a credential enters the child env is through ``provider_env``.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

_log = logging.getLogger("marqov.workflows")

# Minimal-safe default — a small, secret-free set of environment variables.
# Documented here as the single canonical SDK-standalone fallback.
# A host runner should set MARQOV_SCRUB_ALLOWLIST at worker boot so the extra
# vars its runtime images need (e.g. TLS/region settings) are also passed through.
_MINIMAL_SAFE_DEFAULT: tuple[str, ...] = (
    "PATH",
    "LANG",
    "LC_ALL",
    "VIRTUAL_ENV",
    "PYTHONPATH",
    "HOME",
    "TMPDIR",
)

_ALLOWLIST_ENV_VAR = "MARQOV_SCRUB_ALLOWLIST"

_TMP_ROOT = Path(tempfile.gettempdir())


def _read_allowlist() -> tuple[str, ...]:
    """Read the scrub allowlist from the env var set by the host at boot.

    Returns the minimal-safe default when the var is unset.
    """
    raw = os.environ.get(_ALLOWLIST_ENV_VAR)
    if not raw:
        # Loud on purpose: UNDER A HOST RUNNER an unset var signals a version skew
        # (the host must set it at worker boot). The child env then differs from
        # the host-tested one and a task may hit a native ImportError — a silent
        # operational fail-open. Standalone SDK use can ignore this.
        _log.warning(
            "%s is not set — falling back to the minimal-safe env allowlist. "
            "Under a host runner this indicates SDK/host version skew.",
            _ALLOWLIST_ENV_VAR,
        )
        return _MINIMAL_SAFE_DEFAULT
    return tuple(k.strip() for k in raw.split(",") if k.strip())


def build_child_env(
    workdir: Path,
    provider_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a scrubbed environment for a task-body child process.

    Args:
        workdir: Per-job working directory (must already exist). HOME and
            TMPDIR are set to subdirs of this directory so the child never
            writes to the real user HOME or system /tmp.
        provider_env: Explicit provider credentials that the task body needs
            (e.g. AWS keys for a Braket job). The ONLY way a credential enters
            the child env. Pass ``None`` (default) for pure-compute tasks.

    Returns:
        A dict suitable for ``env=`` in ``asyncio.create_subprocess_exec``.
    """
    allowlist = _read_allowlist()
    env: dict[str, str] = {k: os.environ[k] for k in allowlist if k in os.environ}

    # Per-job isolated dirs — child writes here, never to real HOME/TMPDIR.
    home = workdir / "home"
    tmp = workdir / "tmp"
    for d in (home, tmp):
        d.mkdir(parents=True, exist_ok=True)

    # Override HOME/TMPDIR even if they were in the allowlist — the per-job
    # values MUST take precedence to prevent cross-job contamination.
    env["HOME"] = str(home)
    env["TMPDIR"] = str(tmp)

    # Provider creds are injected LAST so nothing in the allowlist can shadow them.
    if provider_env:
        env.update(provider_env)

    return env


def new_task_workdir(node_id: str) -> Path:
    """Create an isolated per-task scratch directory.

    The directory name is prefixed ``marqov_task_`` so a host runner's
    stale-scratch sweep can clean it up.
    """
    # Sanitise node_id for use in a path prefix (keep alphanumeric + dash).
    safe = "".join(c if c.isalnum() or c == "-" else "_" for c in node_id)[:32]
    wd = Path(tempfile.mkdtemp(prefix=f"marqov_task_{safe}_", dir=_TMP_ROOT))
    return wd
