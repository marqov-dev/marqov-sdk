"""Tests for marqov.platform public package API (Task 6).

Verifies that:
1. All expected public names are importable from the top-level ``marqov.platform``.
2. ``__all__`` exactly matches the advertised public names.
3. Importing ``marqov`` (the core SDK) does NOT trigger a side-import of
   ``marqov.platform`` (laziness guarantee — tested via a subprocess so that
   prior imports in this test process cannot pollute the check).
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# 1. Import surface — all public names importable from marqov.platform
# ---------------------------------------------------------------------------

# Import each public name individually so that failures name the exact symbol.
from marqov.platform import Backend
from marqov.platform import Job
from marqov.platform import JobStatus
from marqov.platform import MarqovClient
from marqov.platform import MarqovPlatformError
from marqov.platform import PaidBackendNotSupportedYet
from marqov.platform import PlatformInfo
from marqov.platform import PlatformResult
from marqov.platform import AuthenticationError
from marqov.platform import PermissionTierError
from marqov.platform import BackendUnavailable
from marqov.platform import InvalidProgram
from marqov.platform import JobFailed
from marqov.platform import RateLimited
from marqov.platform import TransportError
from marqov.platform import is_terminal


# The full set we expect to be public.
_EXPECTED_PUBLIC_NAMES: frozenset[str] = frozenset(
    [
        "MarqovClient",
        "Job",
        "JobStatus",
        "Backend",
        "PlatformResult",
        "PlatformInfo",
        "is_terminal",
        "MarqovPlatformError",
        "AuthenticationError",
        "PermissionTierError",
        "BackendUnavailable",
        "PaidBackendNotSupportedYet",
        "InvalidProgram",
        "JobFailed",
        "RateLimited",
        "TransportError",
    ]
)


class TestPublicImports:
    """Confirm every expected name is importable from the package."""

    def test_marqov_client_importable(self) -> None:
        assert MarqovClient is not None

    def test_job_importable(self) -> None:
        assert Job is not None

    def test_job_status_importable(self) -> None:
        assert JobStatus is not None

    def test_backend_importable(self) -> None:
        assert Backend is not None

    def test_platform_result_importable(self) -> None:
        assert PlatformResult is not None

    def test_platform_info_importable(self) -> None:
        assert PlatformInfo is not None

    def test_is_terminal_importable(self) -> None:
        assert is_terminal is not None

    def test_marqov_platform_error_importable(self) -> None:
        assert MarqovPlatformError is not None

    def test_authentication_error_importable(self) -> None:
        assert AuthenticationError is not None

    def test_permission_tier_error_importable(self) -> None:
        assert PermissionTierError is not None

    def test_backend_unavailable_importable(self) -> None:
        assert BackendUnavailable is not None

    def test_paid_backend_not_supported_yet_importable(self) -> None:
        assert PaidBackendNotSupportedYet is not None

    def test_invalid_program_importable(self) -> None:
        assert InvalidProgram is not None

    def test_job_failed_importable(self) -> None:
        assert JobFailed is not None

    def test_rate_limited_importable(self) -> None:
        assert RateLimited is not None

    def test_transport_error_importable(self) -> None:
        assert TransportError is not None


# ---------------------------------------------------------------------------
# 2. __all__ matches exactly the expected public names
# ---------------------------------------------------------------------------


class TestDunderAll:
    """Verify that ``__all__`` declares exactly the expected names."""

    def test_all_is_defined(self) -> None:
        import marqov.platform as pkg

        assert hasattr(pkg, "__all__"), "__all__ is not defined on marqov.platform"

    def test_all_contains_expected_names(self) -> None:
        import marqov.platform as pkg

        actual = frozenset(pkg.__all__)
        assert actual == _EXPECTED_PUBLIC_NAMES, (
            f"__all__ mismatch.\n"
            f"  Extra (in __all__ but not expected): {actual - _EXPECTED_PUBLIC_NAMES}\n"
            f"  Missing (expected but not in __all__): {_EXPECTED_PUBLIC_NAMES - actual}"
        )

    def test_all_names_are_actually_present(self) -> None:
        """Every name in ``__all__`` must be a real attribute of the package."""
        import marqov.platform as pkg

        for name in pkg.__all__:
            assert hasattr(pkg, name), (
                f"'{name}' is listed in __all__ but not an attribute of marqov.platform"
            )


# ---------------------------------------------------------------------------
# 3. Laziness: import marqov does NOT side-import marqov.platform
# ---------------------------------------------------------------------------


class TestLaziness:
    """Confirm the core marqov SDK does not eagerly import marqov.platform."""

    def test_import_marqov_does_not_load_platform(self) -> None:
        """Run a subprocess so no prior imports in this process can pollute the check."""
        script = (
            "import marqov, sys; "
            "assert 'marqov.platform' not in sys.modules, "
            "'marqov.platform was imported as a side-effect of importing marqov'"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"Laziness check failed (returncode={result.returncode}).\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )


# ---------------------------------------------------------------------------
# 4. Boundary: core SDK files never import marqov.platform (the other
#    direction of the laziness guarantee — see docs/design/platform-client-
#    boundary.md's "two-direction rule"). Mirrors the grep-gate pattern in
#    tests/test_no_private_references.py.
# ---------------------------------------------------------------------------

_CORE_PATHS = (
    "marqov/circuits.py",
    "marqov/device.py",
    "marqov/backends.py",
    "marqov/executors",
    "marqov/workflows",
)
_PLATFORM_IMPORT = re.compile(r"^\s*(from\s+marqov\.platform\s+import|import\s+marqov\.platform)")


def _iter_core_files() -> Iterator[Path]:
    root = Path(__file__).resolve().parent.parent
    for rel in _CORE_PATHS:
        base = root / rel
        if base.is_file():
            yield base
        elif base.is_dir():
            yield from base.rglob("*.py")


class TestCoreDoesNotImportPlatform:
    """Confirm marqov.platform never leaks into the core SDK (the reverse
    direction of TestLaziness above)."""

    def test_core_files_never_import_platform(self) -> None:
        offenders: list[str] = []
        for path in _iter_core_files():
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if _PLATFORM_IMPORT.match(line):
                    offenders.append(f"{path}:{lineno}: {line.strip()}")
        assert not offenders, (
            f"{len(offenders)} core-SDK import(s) of marqov.platform found — this "
            "breaks the boundary documented in docs/design/platform-client-boundary.md "
            "(platform is a consumer of the core, never a dependency of it):\n  "
            + "\n  ".join(offenders)
        )
