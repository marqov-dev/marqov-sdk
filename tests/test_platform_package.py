"""Tests for marqov.platform public package API (Task 6).

Verifies that:
1. All expected public names are importable from the top-level ``marqov.platform``.
2. ``__all__`` exactly matches the advertised public names.
3. Importing ``marqov`` (the core SDK) does NOT trigger a side-import of
   ``marqov.platform`` (laziness guarantee — tested via a subprocess so that
   prior imports in this test process cannot pollute the check).
4. No core SDK file imports ``marqov.platform`` (the reverse direction of the
   same boundary — an AST scan over everything under ``marqov/`` except the
   platform subpackage itself).
"""

from __future__ import annotations

import ast
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
#    boundary.md's "two-direction rule").
#
#    This parses the AST rather than grepping raw lines (the grep-gate pattern
#    of tests/test_no_private_references.py). Two reasons: text in docstrings
#    and comments cannot trip it, and every real import form is covered —
#    submodule from-imports, relative imports, `from marqov import platform`,
#    and aliased plain imports, at module level or inside a function body.
#
#    ``if TYPE_CHECKING:`` blocks are deliberately NOT exempt: the boundary is
#    absolute, so a type-only import is a violation too. Use a string
#    annotation or a platform-free protocol instead.
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
_CORE_ROOT = _ROOT / "marqov"
# Denylist, not an allowlist: everything under marqov/ is core except the
# platform subpackage itself, so a newly added core module is covered the
# moment it lands and there is no path list to drift out of sync.
_EXCLUDED_DIRS = frozenset({"platform"})
_BANNED_MODULE = "marqov.platform"
# Dynamic-import helpers whose first string-literal argument is inspected.
# Best-effort by design: a module name assembled at runtime is out of scope.
_DYNAMIC_IMPORT_FUNCS = frozenset({"import_module", "__import__"})
# Floor on the scan size, so a moved/renamed tree fails loudly instead of
# passing vacuously on zero files (the core tree is ~36 files today).
_MIN_SCANNED_FILES = 30


def _iter_core_files() -> Iterator[Path]:
    """Every ``.py`` file under ``marqov/`` except the platform subpackage."""
    for path in sorted(_CORE_ROOT.rglob("*.py")):
        # Directory components only — a core module that merely happens to be
        # named platform.py is still core, and still scanned.
        if _EXCLUDED_DIRS.isdisjoint(path.relative_to(_CORE_ROOT).parts[:-1]):
            yield path


def _package_parts(path: Path) -> list[str]:
    """Dotted parts of the package that ``path`` lives in.

    This is the base a relative import resolves against: ``marqov/device.py``
    and ``marqov/__init__.py`` both resolve against ``marqov``, while
    ``marqov/executors/factory.py`` resolves against ``marqov.executors``.
    """
    return list(path.relative_to(_ROOT).with_suffix("").parts)[:-1]


def _resolve_from(node: ast.ImportFrom, package: list[str]) -> str | None:
    """Absolute module name a ``from ... import`` targets, or None if unresolvable."""
    if not node.level:
        return node.module
    base = package[: len(package) - node.level + 1]
    if not base:  # more dots than the package has levels
        return None
    return ".".join([*base, node.module] if node.module else base)


def _is_banned(module: str) -> bool:
    return module == _BANNED_MODULE or module.startswith(f"{_BANNED_MODULE}.")


def _platform_references(tree: ast.Module, package: list[str]) -> Iterator[tuple[int, str]]:
    """Yield ``(lineno, source-ish description)`` for each marqov.platform reference."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_banned(alias.name):
                    yield node.lineno, f"import {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            module = _resolve_from(node, package)
            if module is None:
                continue
            prefix = f"from {'.' * node.level}{node.module or ''} import"
            if _is_banned(module):
                yield node.lineno, f"{prefix} {', '.join(a.name for a in node.names)}"
                continue
            # `from marqov import platform` / `from . import platform`.
            for alias in node.names:
                if _is_banned(f"{module}.{alias.name}"):
                    yield node.lineno, f"{prefix} {alias.name}"
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name not in _DYNAMIC_IMPORT_FUNCS or not node.args:
                continue
            arg = node.args[0]
            # Prefix match: best-effort cover for the dynamic-import escape hatch.
            if (
                isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
                and arg.value.startswith(_BANNED_MODULE)
            ):
                yield node.lineno, f"{name}({arg.value!r})"


class TestCoreDoesNotImportPlatform:
    """Confirm marqov.platform never leaks into the core SDK (the reverse
    direction of TestLaziness above)."""

    def test_scan_is_not_vacuous(self) -> None:
        """A moved or renamed core tree must fail here, not silently scan nothing."""
        scanned = list(_iter_core_files())
        assert len(scanned) > _MIN_SCANNED_FILES, (
            f"Only {len(scanned)} core file(s) found under {_CORE_ROOT} — the boundary "
            "scan below would pass vacuously. Did the package move or get renamed?"
        )

    def test_core_files_never_import_platform(self) -> None:
        offenders: list[str] = []
        for path in _iter_core_files():
            rel = path.relative_to(_ROOT)
            source = path.read_text(encoding="utf-8", errors="replace")
            try:
                tree = ast.parse(source, filename=str(path))
            except SyntaxError as exc:
                offenders.append(f"{rel}:{exc.lineno}: unparseable ({exc.msg})")
                continue
            for lineno, what in _platform_references(tree, _package_parts(path)):
                offenders.append(f"{rel}:{lineno}: {what}")
        assert not offenders, (
            f"{len(offenders)} core-SDK reference(s) to marqov.platform found — this "
            "breaks the boundary documented in docs/design/platform-client-boundary.md "
            "(platform is a consumer of the core, never a dependency of it). Type-only "
            "imports under `if TYPE_CHECKING:` count too:\n  " + "\n  ".join(offenders)
        )
