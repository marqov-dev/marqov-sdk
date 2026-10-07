"""Pytest configuration for test suite."""

# Import qiskit eagerly, once, at collection time — before any test enters a
# ``patch.dict("sys.modules", ...)`` block. qiskit imports its standard-gate
# library lazily; if that first import happens *inside* a patched sys.modules
# context, patch.dict deletes the freshly-added qiskit submodule entries on
# exit, leaving qiskit half-initialized. Subsequent qiskit use then raises
# (e.g. ``TypeError: invalid input: Instruction(name='rcccx_dg', ...)``) when
# tests are run in isolation. Loading qiskit here makes that first import a
# clean no-op everywhere downstream. See issue #47.
try:
    import qiskit  # noqa: F401
    from qiskit import qasm2, qasm3  # noqa: F401
except ImportError:
    # qiskit is an optional dependency; tests that need it will skip/fail on
    # their own terms rather than being blocked by this safeguard.
    pass


def pytest_collection_finish(session):
    """Release-only guard: collection and child interpreters must use the wheel."""
    import os
    if os.environ.get("MARQOV_REQUIRE_INSTALLED") != "1":
        return
    import importlib.util
    import subprocess
    import sys
    from pathlib import Path

    import pytest

    tool = Path(__file__).resolve().parents[1] / "tools/verify_sdk_against_fork.py"
    spec = importlib.util.spec_from_file_location("release_fork_guard", tool)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        module.verify_installed()
        # Several tests start Python from rootpath. PYTHONSAFEPATH must also
        # protect these children; checking only the parent misses this collision.
        result = subprocess.run(
            [sys.executable, "-c", f"import runpy; runpy.run_path({str(tool)!r}, run_name='__main__')"],
            cwd=session.config.rootpath,
            text=True, capture_output=True, check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    except AssertionError as exc:
        pytest.exit(f"installed-wheel guard failed: {exc}", returncode=3)
