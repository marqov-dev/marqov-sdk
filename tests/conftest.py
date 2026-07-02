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
