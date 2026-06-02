"""Pytest configuration for test suite.

Mocks optional heavy dependencies that are not required for simulation tests
to allow the test suite to run without a full environment install.
"""

import sys
import types


class _AutoAttrModule(types.ModuleType):
    """A module stub that returns new instances of itself for any attribute access.

    This allows 'from some_stub import SomeClass' to succeed without error.
    """

    def __getattr__(self, name: str) -> "_AutoAttrModule":
        child = _AutoAttrModule(f"{self.__name__}.{name}")
        setattr(self, name, child)
        return child

    def __call__(self, *args, **kwargs):  # type: ignore[override]
        return _AutoAttrModule(f"{self.__name__}.__call__")

    def __iadd__(self, other: object) -> "_AutoAttrModule":
        return self

    def __add__(self, other: object) -> "_AutoAttrModule":
        return self

    def __len__(self) -> int:
        return 0

    def __iter__(self):  # type: ignore[override]
        return iter([])

    def __repr__(self) -> str:
        return f"<MockModule '{self.__name__}'>"


def _mock(name: str) -> _AutoAttrModule:
    """Create and register a mock module by name."""
    mod = _AutoAttrModule(name)
    mod.__package__ = name.split(".")[0]
    mod.__spec__ = None  # type: ignore[assignment]
    mod.__path__ = []  # type: ignore[assignment]
    mod.__file__ = None  # type: ignore[assignment]
    mod.__loader__ = None  # type: ignore[assignment]
    return mod


# Stub out heavy optional dependencies before any marqov imports.
# These are cloud/hardware SDK dependencies not needed for pure unit tests.
_STUB_MODULES = [
    "quantumflow",
    "boto3",
    "botocore",
    "braket",
    "braket.circuits",
    "braket.circuits.circuit",
    "braket.devices",
    "braket.aws",
    "braket.aws.aws_device",
    "braket.aws.aws_quantum_task",
    "amazon_braket_sdk",
    "azure",
    "azure.quantum",
    "azure.quantum.job",
    "azure.quantum.target",
    "temporalio",
    "temporalio.client",
    "temporalio.worker",
    "temporalio.activity",
    "temporalio.workflow",
    "temporalio.worker.workflow_sandbox",
    "temporalio.common",
    "covalent",
    "covalent_braket_plugin",
    "supabase",
    "sentry_sdk",
    "structlog",
    "pydantic",
    "pydantic.fields",
    "pydantic_settings",
    "cloudpickle",
]

for _mod_name in _STUB_MODULES:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = _mock(_mod_name)



# Specialized qiskit stub: qasm2.dumps() and qasm3.dumps() must return a
# valid OpenQASM string so SimulationExecutor can parse the qubit count.
_QASM2_STUB = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nh q[0];\n'
_QASM3_STUB = 'OPENQASM 3.0;\nqubit[1] q;\nh q[0];\n'


def _qasm2_dumps(*args, **kwargs):  # type: ignore[misc]
    return _QASM2_STUB


def _qasm3_dumps(*args, **kwargs):  # type: ignore[misc]
    return _QASM3_STUB


_qiskit_mod = _mock("qiskit")
_qiskit_qasm2 = _mock("qiskit.qasm2")
_qiskit_qasm2.dumps = _qasm2_dumps  # type: ignore[attr-defined]
_qiskit_qasm3 = _mock("qiskit.qasm3")
_qiskit_qasm3.dumps = _qasm3_dumps  # type: ignore[attr-defined]
_qiskit_mod.qasm2 = _qiskit_qasm2  # type: ignore[attr-defined]
_qiskit_mod.qasm3 = _qiskit_qasm3  # type: ignore[attr-defined]

for _mod_name, _mod_obj in [
    ("qiskit", _qiskit_mod),
    ("qiskit.qasm2", _qiskit_qasm2),
    ("qiskit.qasm3", _qiskit_qasm3),
]:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = _mod_obj

import os

# Add platform/src to path so worker modules are importable in tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "platform", "src"))
