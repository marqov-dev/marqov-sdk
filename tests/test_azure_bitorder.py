"""Tests for Azure bit-order normalization on both execution paths.

These drive the executor's own conversion code (`_execute_cirq` through
`execute()`, and `_execute_qiskit` directly) with fake Azure services that
return genuine framework result objects, so the assertions fail if the
conversion changes. The probe is asymmetric on purpose: X on qubit 0 of a
2-qubit circuit must come back as "10" (qubit 0 leftmost) on both paths.

See marqov-sdk#131.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from qiskit.result import Result  # type: ignore[import-untyped]

from marqov.circuits import Circuit
from marqov.executors.azure import AzureQuantumExecutor, AzureQuantumExecutorConfig


@pytest.fixture
def cirq_module() -> Any:
    """The cirq package, or skip: cirq is an optional backend extra."""
    return pytest.importorskip("cirq")


def _executor(framework: str, backend: Any) -> AzureQuantumExecutor:
    """Build an executor whose cached backend is a fake, so Azure is never reached."""
    config = AzureQuantumExecutorConfig(
        subscription_id="sub",
        resource_group="rg",
        workspace_name="ws",
        location="eastus",
        target="ionq.simulator",
        framework=framework,
    )
    executor = AzureQuantumExecutor(config)
    # _backend holds the lazily created Cirq service or Qiskit backend; seeding
    # it short-circuits workspace creation.
    cast(Any, executor)._backend = backend
    return executor


class _FakeCirqService:
    """Stand-in for azure.quantum.cirq.AzureQuantumService.

    Like the real service, `run()` submits, waits, and returns the finished
    `cirq.Result` itself rather than a job handle.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[Any, int]] = []

    def run(self, program: Any, repetitions: int = 1, **kwargs: Any) -> Any:
        import cirq

        self.calls.append((program, repetitions))
        return cirq.Simulator().run(program, repetitions=repetitions)


class _FakeQiskitJob:
    """Stand-in for a Qiskit job returned by backend.run()."""

    def __init__(self, result: Any) -> None:
        self._result = result

    def job_id(self) -> str:
        return "fake-job-id"

    def result(self) -> Any:
        return self._result

    def properties(self) -> Any:
        return None


class _FakeQiskitBackend:
    """Stand-in for an Azure Quantum Qiskit backend."""

    def __init__(self, result: Any) -> None:
        self._result = result
        self.calls: list[tuple[Any, int]] = []

    def run(self, circuit: Any, shots: int = 1, **kwargs: Any) -> _FakeQiskitJob:
        self.calls.append((circuit, shots))
        return _FakeQiskitJob(self._result)


def _qiskit_result(counts: dict[str, int], creg_sizes: list[list[Any]], shots: int) -> Result:
    """Build a genuine qiskit Result from raw (hex-keyed) counts."""
    memory_slots = sum(size for _, size in creg_sizes)
    return Result.from_dict(
        {
            "backend_name": "fake-azure-backend",
            "backend_version": "0.0.0",
            "qobj_id": "fake-qobj",
            "job_id": "fake-job-id",
            "success": True,
            "results": [
                {
                    "shots": shots,
                    "success": True,
                    "data": {"counts": counts},
                    "header": {"memory_slots": memory_slots, "creg_sizes": creg_sizes},
                }
            ],
        }
    )


class TestCirqExecutionPath:
    """The Cirq path converts a real cirq.Result into Marqov counts."""

    @pytest.mark.asyncio
    async def test_x_on_qubit_0_is_leftmost(self, cirq_module: Any) -> None:
        """X(0) on 2 qubits: cirq histogram key 2 must become '10'."""
        service = _FakeCirqService()
        executor = _executor("cirq", service)

        circuit = Circuit().x(0).z(1)
        result = await executor.execute(circuit, shots=100)

        assert result.counts == {"10": 100}
        assert service.calls[0][1] == 100

    @pytest.mark.asyncio
    async def test_x_on_qubit_0_of_three_qubits(self, cirq_module: Any) -> None:
        """X(0) on 3 qubits must become '100', not '001'."""
        executor = _executor("cirq", _FakeCirqService())

        circuit = Circuit().x(0).z(1).z(2)
        result = await executor.execute(circuit, shots=64)

        assert result.counts == {"100": 64}

    @pytest.mark.asyncio
    async def test_bell_state_counts_are_correlated(self, cirq_module: Any) -> None:
        """A Bell state yields only '00' and '11', summing to the shot count."""
        executor = _executor("cirq", _FakeCirqService())

        circuit = Circuit().h(0).cnot(0, 1)
        result = await executor.execute(circuit, shots=200)

        assert set(result.counts) <= {"00", "11"}
        assert sum(result.counts.values()) == 200

    @pytest.mark.asyncio
    async def test_no_timeout_path_uses_the_result_directly(self, cirq_module: Any) -> None:
        """With timeout_seconds=None the run() result is used as-is, as with a timeout."""
        service = _FakeCirqService()
        executor = _executor("cirq", service)
        executor.config.timeout_seconds = None

        result = await executor.execute(Circuit().x(0).z(1), shots=50)

        assert result.counts == {"10": 50}
        assert service.calls[0][1] == 50

    @pytest.mark.asyncio
    async def test_job_id_is_none(self, cirq_module: Any) -> None:
        """service.run() returns a result, so no job id can be recorded."""
        executor = _executor("cirq", _FakeCirqService())

        result = await executor.execute(Circuit().x(0).z(1), shots=10)

        assert result.metadata["job_id"] is None
        assert executor._current_job_id is None
        assert result.metadata["framework"] == "cirq"
        assert result.shots == 10


class TestQiskitExecutionPath:
    """The Qiskit path reverses Qiskit's little-endian bitstrings."""

    @pytest.mark.asyncio
    async def test_x_on_qubit_0_is_leftmost(self) -> None:
        """X(0) on 2 qubits: Qiskit reports '01', Marqov reports '10'."""
        executor = _executor(
            "qiskit", _FakeQiskitBackend(_qiskit_result({"0x1": 100}, [["c", 2]], shots=100))
        )

        result = await executor._execute_qiskit(Circuit().x(0).z(1), shots=100)

        assert result.counts == {"10": 100}
        assert result.metadata["job_id"] == "fake-job-id"

    @pytest.mark.asyncio
    async def test_multi_register_spaces_are_stripped(self) -> None:
        """Two 1-bit registers report '0 1', which must become '10'."""
        executor = _executor(
            "qiskit",
            _FakeQiskitBackend(_qiskit_result({"0x1": 100}, [["c0", 1], ["c1", 1]], shots=100)),
        )

        result = await executor._execute_qiskit(Circuit().x(0).z(1), shots=100)

        assert result.counts == {"10": 100}

    @pytest.mark.asyncio
    async def test_three_qubit_probe(self) -> None:
        """X(0) on 3 qubits: Qiskit '001' becomes '100'."""
        executor = _executor(
            "qiskit", _FakeQiskitBackend(_qiskit_result({"0x1": 50}, [["c", 3]], shots=50))
        )

        result = await executor._execute_qiskit(Circuit().x(0).z(1).z(2), shots=50)

        assert result.counts == {"100": 50}

    @pytest.mark.asyncio
    async def test_bell_state_palindromes_are_unchanged(self) -> None:
        """Bell-state bitstrings are palindromes and survive the reversal."""
        executor = _executor(
            "qiskit",
            _FakeQiskitBackend(_qiskit_result({"0x0": 500, "0x3": 500}, [["c", 2]], shots=1000)),
        )

        result = await executor._execute_qiskit(Circuit().h(0).cnot(0, 1), shots=1000)

        assert result.counts == {"00": 500, "11": 500}


class TestPathsAgree:
    """Both frameworks report the same bitstring for the same probe."""

    @pytest.mark.asyncio
    async def test_same_probe_same_counts(self, cirq_module: Any) -> None:
        cirq_executor = _executor("cirq", _FakeCirqService())
        qiskit_executor = _executor(
            "qiskit", _FakeQiskitBackend(_qiskit_result({"0x1": 100}, [["c", 2]], shots=100))
        )

        probe = Circuit().x(0).z(1)
        cirq_result = await cirq_executor.execute(probe, shots=100)
        qiskit_result = await qiskit_executor._execute_qiskit(probe, shots=100)

        assert cirq_result.counts == qiskit_result.counts == {"10": 100}
