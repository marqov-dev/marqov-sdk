"""Tests for the Rigetti QCS executor.

The unit tests here use an injected ``QuantumComputer`` double and an injected
processor-list callable, so they never start a QVM, touch QCS or need credentials.
They cover result normalisation, error handling, the empty-circuit path and the
``get_status`` mapping. A real end-to-end QVM test lives in
``tests/integration/test_rigetti_qvm.py``.
"""

from typing import Any

import pytest

from marqov.circuits import Circuit, bell_state
from marqov.executors import ExecutionResult, ExecutorFactory, RigettiExecutor
from marqov.executors.base import DeviceStatus
from marqov.executors.rigetti import RigettiExecutorConfig


class _FakeResult:
    """Stand-in for a pyquil execution result exposing ``get_register_map``."""

    def __init__(self, readout: list[list[int]]) -> None:
        self._readout = readout

    def get_register_map(self) -> dict[str, Any]:
        return {"ro": self._readout}


class _FakeQC:
    """Stand-in QuantumComputer: ``compile`` is a no-op, ``run`` returns canned data."""

    def __init__(self, result: Any = None, *, run_error: Exception | None = None) -> None:
        self._result = result
        self._run_error = run_error
        self.compiled: Any = None

    def compile(self, program: Any) -> Any:
        self.compiled = program
        return program

    def run(self, executable: Any) -> Any:
        if self._run_error is not None:
            raise self._run_error
        return self._result


class TestRigettiExecutorConfig:
    """Tests for RigettiExecutorConfig."""

    def test_defaults(self) -> None:
        """Config defaults to a local 2-qubit QVM."""
        config = RigettiExecutorConfig()
        assert config.quantum_processor_id == "2q-qvm"
        assert config.as_qvm is None
        assert config.compiler_timeout_seconds == 30.0
        assert config.execution_timeout_seconds == 30.0
        assert config.timeout_seconds is None


class TestIsQvm:
    """Tests for QVM vs QPU detection."""

    def test_name_ending_in_qvm_is_qvm(self) -> None:
        """A processor id ending in -qvm is treated as a QVM."""
        executor = RigettiExecutor(RigettiExecutorConfig(quantum_processor_id="9q-square-qvm"))
        assert executor._is_qvm() is True

    def test_qpu_name_is_not_qvm(self) -> None:
        """A bare QPU id is not treated as a QVM."""
        executor = RigettiExecutor(RigettiExecutorConfig(quantum_processor_id="Ankaa-3"))
        assert executor._is_qvm() is False

    def test_as_qvm_override_wins(self) -> None:
        """An explicit as_qvm overrides the name-based inference."""
        executor = RigettiExecutor(
            RigettiExecutorConfig(quantum_processor_id="Ankaa-3", as_qvm=True)
        )
        assert executor._is_qvm() is True


class TestResultNormalisation:
    """Tests for converting a register map into counts."""

    def test_counts_qubit0_is_leftmost(self) -> None:
        """Bitstrings put qubit 0 on the left, matching the SDK convention."""
        # ro[0]=1, ro[1]=0 -> "10": qubit 0 measured 1, qubit 1 measured 0.
        readout = [[1, 0], [1, 0], [1, 1]]
        counts = RigettiExecutor._result_to_counts(_FakeResult(readout), 2)
        assert counts == {"10": 2, "11": 1}

    def test_counts_sum_to_shots(self) -> None:
        """Every shot lands in exactly one bin."""
        readout = [[0, 0], [1, 1], [0, 0], [1, 1], [0, 0]]
        counts = RigettiExecutor._result_to_counts(_FakeResult(readout), 2)
        assert sum(counts.values()) == 5
        assert counts == {"00": 3, "11": 2}

    def test_zero_qubits_returns_empty(self) -> None:
        """A zero-qubit run has no counts."""
        assert RigettiExecutor._result_to_counts(_FakeResult([]), 0) == {}

    def test_missing_ro_register_returns_empty(self) -> None:
        """A result without a ``ro`` register yields empty counts, not an error."""

        class _NoRo:
            def get_register_map(self) -> dict[str, Any]:
                return {}

        assert RigettiExecutor._result_to_counts(_NoRo(), 2) == {}


class TestExecute:
    """Tests for the execute() path with an injected QuantumComputer."""

    @pytest.mark.asyncio
    async def test_execute_returns_normalised_result(self) -> None:
        """execute() compiles, runs and normalises into an ExecutionResult."""
        pytest.importorskip("pyquil")
        result = _FakeResult([[0, 0], [1, 1], [0, 0], [1, 1]])
        qc = _FakeQC(result)
        executor = RigettiExecutor(RigettiExecutorConfig(quantum_processor_id="2q-qvm"), qc=qc)

        out = await executor.execute(bell_state(), shots=4)

        assert isinstance(out, ExecutionResult)
        assert out.counts == {"00": 2, "11": 2}
        assert out.backend == "2q-qvm"
        assert out.shots == 4
        assert out.raw_result is result
        assert out.metadata["provider"] == "rigetti"
        assert out.metadata["quantum_processor_id"] == "2q-qvm"
        assert out.metadata["as_qvm"] is True
        assert out.metadata["num_qubits"] == 2
        # The program handed to the backend declares a readout and measures.
        assert qc.compiled is not None
        assert "MEASURE" in qc.compiled.out()
        assert "DECLARE ro BIT[2]" in qc.compiled.out()

    @pytest.mark.asyncio
    async def test_execute_wraps_backend_error_in_runtimeerror(self) -> None:
        """A backend failure is normalised to RuntimeError naming the processor."""
        pytest.importorskip("pyquil")
        qc = _FakeQC(run_error=ValueError("quilc down"))
        executor = RigettiExecutor(RigettiExecutorConfig(quantum_processor_id="2q-qvm"), qc=qc)

        with pytest.raises(RuntimeError, match="2q-qvm"):
            await executor.execute(bell_state(), shots=10)

    @pytest.mark.asyncio
    async def test_execute_empty_circuit_returns_empty_counts(self) -> None:
        """An empty circuit returns empty counts without touching the backend."""
        # No qc injected and none created: the empty-circuit path must not need one.
        executor = RigettiExecutor(RigettiExecutorConfig(quantum_processor_id="2q-qvm"))
        out = await executor.execute(Circuit(), shots=100)
        assert out.counts == {}
        assert out.shots == 100
        assert out.backend == "2q-qvm"

    @pytest.mark.asyncio
    async def test_execute_rejects_non_marqov_circuit(self) -> None:
        """A non-Marqov circuit is rejected with a helpful TypeError."""
        executor = RigettiExecutor(RigettiExecutorConfig(), qc=_FakeQC(_FakeResult([])))
        with pytest.raises(TypeError):
            await executor.execute("not a circuit", shots=10)  # type: ignore[arg-type]


class TestGetStatus:
    """Tests for device-availability reporting."""

    @pytest.mark.asyncio
    async def test_qvm_is_always_online(self) -> None:
        """A QVM is a local simulator: always online, no queue."""
        executor = RigettiExecutor(RigettiExecutorConfig(quantum_processor_id="2q-qvm"))
        status = await executor.get_status()
        assert status == DeviceStatus(status="online", queue_depth=0, queue_time_seconds=0)

    @pytest.mark.asyncio
    async def test_qpu_online_when_listed(self) -> None:
        """A QPU present in the live processor list reports online."""
        executor = RigettiExecutor(
            RigettiExecutorConfig(quantum_processor_id="Ankaa-3"),
            list_processors=lambda: ["Ankaa-3", "Aspen-M-3"],
        )
        status = await executor.get_status()
        assert status.status == "online"

    @pytest.mark.asyncio
    async def test_qpu_offline_when_not_listed(self) -> None:
        """A QPU missing from the live list reports offline."""
        executor = RigettiExecutor(
            RigettiExecutorConfig(quantum_processor_id="Ankaa-3"),
            list_processors=lambda: ["Aspen-M-3"],
        )
        status = await executor.get_status()
        assert status.status == "offline"

    @pytest.mark.asyncio
    async def test_qpu_maintenance_on_error(self) -> None:
        """If the processor list cannot be fetched, degrade to maintenance."""

        def _boom() -> list[str]:
            raise ConnectionError("QCS unreachable")

        executor = RigettiExecutor(
            RigettiExecutorConfig(quantum_processor_id="Ankaa-3"),
            list_processors=_boom,
        )
        status = await executor.get_status()
        assert status.status == "maintenance"


class TestFactoryRegistration:
    """Tests that the factory knows about the Rigetti provider."""

    def test_provider_is_supported(self) -> None:
        """Rigetti QCS appears in the supported-provider list."""
        assert "Rigetti QCS" in ExecutorFactory.get_supported_providers()
        assert ExecutorFactory.is_provider_supported("Rigetti QCS") is True

    def test_factory_creates_rigetti_executor(self) -> None:
        """The factory builds a RigettiExecutor from a Rigetti QCS config."""
        executor = ExecutorFactory.create_executor(
            "2q-qvm",
            {"provider": "Rigetti QCS"},
        )
        assert isinstance(executor, RigettiExecutor)
        # Processor id falls back to the backend slug when not given.
        assert executor.config.quantum_processor_id == "2q-qvm"

    def test_factory_forwards_config_fields(self) -> None:
        """Explicit config fields reach the executor config."""
        executor = ExecutorFactory.create_executor(
            "ankaa",
            {
                "provider": "Rigetti QCS",
                "quantum_processor_id": "Ankaa-3",
                "as_qvm": False,
                "timeout_seconds": 12.0,
            },
        )
        assert isinstance(executor, RigettiExecutor)
        assert executor.config.quantum_processor_id == "Ankaa-3"
        assert executor.config.as_qvm is False
        assert executor.config.timeout_seconds == 12.0


# Integration template (run against a real local QVM):
#
#   1. Start the containers (see CONTRIBUTING.md §4):
#        docker run -d -p 5555:5555 rigetti/quilc -R
#        docker run -d -p 5000:5000 rigetti/qvm -S
#   2. Run the end-to-end test:
#        MARQOV_QVM_AVAILABLE=1 pytest tests/integration/test_rigetti_qvm.py
#
# That test submits a real Bell program through quilc + qvm and asserts only the
# correlated outcomes appear. See tests/integration/test_rigetti_qvm.py.
