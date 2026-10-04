"""Real vendor local-model checks and explicit submission boundaries."""

from unittest.mock import Mock

import pytest

pytest.importorskip("qiskit_alice_bob_provider")
from qiskit import QuantumCircuit

from marqov.circuits import Circuit
from marqov.executors import AliceBobExecutor, AliceBobExecutorConfig, ExecutorFactory
from marqov.executors.alice_bob import AliceBobExecutionError


@pytest.mark.asyncio
async def test_real_basis_counts_and_provenance():
    executor = ExecutorFactory.create_executor("alice-bob-local", {"provider": "Alice & Bob"})
    result = await executor.execute(Circuit().x(0).cz(0, 1), shots=32, seed=7)
    assert result.counts == {"10": 32}
    assert result.metadata["vendor"] == "Alice & Bob"
    assert result.metadata["access_path"] == "local"
    assert result.metadata["reproducibility"]["packages"]["qiskit-alice-bob-provider"] == "1.2.0"


@pytest.mark.asyncio
async def test_native_initialization_and_delay_are_preserved():
    circuit = QuantumCircuit(1, 1)
    circuit.initialize("+")
    circuit.delay(1, 0, unit="ms")
    circuit.h(0)
    circuit.measure(0, 0)
    result = await AliceBobExecutor().execute_native(circuit, shots=32, seed=7)
    assert result.counts == {"0": 32}
    assert circuit.count_ops()["delay"] == 1


@pytest.mark.asyncio
async def test_permuted_measurements_refused_before_submission():
    executor = AliceBobExecutor()
    executor._backend = Mock()
    circuit = QuantumCircuit(2, 2)
    circuit.measure(0, 1)
    circuit.measure(1, 0)
    with pytest.raises(NotImplementedError):
        await executor.execute_native(circuit, shots=32)
    executor._backend.run.assert_not_called()


@pytest.mark.asyncio
async def test_unresolved_job_retains_id_without_resubmitting(monkeypatch):
    executor = AliceBobExecutor()
    executor.config.mode = "remote"
    backend = Mock()
    backend.run.return_value.job_id.return_value = "retained-job"
    backend.run.return_value.result.side_effect = TimeoutError()
    executor._backend = backend
    monkeypatch.setattr("qiskit.transpile", lambda circuit, *args, **kwargs: circuit)
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    with pytest.raises(AliceBobExecutionError) as error:
        await executor.execute_native(circuit, shots=32)
    assert error.value.job_id == "retained-job"
    assert backend.run.call_count == 1


@pytest.mark.parametrize("timeout", [0, float("nan"), float("inf"), True])
def test_bad_timeout_refused(timeout):
    with pytest.raises(ValueError):
        AliceBobExecutorConfig(mode="remote", api_key="private-test", timeout_seconds=timeout)


def test_credentials_not_logged_or_used_locally():
    config = AliceBobExecutorConfig(mode="remote", api_key="private-test")
    assert "private-test" not in repr(config)
    with pytest.raises(ValueError):
        AliceBobExecutorConfig(api_key="private-test")
