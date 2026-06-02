"""Tests for the IonQ Direct executor."""

import pytest

from marqov.circuits import bell_state
from marqov.executors import ExecutionResult, IonQExecutor, IonQExecutorConfig
from marqov.executors.factory import ExecutorFactory
from marqov.executors.ionq import IonQExecutor as DirectIonQExecutor


class FakeIonQClient:
    """Mock IonQ client with the subset of calls used by IonQExecutor."""

    def __init__(self) -> None:
        self.created_payloads: list[dict] = []
        self.cancelled_job_id: str | None = None
        self.job = {
            "id": "job-123",
            "status": "completed",
            "result": {"counts": {"00": 6, "11": 4}},
            "execution_time_ms": 12.5,
        }
        self.backend = {
            "status": "available",
            "queue_depth": 3,
            "queue_time_seconds": 90,
        }

    def create_job(self, payload: dict) -> dict:
        self.created_payloads.append(payload)
        return {"id": "job-123"}

    def get_job(self, job_id: str) -> dict:
        assert job_id == "job-123"
        return self.job

    def cancel_job(self, job_id: str) -> dict:
        self.cancelled_job_id = job_id
        return {"status": "canceled"}

    def get_backend(self, backend: str) -> dict:
        assert backend == "simulator"
        return self.backend


@pytest.mark.asyncio
async def test_execute_submits_qasm_payload_and_returns_result() -> None:
    client = FakeIonQClient()
    executor = IonQExecutor(IonQExecutorConfig(backend="simulator", client=client))

    result = await executor.execute(bell_state(), shots=10)

    assert isinstance(result, ExecutionResult)
    assert result.counts == {"00": 6, "11": 4}
    assert result.backend == "simulator"
    assert result.execution_time_ms == 12.5
    assert result.shots == 10
    assert result.metadata["job_id"] == "job-123"
    assert result.metadata["provider"] == "IonQ Direct"

    payload = client.created_payloads[0]
    assert payload["target"] == "simulator"
    assert payload["shots"] == 10
    assert payload["input"]["format"] == "qasm"
    assert payload["input"]["data"].startswith("OPENQASM")


@pytest.mark.asyncio
async def test_execute_converts_probabilities_to_counts() -> None:
    client = FakeIonQClient()
    client.job = {
        "id": "job-123",
        "status": "completed",
        "result": {"probabilities": {"0": 0.25, "1": 0.75}},
    }
    executor = IonQExecutor(IonQExecutorConfig(backend="simulator", client=client))

    result = await executor.execute(bell_state(), shots=20)

    assert result.counts == {"0": 5, "1": 15}


@pytest.mark.asyncio
async def test_get_status_maps_available_backend_to_device_status() -> None:
    client = FakeIonQClient()
    executor = IonQExecutor(IonQExecutorConfig(backend="simulator", client=client))

    status = await executor.get_status()

    assert status.status == "online"
    assert status.queue_depth == 3
    assert status.queue_time_seconds == 90


@pytest.mark.asyncio
async def test_cancel_delegates_to_client() -> None:
    client = FakeIonQClient()
    executor = IonQExecutor(IonQExecutorConfig(backend="simulator", client=client))

    result = await executor.cancel("job-123")

    assert result is True
    assert client.cancelled_job_id == "job-123"


def test_factory_registers_ionq_direct_provider() -> None:
    client = FakeIonQClient()
    executor = ExecutorFactory.create_executor(
        "simulator",
        {"provider": "IonQ Direct", "client": client},
    )

    assert isinstance(executor, DirectIonQExecutor)
    assert ExecutorFactory.is_provider_supported("IonQ Direct")


def test_rest_payload_uses_ionq_v04_qasm_shape() -> None:
    payload = DirectIonQExecutor._to_ionq_api_payload(
        {
            "backend": "simulator",
            "shots": 10,
            "input": {"format": "qasm", "data": "OPENQASM 3.0;"},
            "metadata": {"suite": "unit"},
        }
    )

    assert payload == {
        "type": "ionq.circuit.v1",
        "backend": "simulator",
        "shots": 10,
        "input": {"qasm": "OPENQASM 3.0;"},
        "metadata": {"suite": "unit"},
    }
