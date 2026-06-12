"""Tests for IonQExecutor (IonQ Direct API).

These tests use an in-memory fake that implements the same interface as
_IonQRestClient, injected via IonQExecutorConfig.client, so no real HTTP
calls or IonQ credentials are needed.
"""

from __future__ import annotations

import pytest

from marqov.circuits import bell_state
from marqov.executors.factory import ExecutorFactory
from marqov.executors.ionq import (
    IonQExecutor,
    IonQExecutorConfig,
    _probabilities_to_counts,
)


class FakeIonQClient:
    """In-memory stand-in for _IonQRestClient.

    Configure `job_sequence` with the list of /jobs and /jobs/{id} responses
    to return in order (first is the create_job response, subsequent are
    get_job polls). `results` is returned from get_results. `backend_info`
    is returned from get_backend.
    """

    def __init__(self, job_sequence, results=None, backend_info=None):
        self.job_sequence = list(job_sequence)
        self.results = results or {}
        self.backend_info = backend_info or {}
        self.created_bodies = []
        self.canceled_job_ids = []
        self.get_job_calls = 0

    def create_job(self, body):
        self.created_bodies.append(body)
        return self.job_sequence[0]

    def get_job(self, job_id):
        self.get_job_calls += 1
        # index 0 is the create_job response; subsequent polls advance.
        idx = min(self.get_job_calls, len(self.job_sequence) - 1)
        return self.job_sequence[idx]

    def get_results(self, job_id):
        return self.results

    def cancel_job(self, job_id):
        self.canceled_job_ids.append(job_id)
        return {"id": job_id, "status": "canceled"}

    def get_backend(self, name):
        return self.backend_info


@pytest.fixture
def circuit():
    return bell_state()


@pytest.mark.asyncio
async def test_execute_success_maps_probabilities_to_counts(circuit):
    client = FakeIonQClient(
        job_sequence=[
            {"id": "job-1", "status": "completed", "target": "simulator", "execution_time": 42},
        ],
        results={"0": 0.5, "3": 0.5},
    )
    config = IonQExecutorConfig(backend="simulator", client=client)
    executor = IonQExecutor(config)

    result = await executor.execute(circuit, shots=1000)

    assert result.counts == {"00": 500, "11": 500}
    assert result.backend == "simulator"
    assert result.shots == 1000
    assert result.execution_time_ms == 42
    assert result.metadata["job_id"] == "job-1"

    # qasm was actually generated and submitted
    submitted = client.created_bodies[0]
    assert submitted["target"] == "simulator"
    assert submitted["shots"] == 1000
    assert submitted["input"]["format"] == "qasm"
    assert "OPENQASM" in submitted["input"]["data"]


@pytest.mark.asyncio
async def test_execute_polls_until_completion(circuit):
    client = FakeIonQClient(
        job_sequence=[
            {"id": "job-2", "status": "submitted", "target": "simulator"},
            {"id": "job-2", "status": "running", "target": "simulator"},
            {"id": "job-2", "status": "completed", "target": "simulator", "execution_time": 10},
        ],
        results={"0": 1.0},
    )
    config = IonQExecutorConfig(backend="simulator", client=client, poll_interval_seconds=0)
    executor = IonQExecutor(config)

    result = await executor.execute(circuit, shots=100)

    assert result.metadata["job_id"] == "job-2"
    assert client.get_job_calls >= 2


@pytest.mark.asyncio
async def test_execute_raises_on_job_failure(circuit):
    client = FakeIonQClient(
        job_sequence=[
            {
                "id": "job-3",
                "status": "failed",
                "target": "simulator",
                "failure": {"error": "circuit too large", "code": "qubit_limit_exceeded"},
            },
        ],
        results={},
    )
    config = IonQExecutorConfig(backend="simulator", client=client)
    executor = IonQExecutor(config)

    with pytest.raises(RuntimeError) as exc_info:
        await executor.execute(circuit, shots=100)

    assert "circuit too large" in str(exc_info.value)
    assert "qubit_limit_exceeded" in str(exc_info.value)


@pytest.mark.asyncio
async def test_cancel_success_and_failure(circuit):
    client = FakeIonQClient(job_sequence=[{"id": "job-4", "status": "completed"}])
    config = IonQExecutorConfig(backend="simulator", client=client)
    executor = IonQExecutor(config)

    assert await executor.cancel("job-4") is True
    assert client.canceled_job_ids == ["job-4"]

    class BrokenClient(FakeIonQClient):
        def cancel_job(self, job_id):
            raise RuntimeError("network error")

    broken_executor = IonQExecutor(IonQExecutorConfig(backend="simulator", client=BrokenClient(job_sequence=[{}])))
    assert await broken_executor.cancel("job-5") is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw_status", "expected_status"),
    [
        ("available", "online"),
        ("running", "online"),
        ("calibrating", "maintenance"),
        ("unavailable", "offline"),
        ("reserved", "offline"),
        ("something_new", "maintenance"),
    ],
)
async def test_get_status_maps_backend_status(raw_status, expected_status):
    client = FakeIonQClient(
        job_sequence=[{}],
        backend_info={"status": raw_status, "backlog": 3, "average_queue_time": 120},
    )
    config = IonQExecutorConfig(backend="qpu.aria-1", client=client)
    executor = IonQExecutor(config)

    status = await executor.get_status()

    assert status.status == expected_status
    assert status.queue_depth == 3
    assert status.queue_time_seconds == 120


@pytest.mark.asyncio
async def test_get_status_returns_maintenance_on_error():
    class BrokenClient(FakeIonQClient):
        def get_backend(self, name):
            raise RuntimeError("network error")

    config = IonQExecutorConfig(backend="simulator", client=BrokenClient(job_sequence=[{}]))
    executor = IonQExecutor(config)

    status = await executor.get_status()

    assert status.status == "maintenance"
    assert status.queue_depth is None
    assert status.queue_time_seconds is None


def test_execute_without_api_key_raises():
    config = IonQExecutorConfig(backend="simulator")  # no api_key, no client
    executor = IonQExecutor(config)

    with pytest.raises(ValueError, match="API key"):
        executor._get_client()


def test_probabilities_to_counts_bit_ordering():
    # 3-qubit example from IonQ's multicircuit docs: {"0": 0.5, "6": 0.5}
    counts = _probabilities_to_counts({"0": 0.5, "6": 0.5}, num_qubits=3, shots=1000)
    assert counts == {"000": 500, "110": 500}


def test_factory_creates_ionq_executor():
    backend_config = {
        "provider": "IonQ Direct",
        "api_key": "test-key",
        "backend": "simulator",
    }
    executor = ExecutorFactory.create_executor("ionq-simulator", backend_config)

    assert isinstance(executor, IonQExecutor)
    assert executor.config.backend == "simulator"
    assert executor.config.api_key == "test-key"


def test_factory_reports_ionq_direct_supported():
    assert ExecutorFactory.is_provider_supported("IonQ Direct") is True
    assert "IonQ Direct" in ExecutorFactory.get_supported_providers()
