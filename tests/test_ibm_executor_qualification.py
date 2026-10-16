"""Real IBM Runtime SamplerV2 local execution and submitted-job identity."""

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip("qiskit_ibm_runtime")
pytest.importorskip("qiskit_aer")
import qiskit_ibm_runtime
from qiskit_aer import AerSimulator
from marqov.circuits import Circuit
from marqov.executors import IBMExecutor, IBMExecutorConfig
from marqov.executors.ibm import IBMExecutionError


@pytest.mark.asyncio
@pytest.mark.parametrize("qubit,key", [(0, "10"), (1, "01")])
async def test_real_sampler_local_counts_and_provenance(qubit, key):
    executor = IBMExecutor(IBMExecutorConfig(backend_name="aer_simulator"))
    # Explicit offline qualification backend, never a remote failure fallback.
    executor._backend = AerSimulator(seed_simulator=7)
    result = await executor.execute(Circuit().x(qubit).cz(0, 1), shots=32)
    assert result.counts == {key: 32}
    assert result.metadata["access_path"] == "local"
    assert result.metadata["engine"] == "SamplerV2"
    assert result.metadata["job_id"]
    assert len(result.metadata["reproducibility"]["counts_sha256"]) == 64


@pytest.mark.asyncio
async def test_unresolved_result_retains_submitted_job(monkeypatch):
    executor = IBMExecutor(IBMExecutorConfig(backend_name="configured-provider-device"))
    executor._backend = Mock()
    monkeypatch.setattr(executor, "_transpile_sync", lambda circuit, backend: circuit)
    job = Mock()
    job.job_id.return_value = "existing-provider-job"
    job.result.side_effect = TimeoutError()
    submit = Mock(return_value=job)
    monkeypatch.setattr(executor, "_submit_sampler_sync", submit)
    with pytest.raises(IBMExecutionError) as error:
        await executor.execute(Circuit().h(0), shots=32)
    assert error.value.job_id == "existing-provider-job"
    assert submit.call_count == 1


def test_token_not_in_config_repr():
    assert "private-test" not in repr(IBMExecutorConfig(backend_name="test", token="private-test"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "shots,accepted",
    [(np.int64(16), True), (np.uint8(16), True), (True, False), (3.0, False), (0, False)],
)
async def test_shots_accept_numpy_integers_only(shots, accepted):
    executor = IBMExecutor(IBMExecutorConfig(backend_name="aer_simulator"))
    executor._backend = AerSimulator(seed_simulator=7)
    if not accepted:
        with pytest.raises(ValueError, match="positive integer"):
            await executor.execute(Circuit().x(0), shots=shots)
        return
    result = await executor.execute(Circuit().x(0), shots=shots)
    assert result.counts == {"1": 16}
    assert type(result.shots) is int


def _stub_executor(monkeypatch, job, timeout_seconds=None):
    executor = IBMExecutor(
        IBMExecutorConfig(backend_name="configured-provider-device", timeout_seconds=timeout_seconds)
    )
    executor._backend = Mock()
    monkeypatch.setattr(executor, "_transpile_sync", lambda circuit, backend: circuit)
    monkeypatch.setattr(executor, "_submit_sampler_sync", lambda circuit, backend, shots: job)
    return executor


def test_timeout_returns_from_asyncio_run_without_joining_result_thread(monkeypatch):
    finished = threading.Event()

    def slow_result():
        try:
            time.sleep(1.5)  # stands in for a queued IBM job
        finally:
            finished.set()

    job = Mock()
    job.job_id.return_value = "queued-provider-job"
    job.result.side_effect = slow_result
    executor = _stub_executor(monkeypatch, job, timeout_seconds=0.05)

    async def main():
        with pytest.raises(IBMExecutionError) as error:
            await executor.execute(Circuit().h(0), shots=32)
        assert error.value.job_id == "queued-provider-job"
        return time.perf_counter()

    raised_at = asyncio.run(main())
    assert time.perf_counter() - raised_at < 0.5
    assert not finished.is_set()
    job.cancel.assert_not_called()
    finished.wait(5)


@pytest.mark.asyncio
async def test_cancelled_call_retains_its_own_job_under_concurrency(monkeypatch):
    release = threading.Event()
    started = {"job-A": threading.Event(), "job-B": threading.Event()}

    def make_job(job_id):
        job = Mock()
        job.job_id.return_value = job_id

        def result():
            started[job_id].set()
            release.wait(10)
            return Mock()

        job.result.side_effect = result
        return job

    jobs = iter([make_job("job-A"), make_job("job-B")])

    class FakeSampler:
        def __init__(self, mode):
            pass

        def run(self, circuits, shots):
            return next(jobs)

    monkeypatch.setattr(qiskit_ibm_runtime, "SamplerV2", FakeSampler)
    executor = IBMExecutor(IBMExecutorConfig(backend_name="configured-provider-device"))
    executor._backend = Mock()
    monkeypatch.setattr(executor, "_transpile_sync", lambda circuit, backend: circuit)

    async def wait_started(job_id):
        while not started[job_id].is_set():
            await asyncio.sleep(0.005)

    call_a = asyncio.create_task(executor.execute(Circuit().h(0), shots=8))
    call_b = None
    try:
        await asyncio.wait_for(wait_started("job-A"), 5)
        call_b = asyncio.create_task(executor.execute(Circuit().h(0), shots=8))
        await asyncio.wait_for(wait_started("job-B"), 5)
        assert executor._current_job_id == "job-B"
        call_a.cancel()
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await call_a
        assert cancelled.value.remote_job["job_id"] == "job-A"
        assert any("job-A" in note for note in cancelled.value.__notes__)
        assert not call_b.done()
    finally:
        release.set()
        if call_b is not None:
            call_b.cancel()
            with pytest.raises(asyncio.CancelledError):
                await call_b


def _sampler_result(**registers):
    data = SimpleNamespace(keys=lambda: list(registers), **{
        name: SimpleNamespace(get_counts=lambda counts=counts: counts)
        for name, counts in registers.items()
    })
    return [SimpleNamespace(data=data)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result,expected",
    [
        (_sampler_result(c0={"0": 8}, c1={"1": 8}), NotImplementedError),
        (_sampler_result(meas={"0": 7}), ValueError),
    ],
    ids=["multi-register", "shot-mismatch"],
)
async def test_sdk_refusal_of_resolved_result_is_not_execution_error(
    monkeypatch, result, expected
):
    job = Mock()
    job.job_id.return_value = "resolved-provider-job"
    job.result.return_value = result
    executor = _stub_executor(monkeypatch, job)
    with pytest.raises(expected) as error:
        await executor.execute(Circuit().h(0), shots=8)
    assert not isinstance(error.value, IBMExecutionError)
    assert "resolved-provider-job" in str(error.value)


@pytest.mark.asyncio
async def test_undecodable_resolved_result_keeps_type_and_notes_job(monkeypatch):
    job = Mock()
    job.job_id.return_value = "resolved-provider-job"
    job.result.return_value = []  # result[0] raises IndexError in extraction
    executor = _stub_executor(monkeypatch, job)
    with pytest.raises(IndexError) as error:
        await executor.execute(Circuit().h(0), shots=8)
    assert any("resolved-provider-job" in note for note in error.value.__notes__)
