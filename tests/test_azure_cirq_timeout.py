"""Real Azure Cirq service/job wrappers with offline submission and polling."""

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import cirq
import pytest
from azure.quantum.cirq import AzureQuantumService
from azure.quantum.cirq.job import Job

from marqov.circuits import Circuit
from marqov.executors.azure import AzureQuantumExecutor, AzureQuantumExecutorConfig


class OfflineJob(Job):
    def __init__(self, job_id, *, complete=False, cancel=None):
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.budgets = []
        self.workspace = Mock()
        if cancel is not None:
            self.workspace.cancel_job.side_effect = cancel
        self.native = SimpleNamespace(id=job_id, workspace=self.workspace, get_results=self.poll)
        target = SimpleNamespace(_to_cirq_result=lambda **kw: kw["result"])
        super().__init__(self.native, program=cirq.Circuit(), target=target)
        if complete:
            self.release.set()

    def poll(self, *, timeout_secs):
        self.budgets.append(timeout_secs)
        self.started.set()
        try:
            self.release.wait(1)
            return cirq.ResultDict(measurements={"result": [[1, 0]] * 7})
        finally:
            self.finished.set()


def executor(jobs, *, timeout=0.03, submit=None):
    pending = iter(jobs)
    target = Mock()
    target.submit.side_effect = submit or (lambda **kw: next(pending))
    service = AzureQuantumService(workspace=Mock(), default_target="ionq.simulator")
    service.get_target = Mock(return_value=target)
    ex = AzureQuantumExecutor(
        AzureQuantumExecutorConfig(
            subscription_id="subscription",
            resource_group="group",
            workspace_name="workspace",
            location="eastus",
            target="ionq.simulator",
            framework="cirq",
            timeout_seconds=timeout,
        )
    )
    ex._backend = service
    return ex, target


async def wait_event(event):
    async with asyncio.timeout(2):
        while not event.is_set():
            await asyncio.sleep(0.001)


def identity(error, job_id):
    assert error.remote_job["job_id"] == job_id
    assert error.remote_job["framework"] == "cirq"
    assert error.remote_job["workspace_name"] == "workspace"
    assert error.remote_job["subscription_id"] == "subscription"
    assert error.remote_job["backend"] == "ionq.simulator"
    assert error.remote_job["submission_status"] == ("submitted" if job_id else "unknown")


@pytest.mark.parametrize("interruption", ["timeout", "cancel"])
def test_asyncio_run_returns_without_joining_and_cancels_owned_job(interruption):
    job = OfflineJob("owned")
    ex, _ = executor([job], timeout=0.03 if interruption == "timeout" else None)
    errors = []

    async def run():
        if interruption == "timeout":
            with pytest.raises(TimeoutError) as caught:
                await ex.execute(Circuit().x(0).z(1), shots=7)
        else:
            call = asyncio.create_task(ex.execute(Circuit().x(0).z(1), shots=7))
            await wait_event(job.started)
            call.cancel("original")
            with pytest.raises(asyncio.CancelledError, match="original") as caught:
                await call
        errors.append(caught.value)

    start = time.perf_counter()
    try:
        asyncio.run(run())
        assert time.perf_counter() - start < 0.4
        assert not job.finished.is_set()
        job.workspace.cancel_job.assert_called_once_with(job.native)
        identity(errors[0], job.job_id())
    finally:
        job.release.set()
        assert job.finished.wait(2)


@pytest.mark.asyncio
async def test_concurrent_completion_does_not_replace_cancel_or_recovery_identity():
    slow, fast = OfflineJob("slow"), OfflineJob("fast", complete=True)
    ex, _ = executor([slow, fast], timeout=0.05)
    call = asyncio.create_task(ex.execute(Circuit().x(0).z(1)))
    try:
        await wait_event(slow.started)
        result = await ex.execute(Circuit().x(0).z(1), shots=7)
        assert result.metadata["job_id"] == ex._current_job_id == fast.job_id()
        with pytest.raises(TimeoutError) as caught:
            await call
        identity(caught.value, slow.job_id())
        slow.workspace.cancel_job.assert_called_once_with(slow.native)
        fast.workspace.cancel_job.assert_not_called()
    finally:
        slow.release.set()
        await wait_event(slow.finished)


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [None, 1.0])
async def test_success_preserves_native_result_and_vendor_budget(timeout):
    from azure.quantum.job.base_job import DEFAULT_TIMEOUT

    job = OfflineJob("success", complete=True)
    ex, target = executor([job], timeout=timeout)
    result = await ex.execute(Circuit().x(0).z(1), shots=7)
    assert result.counts == {"10": 7}
    assert isinstance(result.raw_result, cirq.Result)
    assert result.metadata["job_id"] == "success"
    assert result.metadata["submission_status"] == "submitted"
    assert result.shots == 7
    assert result.metadata["wall_time_ms"] >= 0
    assert job.budgets == [DEFAULT_TIMEOUT if timeout is None else timeout]
    assert target.submit.call_count == 1
    job.workspace.cancel_job.assert_not_called()


@pytest.mark.asyncio
async def test_failed_cancel_preserves_error_and_identity(caplog):
    job = OfflineJob("failed-cleanup", cancel=RuntimeError("offline cancel failure"))
    ex, _ = executor([job])
    try:
        with pytest.raises(TimeoutError) as caught:
            await ex.execute(Circuit().x(0).z(1))
        identity(caught.value, job.job_id())
        assert "not confirmed" in caplog.text
        job.workspace.cancel_job.assert_called_once_with(job.native)
    finally:
        job.release.set()
        await wait_event(job.finished)


@pytest.mark.asyncio
@pytest.mark.parametrize("repeat", [False, True])
async def test_slow_cleanup_and_repeated_cancellation_preserve_identity(monkeypatch, repeat):
    from marqov.executors import _blocking

    monkeypatch.setattr(_blocking, "_CANCEL_WAIT_SECONDS", 0.03)
    started, release = threading.Event(), threading.Event()

    def cancel(native):
        started.set()
        release.wait(1)

    job = OfflineJob("slow-cleanup", cancel=cancel)
    ex, _ = executor([job], timeout=None)
    call = asyncio.create_task(ex.execute(Circuit().x(0).z(1)))
    try:
        await wait_event(job.started)
        start = time.perf_counter()
        call.cancel("first")
        await wait_event(started)
        if repeat:
            call.cancel("second")
        with pytest.raises(asyncio.CancelledError, match="first") as caught:
            await call
        assert time.perf_counter() - start < 0.4
        identity(caught.value, job.job_id())
        job.workspace.cancel_job.assert_called_once_with(job.native)
    finally:
        release.set()
        job.release.set()
        await wait_event(job.finished)


@pytest.mark.asyncio
async def test_submission_error_is_unknown_and_not_replayed():
    submit = Mock(side_effect=RuntimeError("submission response lost"))
    ex, _ = executor([], submit=submit)
    with pytest.raises(RuntimeError, match="submission response lost") as caught:
        await ex.execute(Circuit().x(0))
    identity(caught.value, None)
    submit.assert_called_once()


@pytest.mark.asyncio
async def test_result_error_retains_job_without_cancellation():
    job = OfflineJob("failed-result")
    job.native.get_results = Mock(side_effect=RuntimeError("provider failed"))
    ex, _ = executor([job])
    with pytest.raises(RuntimeError, match="provider failed") as caught:
        await ex.execute(Circuit().x(0))
    identity(caught.value, job.job_id())
    job.workspace.cancel_job.assert_not_called()


@pytest.mark.asyncio
async def test_ionq_provider_result_keeps_vendor_conversion():
    from azure.quantum.cirq.targets.ionq import IonQTarget
    from cirq_ionq import Job as IonQJob
    from cirq_ionq.results import SimulatorResult

    client = Mock()
    job = IonQJob(client=client, job_dict={"id": "ionq-native", "status": "completed"})
    job.results = Mock(
        return_value=[
            SimulatorResult(
                probabilities={2: 1.0},
                num_qubits=2,
                measurement_dict={"result": [0, 1]},
                repetitions=7,
            )
        ]
    )
    ex, target = executor([job], timeout=1)
    target._to_cirq_result = IonQTarget._to_cirq_result
    result = await ex.execute(Circuit().x(0).z(1), shots=7)
    assert result.counts == {"10": 7}
    assert isinstance(result.raw_result, cirq.Result)
    assert result.metadata["job_id"] == "ionq-native"
    job.results.assert_called_once_with(timeout_seconds=1)
    client.cancel_job.assert_not_called()


@pytest.mark.asyncio
async def test_real_ionq_polling_timeout_cancels_its_native_id():
    from azure.quantum.cirq.targets.ionq import IonQTarget
    from cirq_ionq import Job as IonQJob

    client = Mock()
    running = {"id": "ionq-running", "status": "running"}
    client.get_job.return_value = running
    client.cancel_job.return_value = running
    job = IonQJob(client=client, job_dict=running)
    original = job.results
    finished = threading.Event()

    def poll(**kwargs):
        try:
            return original(**kwargs)
        finally:
            finished.set()

    job.results = poll
    ex, target = executor([job], timeout=0.03)
    target._to_cirq_result = IonQTarget._to_cirq_result
    with pytest.raises(TimeoutError) as caught:
        await ex.execute(Circuit().x(0).z(1))
    identity(caught.value, job.job_id())
    client.cancel_job.assert_called_once_with(job_id="ionq-running")
    await wait_event(finished)


@pytest.mark.asyncio
async def test_interrupted_submission_retains_unknown_acceptance():
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    job = OfflineJob("late", complete=True)

    def submit(**kwargs):
        started.set()
        try:
            release.wait(1)
            return job
        finally:
            finished.set()

    ex, target = executor([], submit=submit)
    call = asyncio.create_task(ex.execute(Circuit().x(0)))
    try:
        await wait_event(started)
        call.cancel("submission interrupted")
        with pytest.raises(asyncio.CancelledError, match="submission interrupted") as caught:
            await call
        identity(caught.value, None)
        target.submit.assert_called_once()
        job.workspace.cancel_job.assert_not_called()
    finally:
        release.set()
        await wait_event(finished)


@pytest.mark.asyncio
async def test_legacy_ionq_vendor_poll_error_is_mapped_and_cancelled_with_default_budget():
    from azure.quantum.job.base_job import DEFAULT_TIMEOUT

    job = SimpleNamespace(
        job_id=lambda: "legacy-ionq",
        status=lambda: "running",
        cancel=Mock(),
        results=Mock(
            side_effect=RuntimeError(
                "Job was not completed successful. Instead had status: running"
            )
        ),
    )
    ex, _ = executor([job], timeout=None)
    with pytest.raises(TimeoutError, match="wait time has exceeded") as caught:
        await ex.execute(Circuit().x(0))
    identity(caught.value, "legacy-ionq")
    job.results.assert_called_once_with(timeout_seconds=DEFAULT_TIMEOUT)
    job.cancel.assert_called_once_with()
