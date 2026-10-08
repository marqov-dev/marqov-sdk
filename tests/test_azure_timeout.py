"""Azure Qiskit interruption and recovery identity through real execute paths."""

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from marqov.circuits import Circuit
from marqov.executors.azure import AzureQuantumExecutor, AzureQuantumExecutorConfig


class Job:
    def __init__(self, job_id, *, complete=False, cancel=None):
        self.id = job_id
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.timeouts = []
        self.cancel = cancel or Mock()
        if complete:
            self.release.set()

    def job_id(self):
        return self.id

    def result(self, timeout=None):
        self.timeouts.append(timeout)
        self.started.set()
        try:
            self.release.wait(1)
            return SimpleNamespace(success=True, get_counts=lambda: {"001": 7})
        finally:
            self.finished.set()

    def properties(self):
        return SimpleNamespace(execution_time=0.012, cost_estimate="offline-estimate")


def executor(jobs, *, timeout=0.03, run=None):
    pending = iter(jobs)
    ex = AzureQuantumExecutor(
        AzureQuantumExecutorConfig(
            subscription_id="subscription",
            resource_group="group",
            workspace_name="workspace",
            location="eastus",
            target="ionq.simulator",
            timeout_seconds=timeout,
        )
    )
    ex._backend = SimpleNamespace(run=run or (lambda *a, **kw: next(pending)))
    return ex


def assert_identity(error, job_id, *, status="submitted"):
    assert error.remote_job == {
        "provider": "Azure Quantum",
        "framework": "qiskit",
        "subscription_id": "subscription",
        "resource_group": "group",
        "workspace_name": "workspace",
        "location": "eastus",
        "backend": "ionq.simulator",
        "job_id": job_id,
        "submission_status": status,
    }
    assert error.__notes__


async def wait_event(event):
    async with asyncio.timeout(2):
        while not event.is_set():
            await asyncio.sleep(0.001)


@pytest.mark.parametrize("interruption", ["timeout", "cancel"])
def test_asyncio_run_returns_without_joining_and_retains_identity(interruption):
    job = Job("owned-job")
    ex = executor([job], timeout=0.03 if interruption == "timeout" else None)

    errors = []

    async def run():
        if interruption == "timeout":
            with pytest.raises(TimeoutError) as caught:
                await ex.execute(Circuit().x(0))
        else:
            call = asyncio.create_task(ex.execute(Circuit().x(0)))
            await wait_event(job.started)
            call.cancel("original interruption")
            with pytest.raises(asyncio.CancelledError, match="original interruption") as caught:
                await call
        errors.append(caught.value)

    start = time.perf_counter()
    try:
        asyncio.run(run())
        assert time.perf_counter() - start < 0.4
        assert not job.finished.is_set()
        job.cancel.assert_called_once_with()
        assert_identity(errors[0], job.id)
        assert job.timeouts == [ex.config.timeout_seconds]
    finally:
        job.release.set()
        assert job.finished.wait(2)


@pytest.mark.asyncio
async def test_concurrent_completion_cannot_replace_interrupted_identity():
    slow, fast = Job("slow"), Job("fast", complete=True)
    ex = executor([slow, fast], timeout=0.05)
    call = asyncio.create_task(ex.execute(Circuit().x(0)))
    try:
        await wait_event(slow.started)
        result = await ex.execute(Circuit().x(0), shots=7)
        assert result.metadata["job_id"] == ex._current_job_id == fast.id
        with pytest.raises(TimeoutError) as caught:
            await call
        assert_identity(caught.value, slow.id)
        slow.cancel.assert_called_once_with()
        fast.cancel.assert_not_called()
    finally:
        slow.release.set()
        await wait_event(slow.finished)


@pytest.mark.asyncio
async def test_failed_cleanup_keeps_timeout_and_identity(caplog):
    job = Job("failed-cleanup", cancel=Mock(side_effect=RuntimeError("offline failure")))
    ex = executor([job])
    try:
        with pytest.raises(TimeoutError) as caught:
            await ex.execute(Circuit().x(0))
        assert_identity(caught.value, job.id)
        assert "not confirmed" in caplog.text
        job.cancel.assert_called_once_with()
    finally:
        job.release.set()
        await wait_event(job.finished)


@pytest.mark.asyncio
async def test_repeated_cancellation_keeps_first_error_and_identity(caplog):
    started, release = threading.Event(), threading.Event()

    def cancel():
        started.set()
        release.wait(1)

    job = Job("repeat", cancel=Mock(side_effect=cancel))
    ex = executor([job], timeout=None)
    call = asyncio.create_task(ex.execute(Circuit().x(0)))
    try:
        await wait_event(job.started)
        call.cancel("first")
        await wait_event(started)
        call.cancel("second")
        with pytest.raises(asyncio.CancelledError, match="first") as caught:
            await call
        assert_identity(caught.value, job.id)
        job.cancel.assert_called_once_with()
    finally:
        release.set()
        job.release.set()
        await wait_event(job.finished)


@pytest.mark.asyncio
async def test_vendor_timeout_retains_identity_and_requests_cancellation():
    job = Job("vendor-timeout")
    job.result = Mock(side_effect=TimeoutError("vendor budget exhausted"))
    ex = executor([job], timeout=1)
    with pytest.raises(TimeoutError, match="vendor budget exhausted") as caught:
        await ex.execute(Circuit().x(0))
    assert_identity(caught.value, job.id)
    job.result.assert_called_once_with(timeout=1)
    job.cancel.assert_called_once_with()


@pytest.mark.asyncio
async def test_result_failure_retains_identity_without_requesting_cancellation():
    job = Job("failed-result")
    job.result = Mock(side_effect=RuntimeError("result unavailable"))
    ex = executor([job])
    with pytest.raises(RuntimeError, match="result unavailable") as caught:
        await ex.execute(Circuit().x(0))
    assert_identity(caught.value, job.id)
    job.cancel.assert_not_called()


@pytest.mark.asyncio
async def test_success_preserves_counts_raw_timing_and_no_timeout_defaults():
    job = Job("success", complete=True)
    ex = executor([job], timeout=None)
    result = await ex.execute(Circuit().x(0).z(1).z(2), shots=7)
    assert result.counts == {"100": 7}
    assert result.raw_result.get_counts() == {"001": 7}
    assert result.execution_time_ms == 12
    assert result.metadata["cost_estimate"] == "offline-estimate"
    assert result.metadata["workspace_name"] == "workspace"
    assert result.metadata["job_id"] == job.id
    assert result.metadata["wall_time_ms"] >= 0
    job.cancel.assert_not_called()
    assert job.timeouts == [None]


@pytest.mark.asyncio
async def test_interrupted_submission_is_explicitly_unknown_and_not_replayed():
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    job = Job("late-job", complete=True)

    def run(*args, **kwargs):
        started.set()
        try:
            release.wait(1)
            return job
        finally:
            finished.set()

    submit = Mock(side_effect=run)
    ex = executor([], run=submit)
    call = asyncio.create_task(ex.execute(Circuit().x(0)))
    try:
        await wait_event(started)
        call.cancel("submission interrupted")
        with pytest.raises(asyncio.CancelledError, match="submission interrupted") as caught:
            await call
        assert_identity(caught.value, None, status="unknown")
        assert "do not replay" in " ".join(caught.value.__notes__)
        submit.assert_called_once()
        job.cancel.assert_not_called()
    finally:
        release.set()
        await wait_event(finished)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_invalid_budget_rejected(timeout):
    with pytest.raises(ValueError, match="timeout_seconds"):
        executor([], timeout=timeout)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["failed-status", "counts"])
async def test_post_submission_result_processing_errors_keep_job(failure):
    job = Job("processing-failure", complete=True)
    raw = SimpleNamespace(success=failure != "failed-status", status="Failed")
    raw.get_counts = Mock(side_effect=ValueError("counts malformed"))
    job.result = Mock(return_value=raw)
    ex = executor([job])
    with pytest.raises((RuntimeError, ValueError)) as caught:
        await ex.execute(Circuit().x(0))
    assert_identity(caught.value, job.id)
    job.cancel.assert_not_called()


@pytest.mark.asyncio
async def test_slow_cleanup_is_bounded_and_preserves_identity(monkeypatch, caplog):
    from marqov.executors import _blocking

    monkeypatch.setattr(_blocking, "_CANCEL_WAIT_SECONDS", 0.03)
    started, release = threading.Event(), threading.Event()

    def cancel():
        started.set()
        release.wait(1)

    job = Job("slow-cleanup", cancel=Mock(side_effect=cancel))
    ex = executor([job])
    start = time.perf_counter()
    try:
        with pytest.raises(TimeoutError) as caught:
            await ex.execute(Circuit().x(0))
        assert_identity(caught.value, job.id)
        assert started.is_set()
        assert time.perf_counter() - start < 0.4
        assert "not confirmed" in caplog.text
        job.cancel.assert_called_once_with()
    finally:
        release.set()
        job.release.set()
        await wait_event(job.finished)


@pytest.mark.asyncio
async def test_submission_error_keeps_unknown_acceptance_without_replay():
    run = Mock(side_effect=RuntimeError("submission response lost"))
    ex = executor([], run=run)
    with pytest.raises(RuntimeError, match="submission response lost") as caught:
        await ex.execute(Circuit().x(0))
    assert_identity(caught.value, None, status="unknown")
    run.assert_called_once()


@pytest.mark.asyncio
async def test_recovery_identity_does_not_follow_mutated_config():
    job = Job("snapshot")
    ex = executor([job], timeout=0.05)
    call = asyncio.create_task(ex.execute(Circuit().x(0)))
    try:
        await wait_event(job.started)
        ex.config.workspace_name = "different-workspace"
        ex.config.target = "different-target"
        with pytest.raises(TimeoutError) as caught:
            await call
        assert_identity(caught.value, job.id)
    finally:
        job.release.set()
        await wait_event(job.finished)


def test_workspace_and_backend_creation_use_same_captured_route(monkeypatch):
    import sys

    ex = executor([])
    ex.config.workspace_name = "changed"
    ex.config.target = "changed-target"
    workspace = Mock()
    create_workspace = Mock(return_value=workspace)
    provider = Mock()
    create_provider = Mock(return_value=provider)
    monkeypatch.setitem(sys.modules, "azure.quantum", SimpleNamespace(Workspace=create_workspace))
    monkeypatch.setitem(
        sys.modules, "azure.quantum.qiskit", SimpleNamespace(AzureQuantumProvider=create_provider)
    )
    assert ex._create_workspace_sync() is workspace
    create_workspace.assert_called_once_with(
        subscription_id="subscription", resource_group="group", name="workspace", location="eastus"
    )
    ex._get_qiskit_backend_sync(workspace)
    create_provider.assert_called_once_with(workspace=workspace)
    provider.get_backend.assert_called_once_with("ionq.simulator")
