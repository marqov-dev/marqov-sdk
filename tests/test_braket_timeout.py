"""Timeout and cancellation through real execute and provider-shaped offline tasks."""

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from braket.aws import AwsQuantumTask

from marqov.circuits import Circuit
from marqov.executors.braket import BraketExecutor, BraketExecutorConfig


class Task:
    def __init__(self, arn, *, complete=False):
        self.id = arn
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.loop = None
        if complete:
            self.release.set()

    def result(self):
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = None
        self.started.set()
        try:
            self.release.wait(1.0)  # Safety limit only; normally released by the test.
            return SimpleNamespace(measurement_counts={"0": 7})
        finally:
            self.finished.set()

    def metadata(self, **kwargs):
        return {"status": "COMPLETED", "shots": 7}


def executor(tasks, *, timeout=0.03, cancel=None):
    options = []
    pending = iter(tasks)

    def run(circuit, **kwargs):
        options.append(kwargs)
        return next(pending)

    device = SimpleNamespace(name="offline", run=run)
    client = SimpleNamespace(cancel_quantum_task=cancel or Mock(return_value={}))
    config = BraketExecutorConfig(device_arn="offline", s3_bucket="unused", timeout_seconds=timeout)
    ex = BraketExecutor(config)
    ex._get_device = AsyncMock(return_value=device)
    ex._aws_session = SimpleNamespace(braket_client=client)
    return ex, options, client.cancel_quantum_task


async def wait_event(event):
    async with asyncio.timeout(2):
        while not event.is_set():
            await asyncio.sleep(0.001)


@pytest.mark.parametrize("interruption", ["timeout", "cancel"])
def test_asyncio_run_returns_without_joining_result_worker(interruption):
    task = Task("offline-task-one")
    ex, options, cancel = executor([task], timeout=0.03 if interruption == "timeout" else None)

    async def run():
        if interruption == "timeout":
            with pytest.raises(TimeoutError):
                await ex.execute(Circuit().x(0), shots=7)
        else:
            call = asyncio.create_task(ex.execute(Circuit().x(0), shots=7))
            await wait_event(task.started)
            call.cancel("caller stopped")
            with pytest.raises(asyncio.CancelledError, match="caller stopped"):
                await call

    start = time.perf_counter()
    try:
        asyncio.run(run())
        assert time.perf_counter() - start < 0.4
        assert not task.finished.is_set()  # Result thread was not joined.
        cancel.assert_called_once_with(quantumTaskArn=task.id)
        if interruption == "timeout":
            assert options[0]["poll_timeout_seconds"] == 0.03
            assert options[0]["poll_interval_seconds"] == 0.03
    finally:
        task.release.set()
        assert task.finished.wait(2)
        # finished is set just before the adapter closes the worker loop.
        deadline = time.monotonic() + 2
        while task.loop is not None and not task.loop.is_closed() and time.monotonic() < deadline:
            time.sleep(0.001)


@pytest.mark.asyncio
async def test_concurrent_completion_does_not_overwrite_timeout_cancel_arn():
    slow, fast = Task("slow-task"), Task("fast-task", complete=True)
    ex, _, cancel = executor([slow, fast], timeout=0.05)
    try:
        pending = asyncio.create_task(ex.execute(Circuit().x(0), shots=7))
        await wait_event(slow.started)
        result = await ex.execute(Circuit().x(0), shots=7)
        assert result.metadata["task_arn"] == ex._current_task_arn == fast.id
        with pytest.raises(TimeoutError):
            await pending
        cancel.assert_called_once_with(quantumTaskArn=slow.id)
        assert result.counts == {"0": 7}
    finally:
        slow.release.set()
        await wait_event(slow.finished)


@pytest.mark.asyncio
async def test_failed_cancel_preserves_original_timeout(caplog):
    task = Task("failed-cleanup-task")
    cancel = Mock(side_effect=RuntimeError("offline cancel failure"))
    ex, _, _ = executor([task], cancel=cancel)
    try:
        with pytest.raises(TimeoutError):
            await ex.execute(Circuit().x(0))
        cancel.assert_called_once_with(quantumTaskArn=task.id)
        assert task.id in caplog.text
        assert "not confirmed" in caplog.text
    finally:
        task.release.set()
        await wait_event(task.finished)


@pytest.mark.asyncio
async def test_slow_cancel_has_bounded_wait_and_preserves_timeout(monkeypatch, caplog):
    from marqov.executors import _blocking

    monkeypatch.setattr(_blocking, "_CANCEL_WAIT_SECONDS", 0.03)
    task = Task("slow-cleanup-task")
    started, release = threading.Event(), threading.Event()

    def cancel(**kwargs):
        started.set()
        release.wait(1)

    ex, _, _ = executor([task], cancel=cancel)
    start = time.perf_counter()
    try:
        with pytest.raises(TimeoutError):
            await ex.execute(Circuit().x(0))
        assert started.is_set()
        assert not release.is_set()
        assert time.perf_counter() - start < 0.4
        assert "not confirmed" in caplog.text
    finally:
        release.set()
        task.release.set()
        await wait_event(task.finished)


@pytest.mark.asyncio
async def test_repeated_cancellation_does_not_mask_first_interruption(caplog):
    task = Task("repeated-cancel-task")
    started, release = threading.Event(), threading.Event()
    ids = []

    def cancel(**kwargs):
        ids.append(kwargs["quantumTaskArn"])
        started.set()
        release.wait(1)

    ex, _, _ = executor([task], timeout=None, cancel=cancel)
    call = asyncio.create_task(ex.execute(Circuit().x(0)))
    try:
        await wait_event(task.started)
        call.cancel("first interruption")
        await wait_event(started)
        call.cancel("second interruption")
        with pytest.raises(asyncio.CancelledError, match="first interruption"):
            await call
        assert ids == [task.id]
    finally:
        release.set()
        task.release.set()
        await wait_event(task.finished)


@pytest.mark.asyncio
async def test_success_without_timeout_preserves_vendor_defaults_and_raw_result():
    task = Task("successful-task", complete=True)
    ex, options, cancel = executor([task], timeout=None)
    result = await ex.execute(Circuit().x(0), shots=7)
    assert "poll_timeout_seconds" not in options[0]
    assert "poll_interval_seconds" not in options[0]
    assert result.raw_result.measurement_counts == result.counts == {"0": 7}
    assert task.loop.is_closed()
    cancel.assert_not_called()


@pytest.mark.asyncio
async def test_real_vendor_poller_gets_budget_and_stops_without_network():
    session = Mock()
    session.get_quantum_task.return_value = {"status": "RUNNING"}
    session.braket_client.cancel_quantum_task.return_value = {}
    made = []
    ex, _, _ = executor([], timeout=0.03)
    ex.config.poll_interval_seconds = 0.005
    ex._aws_session = session

    def run(circuit, **options):
        made.append(
            AwsQuantumTask(
                "arn:aws:braket:us-east-1:123456789012:quantum-task/offline",
                aws_session=session,
                quiet=True,
                poll_timeout_seconds=options["poll_timeout_seconds"],
                poll_interval_seconds=options["poll_interval_seconds"],
            )
        )
        return made[-1]

    ex._get_device = AsyncMock(return_value=SimpleNamespace(name="offline", run=run))
    with pytest.raises(TimeoutError):
        await ex.execute(Circuit().x(0))
    session.braket_client.cancel_quantum_task.assert_called_once_with(quantumTaskArn=made[0].id)
    async with asyncio.timeout(2):
        while not hasattr(made[0], "_future") or not made[0]._future.get_loop().is_closed():
            await asyncio.sleep(0.001)
    assert made[0]._future.done()
    assert session.get_quantum_task.call_count > 0


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_invalid_timeout_is_rejected_before_any_device_access(timeout):
    with pytest.raises(ValueError, match="timeout_seconds"):
        BraketExecutor(
            BraketExecutorConfig(device_arn="offline", s3_bucket="unused", timeout_seconds=timeout)
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "timeout", "error", "cancelled"),
    [
        ("FAILED", 0.03, RuntimeError, False),
        ("CANCELLED", 0.03, RuntimeError, False),
        ("RUNNING", 0.03, TimeoutError, True),
        ("RUNNING", None, RuntimeError, False),
    ],
)
async def test_none_vendor_result_is_explicit(status, timeout, error, cancelled):
    task = SimpleNamespace(
        id="empty-result-task", result=lambda: None, metadata=lambda **kw: {"status": status}
    )
    ex, _, cancel = executor([task], timeout=timeout)
    with pytest.raises(error, match=task.id):
        await ex.execute(Circuit().x(0))
    if cancelled:
        cancel.assert_called_once_with(quantumTaskArn=task.id)
    else:
        cancel.assert_not_called()


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf"), True])
def test_invalid_active_poll_interval_is_rejected_before_submission(interval):
    with pytest.raises(ValueError, match="poll_interval_seconds"):
        BraketExecutor(
            BraketExecutorConfig(
                device_arn="offline",
                s3_bucket="unused",
                timeout_seconds=1,
                poll_interval_seconds=interval,
            )
        )
