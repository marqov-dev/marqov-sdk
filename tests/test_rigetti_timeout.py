"""Rigetti interruption through real QPU submit/cancel primitives without QCS."""

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pyquil.api import QPU, EncryptedProgram

from marqov.circuits import Circuit
from marqov.executors.rigetti import RigettiExecutor, RigettiExecutorConfig


class Poll:
    def __init__(self, *, complete=False):
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        if complete:
            self.release.set()

    def __call__(self, response):
        self.started.set()
        try:
            self.release.wait(1)
            return SimpleNamespace(get_register_map=lambda: {"ro": [[1, 0]] * 7})
        finally:
            self.finished.set()


def executor(monkeypatch, ids, polls, *, timeout=0.03, cancel=None, compile=None, submit=None):
    from pyquil.api import _qpu

    client = Mock()
    qpu = QPU(quantum_processor_id="actual-qpu", client_configuration=client)
    pending = iter(ids)
    submit_api = Mock(side_effect=submit or (lambda **kw: [next(pending)]))
    cancel_api = Mock(side_effect=cancel)
    monkeypatch.setattr(_qpu, "submit_with_parameter_batch", submit_api)
    monkeypatch.setattr(_qpu, "cancel_job", cancel_api)
    pending_polls = iter(polls)
    qpu.get_result = lambda response: next(pending_polls)(response)
    qc = SimpleNamespace(
        qam=qpu,
        compile=compile or (lambda p: EncryptedProgram("offline", {}, {})),
        run=qpu.run,
    )
    ex = RigettiExecutor(
        RigettiExecutorConfig(
            quantum_processor_id="requested-alias",
            as_qvm=False,
            timeout_seconds=timeout,
        ),
        qc=qc,
    )
    return ex, qpu, client, submit_api, cancel_api


async def wait_event(event):
    async with asyncio.timeout(2):
        while not event.is_set():
            await asyncio.sleep(0.001)


@pytest.mark.parametrize("interruption", ["timeout", "cancel"])
def test_asyncio_run_returns_without_joining_and_cancels_exact_native_job(
    monkeypatch, interruption
):
    poll = Poll()
    ex, qpu, client, _, cancel = executor(
        monkeypatch, ["owned"], [poll], timeout=0.05 if interruption == "timeout" else None
    )
    errors = []

    async def run():
        if interruption == "timeout":
            with pytest.raises(TimeoutError) as caught:
                await ex.execute(Circuit().x(0).z(1), shots=7)
        else:
            call = asyncio.create_task(ex.execute(Circuit().x(0).z(1)))
            await wait_event(poll.started)
            call.cancel("original")
            with pytest.raises(asyncio.CancelledError, match="original") as caught:
                await call
        errors.append(caught.value)

    start = time.perf_counter()
    try:
        asyncio.run(run())
        assert time.perf_counter() - start < 0.4
        assert not poll.finished.is_set()
        cancel.assert_called_once_with("owned", "actual-qpu", client, qpu.execution_options)
        assert errors[0].remote_job["job_id"] == "owned"
        assert errors[0].remote_job["quantum_processor_id"] == "actual-qpu"
        assert errors[0].remote_job["submission_status"] == "submitted"
        assert errors[0].__notes__
    finally:
        poll.release.set()
        assert poll.finished.wait(2)


@pytest.mark.asyncio
async def test_concurrent_success_does_not_replace_owned_handle(monkeypatch):
    slow, fast = Poll(), Poll(complete=True)
    ex, qpu, client, _, cancel = executor(monkeypatch, ["slow", "fast"], [slow, fast], timeout=0.1)
    call = asyncio.create_task(ex.execute(Circuit().x(0).z(1)))
    try:
        await wait_event(slow.started)
        result = await ex.execute(Circuit().x(0).z(1), shots=7)
        assert result.counts == {"10": 7}
        assert result.metadata["job_id"] == "fast"
        assert result.backend == "actual-qpu"
        with pytest.raises(TimeoutError) as caught:
            await call
        assert caught.value.remote_job["job_id"] == "slow"
        cancel.assert_called_once_with("slow", "actual-qpu", client, qpu.execution_options)
    finally:
        slow.release.set()
        await wait_event(slow.finished)


@pytest.mark.asyncio
async def test_failed_cleanup_keeps_timeout_and_recent_handle(monkeypatch, caplog):
    poll = Poll()
    ex, _, _, _, cancel = executor(
        monkeypatch, ["failed"], [poll], cancel=RuntimeError("offline failure")
    )
    try:
        with pytest.raises(TimeoutError) as caught:
            await ex.execute(Circuit().x(0))
        assert caught.value.remote_job["job_id"] == "failed"
        assert "not confirmed" in caplog.text
        cancel.side_effect = None
        assert await ex.cancel("failed") is True
        assert cancel.call_count == 2  # The second attempt is an explicit user request.
        assert await ex.cancel("other-instance") is False
    finally:
        poll.release.set()
        await wait_event(poll.finished)


@pytest.mark.asyncio
@pytest.mark.parametrize("repeat", [False, True])
async def test_slow_cleanup_and_repeated_cancel_keep_original(monkeypatch, repeat):
    from marqov.executors import _blocking

    monkeypatch.setattr(_blocking, "_CANCEL_WAIT_SECONDS", 0.03)
    started, release = threading.Event(), threading.Event()

    def cancel(*args):
        started.set()
        release.wait(1)

    poll = Poll()
    ex, _, _, _, cancel_api = executor(
        monkeypatch, ["slow-cleanup"], [poll], timeout=None, cancel=cancel
    )
    call = asyncio.create_task(ex.execute(Circuit().x(0)))
    try:
        await wait_event(poll.started)
        start = time.perf_counter()
        call.cancel("first")
        await wait_event(started)
        if repeat:
            call.cancel("second")
        with pytest.raises(asyncio.CancelledError, match="first") as caught:
            await call
        assert time.perf_counter() - start < 0.4
        assert caught.value.remote_job["job_id"] == "slow-cleanup"
        cancel_api.assert_called_once()
    finally:
        release.set()
        poll.release.set()
        await wait_event(poll.finished)


@pytest.mark.asyncio
async def test_deadline_then_caller_cancel_during_cleanup_preserves_timeout(monkeypatch):
    started, release = threading.Event(), threading.Event()

    def cancel(*args):
        started.set()
        release.wait(1)

    poll = Poll()
    ex, _, _, _, cancel_api = executor(monkeypatch, ["timed-out"], [poll], cancel=cancel)
    call = asyncio.create_task(ex.execute(Circuit().x(0)))
    try:
        await wait_event(started)
        call.cancel("second interruption after timeout")
        with pytest.raises(TimeoutError) as caught:
            await call
        assert caught.value.remote_job["job_id"] == "timed-out"
        cancel_api.assert_called_once()
    finally:
        release.set()
        poll.release.set()
        await wait_event(poll.finished)


@pytest.mark.asyncio
async def test_timeout_during_compile_does_not_submit_later(monkeypatch):
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def compile(program):
        started.set()
        try:
            release.wait(1)
            return EncryptedProgram("offline", {}, {})
        finally:
            finished.set()

    ex, _, _, submit, cancel = executor(monkeypatch, [], [], compile=compile)
    try:
        with pytest.raises(TimeoutError) as caught:
            await ex.execute(Circuit().x(0))
        assert caught.value.remote_job["phase"] == "compilation"
        assert caught.value.remote_job["submission_status"] == "not_submitted"
        submit.assert_not_called()
        cancel.assert_not_called()
    finally:
        release.set()
        await wait_event(finished)
        submit.assert_not_called()


@pytest.mark.asyncio
async def test_interrupted_native_submission_is_unknown_without_replay(monkeypatch):
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def submit(**kwargs):
        started.set()
        try:
            release.wait(1)
            return ["late"]
        finally:
            finished.set()

    ex, _, _, submit_api, cancel = executor(monkeypatch, [], [], timeout=None, submit=submit)
    call = asyncio.create_task(ex.execute(Circuit().x(0)))
    try:
        await wait_event(started)
        call.cancel("submission interrupted")
        with pytest.raises(asyncio.CancelledError) as caught:
            await call
        assert caught.value.remote_job["job_id"] is None
        assert caught.value.remote_job["submission_status"] == "unknown"
        submit_api.assert_called_once()
        cancel.assert_not_called()
    finally:
        release.set()
        await wait_event(finished)


@pytest.mark.asyncio
async def test_qvm_run_only_timeout_reports_missing_handle():
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def run(program):
        started.set()
        try:
            release.wait(1)
        finally:
            finished.set()

    qc = SimpleNamespace(compile=lambda p: p, run=run)
    ex = RigettiExecutor(RigettiExecutorConfig(timeout_seconds=0.03), qc=qc)
    try:
        with pytest.raises(TimeoutError) as caught:
            await ex.execute(Circuit().x(0))
        assert caught.value.remote_job["job_id"] is None
        assert caught.value.remote_job["cancellation_supported"] is False
        assert await ex.cancel("unavailable") is False
    finally:
        release.set()
        await wait_event(finished)


@pytest.mark.asyncio
async def test_completed_jobs_are_removed_and_recent_cache_is_bounded(monkeypatch):
    def timeout(response):
        raise TimeoutError("vendor timeout")

    ex, _, _, _, cancel = executor(
        monkeypatch, [str(i) for i in range(33)], [timeout] * 33, timeout=None
    )
    for _ in range(33):
        with pytest.raises(TimeoutError):
            await ex.execute(Circuit().x(0))
    assert await ex.cancel("0") is False
    assert await ex.cancel("32") is True
    assert cancel.call_count == 34

    success = Poll(complete=True)
    ex, _, _, _, cancel = executor(monkeypatch, ["completed"], [success], timeout=None)
    result = await ex.execute(Circuit().x(0).z(1), shots=7)
    assert result.counts == {"10": 7}
    assert await ex.cancel("completed") is False
    cancel.assert_not_called()


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_invalid_overall_budget_rejected(timeout):
    with pytest.raises(ValueError, match="timeout_seconds"):
        RigettiExecutorConfig(timeout_seconds=timeout)


@pytest.mark.asyncio
async def test_explicit_active_cancel_uses_original_client(monkeypatch):
    poll = Poll()
    ex, qpu, client, _, cancel = executor(monkeypatch, ["active"], [poll], timeout=None)
    call = asyncio.create_task(ex.execute(Circuit().x(0).z(1), shots=7))
    try:
        await wait_event(poll.started)
        ex.config.quantum_processor_id = "changed-after-submit"
        assert await ex.cancel("active") is True
        cancel.assert_called_once_with("active", "actual-qpu", client, qpu.execution_options)
        poll.release.set()
        result = await call
        assert result.metadata["requested_quantum_processor_id"] == "requested-alias"
        assert result.backend == "actual-qpu"
        assert result.raw_result.get_register_map() == {"ro": [[1, 0]] * 7}
    finally:
        poll.release.set()
        await call


@pytest.mark.asyncio
async def test_result_conversion_failure_retains_native_identity(monkeypatch):
    def malformed(response):
        return SimpleNamespace(get_register_map=lambda: {"ro": [["not-a-bit"]]})

    ex, _, _, _, cancel = executor(monkeypatch, ["finished"], [malformed], timeout=None)
    with pytest.raises(RuntimeError, match="actual-qpu") as caught:
        await ex.execute(Circuit().x(0))
    assert isinstance(caught.value.__cause__, ValueError)
    assert caught.value.remote_job["job_id"] == "finished"
    assert await ex.cancel("finished") is False
    cancel.assert_not_called()
