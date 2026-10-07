"""Submission cancellation races through a blocking, provider-shaped transport."""

import asyncio
import threading
from unittest.mock import Mock, patch

import pytest

from marqov.circuits import Circuit
from marqov.executors.ionq import IonQExecutor, IonQExecutorConfig


class Transport:
    def __init__(self):
        self.post_started = threading.Event()
        self.release_post = threading.Event()
        self.put_started = threading.Event()
        self.release_put = threading.Event()
        self.release_put.set()
        self.requests = []
        self.created = []
        self.cancelled = []
        self.response = {"id": "job-interrupted"}
        self.post_error = None
        self.put_error = None

    def request(self, method, url, **kwargs):
        self.requests.append((method, url, kwargs))
        if method == "POST":
            if kwargs["json"]["shots"] == 11:
                self.created.append("job-success")
                data = {"id": "job-success"}
            else:
                self.post_started.set()
                assert self.release_post.wait(3), "test did not release POST"
                if self.post_error:
                    raise self.post_error
                data = self.response
                if isinstance(data.get("id"), str):
                    self.created.append(data["id"])
        elif method == "PUT":
            self.cancelled.append(url)
            self.put_started.set()
            assert self.release_put.wait(3), "test did not release PUT"
            if self.put_error:
                raise self.put_error
            data = {}
        else:
            data = {"status": "completed", "data": {"histogram": {"0": 1.0}}}
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = data
        return response


@pytest.fixture
def transport():
    fake = Transport()
    yield fake
    fake.release_post.set()
    fake.release_put.set()


async def wait_event(event):
    async with asyncio.timeout(2):
        while not event.is_set():
            await asyncio.sleep(0.001)


def execute(executor, shots=7):
    return asyncio.create_task(executor.execute(Circuit().h(0), shots=shots))


@pytest.mark.asyncio
async def test_cancelled_submit_returns_before_post_and_cleans_exact_job(transport, caplog):
    executor = IonQExecutor(IonQExecutorConfig(api_key="offline-secret"), session=transport)
    transport.release_put.clear()
    with patch.object(executor, "_circuit_to_qasm", return_value=("OPENQASM 2.0;", 1)):
        task = execute(executor)
        await wait_event(transport.post_started)
        task.cancel("caller stopped")
        task.cancel("caller stopped again")
        with pytest.raises(asyncio.CancelledError) as error:
            async with asyncio.timeout(0.5):
                await task
        assert not transport.created  # POST remains blocked; caller did not wait.
        assert "Do not automatically resubmit" in " ".join(error.value.__notes__)
        transport.release_post.set()
        await wait_event(transport.put_started)
        assert task.cancel() is False  # Cleanup survives an already-finished caller.
        assert transport.cancelled == [
            "https://api.ionq.co/v0.3/jobs/job-interrupted/status/cancel"
        ]
        assert executor._current_job_id == "job-interrupted"
        assert [m for m, _, _ in transport.requests].count("POST") == 1
        assert all(options["timeout"] == 30 for _, _, options in transport.requests)
        assert "offline-secret" not in caplog.text
        transport.release_put.set()


@pytest.mark.asyncio
async def test_concurrent_success_is_not_cancelled_by_other_submission(transport):
    executor = IonQExecutor(IonQExecutorConfig(api_key="offline"), session=transport)
    with patch.object(executor, "_circuit_to_qasm", return_value=("OPENQASM 2.0;", 1)):
        interrupted = execute(executor)
        await wait_event(transport.post_started)
        successful = await execute(executor, shots=11)
        assert successful.metadata["job_id"] == executor._current_job_id == "job-success"
        interrupted.cancel()
        with pytest.raises(asyncio.CancelledError):
            await interrupted
        transport.release_post.set()
        await wait_event(transport.put_started)
        assert transport.cancelled == [
            "https://api.ionq.co/v0.3/jobs/job-interrupted/status/cancel"
        ]
        assert successful.counts == {"0": 11}


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [{}, {"id": None}, {"id": ""}])
async def test_interrupted_submit_without_id_reports_uncertainty(transport, response, caplog):
    transport.response = response
    executor = IonQExecutor(IonQExecutorConfig(api_key="offline"), session=transport)
    with patch.object(executor, "_circuit_to_qasm", return_value=("OPENQASM 2.0;", 1)):
        task = execute(executor)
        await wait_event(transport.post_started)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        transport.release_post.set()
        async with asyncio.timeout(2):
            while "ended without a usable job ID" not in caplog.text:
                await asyncio.sleep(0.001)
        assert "acceptance is unknown" in caplog.text
        assert transport.cancelled == []
        assert executor._current_job_id is None


@pytest.mark.asyncio
async def test_interrupted_submit_transport_error_is_not_replayed(transport, caplog):
    transport.post_error = RuntimeError("lost response")
    executor = IonQExecutor(IonQExecutorConfig(api_key="offline"), session=transport)
    with patch.object(executor, "_circuit_to_qasm", return_value=("OPENQASM 2.0;", 1)):
        task = execute(executor)
        await wait_event(transport.post_started)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        transport.release_post.set()
        async with asyncio.timeout(2):
            while "acceptance is unknown" not in caplog.text:
                await asyncio.sleep(0.001)
        assert [m for m, _, _ in transport.requests] == ["POST"]


@pytest.mark.asyncio
async def test_normal_submit_error_preserves_original_exception(transport):
    failure = RuntimeError("submission failed")
    transport.post_error = failure
    transport.release_post.set()
    executor = IonQExecutor(IonQExecutorConfig(api_key="offline"), session=transport)
    with patch.object(executor, "_circuit_to_qasm", return_value=("OPENQASM 2.0;", 1)):
        with pytest.raises(RuntimeError) as error:
            await execute(executor)
        assert error.value is failure
    assert transport.cancelled == []


@pytest.mark.asyncio
async def test_failed_cleanup_reports_job_without_masking_cancellation(transport, caplog):
    transport.put_error = RuntimeError("cancel failed")
    executor = IonQExecutor(IonQExecutorConfig(api_key="offline"), session=transport)
    with patch.object(executor, "_circuit_to_qasm", return_value=("OPENQASM 2.0;", 1)):
        task = execute(executor)
        await wait_event(transport.post_started)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        transport.release_post.set()
        async with asyncio.timeout(2):
            while "cancellation request failed" not in caplog.text:
                await asyncio.sleep(0.001)
        assert "job-interrupted" in caplog.text
        assert "may still be running" in caplog.text
        assert len(transport.cancelled) == 1


@pytest.mark.asyncio
async def test_cancel_after_worker_returns_still_claims_cleanup_once(
    transport, monkeypatch, caplog
):
    """Cancel after worker result delivery but before the awaiting task resumes."""
    transport.release_post.set()
    executor = IonQExecutor(IonQExecutorConfig(api_key="offline"), session=transport)
    loop = asyncio.get_running_loop()
    original_run = loop.run_in_executor

    def run(pool, function, *args):
        if function.__name__ != "submit":
            return original_run(pool, function, *args)
        result = function(*args)
        future = loop.create_future()
        loop.call_soon(future.set_result, result)
        loop.call_soon(asyncio.current_task().cancel, "after worker returned")
        return future

    monkeypatch.setattr(loop, "run_in_executor", run)
    with patch.object(executor, "_circuit_to_qasm", return_value=("OPENQASM 2.0;", 1)):
        with pytest.raises(asyncio.CancelledError):
            await execute(executor)
        await wait_event(transport.put_started)
        assert "job ID 'job-interrupted'" in caplog.text
        assert len(transport.cancelled) == 1


@pytest.mark.asyncio
async def test_cleanup_uses_submission_endpoint_and_auth_snapshot(transport):
    config = IonQExecutorConfig(api_key="original-offline", base_url="https://original.invalid")
    executor = IonQExecutor(config, session=transport)
    with patch.object(executor, "_circuit_to_qasm", return_value=("OPENQASM 2.0;", 1)):
        task = execute(executor)
        await wait_event(transport.post_started)
        config.base_url = "https://changed.invalid"
        config.api_key = "changed-offline"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        transport.release_post.set()
        await wait_event(transport.put_started)
        assert transport.cancelled == [
            "https://original.invalid/jobs/job-interrupted/status/cancel"
        ]
        assert all(
            options["headers"]["Authorization"] == "apiKey original-offline"
            for _, _, options in transport.requests
        )


@pytest.mark.asyncio
async def test_cleanup_scheduling_failure_preserves_cancellation(transport, monkeypatch, caplog):
    transport.release_post.set()
    executor = IonQExecutor(IonQExecutorConfig(api_key="offline"), session=transport)
    loop = asyncio.get_running_loop()

    def run(pool, function, *args):
        if function.__name__ != "submit":
            raise RuntimeError("executor shut down")
        result = function(*args)
        future = loop.create_future()
        loop.call_soon(future.set_result, result)
        loop.call_soon(asyncio.current_task().cancel, "original cancellation")
        return future

    monkeypatch.setattr(loop, "run_in_executor", run)
    with (
        patch.object(executor, "_circuit_to_qasm", return_value=("OPENQASM 2.0;", 1)),
        pytest.raises(asyncio.CancelledError, match="original cancellation"),
    ):
        await execute(executor)
    assert "cleanup could not be scheduled" in caplog.text
    assert "job-interrupted" in caplog.text
    assert transport.cancelled == []


@pytest.mark.asyncio
async def test_normal_submit_rejects_non_string_id_without_polling(transport):
    transport.response = {"id": 123}
    transport.release_post.set()
    executor = IonQExecutor(IonQExecutorConfig(api_key="offline"), session=transport)
    with (
        patch.object(executor, "_circuit_to_qasm", return_value=("OPENQASM 2.0;", 1)),
        pytest.raises(ValueError, match="no usable job ID"),
    ):
        await execute(executor)
    assert [method for method, _, _ in transport.requests] == ["POST"]


@pytest.mark.asyncio
async def test_cancellation_before_submission_worker_starts_never_posts(transport, monkeypatch):
    executor = IonQExecutor(IonQExecutorConfig(api_key="offline"), session=transport)
    loop = asyncio.get_running_loop()
    queued = threading.Event()
    original_run = loop.run_in_executor

    def run(pool, function, *args):
        if function.__name__ != "submit":
            return original_run(pool, function, *args)
        queued.set()
        return loop.create_future()  # A queued submission has not contacted the provider.

    monkeypatch.setattr(loop, "run_in_executor", run)
    with patch.object(executor, "_circuit_to_qasm", return_value=("OPENQASM 2.0;", 1)):
        task = execute(executor)
        await wait_event(queued)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert transport.requests == []
