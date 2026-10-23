"""Interruption logging for BlockingCalls.wait."""

import asyncio
import logging
import threading

import pytest

from marqov.executors._blocking import BlockingCalls

INTERRUPTED = "interrupted; cancellation requested, provider state unconfirmed"


def blocked_result(release: threading.Event):
    def result():
        release.wait(1)
        return "late"

    return result


def warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


@pytest.mark.asyncio
async def test_timeout_interrupt_warns_with_job_id_when_cancel_succeeds(caplog):
    release = threading.Event()
    seen_before_cancel: list[list[str]] = []

    def cancel():
        seen_before_cancel.append(warnings(caplog))

    caplog.set_level(logging.WARNING, logger="marqov.executors._blocking")
    try:
        with BlockingCalls() as calls:
            with pytest.raises(TimeoutError):
                await calls.wait(
                    blocked_result(release), cancel, job_id="job-ok", timeout=0.02
                )
    finally:
        release.set()
    assert warnings(caplog) == [f"Remote task 'job-ok' {INTERRUPTED}"]
    # Logged before cleanup runs, so a cleanup hang cannot suppress it.
    assert seen_before_cancel == [[f"Remote task 'job-ok' {INTERRUPTED}"]]


@pytest.mark.asyncio
async def test_caller_cancel_interrupt_warns_with_job_id_when_cancel_succeeds(caplog):
    release, started = threading.Event(), threading.Event()
    cancelled = threading.Event()

    def result():
        started.set()
        release.wait(1)

    caplog.set_level(logging.WARNING, logger="marqov.executors._blocking")
    try:
        with BlockingCalls() as calls:
            task = asyncio.create_task(
                calls.wait(result, cancelled.set, job_id="job-ctrl-c", timeout=None)
            )
            await asyncio.to_thread(started.wait, 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    finally:
        release.set()
    assert cancelled.is_set()
    assert warnings(caplog) == [f"Remote task 'job-ctrl-c' {INTERRUPTED}"]
