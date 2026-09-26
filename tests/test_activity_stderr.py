"""Tests for draining the task child's stderr in ``execute_task``.

The activity starts the child with ``stderr=PIPE``. If nothing reads that pipe
while the child runs, the child blocks in ``write()`` once the OS pipe fills and
``proc.wait()`` never returns, so a chatty task deadlocks the activity until its
Temporal timeout. These tests exercise the real pipe with a real child process:
a mocked subprocess would not catch the deadlock.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging

import cloudpickle
import pytest
from unittest.mock import MagicMock, patch

import marqov.workflows.activity as activity_mod
from marqov.workflows.activity import MAX_STDERR_TAIL_BYTES, execute_task


def _enc(fn) -> str:
    return base64.b64encode(cloudpickle.dumps(fn)).decode()


@pytest.fixture
def _mock_activity_context():
    """Mock Temporal activity context for testing outside Temporal."""
    mock_info = MagicMock()
    mock_info.workflow_id = "test-workflow"
    mock_info.activity_id = "test-activity"
    with (
        patch("temporalio.activity.heartbeat") as mock_hb,
        patch("temporalio.activity.info", return_value=mock_info),
        patch("temporalio.activity.logger"),
    ):
        yield mock_hb


@pytest.mark.asyncio
async def test_chatty_task_completes(_mock_activity_context):
    """A task writing 200 KiB to stderr must still complete promptly."""

    def noisy_then_return():
        import sys

        sys.stderr.write("x" * 200_000)
        sys.stderr.flush()
        return 42

    result_json = await asyncio.wait_for(
        execute_task("node-noisy", _enc(noisy_then_return), json.dumps([]), json.dumps({})),
        timeout=10,
    )

    result = json.loads(result_json)
    assert result["node_id"] == "node-noisy"
    assert result["result"] == 42


@pytest.mark.asyncio
async def test_no_result_path_reports_stderr(_mock_activity_context):
    """The no-result error must still carry the child's stderr tail."""

    def marker_then_die():
        import os
        import sys

        sys.stderr.write("MARQOV-STDERR-MARKER\n")
        sys.stderr.flush()
        os._exit(1)

    with pytest.raises(RuntimeError) as excinfo:
        await asyncio.wait_for(
            execute_task("node-marker", _enc(marker_then_die), json.dumps([]), json.dumps({})),
            timeout=10,
        )

    assert "MARQOV-STDERR-MARKER" in str(excinfo.value)


@pytest.mark.asyncio
async def test_stderr_tail_is_bounded(_mock_activity_context):
    """A flood of stderr must be truncated to the tail cap and report the drop."""

    def flood_then_die():
        import os
        import sys

        sys.stderr.write("A" * 200_000)
        sys.stderr.write("TAIL-END")
        sys.stderr.flush()
        os._exit(1)

    with pytest.raises(RuntimeError) as excinfo:
        await asyncio.wait_for(
            execute_task("node-flood", _enc(flood_then_die), json.dumps([]), json.dumps({})),
            timeout=10,
        )

    message = str(excinfo.value)
    assert len(message) <= MAX_STDERR_TAIL_BYTES + 512, (
        f"error message is {len(message)} bytes, above the {MAX_STDERR_TAIL_BYTES}-byte tail cap"
    )
    # The newest bytes are kept, the oldest are dropped, and the drop is stated.
    assert message.endswith("TAIL-END")
    assert "dropped" in message


@pytest.mark.asyncio
async def test_heartbeat_failure_is_not_silent(caplog):
    """A raising ``activity.heartbeat`` must be logged, not silently discarded."""

    def slow_func():
        import time

        time.sleep(0.4)
        return "done"

    mock_info = MagicMock()
    mock_info.workflow_id = "test-workflow"
    mock_info.activity_id = "test-activity"

    original_interval = activity_mod._HEARTBEAT_INTERVAL_S
    activity_mod._HEARTBEAT_INTERVAL_S = 0.05
    try:
        with (
            patch("temporalio.activity.heartbeat", side_effect=RuntimeError("heartbeat exploded")),
            patch("temporalio.activity.info", return_value=mock_info),
            caplog.at_level(logging.ERROR),
        ):
            await asyncio.wait_for(
                execute_task("node-hb", _enc(slow_func), json.dumps([]), json.dumps({})),
                timeout=20,
            )
    finally:
        activity_mod._HEARTBEAT_INTERVAL_S = original_interval

    assert "heartbeat exploded" in caplog.text
