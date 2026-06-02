"""Tests for activity heartbeating and cancellation handling.

The project conftest.py stubs heavy dependencies (cloudpickle, temporalio)
with mock modules. This test needs the real packages, so we restore them
from the actual installed packages before importing the activity module.
"""
import asyncio
import importlib
import json
import sys
import time

import pytest
from unittest.mock import patch, MagicMock


def _restore_real_module(name: str) -> None:
    """Remove mock module and all sub-modules, then import the real one."""
    keys_to_remove = [k for k in sys.modules if k == name or k.startswith(name + ".")]
    for k in keys_to_remove:
        del sys.modules[k]
    importlib.import_module(name)


# Restore real cloudpickle and temporalio before importing activity module.
_restore_real_module("cloudpickle")
_restore_real_module("temporalio")

# Now we can safely import - these will use real packages.
import base64
import cloudpickle
from temporalio import activity

# Re-import activity module so it picks up real cloudpickle/temporalio.
if "marqov.workflows.activity" in sys.modules:
    del sys.modules["marqov.workflows.activity"]
from marqov.workflows.activity import execute_task


@pytest.fixture
def _mock_activity_context():
    """Mock Temporal activity context for testing outside Temporal."""
    mock_info = MagicMock()
    mock_info.workflow_id = "test-workflow"
    mock_info.activity_id = "test-activity"
    with patch("temporalio.activity.heartbeat") as mock_hb, \
         patch("temporalio.activity.info", return_value=mock_info), \
         patch("temporalio.activity.logger") as mock_logger:
        yield mock_hb


@pytest.mark.asyncio
async def test_execute_task_completes_normally(_mock_activity_context):
    """Heartbeat loop should not interfere with normal execution."""

    def add(a, b):
        return a + b

    func_ref = base64.b64encode(cloudpickle.dumps(add)).decode()
    args_json = json.dumps([1, 2])
    kwargs_json = json.dumps({})

    result_json = await execute_task("node-1", func_ref, args_json, kwargs_json)
    result = json.loads(result_json)

    assert result["node_id"] == "node-1"
    assert result["result"] == 3


@pytest.mark.asyncio
async def test_execute_task_heartbeat_fires_during_long_task(_mock_activity_context):
    """Heartbeat should fire at least once during a task longer than the interval."""
    mock_hb = _mock_activity_context

    def slow_func():
        time.sleep(0.3)
        return 42

    func_ref = base64.b64encode(cloudpickle.dumps(slow_func)).decode()
    args_json = json.dumps([])
    kwargs_json = json.dumps({})

    # Patch interval to 0.05s so heartbeats fire during the 0.3s task
    import marqov.workflows.activity as activity_mod
    original_interval = activity_mod._HEARTBEAT_INTERVAL_S
    activity_mod._HEARTBEAT_INTERVAL_S = 0.05
    try:
        result_json = await execute_task("node-slow", func_ref, args_json, kwargs_json)
    finally:
        activity_mod._HEARTBEAT_INTERVAL_S = original_interval

    result = json.loads(result_json)
    assert result["result"] == 42
    assert mock_hb.call_count >= 1, f"Expected at least 1 heartbeat, got {mock_hb.call_count}"


@pytest.mark.asyncio
async def test_execute_task_cancellation_propagates(_mock_activity_context):
    """CancelledError should propagate when outer task is cancelled."""

    def very_slow_func():
        time.sleep(60)
        return "should not reach here"

    func_ref = base64.b64encode(cloudpickle.dumps(very_slow_func)).decode()
    args_json = json.dumps([])
    kwargs_json = json.dumps({})

    task = asyncio.create_task(
        execute_task("node-cancel", func_ref, args_json, kwargs_json)
    )
    await asyncio.sleep(0.1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
