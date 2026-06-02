"""Temporal activities for task execution.

This module contains all marqov-aware code that runs inside Temporal activities.
Activities are NOT sandboxed like workflows, so they can safely import quantum
libraries (quantumflow, sympy, etc.).

The key architectural principle:
- Workflows = pure coordination (no marqov imports)
- Activities = all computation (imports anything)
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import cloudpickle
from temporalio import activity

# Heartbeat interval for execute_task. Temporal throttles forwarding to 80%
# of heartbeat_timeout (48s for 60s timeout), so the send interval just needs
# to be well under 48s. 10s gives ~4 heartbeats per forward window.
_HEARTBEAT_INTERVAL_S = 10


def _deserialize_value(value: Any) -> Any:
    """Deserialize a value from JSON transport.

    Handles cloudpickle-encoded complex objects.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    elif isinstance(value, list):
        return [_deserialize_value(item) for item in value]
    elif isinstance(value, dict):
        if value.get("__cloudpickle__"):
            return cloudpickle.loads(base64.b64decode(value["data"]))
        return {k: _deserialize_value(v) for k, v in value.items()}
    else:
        return value


def _serialize_value(value: Any) -> Any:
    """Serialize a value for JSON transport.

    Uses cloudpickle for complex objects, JSON-compatible types pass through.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    elif isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    elif isinstance(value, dict):
        # Check for our special marker types
        if value.get("__cloudpickle__"):
            return value
        return {k: _serialize_value(v) for k, v in value.items()}
    else:
        # Complex object - use cloudpickle
        return {
            "__cloudpickle__": True,
            "data": base64.b64encode(cloudpickle.dumps(value)).decode("utf-8"),
        }


@activity.defn
async def execute_task(
    node_id: str,
    func_ref: str,
    args_json: str,
    kwargs_json: str,
) -> str:
    """Execute a single task node.

    This activity receives serialized function and arguments,
    executes the function, and returns the serialized result.

    All marqov imports happen here, inside the activity,
    which is NOT subject to Temporal's workflow sandbox.

    Args:
        node_id: Unique identifier for this node.
        func_ref: Base64-encoded cloudpickle of the function.
        args_json: JSON-encoded list of arguments.
        kwargs_json: JSON-encoded dict of keyword arguments.

    Returns:
        JSON-encoded result with node_id and serialized value.
    """
    # Deserialize the function
    func_bytes = base64.b64decode(func_ref)
    func = cloudpickle.loads(func_bytes)

    # Deserialize arguments
    args_raw = json.loads(args_json)
    kwargs_raw = json.loads(kwargs_json)

    args = [_deserialize_value(arg) for arg in args_raw]
    kwargs = {k: _deserialize_value(v) for k, v in kwargs_raw.items()}

    # Run user code in a separate thread for complete event loop isolation.
    # Libraries like PennyLane's Braket plugin call asyncio.run() internally,
    # which conflicts with Temporal's event loop. A separate thread gets its
    # own clean event loop, avoiding any interference. (#728)
    #
    # Heartbeat every 10s so Temporal can deliver CancelledError when the
    # workflow is cancelled. The thread itself is NOT interruptible — we
    # abandon it and let it finish in the background. The important thing
    # is that the activity reports cancellation immediately.

    async def _heartbeat_loop() -> None:
        """Send heartbeats to Temporal until cancelled."""
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL_S)
            activity.heartbeat(f"executing {node_id}")

    if asyncio.iscoroutinefunction(func):
        def _run_async() -> Any:
            return asyncio.run(func(*args, **kwargs))
        task = asyncio.create_task(asyncio.to_thread(_run_async))
    else:
        task = asyncio.create_task(asyncio.to_thread(func, *args, **kwargs))

    heartbeat_task = asyncio.create_task(_heartbeat_loop())

    try:
        result = await task
    except asyncio.CancelledError:
        # Workflow was cancelled via Temporal.
        # The thread is still running but we report cancellation immediately.
        activity.logger.warning(
            "Activity cancelled for node %s",
            node_id,
        )
        raise
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except (asyncio.CancelledError, Exception):
            # CancelledError is the normal case (we just cancelled it).
            # Other exceptions (e.g. heartbeat SDK error) should not
            # mask the main task's result or error.
            pass

    # Serialize and return result
    return json.dumps({
        "node_id": node_id,
        "result": _serialize_value(result),
    })


@activity.defn
async def prepare_node_inputs(
    node_data_json: str,
    completed_results_json: str,
) -> str:
    """Prepare inputs for a node by resolving dependencies.

    This activity resolves proxy references in arguments by looking up
    results from previously completed nodes.

    Args:
        node_data_json: JSON with node's args, kwargs, and dependency info.
        completed_results_json: JSON dict of node_id -> result for completed nodes.

    Returns:
        JSON with resolved args and kwargs ready for execution.
    """
    node_data = json.loads(node_data_json)
    completed = json.loads(completed_results_json)

    def resolve_arg(arg: Any) -> Any:
        """Recursively resolve proxy references."""
        if isinstance(arg, dict) and arg.get("__proxy__"):
            node_id = arg["node_id"]
            if node_id not in completed:
                raise ValueError(f"Dependency {node_id} not yet computed")
            return completed[node_id]
        elif isinstance(arg, list):
            return [resolve_arg(item) for item in arg]
        elif isinstance(arg, dict):
            return {k: resolve_arg(v) for k, v in arg.items()}
        return arg

    resolved_args = [resolve_arg(arg) for arg in node_data["args"]]
    resolved_kwargs = {k: resolve_arg(v) for k, v in node_data["kwargs"].items()}

    return json.dumps({
        "node_id": node_data["node_id"],
        "func_ref": node_data["func_ref"],
        "args": resolved_args,
        "kwargs": resolved_kwargs,
    })
