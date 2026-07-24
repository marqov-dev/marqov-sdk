"""Temporal activities for task execution.

This module contains all marqov-aware code that runs inside Temporal activities.
Activities are NOT sandboxed like workflows, so they can safely import quantum
libraries (quantumflow, sympy, etc.).

The key architectural principle:
- Workflows = pure coordination (no marqov imports)
- Activities = all computation (imports anything)

The task body (cloudpickle.loads + run) executes in an isolated,
**scrubbed subprocess** — the activity process itself never calls
cloudpickle.loads on func_ref or the result.  The result is forwarded
opaquely (as a JSON blob) so Temporal carries the bytes without this
process deserialising them.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any

from temporalio import activity

from marqov.workflows._child_env import build_child_env, new_task_workdir

# Heartbeat interval for execute_task. Temporal throttles forwarding to 80%
# of heartbeat_timeout (48s for 60s timeout), so the send interval just needs
# to be well under 48s. 10s gives ~4 heartbeats per forward window.
_HEARTBEAT_INTERVAL_S = 10

# Path to the child entry-point (same package, found via __file__).
_CHILD_SCRIPT = str(Path(__file__).parent / "_task_child.py")

# Size caps for child output files.  A child must never return so much data
# that reading it exhausts the worker process.
# error.json is small and structured (our code); 64 KiB is generous.
# result.json cap is aligned with Temporal's default gRPC payload limit (~4 MiB):
# a larger result would pass this cap but then be rejected by Temporal itself — a
# confusing failure at demo scale. Kept UNDER 4 MiB for envelope/protocol headroom.
# Results larger than this need spill-to-S3 + a reference (follow-up, not this build).
MAX_ERROR_BYTES: int = 64 * 1024         # 64 KiB
MAX_RESULT_BYTES: int = 3 * 1024 * 1024  # 3 MiB — under Temporal's ~4 MiB gRPC limit


def _deserialize_value(value: Any) -> Any:
    """Deserialize a value from JSON transport.

    Handles cloudpickle-encoded complex objects.

    NOTE: This function is used only by prepare_node_inputs (proxy resolution)
    and by tests.  execute_task MUST NOT call it — results are forwarded opaquely.
    """
    import cloudpickle

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
    import cloudpickle

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


def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
    """SIGKILL the child's whole process group.

    ``start_new_session=True`` makes pgid == pid, so grandchildren die with it
    instead of lingering (mirrors dwave_executor._kill_process_group).
    """
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass


@activity.defn
async def execute_task(
    node_id: str,
    func_ref: str,
    args_json: str,
    kwargs_json: str,
    provider_env: dict[str, str] | None = None,
) -> str:
    """Execute a single task node in a scrubbed subprocess.

    The activity process NEVER calls cloudpickle.loads on func_ref, args, or
    the result — all deserialization happens inside a scrubbed child process.
    The result is returned **opaquely** as a JSON string containing the raw
    serialized blob produced by the child.

    Args:
        node_id: Unique identifier for this node.
        func_ref: Base64-encoded cloudpickle of the function.
        args_json: JSON-encoded list of arguments.
        kwargs_json: JSON-encoded dict of keyword arguments.
        provider_env: Optional provider credentials to inject into the child env
            (e.g. AWS keys for a Braket task).  Never inherited from the parent
            process env — only what is explicitly passed here enters the child.

    Returns:
        JSON-encoded result with node_id and opaque result blob.
    """
    workdir = new_task_workdir(node_id)
    result_path = workdir / "result.json"

    # Write inputs for the child.  func_ref is base64-encoded; write the text
    # so the child can decode it (writing decoded bytes would require the child
    # to know the encoding; keeping b64 is simpler and avoids double-decode).
    (workdir / "node_id").write_text(node_id)
    (workdir / "func_ref").write_text(func_ref)
    (workdir / "args.json").write_text(args_json)
    (workdir / "kwargs.json").write_text(kwargs_json)

    child_env = build_child_env(workdir, provider_env=provider_env)
    # The child needs to know its workdir.
    child_env["MARQOV_TASK_WORKDIR"] = str(workdir)

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        _CHILD_SCRIPT,
        stdout=asyncio.subprocess.DEVNULL,  # child logs go to stderr; don't mix
        stderr=asyncio.subprocess.PIPE,
        env=child_env,
        start_new_session=True,
    )

    async def _heartbeat_loop() -> None:
        """Send heartbeats to Temporal until cancelled."""
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL_S)
            activity.heartbeat(f"executing {node_id}")

    heartbeat_task = asyncio.create_task(_heartbeat_loop())
    child_task = asyncio.create_task(proc.wait())

    try:
        await child_task
    except asyncio.CancelledError:
        # Activity was cancelled by Temporal — kill the child and propagate.
        _kill_process_group(proc)
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        activity.logger.warning("Activity cancelled for node %s — child killed", node_id)
        raise
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except (asyncio.CancelledError, Exception):
            pass

    # --- Separate error channel -------------------------------------------
    # The child writes failures to error.json and successes to result.json.
    # Reading error.json is safe: it's small, structured, and written by our
    # own child wrapper code (not by user task code).
    # The success path forwards result.json OPAQUELY (bytes → string) without
    # ever calling json.loads on the result content.
    error_path = workdir / "error.json"
    if error_path.exists():
        error_size = error_path.stat().st_size
        if error_size > MAX_ERROR_BYTES:
            raise RuntimeError(
                f"Task {node_id} error.json size {error_size} exceeds limit "
                f"{MAX_ERROR_BYTES} bytes."
            )
        error_data = json.loads(error_path.read_text())
        raise RuntimeError(
            f"Task {node_id} failed in child process: {error_data.get('error', '(no message)')}"
        )

    if result_path.exists():
        result_size = result_path.stat().st_size
        if result_size > MAX_RESULT_BYTES:
            raise RuntimeError(
                f"Task {node_id} result.json size {result_size} exceeds the "
                f"{MAX_RESULT_BYTES}-byte limit (aligned with Temporal's ~4 MiB gRPC "
                f"payload limit). Large results must spill to S3 + pass a reference "
                f"(follow-up); returning multi-MiB results inline is unsupported."
            )
        # Forward result.json content OPAQUELY as a raw string — do NOT
        # json.loads the result value here. The activity wraps it in an
        # envelope so the caller can extract node_id and the opaque blob.
        result_text = result_path.read_text()
        # We need the node_id in the envelope but must not parse result content.
        # Produce the envelope by string construction, not json.loads+json.dumps.
        # result_text is already valid JSON: {"node_id": ..., "result": <blob>}
        # We just forward it directly — the schema matches what callers expect.
        return result_text

    # No result file — child crashed before writing anything.
    rc = proc.returncode
    stderr_bytes = b""
    if proc.stderr is not None:
        try:
            stderr_bytes = await asyncio.wait_for(proc.stderr.read(), timeout=2)
        except (asyncio.TimeoutError, Exception):
            pass
    stderr_snippet = stderr_bytes.decode(errors="replace")[-2000:] if stderr_bytes else ""
    raise RuntimeError(
        f"Task {node_id} child exited with code {rc} and no result.\n"
        f"Stderr: {stderr_snippet}"
    )


@activity.defn
async def prepare_node_inputs(
    node_data_json: str,
    completed_results_json: str,
) -> str:
    """Prepare inputs for a node by resolving dependencies.

    This activity resolves proxy references in arguments by looking up
    results from previously completed nodes.

    Upstream results are kept **opaque** — they are passed through as-is
    (still in their __cloudpickle__ envelope) without deserialization.
    The child process for the next execute_task call will deserialize them.

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
            # Pass the upstream result through opaquely — do NOT deserialize.
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
