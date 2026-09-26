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
from collections import deque
from pathlib import Path
from typing import Any

from temporalio import activity
from temporalio.exceptions import ApplicationError

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

# The child's stderr is drained continuously (see _StderrTail) so that a chatty
# task cannot fill the OS pipe and block forever in write().  Only the last
# MAX_STDERR_TAIL_BYTES are kept: a runaway task must not be able to grow the
# worker's memory, and the tail is the part that explains a crash.
MAX_STDERR_TAIL_BYTES: int = 32 * 1024   # 32 KiB


class _StderrTail:
    """Bounded ring buffer over a child's stderr.

    Keeps at most ``limit`` bytes (the most recent ones) and counts how many
    earlier bytes were discarded, so a truncated tail is never mistaken for
    the child's whole output.
    """

    def __init__(self, limit: int = MAX_STDERR_TAIL_BYTES) -> None:
        self._limit = limit
        self._chunks: deque[bytes] = deque()
        self._size = 0
        self.dropped = 0

    def feed(self, data: bytes) -> None:
        """Append a chunk, evicting the oldest bytes past the limit."""
        self._chunks.append(data)
        self._size += len(data)
        while self._chunks and self._size - len(self._chunks[0]) >= self._limit:
            oldest = self._chunks.popleft()
            self._size -= len(oldest)
            self.dropped += len(oldest)
        if self._size > self._limit:
            excess = self._size - self._limit
            head = self._chunks.popleft()
            self._chunks.appendleft(head[excess:])
            self._size -= excess
            self.dropped += excess

    def text(self) -> str:
        """Decode the retained tail, replacing any bytes split by truncation."""
        return b"".join(self._chunks).decode(errors="replace")


async def _drain_stderr(stream: asyncio.StreamReader, tail: _StderrTail) -> None:
    """Read ``stream`` to EOF, keeping only the last bytes in ``tail``."""
    while True:
        chunk = await stream.read(8192)
        if not chunk:
            return
        tail.feed(chunk)


# Bounds for child-derived text quoted back in failure messages.  The child
# wrote up to MAX_RESULT_BYTES of attacker-chosen content, and an oversized
# failure message would blow Temporal's payload limit: the failure response is
# then rejected and the non-retryable signal is lost.  Every fragment of a
# validation message that came from the child goes through _clip first.
_MAX_QUOTED_CHARS: int = 200
_MAX_QUOTED_KEYS: int = 10
_MAX_QUOTED_KEY_CHARS: int = 40
_MAX_QUOTED_KEY_LIST_CHARS: int = 400


def _clip(text: str, limit: int = _MAX_QUOTED_CHARS) -> str:
    """Return ``text`` truncated to ``limit`` characters, marking truncation."""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _clip_keys(keys: list[str]) -> str:
    """Render an envelope's keys for a message, bounded in count and length.

    Both the number of keys and each key are bounded, and so is the rendered
    list: the child chooses how many keys it writes and how long each one is.
    """
    shown = [_clip(repr(key), _MAX_QUOTED_KEY_CHARS) for key in keys[:_MAX_QUOTED_KEYS]]
    if len(keys) > _MAX_QUOTED_KEYS:
        shown.append(f"... ({len(keys)} keys total)")
    return "[" + _clip(", ".join(shown), _MAX_QUOTED_KEY_LIST_CHARS) + "]"


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
    The result VALUE is returned **opaquely** as a JSON string containing the
    raw serialized blob produced by the child.  The result ENVELOPE is parsed
    and validated here (shape plus node_id), because the child owns its
    workdir and must not be able to name a different node.

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

    Raises:
        ApplicationError: Non-retryable, if the child wrote a result envelope
            that is not a JSON object with exactly ``node_id`` and ``result``,
            or whose ``node_id`` does not match this activity's ``node_id``.
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
        try:
            while True:
                await asyncio.sleep(_HEARTBEAT_INTERVAL_S)
                activity.heartbeat(f"executing {node_id}")
        except asyncio.CancelledError:
            raise
        except Exception:
            # Heartbeating has stopped. Without this log the operator would see
            # only a Temporal heartbeat timeout, never the actual cause.
            activity.logger.exception("Heartbeat loop failed for node %s", node_id)
            raise

    heartbeat_task = asyncio.create_task(_heartbeat_loop())
    child_task = asyncio.create_task(proc.wait())

    # Drain the child's stderr for the whole of its lifetime.  asyncio's
    # StreamReader stops reading once its buffer hits its limit, so an
    # undrained pipe fills, the child blocks in write() and proc.wait() never
    # returns.  This is the ONLY reader of proc.stderr.
    stderr_tail = _StderrTail()
    stderr_task: asyncio.Task[None] | None = None
    if proc.stderr is not None:
        stderr_task = asyncio.create_task(_drain_stderr(proc.stderr, stderr_tail))

    try:
        await child_task
        if stderr_task is not None:
            # The child has exited; let the reader reach EOF so the tail is
            # complete before it is used in an error message.
            try:
                await asyncio.wait_for(stderr_task, timeout=2)
            except asyncio.TimeoutError:
                pass
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
        except asyncio.CancelledError:
            pass
        except Exception:
            # Already reported by _heartbeat_loop; do not let it mask the
            # activity's own outcome (result, child error, or cancellation).
            pass
        if stderr_task is not None:
            stderr_task.cancel()
            try:
                await stderr_task
            except asyncio.CancelledError:
                pass
            except Exception:
                activity.logger.exception("Stderr reader failed for node %s", node_id)

    # --- Separate error channel -------------------------------------------
    # The child writes failures to error.json and successes to result.json.
    # Reading error.json is safe: it's small, structured, and written by our
    # own child wrapper code (not by user task code).
    # The success path validates the result ENVELOPE and forwards the result
    # VALUE opaquely: the activity never calls cloudpickle.loads on it.
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
        # The child owns its workdir, so result.json is whatever the child
        # chose to write. Validate the ENVELOPE before returning it: the
        # parent must not forward a node_id the child picked, and must not
        # hand malformed bytes to the workflow, where a json.loads failure
        # would wedge the workflow task (marqov-sdk#141).
        #
        # Parsing the envelope here is bounded: the MAX_RESULT_BYTES check
        # above runs first, so json.loads never sees more than the cap. The
        # security property that matters is unchanged: the activity still
        # never cloudpickle.loads the child's result. The result VALUE stays
        # opaque, it is re-serialised as-is and deserialised only by the next
        # task's child.
        #
        # read_text and json.loads are both inside the try: non-UTF-8 bytes
        # raise UnicodeDecodeError and deeply nested input raises
        # RecursionError, and neither is a JSONDecodeError. Uncaught, they
        # would surface as ordinary retryable activity failures rather than
        # the non-retryable failure a malformed envelope deserves.
        # (JSONDecodeError and UnicodeDecodeError are both ValueError.)
        try:
            envelope = json.loads(result_path.read_text())
        except (ValueError, RecursionError) as exc:
            raise ApplicationError(
                f"Task {node_id} wrote a result.json that could not be parsed as "
                f"JSON: {_clip(f'{type(exc).__name__}: {exc}')}",
                non_retryable=True,
            ) from exc

        if not isinstance(envelope, dict):
            raise ApplicationError(
                f"Task {node_id} wrote a result.json that is not a JSON object "
                f"(got {type(envelope).__name__}).",
                non_retryable=True,
            )

        expected_keys = {"node_id", "result"}
        actual_keys = set(envelope)
        if actual_keys != expected_keys:
            raise ApplicationError(
                f"Task {node_id} wrote a result envelope with keys "
                f"{_clip_keys(sorted(actual_keys))}; expected exactly "
                f"{sorted(expected_keys)}.",
                non_retryable=True,
            )

        claimed_node_id = envelope["node_id"]
        if claimed_node_id != node_id:
            raise ApplicationError(
                f"Task {node_id} wrote a result envelope claiming node_id "
                f"{_clip(repr(claimed_node_id))}; expected {node_id!r}. Refusing to "
                f"forward a result that could overwrite another node's value.",
                non_retryable=True,
            )

        # Re-serialise from the parsed object with the activity's own node_id,
        # so the returned bytes are the activity's envelope, not the child's.
        return json.dumps({"node_id": node_id, "result": envelope["result"]})

    # No result file — child crashed before writing anything.
    rc = proc.returncode
    stderr_snippet = stderr_tail.text()
    dropped_note = (
        f" (last {MAX_STDERR_TAIL_BYTES} bytes; {stderr_tail.dropped} earlier bytes dropped)"
        if stderr_tail.dropped
        else ""
    )
    raise RuntimeError(
        f"Task {node_id} child exited with code {rc} and no result.\n"
        f"Stderr{dropped_note}: {stderr_snippet}"
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
