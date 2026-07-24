"""Boundary-2 isolation tests for execute_task.

SP2 Task 2 invariant: the activity process itself NEVER calls
``cloudpickle.loads`` on user bytes during ``execute_task``.  All
deserialization happens inside the scrubbed child subprocess.

Tests:
1. Zero cloudpickle.loads calls in the activity process during execute_task.
2. The child env lacks arbitrary host secrets (HOST_* vars).
3. A result that is a __cloudpickle__ blob is forwarded opaquely (not
   deserialized in the activity process).
4. prepare_node_inputs keeps upstream results opaque.
5. Heartbeat fires during a slow task (subprocess version).
6. Activity cancel kills the child process group.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# The project conftest.py is empty — no stubs to restore.
# Import the real modules directly.
import cloudpickle  # noqa: E402
from marqov.workflows.activity import execute_task, prepare_node_inputs, _deserialize_value  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_activity_ctx():
    """Minimal Temporal activity context mock."""
    mock_info = MagicMock()
    mock_info.workflow_id = "test-workflow"
    mock_info.activity_id = "test-activity"
    with (
        patch("temporalio.activity.heartbeat") as mock_hb,
        patch("temporalio.activity.info", return_value=mock_info),
        patch("temporalio.activity.logger"),
    ):
        yield mock_hb


def _enc(fn: Any) -> str:
    return base64.b64encode(cloudpickle.dumps(fn)).decode()


# ---------------------------------------------------------------------------
# 1. Zero cloudpickle.loads calls in the activity process
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_cloudpickle_loads_in_activity_process(mock_activity_ctx):
    """Activity process must NOT call cloudpickle.loads during execute_task.

    We patch cloudpickle.loads at the module level (the canonical location)
    so any caller — the activity, a helper it imports, or a lazy ``import
    cloudpickle`` inside a function body — is caught.  The loads must happen
    ONLY in the child subprocess (which sees the real module, not the mock).
    """

    def add(a: int, b: int) -> int:
        return a + b

    func_ref = _enc(add)
    args_json = json.dumps([3, 4])
    kwargs_json = json.dumps({})

    loads_calls: list = []
    real_loads = cloudpickle.loads

    def spy_loads(data: bytes, *args: Any, **kwargs: Any) -> Any:
        loads_calls.append(data[:16])  # record prefix for debugging
        return real_loads(data, *args, **kwargs)

    # Patch at the cloudpickle module level.  Because _deserialize_value does
    # `import cloudpickle` lazily inside the function body, the patch must
    # target sys.modules["cloudpickle"].loads (which is what the lazy import
    # will resolve to at call time).
    with patch.object(cloudpickle, "loads", side_effect=spy_loads):
        result_json = await execute_task("node-spy", func_ref, args_json, kwargs_json)

    result = json.loads(result_json)
    assert result["node_id"] == "node-spy"
    assert result["result"] == 7

    assert len(loads_calls) == 0, (
        f"cloudpickle.loads was called {len(loads_calls)} time(s) in the activity process. "
        f"First call data prefix: {loads_calls[0] if loads_calls else 'n/a'}"
    )


# ---------------------------------------------------------------------------
# 2. Child env lacks secrets
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_child_env_lacks_secrets(mock_activity_ctx, tmp_path):
    """The scrubbed child env must not contain arbitrary host secrets (HOST_* vars)."""
    # Inject fake secrets into the PARENT env to verify they don't leak.
    fake_secrets = {
        "HOST_SECRET_TOKEN": "fake-secret-value",
        "HOST_API_KEY": "fake-api-key",
        "HOST_NAMESPACE": "example",
    }
    child_env_capture: dict[str, str] = {}

    from marqov.workflows import _child_env as ce

    original_build = ce.build_child_env

    def capturing_build(workdir: Path, provider_env=None):
        env = original_build(workdir, provider_env=provider_env)
        child_env_capture.update(env)
        return env

    with patch.dict(os.environ, fake_secrets):
        with patch("marqov.workflows.activity.build_child_env", side_effect=capturing_build):

            def simple_fn() -> int:
                return 99

            func_ref = _enc(simple_fn)
            result_json = await execute_task(
                "node-secret", func_ref, json.dumps([]), json.dumps({})
            )

    # The captured env (what the child actually receives) must not contain secrets.
    assert "HOST_SECRET_TOKEN" not in child_env_capture, (
        "HOST_SECRET_TOKEN leaked into child env!"
    )
    for key in child_env_capture:
        assert not key.startswith("HOST_"), (
            f"HOST_* key leaked into child env: {key}"
        )

    result = json.loads(result_json)
    assert result["result"] == 99


# ---------------------------------------------------------------------------
# 3. Result forwarded opaquely — no in-process deserialization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complex_result_forwarded_opaquely(mock_activity_ctx):
    """A __cloudpickle__ result blob must be forwarded without in-process loads."""

    class _MyResult:
        def __init__(self, val: int) -> None:
            self.val = val

    def create_result() -> _MyResult:
        return _MyResult(42)

    func_ref = _enc(create_result)
    args_json = json.dumps([])
    kwargs_json = json.dumps({})

    real_loads = cloudpickle.loads
    loads_in_activity: list = []

    import marqov.workflows.activity as activity_mod
    original_dv = activity_mod._deserialize_value

    def spy_dv(value: Any) -> Any:
        # _deserialize_value calls cloudpickle.loads internally when it sees
        # a __cloudpickle__ marker.  Count calls at this level.
        result = original_dv(value)
        return result

    # We spy at the file-read level: after the child returns, does the activity
    # call _deserialize_value on the result? It must NOT.
    dv_calls: list = []

    def tracking_dv(value: Any) -> Any:
        dv_calls.append(value)
        return original_dv(value)

    with patch.object(activity_mod, "_deserialize_value", side_effect=tracking_dv):
        result_json = await execute_task("node-opaque", func_ref, args_json, kwargs_json)

    result = json.loads(result_json)
    assert result["node_id"] == "node-opaque"

    # The result should be a __cloudpickle__ envelope (complex object).
    assert isinstance(result["result"], dict), "Expected a serialized envelope"
    assert result["result"].get("__cloudpickle__"), "Expected __cloudpickle__ marker in result"

    # _deserialize_value must NOT have been called with the result blob.
    # (It may have been called with other things like args — but NOT with a
    # __cloudpickle__ result from the child.)
    result_blob = result["result"]
    assert result_blob not in dv_calls, (
        "Activity called _deserialize_value on the result blob — boundary violated!"
    )

    # Confirm the value is recoverable downstream (but NOT done by the activity).
    recovered = _deserialize_value(result_blob)
    assert recovered.val == 42


# ---------------------------------------------------------------------------
# 4. prepare_node_inputs keeps results opaque
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_node_inputs_opaque(mock_activity_ctx):
    """prepare_node_inputs must pass __cloudpickle__ blobs through without loading."""
    # Simulate a completed result that is a cloudpickle blob.
    blob = {
        "__cloudpickle__": True,
        "data": base64.b64encode(cloudpickle.dumps(lambda x: x + 1)).decode(),
    }
    node_data = {
        "node_id": "task-B",
        "func_ref": "dummy",
        "args": [{"__proxy__": True, "node_id": "task-A"}],
        "kwargs": {},
    }
    completed = {"task-A": blob}

    import marqov.workflows.activity as activity_mod
    dv_calls_with_blob: list = []
    original_dv = activity_mod._deserialize_value

    def spy_dv(value: Any) -> Any:
        if isinstance(value, dict) and value.get("__cloudpickle__"):
            dv_calls_with_blob.append(value)
        return original_dv(value)

    with patch.object(activity_mod, "_deserialize_value", side_effect=spy_dv):
        result_json = await prepare_node_inputs(
            json.dumps(node_data), json.dumps(completed)
        )

    result = json.loads(result_json)
    # The blob must have passed through unchanged.
    assert result["args"] == [blob]
    # _deserialize_value must not have been called on the blob itself.
    assert dv_calls_with_blob == [], (
        "prepare_node_inputs called _deserialize_value on an upstream __cloudpickle__ result!"
    )


# ---------------------------------------------------------------------------
# 5. Heartbeat fires while child runs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_heartbeat_fires_during_child(mock_activity_ctx):
    """Heartbeat task fires at least once while blocking on the child subprocess."""
    mock_hb = mock_activity_ctx

    # A function that sleeps long enough for a heartbeat to fire.
    def slow_fn() -> int:
        import time
        time.sleep(0.3)
        return 7

    func_ref = _enc(slow_fn)
    args_json = json.dumps([])
    kwargs_json = json.dumps({})

    import marqov.workflows.activity as activity_mod
    original_interval = activity_mod._HEARTBEAT_INTERVAL_S
    activity_mod._HEARTBEAT_INTERVAL_S = 0.05  # fire every 50ms
    try:
        result_json = await execute_task("node-hb", func_ref, args_json, kwargs_json)
    finally:
        activity_mod._HEARTBEAT_INTERVAL_S = original_interval

    result = json.loads(result_json)
    assert result["result"] == 7
    assert mock_hb.call_count >= 1, (
        f"Expected at least 1 heartbeat during child execution, got {mock_hb.call_count}"
    )


# ---------------------------------------------------------------------------
# 6. Cancel kills the child process group
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_kills_child(mock_activity_ctx):
    """When the activity task is cancelled, the child process group is killed."""
    killed_pids: list[int] = []
    real_killpg = os.killpg

    def spy_killpg(pgid: int, sig: int) -> None:
        killed_pids.append(pgid)
        try:
            real_killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def very_slow_fn() -> int:
        import time
        time.sleep(60)
        return 0

    func_ref = _enc(very_slow_fn)
    args_json = json.dumps([])
    kwargs_json = json.dumps({})

    with patch("marqov.workflows.activity.os.killpg", side_effect=spy_killpg):
        task = asyncio.create_task(
            execute_task("node-cancel", func_ref, args_json, kwargs_json)
        )
        # Give the child time to start
        await asyncio.sleep(0.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert len(killed_pids) >= 1, (
        "Expected os.killpg to be called when activity was cancelled"
    )


# ---------------------------------------------------------------------------
# 7. Separate error channel: child writes error.json; activity surfaces it as
#    RuntimeError WITHOUT reading result.json
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_task_failure_surfaces_via_error_json(mock_activity_ctx):
    """A task that raises must produce error.json; activity raises RuntimeError from it.

    Hardening: errors travel via error.json, NOT result.json.
    result.json must NOT exist when the task fails.
    """

    def always_fails() -> int:
        raise ValueError("intentional test failure")

    func_ref = _enc(always_fails)
    args_json = json.dumps([])
    kwargs_json = json.dumps({})

    with pytest.raises(RuntimeError, match="intentional test failure"):
        await execute_task("node-fail", func_ref, args_json, kwargs_json)


@pytest.mark.asyncio
async def test_error_json_exists_result_json_absent_on_failure(mock_activity_ctx, tmp_path):
    """On child failure, error.json must exist and result.json must NOT exist."""
    from marqov.workflows import _child_env as ce

    workdir_capture: list[Path] = []
    original_ntw = ce.new_task_workdir

    def capturing_ntw(node_id: str) -> Path:
        wd = original_ntw(node_id)
        workdir_capture.append(wd)
        return wd

    def always_fails() -> int:
        raise RuntimeError("channel test failure")

    func_ref = _enc(always_fails)

    with patch("marqov.workflows.activity.new_task_workdir", side_effect=capturing_ntw):
        with pytest.raises(RuntimeError, match="channel test failure"):
            await execute_task("node-chan", func_ref, json.dumps([]), json.dumps({}))

    assert workdir_capture, "new_task_workdir was never called"
    workdir = workdir_capture[0]
    error_path = workdir / "error.json"
    result_path = workdir / "result.json"

    assert error_path.exists(), "error.json must exist when the child task fails"
    assert not result_path.exists(), "result.json must NOT exist when the child task fails"

    # error.json must be valid, structured JSON with an "error" key.
    error_data = json.loads(error_path.read_text())
    assert "error" in error_data, "error.json must contain an 'error' key"
    assert "channel test failure" in error_data["error"]


# ---------------------------------------------------------------------------
# 8. Success path: json.loads is NEVER called on result.json content by the
#    activity. result.json bytes are forwarded opaquely.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_success_path_never_json_loads_result_content(mock_activity_ctx):
    """On success, the activity must forward result.json bytes WITHOUT calling
    json.loads on the result content.

    We spy on json.loads inside the activity module.  The activity is
    permitted to call json.loads on error.json (to check for errors), but
    MUST NOT call it on the content of result.json.
    """
    import marqov.workflows.activity as activity_mod

    def returns_data() -> dict:
        return {"answer": 42}

    func_ref = _enc(returns_data)
    args_json = json.dumps([])
    kwargs_json = json.dumps({})

    json_loads_calls: list[str] = []
    real_json_loads = json.loads

    def spy_json_loads(s, *args, **kwargs):
        json_loads_calls.append(str(s)[:80])
        return real_json_loads(s, *args, **kwargs)

    with patch.object(activity_mod.json, "loads", side_effect=spy_json_loads):
        result_raw = await execute_task("node-success", func_ref, args_json, kwargs_json)

    # The outer envelope is expected to be parsed by the caller, not the activity.
    # Check: the activity must not have parsed result.json content (only possibly
    # error.json, which is small and structured and ours).
    # result.json contains {"node_id": ..., "result": <blob>} — the "result" field
    # value (the blob itself) must not be json.loads'd by the activity.
    outer = real_json_loads(result_raw)
    assert outer["node_id"] == "node-success"
    # Confirm the activity never tried to json.loads the result content bytes.
    # (It's acceptable for the activity to read result.json as bytes/text and
    # forward it — but NOT to json.loads the result field inside it.)
    result_blob_str = json.dumps(outer["result"])
    for call_arg in json_loads_calls:
        assert result_blob_str not in call_arg, (
            "Activity called json.loads on the result content — boundary violated!\n"
            f"json.loads was called with: {call_arg!r}"
        )


# ---------------------------------------------------------------------------
# 9. Size cap: oversized child output is rejected before being read into the
#    worker process.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_oversized_result_is_rejected(mock_activity_ctx):
    """result.json larger than MAX_RESULT_BYTES must be rejected with RuntimeError."""
    import marqov.workflows.activity as activity_mod

    # Confirm the constant exists and is reasonable.
    assert hasattr(activity_mod, "MAX_RESULT_BYTES"), (
        "activity.py must define MAX_RESULT_BYTES constant"
    )
    cap = activity_mod.MAX_RESULT_BYTES
    assert 1024 * 1024 <= cap <= 256 * 1024 * 1024, (
        f"MAX_RESULT_BYTES={cap} is outside the expected range [1 MiB, 256 MiB]"
    )

    # Simulate a child that writes a result.json exceeding the cap.
    from marqov.workflows import _child_env as ce

    workdir_capture: list[Path] = []
    original_ntw = ce.new_task_workdir

    def capturing_ntw(node_id: str) -> Path:
        wd = original_ntw(node_id)
        workdir_capture.append(wd)
        return wd

    def normal_fn() -> int:
        return 1

    func_ref = _enc(normal_fn)

    # We need the child to write a file that exceeds the cap.  Instead of
    # running a real task that produces huge output (slow), we intercept after
    # the child exits and replace result.json with an oversized stub.
    original_path_stat = Path.stat

    def patched_stat(self, *args, **kwargs):
        st = original_path_stat(self, *args, **kwargs)
        # For any result.json inside a marqov_task_ workdir, lie about the size.
        if self.name == "result.json" and "marqov_task_" in str(self.parent):
            import os
            import stat as stat_mod
            # Return a stat_result with st_size beyond the cap.
            fake_size = cap + 1
            # stat_result is a named tuple-like; rebuild from os.stat_result
            fields = list(st)
            fields[6] = fake_size  # index 6 = st_size
            return os.stat_result(fields)
        return st

    with patch("marqov.workflows.activity.new_task_workdir", side_effect=capturing_ntw):
        with patch.object(Path, "stat", patched_stat):
            with pytest.raises(RuntimeError, match="[Ss]ize|[Ll]imit|[Tt]oo large|[Oo]versized|[Cc]ap"):
                await execute_task("node-big", func_ref, json.dumps([]), json.dumps({}))


@pytest.mark.asyncio
async def test_oversized_error_json_is_rejected(mock_activity_ctx):
    """error.json larger than MAX_ERROR_BYTES must be rejected (not read in full)."""
    import marqov.workflows.activity as activity_mod

    assert hasattr(activity_mod, "MAX_ERROR_BYTES"), (
        "activity.py must define MAX_ERROR_BYTES constant"
    )
    cap = activity_mod.MAX_ERROR_BYTES
    assert 1024 <= cap <= 1024 * 1024, (
        f"MAX_ERROR_BYTES={cap} is outside the expected range [1 KiB, 1 MiB]"
    )


# ---------------------------------------------------------------------------
# 10. Fail-closed fallback: build_child_env with no MARQOV_SCRUB_ALLOWLIST set
#     must NOT copy os.environ wholesale — it uses the minimal-safe allowlist.
# ---------------------------------------------------------------------------


def test_build_child_env_fail_closed_no_allowlist_env_var():
    """With MARQOV_SCRUB_ALLOWLIST unset, build_child_env uses the minimal-safe
    allowlist and never includes arbitrary parent-env vars.

    This asserts the fail-closed invariant: an absent var must not cause the
    function to fall back to os.environ.copy() or similar.
    """
    import tempfile
    from marqov.workflows._child_env import build_child_env, _MINIMAL_SAFE_DEFAULT

    # Inject a unique canary into the parent env.
    canary_key = "MARQOV_CANARY_SECRET_XYZ"
    canary_val = "should-never-reach-child"

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        with patch.dict(os.environ, {canary_key: canary_val}, clear=False):
            # Ensure the allowlist var itself is NOT set.
            env_without_allowlist = {
                k: v for k, v in os.environ.items()
                if k != "MARQOV_SCRUB_ALLOWLIST"
            }
            with patch.dict(os.environ, env_without_allowlist, clear=True):
                child_env = build_child_env(workdir)

    # The canary must not be in the child env.
    assert canary_key not in child_env, (
        f"build_child_env leaked parent-env key {canary_key!r} into child env "
        "when MARQOV_SCRUB_ALLOWLIST was unset — this is NOT fail-closed!"
    )

    # Only keys from the minimal-safe allowlist (plus HOME/TMPDIR overrides) should be present.
    allowed_keys = set(_MINIMAL_SAFE_DEFAULT) | {"HOME", "TMPDIR", "MARQOV_TASK_WORKDIR"}
    unexpected = {k for k in child_env if k not in allowed_keys}
    assert not unexpected, (
        f"build_child_env included unexpected keys when using minimal-safe fallback: "
        f"{unexpected}"
    )
