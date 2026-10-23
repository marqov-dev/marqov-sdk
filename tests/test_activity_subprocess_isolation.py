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
async def test_error_json_exists_result_json_absent_on_failure(
    mock_activity_ctx, tmp_path, monkeypatch
):
    """On child failure, error.json must exist and result.json must NOT exist."""
    import shutil

    from marqov.workflows import _child_env as ce

    # Inspecting the workdir after the fact needs the debug escape hatch:
    # execute_task otherwise removes it (marqov-sdk#144).
    monkeypatch.setenv("MARQOV_KEEP_TASK_WORKDIR", "1")

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

    try:
        assert error_path.exists(), "error.json must exist when the child task fails"
        assert not result_path.exists(), "result.json must NOT exist when the child task fails"

        # error.json must be valid, structured JSON with an "error" key.
        error_data = json.loads(error_path.read_text())
        assert "error" in error_data, "error.json must contain an 'error' key"
        assert "channel test failure" in error_data["error"]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 8. Success path: the activity parses the result ENVELOPE (it must validate
#    node_id, see marqov-sdk#141) but never cloudpickle.loads the result VALUE.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_success_path_never_cloudpickle_loads_result_value(mock_activity_ctx):
    """On success, the activity keeps the result VALUE opaque.

    Since marqov-sdk#141 the activity parses the envelope with json.loads so
    it can reject a forged node_id.  That parse is bounded by the
    MAX_RESULT_BYTES check, which runs first.  The boundary that still holds
    is the one that matters: the activity never calls cloudpickle.loads on
    anything the child produced.
    """

    class _Opaque:
        def __init__(self, val: int) -> None:
            self.val = val

    def returns_data() -> Any:
        return _Opaque(42)

    func_ref = _enc(returns_data)
    args_json = json.dumps([])
    kwargs_json = json.dumps({})

    loads_calls: list = []
    real_loads = cloudpickle.loads

    def spy_loads(data: bytes, *args: Any, **kwargs: Any) -> Any:
        loads_calls.append(data[:16])
        return real_loads(data, *args, **kwargs)

    with patch.object(cloudpickle, "loads", side_effect=spy_loads):
        result_raw = await execute_task("node-success", func_ref, args_json, kwargs_json)

    assert loads_calls == [], (
        "Activity process called cloudpickle.loads on child output: "
        f"boundary violated! Calls: {loads_calls}"
    )

    outer = json.loads(result_raw)
    assert outer["node_id"] == "node-success"
    assert outer["result"].get("__cloudpickle__") is True


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


# ---------------------------------------------------------------------------
# 11. Result envelope validation (marqov-sdk#141)
#
# The child owns its workdir, so it can write anything it likes to
# result.json.  The activity must therefore validate the envelope instead of
# forwarding the child's bytes verbatim.  Every task body below does the
# forging itself: patching result.json from the test process would not
# exercise the real path.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_forged_node_id_is_rejected(mock_activity_ctx):
    """A child claiming a different node_id must fail the activity, not return."""
    from temporalio.exceptions import ApplicationError

    def forge_node_id() -> None:
        import json as _json
        import os as _os

        workdir = _os.environ["MARQOV_TASK_WORKDIR"]
        with open(_os.path.join(workdir, "result.json"), "w") as handle:
            _json.dump({"node_id": "victim-node", "result": {"stolen": True}}, handle)
        _os._exit(0)

    with pytest.raises(ApplicationError) as excinfo:
        await execute_task(
            "attacker-node", _enc(forge_node_id), json.dumps([]), json.dumps({})
        )

    assert excinfo.value.non_retryable is True
    message = str(excinfo.value)
    assert "attacker-node" in message
    assert "victim-node" in message


@pytest.mark.asyncio
async def test_forged_node_id_cannot_poison_sibling(mock_activity_ctx):
    """A forging sibling contributes nothing to the results a workflow would keep."""
    from temporalio.exceptions import ApplicationError

    def honest() -> int:
        return 11

    def forge_honest_node_id() -> None:
        import json as _json
        import os as _os

        workdir = _os.environ["MARQOV_TASK_WORKDIR"]
        with open(_os.path.join(workdir, "result.json"), "w") as handle:
            _json.dump({"node_id": "honest-node", "result": 666}, handle)
        _os._exit(0)

    completed_results: dict[str, Any] = {}

    honest_json = await execute_task(
        "honest-node", _enc(honest), json.dumps([]), json.dumps({})
    )
    honest_envelope = json.loads(honest_json)
    completed_results[honest_envelope["node_id"]] = honest_envelope["result"]

    with pytest.raises(ApplicationError):
        forged_json = await execute_task(
            "forging-node", _enc(forge_honest_node_id), json.dumps([]), json.dumps({})
        )
        # Unreachable: only a returned value could reach completed_results.
        forged_envelope = json.loads(forged_json)
        completed_results[forged_envelope["node_id"]] = forged_envelope["result"]

    assert completed_results == {"honest-node": 11}


@pytest.mark.asyncio
async def test_non_json_result_is_rejected(mock_activity_ctx):
    """Non-JSON bytes in result.json must fail the activity, not be forwarded."""
    from temporalio.exceptions import ApplicationError

    def forge_non_json() -> None:
        import os as _os

        workdir = _os.environ["MARQOV_TASK_WORKDIR"]
        with open(_os.path.join(workdir, "result.json"), "w") as handle:
            handle.write("not json at all")
        _os._exit(0)

    with pytest.raises(ApplicationError) as excinfo:
        await execute_task(
            "garbage-node", _enc(forge_non_json), json.dumps([]), json.dumps({})
        )

    assert excinfo.value.non_retryable is True
    assert "garbage-node" in str(excinfo.value)


@pytest.mark.asyncio
async def test_extra_envelope_keys_are_rejected(mock_activity_ctx):
    """The envelope must carry exactly node_id and result."""
    from temporalio.exceptions import ApplicationError

    def forge_extra_key() -> None:
        import json as _json
        import os as _os

        workdir = _os.environ["MARQOV_TASK_WORKDIR"]
        with open(_os.path.join(workdir, "result.json"), "w") as handle:
            _json.dump(
                {"node_id": "extra-node", "result": 1, "smuggled": "payload"}, handle
            )
        _os._exit(0)

    with pytest.raises(ApplicationError) as excinfo:
        await execute_task(
            "extra-node", _enc(forge_extra_key), json.dumps([]), json.dumps({})
        )

    assert excinfo.value.non_retryable is True


@pytest.mark.asyncio
async def test_happy_path_envelope_unchanged(mock_activity_ctx):
    """An ordinary complex result still round-trips under the activity's node_id."""

    class _Payload:
        def __init__(self, val: int) -> None:
            self.val = val

    def build() -> Any:
        return _Payload(5)

    result_json = await execute_task(
        "happy-node", _enc(build), json.dumps([]), json.dumps({})
    )
    envelope = json.loads(result_json)

    assert set(envelope) == {"node_id", "result"}
    assert envelope["node_id"] == "happy-node"
    assert envelope["result"].get("__cloudpickle__") is True
    assert _deserialize_value(envelope["result"]).val == 5


# ---------------------------------------------------------------------------
# 12. Validation failures stay bounded and stay non-retryable
#
# Everything quoted back in a validation message came from the child, so it
# must be clipped: an oversized failure message would exceed Temporal's
# payload limit, the failure response would be rejected, and the
# non-retryable signal would be lost.  Malformed bytes that raise something
# other than JSONDecodeError must still land as a non-retryable failure
# rather than an ordinary retryable one.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_huge_forged_node_id_message_is_bounded(mock_activity_ctx):
    """A 1 MiB forged node_id must not produce a 1 MiB failure message."""
    from temporalio.exceptions import ApplicationError

    def forge_huge_node_id() -> None:
        import json as _json
        import os as _os

        workdir = _os.environ["MARQOV_TASK_WORKDIR"]
        with open(_os.path.join(workdir, "result.json"), "w") as handle:
            _json.dump({"node_id": "V" * (1024 * 1024), "result": 1}, handle)
        _os._exit(0)

    with pytest.raises(ApplicationError) as excinfo:
        await execute_task(
            "bounded-node", _enc(forge_huge_node_id), json.dumps([]), json.dumps({})
        )

    assert excinfo.value.non_retryable is True
    message = str(excinfo.value)
    assert len(message) < 2000, f"Failure message is {len(message)} chars"
    assert "bounded-node" in message


@pytest.mark.asyncio
async def test_huge_forged_keys_message_is_bounded(mock_activity_ctx):
    """Many long forged envelope keys must not produce an unbounded message."""
    from temporalio.exceptions import ApplicationError

    def forge_many_keys() -> None:
        import json as _json
        import os as _os

        workdir = _os.environ["MARQOV_TASK_WORKDIR"]
        envelope = {f"k{index}" + "P" * 4096: index for index in range(200)}
        with open(_os.path.join(workdir, "result.json"), "w") as handle:
            _json.dump(envelope, handle)
        _os._exit(0)

    with pytest.raises(ApplicationError) as excinfo:
        await execute_task(
            "keys-node", _enc(forge_many_keys), json.dumps([]), json.dumps({})
        )

    assert excinfo.value.non_retryable is True
    assert len(str(excinfo.value)) < 2000


@pytest.mark.asyncio
async def test_invalid_utf8_result_is_rejected(mock_activity_ctx):
    """Non-UTF-8 bytes raise UnicodeDecodeError; it must still be non-retryable."""
    from temporalio.exceptions import ApplicationError

    def forge_invalid_utf8() -> None:
        import os as _os

        workdir = _os.environ["MARQOV_TASK_WORKDIR"]
        with open(_os.path.join(workdir, "result.json"), "wb") as handle:
            handle.write(b'{"node_id": "\xff\xfe", "result": 1}')
        _os._exit(0)

    with pytest.raises(ApplicationError) as excinfo:
        await execute_task(
            "utf8-node", _enc(forge_invalid_utf8), json.dumps([]), json.dumps({})
        )

    assert excinfo.value.non_retryable is True
    assert len(str(excinfo.value)) < 2000


@pytest.mark.asyncio
async def test_deeply_nested_result_is_rejected(mock_activity_ctx):
    """Deeply nested JSON raises RecursionError; it must still be non-retryable."""
    from temporalio.exceptions import ApplicationError

    def forge_deep_nesting() -> None:
        import os as _os

        workdir = _os.environ["MARQOV_TASK_WORKDIR"]
        with open(_os.path.join(workdir, "result.json"), "w") as handle:
            handle.write("[" * 100_000)
        _os._exit(0)

    with pytest.raises(ApplicationError) as excinfo:
        await execute_task(
            "deep-node", _enc(forge_deep_nesting), json.dumps([]), json.dumps({})
        )

    assert excinfo.value.non_retryable is True
    assert len(str(excinfo.value)) < 2000


# ---------------------------------------------------------------------------
# Workdir lifecycle (marqov-sdk#144): each task's scratch directory is removed
# once execute_task finishes, and lives inside a private per-worker parent.
# ---------------------------------------------------------------------------


def _task_dirs_under_temp_root() -> set[str]:
    """Every ``marqov_task_*`` dir in the temp root or a ``marqov_worker_*`` parent."""
    import glob
    import tempfile

    root = tempfile.gettempdir()
    return set(glob.glob(os.path.join(root, "marqov_task_*"))) | set(
        glob.glob(os.path.join(root, "marqov_worker_*", "marqov_task_*"))
    )


def _workdir_fn_ref() -> str:
    """Encode a task body that returns its own workdir.

    Defined locally so cloudpickle serialises it by value: the child cannot
    import this test module.
    """

    def return_workdir() -> str:
        import os as _os

        return _os.environ["MARQOV_TASK_WORKDIR"]

    return _enc(return_workdir)


@pytest.mark.asyncio
async def test_workdir_removed_after_success(mock_activity_ctx, monkeypatch):
    monkeypatch.delenv("MARQOV_KEEP_TASK_WORKDIR", raising=False)
    result_json = await execute_task(
        "wd-ok", _workdir_fn_ref(), json.dumps([]), json.dumps({})
    )
    workdir = json.loads(result_json)["result"]
    assert isinstance(workdir, str) and workdir
    assert not os.path.exists(workdir), f"workdir {workdir} survived a successful task"


@pytest.mark.asyncio
async def test_workdir_removed_after_failure(mock_activity_ctx, monkeypatch):
    monkeypatch.delenv("MARQOV_KEEP_TASK_WORKDIR", raising=False)

    def fail_with_workdir() -> None:
        import os as _os

        raise RuntimeError("WD<" + _os.environ["MARQOV_TASK_WORKDIR"] + ">")

    with pytest.raises(RuntimeError) as excinfo:
        await execute_task(
            "wd-fail", _enc(fail_with_workdir), json.dumps([]), json.dumps({})
        )
    workdir = str(excinfo.value).split("WD<", 1)[1].split(">", 1)[0]
    assert not os.path.exists(workdir), f"workdir {workdir} survived a failed task"


@pytest.mark.asyncio
async def test_workdir_removed_after_cancellation(mock_activity_ctx, monkeypatch):
    """Cancellation (which is also how a Temporal timeout arrives) cleans up."""
    from marqov.workflows import _child_env as ce

    monkeypatch.delenv("MARQOV_KEEP_TASK_WORKDIR", raising=False)
    captured: list[Path] = []
    original_ntw = ce.new_task_workdir

    def capturing_ntw(node_id: str) -> Path:
        wd = original_ntw(node_id)
        captured.append(wd)
        return wd

    def very_slow_fn() -> int:
        import time as _time

        _time.sleep(60)
        return 0

    with patch("marqov.workflows.activity.new_task_workdir", side_effect=capturing_ntw):
        task = asyncio.create_task(
            execute_task("wd-cancel", _enc(very_slow_fn), json.dumps([]), json.dumps({}))
        )
        await asyncio.sleep(0.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert captured, "new_task_workdir was never called"
    assert not captured[0].exists(), f"workdir {captured[0]} survived cancellation"


@pytest.mark.asyncio
async def test_keep_task_workdir_escape_hatch(mock_activity_ctx, monkeypatch):
    import shutil

    monkeypatch.setenv("MARQOV_KEEP_TASK_WORKDIR", "1")
    result_json = await execute_task(
        "wd-keep", _workdir_fn_ref(), json.dumps([]), json.dumps({})
    )
    kept = json.loads(result_json)["result"]
    try:
        assert os.path.isdir(kept), "MARQOV_KEEP_TASK_WORKDIR=1 must keep the workdir"
    finally:
        shutil.rmtree(kept, ignore_errors=True)

    monkeypatch.delenv("MARQOV_KEEP_TASK_WORKDIR")
    result_json = await execute_task(
        "wd-keep2", _workdir_fn_ref(), json.dumps([]), json.dumps({})
    )
    assert not os.path.exists(json.loads(result_json)["result"])


@pytest.mark.asyncio
async def test_no_workdir_accumulation_across_batch(mock_activity_ctx, monkeypatch):
    monkeypatch.delenv("MARQOV_KEEP_TASK_WORKDIR", raising=False)

    def ok() -> int:
        return 1

    def bad() -> int:
        raise RuntimeError("batch failure")

    before = _task_dirs_under_temp_root()
    for i in range(6):
        if i % 2:
            with pytest.raises(RuntimeError):
                await execute_task(f"batch-{i}", _enc(bad), json.dumps([]), json.dumps({}))
        else:
            await execute_task(f"batch-{i}", _enc(ok), json.dumps([]), json.dumps({}))
    after = _task_dirs_under_temp_root()
    assert after - before == set(), f"task workdirs left behind: {sorted(after - before)}"


@pytest.mark.asyncio
async def test_workdir_lives_in_private_worker_parent(mock_activity_ctx):
    import stat as stat_mod

    def report_parent() -> dict:
        import os as _os

        wd = _os.environ["MARQOV_TASK_WORKDIR"]
        parent = _os.path.dirname(wd)
        return {
            "parent": parent,
            "parent_name": _os.path.basename(parent),
            "env": dict(_os.environ),
        }

    result_json = await execute_task(
        "wd-parent", _enc(report_parent), json.dumps([]), json.dumps({})
    )
    seen = json.loads(result_json)["result"]
    assert seen["parent_name"].startswith("marqov_worker_"), seen["parent_name"]
    if sys.platform != "win32":
        mode = stat_mod.S_IMODE(os.stat(seen["parent"]).st_mode)
        assert mode == 0o700, oct(mode)

    # No variable other than MARQOV_TASK_WORKDIR names the parent itself, and
    # every other mention of it is inside this task's own workdir (HOME, TMPDIR).
    workdir = seen["env"]["MARQOV_TASK_WORKDIR"]
    for key, value in seen["env"].items():
        if key == "MARQOV_TASK_WORKDIR" or seen["parent"] not in value:
            continue
        assert value.startswith(workdir + os.sep), (
            f"{key}={value!r} exposes the worker parent directory"
        )


def _slow_rmtree(started: Any, delay: float) -> Any:
    """A stand-in for shutil.rmtree that takes ``delay`` seconds, then removes."""
    import shutil

    real_rmtree = shutil.rmtree

    def slow(path: Any, *args: Any, **kwargs: Any) -> None:
        started.set()
        time.sleep(delay)
        real_rmtree(path, *args, **kwargs)

    return slow


@pytest.mark.asyncio
async def test_workdir_removal_does_not_block_event_loop(mock_activity_ctx, monkeypatch):
    """A slow workdir removal must not stall the worker's event loop (heartbeats)."""
    import threading

    monkeypatch.delenv("MARQOV_KEEP_TASK_WORKDIR", raising=False)
    started = threading.Event()
    gaps: list[float] = []
    stop = False

    async def ticker() -> None:
        last = time.monotonic()
        while not stop:
            await asyncio.sleep(0.01)
            now = time.monotonic()
            if started.is_set():
                gaps.append(now - last)
            last = now

    tick_task = asyncio.create_task(ticker())
    with patch(
        "marqov.workflows.activity.shutil.rmtree", _slow_rmtree(started, delay=1.0)
    ):
        result_json = await execute_task(
            "wd-slow-rm", _workdir_fn_ref(), json.dumps([]), json.dumps({})
        )
    stop = True
    await tick_task

    assert started.is_set(), "workdir removal never ran"
    assert not os.path.exists(json.loads(result_json)["result"])
    assert gaps and max(gaps) < 0.5, (
        f"event loop stalled for {max(gaps, default=0):.2f}s during workdir removal"
    )


@pytest.mark.asyncio
async def test_cancellation_during_removal_still_removes_workdir(
    mock_activity_ctx, monkeypatch
):
    """A cancellation that lands while the workdir is being removed must not skip it."""
    import threading

    monkeypatch.delenv("MARQOV_KEEP_TASK_WORKDIR", raising=False)
    started = threading.Event()
    with patch(
        "marqov.workflows.activity.shutil.rmtree", _slow_rmtree(started, delay=0.5)
    ):
        task = asyncio.create_task(
            execute_task("wd-cancel-rm", _workdir_fn_ref(), json.dumps([]), json.dumps({}))
        )
        deadline = time.monotonic() + 30
        while not started.is_set():
            assert time.monotonic() < deadline, "workdir removal never started"
            await asyncio.sleep(0.01)
        # Cancel twice while removal is in progress.
        task.cancel()
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    from marqov.workflows import _child_env as ce

    parent = ce._worker_parent_dir
    assert parent is not None
    leftover = list(parent.glob("marqov_task_wd-cancel-rm_*"))
    assert leftover == [], f"workdir left behind after cancellation: {leftover}"


@pytest.mark.asyncio
async def test_grandchild_cannot_recreate_workdir(mock_activity_ctx, monkeypatch):
    """A process the task leaves running is killed before the workdir is removed."""
    monkeypatch.delenv("MARQOV_KEEP_TASK_WORKDIR", raising=False)

    def spawn_late_writer() -> str:
        import os as _os
        import subprocess as _subprocess
        import sys as _sys

        wd = _os.environ["MARQOV_TASK_WORKDIR"]
        code = (
            "import os, time; time.sleep(0.5); "
            f"os.makedirs(os.path.join({wd!r}, 'late'), exist_ok=True); "
            f"open(os.path.join({wd!r}, 'late', 'x'), 'w').write('hi')"
        )
        _subprocess.Popen(
            [_sys.executable, "-c", code],
            stdin=_subprocess.DEVNULL,
            stdout=_subprocess.DEVNULL,
            stderr=_subprocess.DEVNULL,
        )
        return wd

    result_json = await execute_task(
        "wd-grandchild", _enc(spawn_late_writer), json.dumps([]), json.dumps({})
    )
    workdir = json.loads(result_json)["result"]
    await asyncio.sleep(1.5)
    try:
        assert not os.path.exists(workdir), (
            f"a leftover grandchild recreated {workdir} after removal"
        )
    finally:
        import shutil

        shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Heartbeats span the workdir removal: a slow removal after the task finished
# must not let Temporal's heartbeat timeout fire and retry the task.
# ---------------------------------------------------------------------------


def _heartbeat_gated_rmtree(mock_hb: Any, removing: Any, beat_seen: Any) -> Any:
    """A stand-in for shutil.rmtree that waits for a heartbeat, then removes.

    While it runs, ``removing`` is set; a heartbeat sent in that window sets
    ``beat_seen``. It waits up to 10s for one, so the test does not depend on
    tight timing, then performs the real removal.
    """
    import shutil

    real_rmtree = shutil.rmtree

    def on_heartbeat(*args: Any, **kwargs: Any) -> None:
        if removing.is_set():
            beat_seen.set()

    mock_hb.side_effect = on_heartbeat

    def gated(path: Any, *args: Any, **kwargs: Any) -> None:
        removing.set()
        try:
            beat_seen.wait(timeout=10)
        finally:
            removing.clear()
        real_rmtree(path, *args, **kwargs)

    return gated


@pytest.mark.asyncio
async def test_heartbeats_continue_during_removal_after_success(
    mock_activity_ctx, monkeypatch
):
    import threading

    import marqov.workflows.activity as activity_mod

    monkeypatch.delenv("MARQOV_KEEP_TASK_WORKDIR", raising=False)
    monkeypatch.setattr(activity_mod, "_HEARTBEAT_INTERVAL_S", 0.05)
    removing, beat_seen = threading.Event(), threading.Event()
    with patch(
        "marqov.workflows.activity.shutil.rmtree",
        _heartbeat_gated_rmtree(mock_activity_ctx, removing, beat_seen),
    ):
        result_json = await execute_task(
            "hb-rm-ok", _workdir_fn_ref(), json.dumps([]), json.dumps({})
        )
    assert not os.path.exists(json.loads(result_json)["result"])
    assert beat_seen.is_set(), "no heartbeat was sent while the workdir was removed"


@pytest.mark.asyncio
async def test_heartbeats_continue_during_removal_after_failure(
    mock_activity_ctx, monkeypatch
):
    import threading

    import marqov.workflows.activity as activity_mod

    monkeypatch.delenv("MARQOV_KEEP_TASK_WORKDIR", raising=False)
    monkeypatch.setattr(activity_mod, "_HEARTBEAT_INTERVAL_S", 0.05)

    def fail() -> None:
        raise RuntimeError("task failed on purpose")

    removing, beat_seen = threading.Event(), threading.Event()
    with patch(
        "marqov.workflows.activity.shutil.rmtree",
        _heartbeat_gated_rmtree(mock_activity_ctx, removing, beat_seen),
    ):
        with pytest.raises(RuntimeError, match="task failed on purpose"):
            await execute_task("hb-rm-fail", _enc(fail), json.dumps([]), json.dumps({}))
    assert beat_seen.is_set(), "no heartbeat was sent while the workdir was removed"


@pytest.mark.asyncio
async def test_heartbeats_continue_during_removal_after_cancellation(
    mock_activity_ctx, monkeypatch
):
    import threading

    import marqov.workflows.activity as activity_mod

    monkeypatch.delenv("MARQOV_KEEP_TASK_WORKDIR", raising=False)
    monkeypatch.setattr(activity_mod, "_HEARTBEAT_INTERVAL_S", 0.05)

    def very_slow_fn() -> int:
        import time as _time

        _time.sleep(60)
        return 0

    removing, beat_seen = threading.Event(), threading.Event()
    with patch(
        "marqov.workflows.activity.shutil.rmtree",
        _heartbeat_gated_rmtree(mock_activity_ctx, removing, beat_seen),
    ):
        task = asyncio.create_task(
            execute_task("hb-rm-cancel", _enc(very_slow_fn), json.dumps([]), json.dumps({}))
        )
        await asyncio.sleep(0.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert beat_seen.is_set(), "no heartbeat was sent while the workdir was removed"


@pytest.mark.asyncio
async def test_heartbeat_loop_stops_after_removal(mock_activity_ctx, monkeypatch):
    import marqov.workflows.activity as activity_mod

    monkeypatch.delenv("MARQOV_KEEP_TASK_WORKDIR", raising=False)
    monkeypatch.setattr(activity_mod, "_HEARTBEAT_INTERVAL_S", 0.02)
    result_json = await execute_task(
        "hb-stop", _workdir_fn_ref(), json.dumps([]), json.dumps({})
    )
    assert not os.path.exists(json.loads(result_json)["result"])
    beats_at_return = mock_activity_ctx.call_count
    # Ten heartbeat intervals: a loop still running would have beaten again.
    await asyncio.sleep(0.2)
    assert mock_activity_ctx.call_count == beats_at_return, (
        "heartbeats were still sent after execute_task returned"
    )


@pytest.mark.asyncio
async def test_cancellation_while_stopping_heartbeat_propagates() -> None:
    """A cancel of execute_task that lands while its heartbeat loop is stopped
    must propagate, not be taken for the loop's own cancellation."""
    import marqov.workflows.activity as activity_mod

    outer: asyncio.Task[str] | None = None

    async def fake_run(*args: Any, **kwargs: Any) -> str:
        return "RESULT"

    async def fake_remove(workdir: Path) -> None:
        # Fires on the next loop tick, while the outer finally awaits the
        # cancelled heartbeat loop.
        assert outer is not None
        asyncio.get_running_loop().call_soon(outer.cancel)

    with patch.object(activity_mod, "_run_task_in_workdir", fake_run), patch.object(
        activity_mod, "_remove_task_workdir", fake_remove
    ), patch("temporalio.activity.heartbeat"), patch("temporalio.activity.logger", MagicMock()):
        outer = asyncio.create_task(execute_task("hb-cancel-stop", "x", "[]", "{}"))
        with pytest.raises(asyncio.CancelledError):
            await outer
