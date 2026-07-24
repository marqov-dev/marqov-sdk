"""Child-process entry point for scrubbed task execution.

This script is invoked by ``execute_task`` as a subprocess.  It runs inside
a minimal, allowlisted environment (no host-provided secrets) and is the
ONLY place ``cloudpickle.loads`` is called on user-supplied bytes.

Protocol (file-based, never stdout):
- Reads:  ``$WORKDIR/func_ref``   — raw bytes, base64-decoded cloudpickle of the fn
          ``$WORKDIR/args.json``  — JSON list of args (may contain __cloudpickle__ blobs)
          ``$WORKDIR/kwargs.json`` — JSON dict of kwargs
- Writes: ``$WORKDIR/result.json`` — JSON: {"node_id": ..., "result": <serialized>}
                                     or  {"node_id": ..., "error": <str>}

Exit codes: 0 = success, 1 = error (error written to result.json).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any


def _deserialize_value(value: Any) -> Any:
    """Deserialize a value from JSON transport (mirrors activity._deserialize_value)."""
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
    """Serialize a value for JSON transport (mirrors activity._serialize_value)."""
    import cloudpickle

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    elif isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    elif isinstance(value, dict):
        if value.get("__cloudpickle__"):
            return value
        return {k: _serialize_value(v) for k, v in value.items()}
    else:
        return {
            "__cloudpickle__": True,
            "data": base64.b64encode(cloudpickle.dumps(value)).decode("utf-8"),
        }


def main() -> int:
    workdir = Path(os.environ.get("MARQOV_TASK_WORKDIR", ""))
    if not workdir or not workdir.is_dir():
        sys.stderr.write("MARQOV_TASK_WORKDIR not set or not a directory\n")
        return 1

    node_id = (workdir / "node_id").read_text().strip()
    result_path = workdir / "result.json"
    error_path = workdir / "error.json"

    def _write_error(msg: str) -> int:
        # Errors go to error.json (separate channel); result.json is never
        # written on failure so the activity can treat result.json as "success only".
        error_path.write_text(json.dumps({"node_id": node_id, "error": msg}))
        return 1

    try:
        import cloudpickle

        func_bytes = base64.b64decode((workdir / "func_ref").read_text().strip())
        func = cloudpickle.loads(func_bytes)

        args_raw: list[Any] = json.loads((workdir / "args.json").read_text())
        kwargs_raw: dict[str, Any] = json.loads((workdir / "kwargs.json").read_text())

        args = [_deserialize_value(a) for a in args_raw]
        kwargs = {k: _deserialize_value(v) for k, v in kwargs_raw.items()}

    except Exception as e:
        return _write_error(f"Deserialization error: {type(e).__name__}: {e}")

    try:
        if asyncio.iscoroutinefunction(func):
            result = asyncio.run(func(*args, **kwargs))
        else:
            result = func(*args, **kwargs)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return _write_error(f"Task error: {type(e).__name__}: {e}\n{tb}")

    try:
        serialized = _serialize_value(result)
        # Atomic write: temp + rename so the parent never reads a partial file.
        tmp_path = result_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps({"node_id": node_id, "result": serialized}))
        tmp_path.rename(result_path)
    except Exception as e:
        return _write_error(f"Serialization error: {type(e).__name__}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
