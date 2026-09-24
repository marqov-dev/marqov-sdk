"""Request building and validation for managed native submission.

The hosted API admits Python ``@task``/``@workflow`` programs through a
versioned, closed request body (``marqov.public-submission/v1``). Everything
here runs before any network call, so a request the server would reject for its
shape is refused locally instead. The server remains the authority: it
re-validates every field and decides admission, funding and budgets.

Observable contract mirrored here:

* ``team_id`` / ``script_id`` / ``Idempotency-Key``: RFC 4122 UUIDs (version
  1-8, RFC variant) or the nil / max UUID; braces and undashed forms are refused.
* ``input``: JSON *text* holding exactly ``{"entrypoint", "args", "kwargs"}``;
  ``entrypoint`` is a Python identifier of at most 128 characters; nesting depth
  at most 32; integers at most 4300 digits; finite numbers only.
* source and input: non-empty, valid UTF-8, at most 1 MiB each; source has no NUL.
* whole body: at most 4 MiB. The body is closed, so no other field may be sent.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from marqov.platform.errors import MarqovPlatformError

PUBLIC_SUBMISSION_PROTOCOL = "marqov.public-submission/v1"
ADMISSION_RECEIPT_PROTOCOL = "marqov.funded-admission-receipt/v1"
PROGRAMMING_MODELS = ("native_workflow", "single_task")

#: Upper bound the server applies to ``cap_cents`` (100 USD per job).
MAX_CAP_CENTS = 10_000
_MAX_ARTIFACT_BYTES = 1024 * 1024
_MAX_BODY_BYTES = 4 * 1024 * 1024
_MAX_DEPTH = 32
_MAX_INTEGER_DIGITS = 4300

_UUID = re.compile(
    r"^(?:[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-8][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}"
    r"|00000000-0000-0000-0000-000000000000|ffffffff-ffff-ffff-ffff-ffffffffffff)$"
)
_ENTRYPOINT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_RUNTIME_KEYS = {"backend", "programming_model", "min_cap_cents", "max_cap_cents"}
_RECEIPT_KEYS = {"protocol_version", "job_id", "submission_id", "source_sha256", "input_sha256"}


def require_uuid(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _UUID.match(value):
        raise ValueError(f"{name} must be a UUID string (e.g. from the Marqov app), got {value!r}")
    return value


def _utf8(text: str, name: str) -> bytes:
    try:
        data = text.encode("utf-8")
    except UnicodeEncodeError as exc:  # lone surrogates cannot be sent as JSON text
        raise ValueError(f"{name} is not valid UTF-8 text") from exc
    if not 0 < len(data) <= _MAX_ARTIFACT_BYTES:
        raise ValueError(f"{name} must be between 1 byte and 1 MiB")
    return data


def _check_value(value: Any, depth: int, where: str) -> None:
    """Allow only values whose JSON encoding the server decodes unchanged."""
    if depth > _MAX_DEPTH:
        raise ValueError(f"{where}: nesting deeper than {_MAX_DEPTH} levels")
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        if len(str(abs(value))) > _MAX_INTEGER_DIGITS:
            raise ValueError(f"{where}: integer has more than {_MAX_INTEGER_DIGITS} digits")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{where}: NaN and infinity are not JSON")
        return
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError(f"{where}: string is not valid UTF-8 text") from exc
        return
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _check_value(item, depth + 1, f"{where}[{i}]")
        return
    if isinstance(value, Mapping):
        if depth + 1 > _MAX_DEPTH:
            raise ValueError(f"{where}: nesting deeper than {_MAX_DEPTH} levels")
        for key, item in value.items():
            # json.dumps would silently turn 1, True or None keys into strings.
            if not isinstance(key, str):
                raise TypeError(f"{where}: dictionary keys must be str, got {type(key).__name__}")
            _check_value(item, depth + 1, f"{where}[{key!r}]")
        return
    raise TypeError(f"{where}: {type(value).__name__} is not JSON-serialisable")


def build_public_submission(
    *,
    team_id: str,
    entrypoint: str,
    cap_cents: int,
    script_id: str | None,
    source: str | None,
    args: Sequence[Any],
    kwargs: Mapping[str, Any] | None,
    programming_model: str,
    backend: str,
    source_sha256: str | None,
) -> tuple[dict[str, Any], str]:
    """Return ``(body, input_text)`` for the versioned submission, or raise."""
    require_uuid(team_id, "team_id")
    if (script_id is None) == (source is None):
        raise ValueError("Pass exactly one of script_id (a saved script) or source (inline code)")
    if not isinstance(entrypoint, str) or not _ENTRYPOINT.match(entrypoint):
        raise ValueError("entrypoint must be a Python identifier of at most 128 characters")
    if programming_model not in PROGRAMMING_MODELS:
        raise ValueError(f"programming_model must be one of {PROGRAMMING_MODELS}")
    if not isinstance(backend, str) or not backend:
        raise ValueError("backend must be a non-empty string")
    if type(cap_cents) is not int or not 1 <= cap_cents <= MAX_CAP_CENTS:
        raise ValueError(f"cap_cents must be an int from 1 to {MAX_CAP_CENTS}")
    if source_sha256 is not None and (not isinstance(source_sha256, str) or not _SHA256.match(source_sha256)):
        raise ValueError("source_sha256 must be 64 lowercase hex characters")
    if not isinstance(args, (list, tuple)):
        raise TypeError("args must be a list or tuple")
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, Mapping):
        raise TypeError("kwargs must be a mapping")

    call = {"entrypoint": entrypoint, "args": list(args), "kwargs": dict(kwargs)}
    _check_value(call["args"], 1, "args")
    _check_value(call["kwargs"], 1, "kwargs")
    input_text = json.dumps(call, allow_nan=False)
    _utf8(input_text, "input")

    if source is not None:
        if not isinstance(source, str):
            raise TypeError("source must be a str")
        if "\x00" in source:
            raise ValueError("source must not contain NUL characters")
        _utf8(source, "source")
        source_field: dict[str, Any] = {"kind": "inline", "content": source}
    else:
        require_uuid(script_id, "script_id")
        source_field = {"kind": "script", "script_id": script_id}

    body: dict[str, Any] = {
        "protocol_version": PUBLIC_SUBMISSION_PROTOCOL,
        "team_id": team_id,
        "backend": backend,
        "programming_model": programming_model,
        "cap_cents": cap_cents,
        "analysis_id": None,
        "source": source_field,
        "input": input_text,
    }
    if source_sha256 is not None:
        body["source_sha256"] = source_sha256
    if len(json.dumps(body, allow_nan=False).encode("utf-8")) > _MAX_BODY_BYTES:
        raise ValueError("submission is larger than 4 MiB")
    return body, input_text


def parse_runtimes(response: Any) -> list[dict[str, Any]]:
    """Validate the discovery response; any unexpected shape fails closed."""
    runtimes = response.get("runtimes") if isinstance(response, dict) else None
    if not isinstance(runtimes, list):
        raise MarqovPlatformError("Runtime discovery returned an unexpected response", code="invalid_response")
    for runtime in runtimes:
        if (not isinstance(runtime, dict) or set(runtime) != _RUNTIME_KEYS
                or not isinstance(runtime["backend"], str)
                or runtime["programming_model"] not in PROGRAMMING_MODELS
                or type(runtime["min_cap_cents"]) is not int or type(runtime["max_cap_cents"]) is not int
                or not 0 < runtime["min_cap_cents"] <= runtime["max_cap_cents"]):
            raise MarqovPlatformError("Runtime discovery returned an unexpected runtime", code="invalid_response")
    return runtimes


def check_receipt(
    response: Any, *, input_text: str, source: str | None, source_sha256: str | None
) -> str:
    """Return the admitted job id, or raise if the receipt cannot be confirmed.

    Always checks the closed structure, protocol and the input hash. Checks the
    source hash only when the source is known locally (inline ``source``, or a
    caller-supplied ``source_sha256``); for a saved script without
    ``source_sha256`` the source content cannot be verified here.
    """
    def sha(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    if (not isinstance(response, dict) or set(response) != _RECEIPT_KEYS
            or response["protocol_version"] != ADMISSION_RECEIPT_PROTOCOL
            or not isinstance(response["job_id"], str) or not _UUID.match(response["job_id"])
            or not isinstance(response["submission_id"], str) or not _UUID.match(response["submission_id"])
            or not isinstance(response["source_sha256"], str) or not _SHA256.match(response["source_sha256"])
            or not isinstance(response["input_sha256"], str) or not _SHA256.match(response["input_sha256"])):
        raise MarqovPlatformError(
            "The platform returned an unrecognised admission receipt; the submission may or may not "
            "have been admitted. Retry with the same idempotency_key to confirm.",
            code="invalid_receipt",
        )
    expected_source = sha(source) if source is not None else source_sha256
    if response["input_sha256"] != sha(input_text) or (
        expected_source is not None and response["source_sha256"] != expected_source
    ):
        raise MarqovPlatformError(
            "The admission receipt does not match the submitted source or input.",
            code="receipt_mismatch",
        )
    return response["job_id"]
