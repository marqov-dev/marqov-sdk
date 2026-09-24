"""Hosted-submission contract checks for ``marqov.platform``.

These pin what the client sends against the hosted API's *observable* request
contract (field names, required headers, status codes, endpoints), captured at
the HTTP session boundary. No network is used.

Scope: only the Marqov-hosted client (``marqov.platform``). Local and provider
execution (``MarqovDevice``, executors, self-hosted ``create_worker`` /
``WorkflowDispatch``) never talk to the hosted API and are out of scope here.

Tests that are expected to fail today are marked ``xfail(strict=True)``: each
reproduces a known mismatch and will turn into a hard failure (XPASS) as soon
as the mismatch is fixed, so the marker must be removed with the fix.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock

import pytest

from marqov.platform import MarqovClient
from marqov.platform.job import Job

# ---------------------------------------------------------------------------
# The hosted contract, as observed on the wire.
# ---------------------------------------------------------------------------

#: Fields the generic ``POST /api/jobs/submit`` body accepts. Unknown keys are
#: silently dropped by the server; one of the three source fields is required.
GENERIC_SUBMIT_FIELDS = {
    "script_id", "script_slug", "inline_code", "framework", "backend", "params",
    "create_capsule", "capsule_name", "execution_mode", "analysis_id",
    "warn_check_ids", "not_before", "hold_reason",
}
GENERIC_SOURCE_FIELDS = {"script_id", "script_slug", "inline_code"}

#: The versioned managed-native body (``marqov.public-submission/v1``). Closed:
#: unknown keys are rejected with 400. A UUID ``Idempotency-Key`` is required.
V1_REQUIRED_FIELDS = {
    "protocol_version", "team_id", "backend", "programming_model",
    "cap_cents", "analysis_id", "source", "input",
}
V1_OPTIONAL_FIELDS = {"source_sha256"}

#: Endpoints the hosted API serves to an API key. ``POST /api/jobs/{id}/cancel``
#: exists but answers 403 ``api_key_not_supported`` to API keys (session only).
API_KEY_ENDPOINTS = {
    ("POST", "/api/jobs/submit"),
    ("GET", "/api/jobs/{id}/status"),
    ("GET", "/api/backends"),
    ("GET", "/api/jobs/managed-runtimes"),
}

JOB = "11111111-1111-4111-8111-111111111111"


def _capturing_client(response_body: dict | None = None, status: int = 200):
    """A real client whose HTTP session records requests instead of sending."""
    client = MarqovClient(api_key="marqey_test_" + "x" * 32, base_url="http://test.invalid")
    sent: list[dict] = []

    def fake_request(method, url, **kwargs):
        sent.append({
            "method": method,
            "path": url.removeprefix("http://test.invalid"),
            "headers": {**client._transport._session.headers, **(kwargs.get("headers") or {})},
            "json": kwargs.get("json"),
            "params": kwargs.get("params"),
        })
        resp = MagicMock()
        resp.status_code = status
        resp.ok = 200 <= status < 300
        resp.headers = {}
        body = response_body if response_body is not None else {
            "job_id": JOB, "id": JOB, "status": "completed", "result": {}, "backends": [],
        }
        resp.json.return_value = body
        resp.text = json.dumps(body)
        return resp

    client._transport._session.request = fake_request
    return client, sent


def _template(path: str) -> str:
    return path.replace(JOB, "{id}")


# ---------------------------------------------------------------------------
# Compatible today (must keep passing).
# ---------------------------------------------------------------------------


def test_string_submit_body_uses_only_generic_contract_fields():
    client, sent = _capturing_client()
    client.submit("from marqov import task\n", backend="marqov-sim", framework="marqov")
    (req,) = sent
    assert (req["method"], req["path"]) == ("POST", "/api/jobs/submit")
    body = req["json"]
    # sdk_version is the only extra key, and the server drops it without error.
    assert set(body) - {"sdk_version"} <= GENERIC_SUBMIT_FIELDS
    assert set(body) & GENERIC_SOURCE_FIELDS == {"inline_code"}
    assert isinstance(body["params"]["shots"], int)


def test_writes_carry_a_uuid_idempotency_key_and_bearer_auth():
    client, sent = _capturing_client()
    client.submit("x = 1\n", backend="marqov-sim", framework="marqov")
    headers = sent[0]["headers"]
    assert headers["Authorization"].startswith("Bearer marqey_")
    uuid.UUID(headers["Idempotency-Key"])  # the versioned contract requires a UUID


@pytest.mark.parametrize("call, expected", [
    (lambda c: c.job(JOB).status(), ("GET", "/api/jobs/{id}/status")),
    (lambda c: c.backends(), ("GET", "/api/backends")),
])
def test_read_calls_target_served_endpoints(call, expected):
    client, sent = _capturing_client()
    call(client)
    assert (sent[0]["method"], _template(sent[0]["path"])) == expected
    assert expected in API_KEY_ENDPOINTS


def test_result_exposes_the_managed_native_result_projection_unchanged():
    """A completed native workflow's ``result`` is a closed projection
    (``marqov.managed-result/v1``). The client must pass it through intact so
    callers can read each output's JSON ``display.value``."""
    managed = {
        "protocol_version": "marqov.managed-result/v1",
        "outputs": [{
            "task_key": "task_0001", "port": "result",
            "artifact": {"artifact_id": JOB, "team_id": JOB, "schema": "python-value/v1",
                         "sha256": "0" * 64, "size_bytes": 64},
            "display": {"protocol_version": "marqov.task-display/v1", "result_sha256": "0" * 64,
                        "state": "available", "value": {"total": 24}},
        }],
    }
    client, _ = _capturing_client({"id": JOB, "status": "completed", "result": managed})
    result = Job(client._transport, JOB).result(timeout=1)
    assert result.raw == managed
    assert result.counts is None  # a workflow result is not a counts histogram


# ---------------------------------------------------------------------------
# Known mismatches (reproduced; fail today).
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="Circuit submit sends a `circuit` field the hosted "
                   "contract does not accept and no source field, so the server returns 400")
def test_circuit_submit_body_carries_an_accepted_source_field():
    pytest.importorskip("qiskit")
    from marqov import Circuit

    client, sent = _capturing_client()
    client.submit(Circuit().h(0).cnot(0, 1), backend="marqov-sim")
    body = sent[0]["json"]
    assert set(body) - {"sdk_version"} <= GENERIC_SUBMIT_FIELDS
    assert set(body) & GENERIC_SOURCE_FIELDS


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="POST /api/jobs/{id}/cancel is session-only; an API "
                   "key receives 403 api_key_not_supported")
def test_cancel_targets_an_endpoint_served_to_api_keys():
    client, sent = _capturing_client()
    client.job(JOB).cancel()
    assert (sent[0]["method"], _template(sent[0]["path"])) in API_KEY_ENDPOINTS


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="the hosted API has no /api/meta endpoint")
def test_platform_info_targets_a_served_endpoint():
    client, sent = _capturing_client({"api_version": "x"})
    client.platform_info()
    assert (sent[0]["method"], sent[0]["path"]) in API_KEY_ENDPOINTS


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="no public client method produces the versioned "
                   "managed-native body; string submit uses the generic body, which the "
                   "hosted API refuses for native @task/@workflow source (422 execution_unavailable)")
def test_some_public_submit_method_produces_the_versioned_native_body():
    client, sent = _capturing_client()
    source = "from marqov import task, workflow\n@task\ndef a():\n    return 1\n@workflow\ndef run():\n    return a()\n"
    client.submit(source, backend="marqov-sim", framework="marqov")
    body = sent[0]["json"]
    assert body.get("protocol_version") == "marqov.public-submission/v1"
    assert V1_REQUIRED_FIELDS <= set(body) <= V1_REQUIRED_FIELDS | V1_OPTIONAL_FIELDS
