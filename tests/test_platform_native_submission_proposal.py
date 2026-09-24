"""PROPOSAL — failing tests for submitting a saved native workflow through the
hosted versioned contract (``marqov.public-submission/v1``).

Nothing here is implemented. These tests pin the proposed public surface so it
can be reviewed before any code is written:

    client.managed_runtimes(team_id) -> list[dict]
    client.submit_native(*, team_id, entrypoint, cap_cents,
                         script_id=None | source=None,
                         args=(), kwargs=None,
                         programming_model="native_workflow",
                         source_sha256=None, idempotency_key=None) -> Job

Additive only: ``submit()``, ``Job``, ``PlatformResult``, workflow capture and
executors are unchanged. The returned ``Job`` is the existing class, so
``status()`` / ``result()`` polling is reused as-is.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock

import pytest

from marqov.platform import MarqovClient
from marqov.platform.errors import MarqovPlatformError
from marqov.platform.job import Job

TEAM = "0f4c2d7e-5b1a-4c3e-9a8d-2e6f1b7c9d30"
SCRIPT = "7b1e9c40-2d3f-4a51-9e8c-5f6a7b8c9d01"
JOB = "5d0a8b7e-3c2f-4e19-8a61-7b4c9e2d1f08"
KEY = "9a2b3c4d-5e6f-4a7b-8c9d-0e1f2a3b4c5d"
RECEIPT = {
    "protocol_version": "marqov.funded-admission-receipt/v1",
    "job_id": JOB, "submission_id": JOB, "source_sha256": "a" * 64, "input_sha256": "b" * 64,
}
SOURCE = (
    "from marqov import task, workflow\n\n\n@task(timeout=60)\ndef alice(seed: int):\n"
    "    values = [seed, seed + 1, seed + 2]\n"
    "    return {\"stage\": \"alice\", \"seed\": seed, \"values\": values, \"total\": sum(values)}\n\n\n"
    "@workflow\ndef native_canary(seed: int = 7):\n"
    "    return {\"schema_version\": 1, \"alice\": alice(seed)}\n"
)

#: The exact body the proposal must send for a saved script. This is the
#: documented hosted contract; it is checked against the server separately.
GOLDEN_SAVED_SCRIPT_BODY = {
    "protocol_version": "marqov.public-submission/v1",
    "team_id": TEAM,
    "backend": "marqov-sim",
    "programming_model": "native_workflow",
    "cap_cents": 100,
    "analysis_id": None,
    "source": {"kind": "script", "script_id": SCRIPT},
    "input": "{\"entrypoint\": \"native_canary\", \"args\": [], \"kwargs\": {\"seed\": 7}}",
}


def _client(response=RECEIPT, status=201):
    client = MarqovClient(api_key="marqey_test_" + "x" * 32, base_url="http://test.invalid")
    sent: list[dict] = []

    def fake_request(method, url, **kwargs):
        sent.append({"method": method, "path": url.removeprefix("http://test.invalid"),
                     "headers": kwargs.get("headers") or {}, "json": kwargs.get("json"),
                     "params": kwargs.get("params")})
        resp = MagicMock()
        resp.status_code, resp.ok, resp.headers = status, 200 <= status < 300, {}
        resp.json.return_value = response
        resp.text = json.dumps(response)
        return resp

    client._transport._session.request = fake_request
    return client, sent


def test_saved_script_submission_sends_the_exact_versioned_body():
    client, sent = _client()
    job = client.submit_native(team_id=TEAM, script_id=SCRIPT, entrypoint="native_canary",
                               kwargs={"seed": 7}, cap_cents=100)
    (req,) = sent
    assert (req["method"], req["path"]) == ("POST", "/api/jobs/submit")
    assert req["json"] == GOLDEN_SAVED_SCRIPT_BODY
    uuid.UUID(req["headers"]["Idempotency-Key"])
    assert isinstance(job, Job) and job.id == JOB


def test_inline_source_and_optional_source_guard():
    client, sent = _client()
    client.submit_native(team_id=TEAM, source=SOURCE, entrypoint="native_canary",
                         cap_cents=100, source_sha256="c" * 64)
    body = sent[0]["json"]
    assert body["source"] == {"kind": "inline", "content": SOURCE}
    assert body["source_sha256"] == "c" * 64


def test_caller_supplied_idempotency_key_is_sent_verbatim_for_cross_process_retry():
    client, sent = _client()
    client.submit_native(team_id=TEAM, script_id=SCRIPT, entrypoint="native_canary",
                         cap_cents=100, idempotency_key=KEY)
    assert sent[0]["headers"]["Idempotency-Key"] == KEY


def test_input_is_json_text_and_preserves_large_integers_exactly():
    client, sent = _client()
    big = 2**64 + 1
    client.submit_native(team_id=TEAM, script_id=SCRIPT, entrypoint="run",
                         args=[big, 1.5], cap_cents=100)
    text = sent[0]["json"]["input"]
    assert isinstance(text, str) and str(big) in text
    assert json.loads(text) == {"entrypoint": "run", "args": [big, 1.5], "kwargs": {}}


@pytest.mark.parametrize("kwargs", [
    {"script_id": SCRIPT, "source": SOURCE},          # both sources
    {},                                               # no source
    {"script_id": SCRIPT, "entrypoint": "1bad"},      # not an identifier
    {"script_id": SCRIPT, "cap_cents": 0},            # cap must be positive
    {"script_id": SCRIPT, "programming_model": "graph"},
    {"script_id": SCRIPT, "args": [float("nan")]},    # not representable as JSON
])
def test_invalid_requests_are_refused_before_any_network_call(kwargs):
    client, sent = _client()
    base = {"team_id": TEAM, "entrypoint": "run", "cap_cents": 100}
    with pytest.raises((ValueError, TypeError)):
        client.submit_native(**{**base, **kwargs})
    assert sent == []


def test_cap_has_no_default():
    client, _ = _client()
    with pytest.raises(TypeError):
        client.submit_native(team_id=TEAM, script_id=SCRIPT, entrypoint="run")  # type: ignore[call-arg]


def test_malformed_receipt_is_not_treated_as_admission():
    client, _ = _client({"job_id": JOB})
    with pytest.raises(MarqovPlatformError):
        client.submit_native(team_id=TEAM, script_id=SCRIPT, entrypoint="run", cap_cents=100)


@pytest.mark.parametrize("status, code", [
    (402, "spend_limit_exceeded"),
    (503, "managed_execution_unavailable"),
    (409, "idempotency_conflict"),
])
def test_refusals_surface_their_server_code(status, code):
    client, _ = _client({"error": {"code": code, "message": "m", "status": status}}, status=status)
    with pytest.raises(MarqovPlatformError) as info:
        client.submit_native(team_id=TEAM, script_id=SCRIPT, entrypoint="run", cap_cents=100)
    assert info.value.code == code


def test_managed_runtimes_reads_the_advisory_discovery_endpoint():
    runtimes = {"runtimes": [{"backend": "marqov-sim", "programming_model": "native_workflow",
                              "min_cap_cents": 51, "max_cap_cents": 10000}]}
    client, sent = _client(runtimes, status=200)
    assert client.managed_runtimes(TEAM) == runtimes["runtimes"]
    assert (sent[0]["method"], sent[0]["path"]) == ("GET", "/api/jobs/managed-runtimes")
    assert sent[0]["params"] == {"team_id": TEAM}


def test_existing_submit_is_unchanged():
    """Backward compatibility: the generic body is still what submit() sends."""
    client, sent = _client({"job_id": JOB}, status=200)
    client.submit("x = 1\n", backend="marqov-sim", framework="marqov")
    assert "protocol_version" not in sent[0]["json"]
