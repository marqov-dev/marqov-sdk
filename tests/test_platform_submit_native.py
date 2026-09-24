"""Tests for MarqovClient.managed_runtimes() and MarqovClient.submit_native().

The client's HTTP session is replaced by a small fake server that records each
request and answers like the hosted API: discovery returns the fixture runtimes,
and admission returns a receipt whose hashes are computed from the body it
received. No network.

The golden bodies in ``tests/fixtures/public_submission_v1`` are the exact wire
bodies this client produces. The same files are checked against the hosted
API's own request handler, so the two sides cannot drift silently.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from marqov.platform import MarqovClient
from marqov.platform.errors import MarqovPlatformError
from marqov.platform.job import Job

FIXTURES = Path(__file__).parent / "fixtures" / "public_submission_v1"
SAVED = json.loads((FIXTURES / "saved_script.json").read_text())
INLINE = json.loads((FIXTURES / "inline_source.json").read_text())
LARGE_INT = json.loads((FIXTURES / "saved_script_large_integer.json").read_text())
SCRIPT = json.loads((FIXTURES / "script_source.json").read_text())
RUNTIMES = json.loads((FIXTURES / "runtimes.json").read_text())

TEAM = SAVED["team_id"]
SOURCE = SCRIPT["content"]
SOURCE_SHA = hashlib.sha256(SOURCE.encode()).hexdigest()
JOB = "5d0a8b7e-3c2f-4e19-8a61-7b4c9e2d1f08"
KEY = "9a2b3c4d-5e6f-4a7b-8c9d-0e1f2a3b4c5d"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class FakeHostedApi:
    """Answers discovery and admission like the hosted API; records requests."""

    def __init__(self, runtimes=RUNTIMES, submit_status=201, submit_body=None, script_content=SOURCE):
        self.runtimes, self.submit_status, self.submit_body = runtimes, submit_status, submit_body
        self.script_content = script_content
        self.sent: list[dict] = []

    def __call__(self, method, url, **kwargs):
        path = url.removeprefix("http://test.invalid")
        self.sent.append({"method": method, "path": path, "headers": kwargs.get("headers") or {},
                          "json": kwargs.get("json"), "params": kwargs.get("params")})
        if path == "/api/jobs/managed-runtimes":
            return self._resp(200, self.runtimes)
        if path == "/api/jobs/submit":
            if self.submit_body is not None:
                return self._resp(self.submit_status, self.submit_body)
            body = kwargs["json"]
            source = body["source"].get("content", self.script_content)
            return self._resp(201, {
                "protocol_version": "marqov.funded-admission-receipt/v1",
                "job_id": JOB, "submission_id": JOB,
                "source_sha256": _sha(source), "input_sha256": _sha(body["input"]),
            })
        if path == f"/api/jobs/{JOB}/status":
            return self._resp(200, {"id": JOB, "status": "completed", "result": {"outputs": []}})
        raise AssertionError(f"unexpected request {method} {path}")

    @staticmethod
    def _resp(status, body):
        resp = MagicMock()
        resp.status_code, resp.ok, resp.headers = status, 200 <= status < 300, {}
        resp.json.return_value = body
        resp.text = json.dumps(body)
        return resp

    def posts(self):
        return [r for r in self.sent if r["method"] == "POST"]


def _client(api: FakeHostedApi | None = None):
    api = api or FakeHostedApi()
    client = MarqovClient(api_key="marqey_test_" + "x" * 32, base_url="http://test.invalid")
    client._transport._session.request = api
    return client, api


BASE = dict(team_id=TEAM, entrypoint="native_canary", cap_cents=100, kwargs={"seed": 7})


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_managed_runtimes_reads_the_discovery_endpoint():
    client, api = _client()
    assert client.managed_runtimes(TEAM) == RUNTIMES["runtimes"]
    (req,) = api.sent
    assert (req["method"], req["path"], req["params"]) == ("GET", "/api/jobs/managed-runtimes", {"team_id": TEAM})
    assert "Idempotency-Key" not in req["headers"]


def test_managed_runtimes_rejects_a_bad_team_id_without_a_request():
    client, api = _client()
    with pytest.raises(ValueError):
        client.managed_runtimes("not-a-team")
    assert api.sent == []


@pytest.mark.parametrize("response", [
    {},
    {"runtimes": None},
    {"runtimes": [{"backend": "marqov-sim"}]},
    {"runtimes": [{**RUNTIMES["runtimes"][0], "extra": 1}]},
    {"runtimes": [{**RUNTIMES["runtimes"][0], "min_cap_cents": 50, "max_cap_cents": 10}]},
])
def test_managed_runtimes_fails_closed_on_an_unexpected_shape(response):
    client, _ = _client(FakeHostedApi(runtimes=response))
    with pytest.raises(MarqovPlatformError) as info:
        client.managed_runtimes(TEAM)
    assert info.value.code == "invalid_response"


# ---------------------------------------------------------------------------
# Wire bodies: exactly the golden fixtures
# ---------------------------------------------------------------------------


def test_saved_script_submission_sends_the_golden_body_after_discovery():
    client, api = _client()
    job = client.submit_native(script_id=SAVED["source"]["script_id"], source_sha256=SOURCE_SHA, **BASE)
    assert [(r["method"], r["path"]) for r in api.sent] == [
        ("GET", "/api/jobs/managed-runtimes"), ("POST", "/api/jobs/submit")]
    post = api.posts()[0]
    assert post["json"] == {**SAVED}
    assert "sdk_version" not in post["json"]  # the versioned body is closed
    uuid.UUID(post["headers"]["Idempotency-Key"])
    assert isinstance(job, Job) and job.id == JOB


def test_inline_source_submission_sends_the_golden_body():
    client, api = _client()
    client.submit_native(source=SOURCE, **BASE)
    assert api.posts()[0]["json"] == INLINE


def test_large_integers_are_sent_exactly():
    client, api = _client()
    client.submit_native(team_id=TEAM, entrypoint="native_canary", cap_cents=100,
                         script_id=SAVED["source"]["script_id"], args=[2**64 + 1, 1.5])
    assert api.posts()[0]["json"] == LARGE_INT
    assert json.loads(LARGE_INT["input"])["args"][0] == 2**64 + 1


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_caller_idempotency_key_is_sent_and_reused_across_transport_retries(monkeypatch):
    import requests

    api = FakeHostedApi()
    calls = {"n": 0}

    def flaky(method, url, **kwargs):
        if method == "POST":
            calls["n"] += 1
            if calls["n"] == 1:
                raise requests.exceptions.ConnectionError("refused")
        return api(method, url, **kwargs)

    client, _ = _client(api)
    client._transport._session.request = flaky
    monkeypatch.setattr("marqov.platform._transport.time.sleep", lambda _s: None)
    client.submit_native(script_id=SAVED["source"]["script_id"], idempotency_key=KEY, **BASE)
    assert calls["n"] == 2
    assert [p["headers"]["Idempotency-Key"] for p in api.posts()] == [KEY]  # only the retry reached the fake
    assert api.posts()[0]["headers"]["Idempotency-Key"] == KEY


def test_a_fresh_key_is_generated_per_submission_when_omitted():
    client, api = _client()
    client.submit_native(script_id=SAVED["source"]["script_id"], **BASE)
    client.submit_native(script_id=SAVED["source"]["script_id"], **BASE)
    keys = [p["headers"]["Idempotency-Key"] for p in api.posts()]
    assert len(set(keys)) == 2 and all(uuid.UUID(k) for k in keys)


# ---------------------------------------------------------------------------
# Local refusals: nothing is sent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("override", [
    {"script_id": SAVED["source"]["script_id"], "source": SOURCE},   # both sources
    {},                                                              # no source
    {"script_id": "7b1e9c40-2d3f-0a51-9e8c-5f6a7b8c9d01"},           # not an RFC UUID
    {"script_id": "{7b1e9c40-2d3f-4a51-9e8c-5f6a7b8c9d01}"},         # braces refused
    {"team_id": "team-1", "script_id": SAVED["source"]["script_id"]},
    {"entrypoint": "1bad", "script_id": SAVED["source"]["script_id"]},
    {"entrypoint": "a" * 129, "script_id": SAVED["source"]["script_id"]},
    {"cap_cents": 0, "script_id": SAVED["source"]["script_id"]},
    {"cap_cents": 10_001, "script_id": SAVED["source"]["script_id"]},
    {"cap_cents": 1.5, "script_id": SAVED["source"]["script_id"]},
    {"cap_cents": True, "script_id": SAVED["source"]["script_id"]},
    {"programming_model": "graph", "script_id": SAVED["source"]["script_id"]},
    {"source_sha256": "ABC", "script_id": SAVED["source"]["script_id"]},
    {"idempotency_key": "retry-1", "script_id": SAVED["source"]["script_id"]},
    {"source": ""},
    {"source": "x = 1\x00"},
    {"source": "\ud800"},
    {"source": "x" * (1024 * 1024 + 1)},
    {"args": [float("nan")], "script_id": SAVED["source"]["script_id"]},
    {"args": [float("inf")], "script_id": SAVED["source"]["script_id"]},
    {"args": [10 ** 4300], "script_id": SAVED["source"]["script_id"]},
    {"args": [object()], "script_id": SAVED["source"]["script_id"]},
    {"args": "abc", "script_id": SAVED["source"]["script_id"]},
    {"kwargs": {1: "int key would be renamed to '1'"}, "script_id": SAVED["source"]["script_id"]},
    {"kwargs": {"nested": {None: 1}}, "script_id": SAVED["source"]["script_id"]},
    {"kwargs": {"deep": [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[1]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]},
     "script_id": SAVED["source"]["script_id"]},
])
def test_invalid_requests_are_refused_before_any_request(override):
    client, api = _client()
    with pytest.raises((ValueError, TypeError)):
        client.submit_native(**{**BASE, **override})
    assert api.sent == []


def test_cap_cents_and_team_id_have_no_defaults():
    client, api = _client()
    with pytest.raises(TypeError):
        client.submit_native(team_id=TEAM, entrypoint="run", script_id=SAVED["source"]["script_id"])  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        client.submit_native(entrypoint="run", cap_cents=100, script_id=SAVED["source"]["script_id"])  # type: ignore[call-arg]
    assert api.sent == []


# ---------------------------------------------------------------------------
# Authoritative discovery: no substitution
# ---------------------------------------------------------------------------


def test_refuses_when_the_requested_runtime_is_not_enabled_and_never_submits():
    client, api = _client(FakeHostedApi(runtimes={"runtimes": []}))
    with pytest.raises(MarqovPlatformError) as info:
        client.submit_native(script_id=SAVED["source"]["script_id"], **BASE)
    assert info.value.code == "runtime_not_enabled"
    assert api.posts() == []


def test_does_not_substitute_another_programming_model():
    single = {"runtimes": [{**RUNTIMES["runtimes"][0], "programming_model": "single_task"}]}
    client, api = _client(FakeHostedApi(runtimes=single))
    with pytest.raises(MarqovPlatformError) as info:
        client.submit_native(script_id=SAVED["source"]["script_id"], **BASE)
    assert info.value.code == "runtime_not_enabled"
    assert "single_task" in str(info.value) and api.posts() == []


def test_does_not_substitute_another_backend():
    client, api = _client()
    with pytest.raises(MarqovPlatformError):
        client.submit_native(script_id=SAVED["source"]["script_id"], backend="sv1", **BASE)
    assert api.posts() == []


@pytest.mark.parametrize("cap", [10, 10_000 + 0])
def test_cap_must_sit_inside_the_discovered_runtime_range(cap):
    narrow = {"runtimes": [{**RUNTIMES["runtimes"][0], "min_cap_cents": 11, "max_cap_cents": 9_999}]}
    client, api = _client(FakeHostedApi(runtimes=narrow))
    with pytest.raises(ValueError, match="between 11 and 9999"):
        client.submit_native(**{**BASE, "cap_cents": cap, "script_id": SAVED["source"]["script_id"]})
    assert api.posts() == []


# ---------------------------------------------------------------------------
# Receipt and refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("receipt, code", [
    ({"job_id": JOB}, "invalid_receipt"),
    ({"protocol_version": "other", "job_id": JOB, "submission_id": JOB,
      "source_sha256": "a" * 64, "input_sha256": "b" * 64}, "invalid_receipt"),
    ({"protocol_version": "marqov.funded-admission-receipt/v1", "job_id": JOB, "submission_id": JOB,
      "source_sha256": "a" * 64, "input_sha256": "b" * 64}, "receipt_mismatch"),
])
def test_an_unconfirmed_receipt_is_not_treated_as_admission(receipt, code):
    client, _ = _client(FakeHostedApi(submit_status=201, submit_body=receipt))
    with pytest.raises(MarqovPlatformError) as info:
        client.submit_native(source=SOURCE, **BASE)
    assert info.value.code == code


@pytest.mark.parametrize("status, code", [
    (402, "spend_limit_exceeded"),
    (409, "idempotency_conflict"),
    (503, "managed_execution_unavailable"),
])
def test_server_refusals_surface_their_code(status, code):
    body = {"error": {"code": code, "message": "m", "status": status}}
    client, _ = _client(FakeHostedApi(submit_status=status, submit_body=body))
    with pytest.raises(MarqovPlatformError) as info:
        client.submit_native(script_id=SAVED["source"]["script_id"], **BASE)
    assert info.value.code == code


# ---------------------------------------------------------------------------
# Polling and results reuse the existing Job
# ---------------------------------------------------------------------------


def test_returned_job_polls_and_reads_through_the_existing_status_endpoint():
    client, api = _client()
    job = client.submit_native(script_id=SAVED["source"]["script_id"], **BASE)
    assert job.result(timeout=1).raw == {"outputs": []}
    assert api.sent[-1]["path"] == f"/api/jobs/{JOB}/status"


def test_legacy_submit_is_unchanged():
    client, api = _client()
    api.submit_body, api.submit_status = {"job_id": JOB}, 200
    client.submit("x = 1\n", backend="marqov-sim", framework="marqov")
    assert "protocol_version" not in api.posts()[0]["json"]
    assert [r["path"] for r in api.sent] == ["/api/jobs/submit"]  # no discovery call


def test_nesting_depth_boundary_matches_the_shared_fixture():
    """The server applies the same boundary; the fixture is checked on both sides."""
    fixture = json.loads((FIXTURES / "depth_limit.json").read_text())
    depth = fixture["kwargs_x_list_nesting_accepted"]

    def nest(n):
        value = 1
        for _ in range(n):
            value = [value]
        return value

    client, api = _client()
    common = dict(team_id=TEAM, entrypoint="run", cap_cents=100, script_id=SAVED["source"]["script_id"])
    client.submit_native(kwargs={"x": nest(depth)}, **common)
    assert api.posts()[0]["json"] == fixture["body"]
    with pytest.raises(ValueError):
        client.submit_native(kwargs={"x": nest(depth + 1)}, **common)
    assert len(api.posts()) == 1


def test_the_documented_example_is_the_example_file_verbatim():
    root = Path(__file__).resolve().parent.parent
    example = (root / "examples" / "platform_native_workflow.py").read_text()
    guide = (root / "docs" / "platform-client" / "native-workflows.md").read_text()
    marker = "<!-- example: examples/platform_native_workflow.py (checked verbatim by tests) -->\n```python\n"
    start = guide.index(marker) + len(marker)
    assert guide[start:guide.index("```", start)] == example
    compile(example, "platform_native_workflow.py", "exec")


def test_the_example_runs_discover_submit_wait_read_against_the_fake_api(monkeypatch, capsys):
    import runpy

    managed = {
        "protocol_version": "marqov.managed-result/v1",
        "outputs": [{
            "task_key": "task_0001", "port": "result",
            "artifact": {"artifact_id": JOB, "team_id": TEAM, "schema": "python-value/v1",
                         "sha256": "0" * 64, "size_bytes": 64},
            "display": {"protocol_version": "marqov.task-display/v1", "result_sha256": "0" * 64,
                        "state": "available", "value": {"schema_version": 1, "total": 24}},
        }],
    }
    api = FakeHostedApi()
    real_call = api.__call__

    def serve(method, url, **kwargs):
        if url.endswith(f"/api/jobs/{JOB}/status"):
            api.sent.append({"method": method, "path": url, "headers": {}, "json": None, "params": None})
            return FakeHostedApi._resp(200, {"id": JOB, "status": "completed", "result": managed})
        return real_call(method, url, **kwargs)

    import marqov.platform as platform_pkg

    real_init = platform_pkg.MarqovClient.__init__

    def init(self, *a, **k):
        real_init(self, api_key="marqey_test_" + "x" * 32, base_url="http://test.invalid")
        self._transport._session.request = serve

    monkeypatch.setattr(platform_pkg.MarqovClient, "__init__", init)
    monkeypatch.setenv("MARQOV_TEAM_ID", TEAM)
    monkeypatch.setenv("MARQOV_SCRIPT_ID", SAVED["source"]["script_id"])
    root = Path(__file__).resolve().parent.parent
    runpy.run_path(str(root / "examples" / "platform_native_workflow.py"), run_name="__main__")

    out = capsys.readouterr().out
    assert "marqov-sim: cap between 11 and 10000 cents" in out
    assert f"Admitted job {JOB}" in out
    assert "task_0001 {'schema_version': 1, 'total': 24}" in out
    post = api.posts()[0]
    assert post["json"]["source"] == {"kind": "script", "script_id": SAVED["source"]["script_id"]}
    assert post["json"]["cap_cents"] == 100


# ---------------------------------------------------------------------------
# Ambiguous outcomes: the effective idempotency key is always recoverable
# ---------------------------------------------------------------------------

API_KEY = "marqey_test_" + "x" * 32


def _recording(handler):
    """A session stub that records every POST's key and delegates to *handler*."""
    api = FakeHostedApi()
    keys: list[str] = []

    def request(method, url, **kwargs):
        if method == "POST":
            keys.append((kwargs.get("headers") or {}).get("Idempotency-Key"))
            return handler(api, method, url, **kwargs)
        return api(method, url, **kwargs)

    client = MarqovClient(api_key=API_KEY, base_url="http://test.invalid")
    client._transport._session.request = request
    return client, keys


def _no_secrets(exc: BaseException) -> None:
    text = str(exc) + repr(exc)
    assert API_KEY not in text
    assert "native_canary(seed" not in text and "def alice" not in text


@pytest.mark.parametrize("caller_key", [None, KEY])
def test_exhausted_connection_retries_expose_the_key_every_attempt_used(monkeypatch, caller_key):
    import requests
    from marqov.platform.errors import TransportError

    def refuse(api, method, url, **kwargs):
        raise requests.exceptions.ConnectionError("connection reset")

    monkeypatch.setattr("marqov.platform._transport.time.sleep", lambda _s: None)
    client, keys = _recording(refuse)
    with pytest.raises(TransportError) as info:
        client.submit_native(source=SOURCE, idempotency_key=caller_key, **BASE)
    assert len(keys) > 1 and len(set(keys)) == 1          # one key, reused on every retry
    assert info.value.idempotency_key == keys[0]
    if caller_key is not None:
        assert keys[0] == caller_key
    uuid.UUID(info.value.idempotency_key)
    assert info.value.idempotency_key in str(info.value)
    _no_secrets(info.value)


def test_write_timeout_exposes_the_key():
    import requests
    from marqov.platform.errors import TransportError

    def slow(api, method, url, **kwargs):
        raise requests.exceptions.ReadTimeout("read timed out")

    client, keys = _recording(slow)
    with pytest.raises(TransportError) as info:
        client.submit_native(source=SOURCE, **BASE)
    assert keys and info.value.idempotency_key == keys[0]
    _no_secrets(info.value)


def test_server_error_exposes_the_key():
    def unavailable(api, method, url, **kwargs):
        return FakeHostedApi._resp(503, {"error": {"code": "managed_execution_unavailable",
                                                   "message": "retry", "status": 503}})

    client, keys = _recording(unavailable)
    with pytest.raises(MarqovPlatformError) as info:
        client.submit_native(source=SOURCE, **BASE)
    assert info.value.code == "managed_execution_unavailable"
    assert info.value.idempotency_key == keys[0]
    _no_secrets(info.value)


def test_unreadable_success_response_is_reported_as_unknown_outcome_with_the_key():
    from marqov.platform.errors import TransportError

    def html(api, method, url, **kwargs):
        resp = FakeHostedApi._resp(201, {})
        resp.json.side_effect = ValueError("Expecting value: line 1 column 1")
        return resp

    client, keys = _recording(html)
    with pytest.raises(TransportError) as info:
        client.submit_native(source=SOURCE, **BASE)
    assert info.value.code == "submission_outcome_unknown"
    assert info.value.idempotency_key == keys[0]
    assert isinstance(info.value.__cause__, ValueError)
    _no_secrets(info.value)


@pytest.mark.parametrize("receipt, code", [
    ({"job_id": JOB}, "invalid_receipt"),
    ({"protocol_version": "marqov.funded-admission-receipt/v1", "job_id": JOB, "submission_id": JOB,
      "source_sha256": "a" * 64, "input_sha256": "b" * 64}, "receipt_mismatch"),
])
def test_receipt_validation_failures_expose_the_key(receipt, code):
    def reply(api, method, url, **kwargs):
        return FakeHostedApi._resp(201, receipt)

    client, keys = _recording(reply)
    with pytest.raises(MarqovPlatformError) as info:
        client.submit_native(source=SOURCE, **BASE)
    assert info.value.code == code
    assert info.value.idempotency_key == keys[0]
    _no_secrets(info.value)


def test_failures_before_sending_carry_no_key():
    client, keys = _recording(lambda *a, **k: pytest.fail("must not send"))
    with pytest.raises(ValueError):
        client.submit_native(source=SOURCE, **{**BASE, "cap_cents": 0})
    empty = FakeHostedApi(runtimes={"runtimes": []})
    client2, _ = _client(empty)
    with pytest.raises(MarqovPlatformError) as info:
        client2.submit_native(source=SOURCE, **BASE)
    assert info.value.code == "runtime_not_enabled" and info.value.idempotency_key is None
    assert keys == []


def test_documented_recovery_resubmits_with_the_same_key(monkeypatch):
    import requests

    monkeypatch.setattr("marqov.platform._transport.time.sleep", lambda _s: None)
    state = {"down": True}

    def flaky(api, method, url, **kwargs):
        if state["down"]:
            raise requests.exceptions.ConnectionError("connection reset")
        return api(method, url, **kwargs)

    client, keys = _recording(flaky)
    request = dict(source=SOURCE, **BASE)
    with pytest.raises(MarqovPlatformError) as info:
        client.submit_native(**request)
    state["down"] = False
    job = client.submit_native(**request, idempotency_key=info.value.idempotency_key)
    assert job.id == JOB
    assert set(keys) == {info.value.idempotency_key}


def test_saved_script_without_source_sha256_cannot_verify_the_source_content():
    """Documented limit: only structure and input hash are checked, so a receipt
    reporting any well-formed source hash is accepted for a bare script_id."""
    def reply(api, method, url, **kwargs):
        body = kwargs["json"]
        return FakeHostedApi._resp(201, {
            "protocol_version": "marqov.funded-admission-receipt/v1", "job_id": JOB,
            "submission_id": JOB, "source_sha256": "f" * 64, "input_sha256": _sha(body["input"])})

    client, _ = _recording(reply)
    assert client.submit_native(script_id=SAVED["source"]["script_id"], **BASE).id == JOB
    with pytest.raises(MarqovPlatformError) as info:   # with source_sha256 the same receipt is refused
        client.submit_native(script_id=SAVED["source"]["script_id"], source_sha256=SOURCE_SHA, **BASE)
    assert info.value.code == "receipt_mismatch"
