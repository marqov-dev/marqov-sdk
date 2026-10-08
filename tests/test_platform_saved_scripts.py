"""Saved-script contracts at the HTTP boundary; no live or paid calls."""

import hashlib
from unittest.mock import MagicMock
from uuid import UUID

import pytest
import requests

from marqov.platform import MarqovClient, TransportError
from marqov.platform.job import Job

SCRIPT = "0d470ec5-2fe4-4d17-9489-cf20cf7bf869"
ANALYSIS = "0f322402-620c-4110-a983-a9fdb79f3d27"
JOB = "f55c1cb9-fa7b-457b-a795-77713d2e78a0"
OPERATION = "86e4bd2e-df75-4ef6-b55c-5637703ec6e4"
SOURCE = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\n'


def client_with_responses(*bodies):
    client = MarqovClient(api_key="fixture", base_url="https://test.invalid")
    calls = []
    remaining = iter(bodies)

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        response = MagicMock()
        response.ok = True
        response.status_code = 200
        response.json.return_value = next(remaining)
        return response

    client._transport._session.request = request
    return client, calls


def test_saved_sequence_uses_observed_wire_fields_and_explicit_operations():
    analysis = {
        "analysis_id": ANALYSIS,
        "content_hash": "55c29d3e",
        "can_run": True,
        "checks": [],
        "cost": {"total_usd": 0},
    }
    client, calls = client_with_responses(
        {"script_id": SCRIPT}, {"source_sha256": "snapshot"}, analysis, {"job_id": JOB}
    )
    saved = client.save_script(SOURCE, name="QPE", idempotency_key=OPERATION)
    options = client.script_execution_options(saved["script_id"], shots=4096)
    observed = client.analyse_script(
        SCRIPT, content=SOURCE, backend="local", shots=4096, circuit_count=1
    )
    job = client.submit_script(
        SCRIPT,
        backend="local",
        analysis_id=observed["analysis_id"],
        shots=4096,
        circuit_count=1,
        idempotency_key=OPERATION,
    )
    assert isinstance(job, Job) and job.id == JOB
    assert observed is analysis and options["source_sha256"] == "snapshot"
    assert [url.removeprefix("https://test.invalid") for _, url, _ in calls] == [
        "/api/scripts/upload",
        f"/api/scripts/{SCRIPT}/execution-options",
        f"/api/scripts/{SCRIPT}/analyse",
        "/api/jobs/submit",
    ]
    assert calls[0][2]["json"] == {
        "name": "QPE",
        "description": "",
        "script_content": SOURCE,
        "script_type": "script",
    }
    assert calls[1][0] == "GET" and calls[1][2]["params"] == {"shots": 4096}
    assert calls[2][2]["json"]["content_hash"] == hashlib.sha256(SOURCE.encode()).hexdigest()
    assert calls[3][2]["json"] == {
        "script_id": SCRIPT,
        "backend": "local",
        "analysis_id": ANALYSIS,
        "params": {"shots": 4096, "circuit_count": 1},
        "warn_check_ids": [],
    }
    assert calls[0][2]["headers"]["Idempotency-Key"] == OPERATION
    assert calls[3][2]["headers"]["Idempotency-Key"] == OPERATION
    UUID(calls[2][2]["headers"]["Idempotency-Key"])
    assert not calls[1][2]["headers"]


def test_ambiguous_submit_is_not_retried_and_retains_operation_key():
    client = MarqovClient(api_key="fixture")
    request = MagicMock(side_effect=requests.exceptions.ReadTimeout("fixture"))
    client._transport._session.request = request
    with pytest.raises(TransportError) as failure:
        client.submit_script(
            SCRIPT, backend="local", analysis_id=ANALYSIS, idempotency_key=OPERATION
        )
    assert request.call_count == 1
    assert failure.value.idempotency_key == OPERATION


@pytest.mark.parametrize(
    "action",
    [
        lambda c: c.save_script("", name="QPE"),
        lambda c: c.save_script(SOURCE, name=""),
        lambda c: c.save_script(SOURCE, name="QPE", script_type="other"),
        lambda c: c.script_execution_options("../jobs", shots=1),
        lambda c: c.script_execution_options(SCRIPT, shots=True),
        lambda c: c.analyse_script(SCRIPT, content=SOURCE, backend="local", circuit_count=0),
        lambda c: c.analyse_script(SCRIPT, content="", backend="local"),
        lambda c: c.submit_script(SCRIPT, backend="local", analysis_id="preview-uuid"),
        lambda c: c.submit_script(SCRIPT, backend="local", analysis_id=ANALYSIS, shots=0),
        lambda c: c.submit_script(
            SCRIPT, backend="local", analysis_id=ANALYSIS, warn_check_ids="warning"
        ),
        lambda c: c.submit_script(
            SCRIPT, backend="local", analysis_id=ANALYSIS, warn_check_ids=[None]
        ),
        lambda c: c.submit_script(
            SCRIPT, backend="local", analysis_id=ANALYSIS, idempotency_key=""
        ),
    ],
)
def test_invalid_input_never_reaches_http(action):
    client = MarqovClient(api_key="fixture")
    request = MagicMock()
    client._transport._session.request = request
    with pytest.raises(ValueError):
        action(client)
    request.assert_not_called()


def test_warnings_require_explicit_caller_selection():
    client, calls = client_with_responses({"job_id": JOB})
    warnings = ["circuit_count"]
    client.submit_script(SCRIPT, backend="local", analysis_id=ANALYSIS, warn_check_ids=warnings)
    assert calls[0][2]["json"]["warn_check_ids"] == warnings
    assert calls[0][2]["json"]["warn_check_ids"] is not warnings


@pytest.mark.parametrize("response", [{}, {"job_id": "not-a-uuid"}, None])
def test_unreadable_submit_receipt_holds_with_effective_operation_key(response):
    client, calls = client_with_responses(response)
    with pytest.raises(TransportError) as failure:
        client.submit_script(SCRIPT, backend="local", analysis_id=ANALYSIS)
    assert len(calls) == 1
    assert failure.value.code == "submission_outcome_unknown"
    assert failure.value.idempotency_key == calls[0][2]["headers"]["Idempotency-Key"]
    UUID(failure.value.idempotency_key)
