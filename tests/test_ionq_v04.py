"""IonQ documented QASM3/artifact contract, offline real HTTP response bytes."""

import base64
import hashlib
import json
from unittest.mock import Mock

import pytest
import requests
from qiskit.quantum_info import Statevector

from marqov.circuits import Circuit
from marqov.executors.ionq import IonQExecutor, IonQExecutorConfig

FORMAT = "ionq.result.probabilities.json.v2"


class Transport:
    def __init__(self, distribution, *, job=None, body=None):
        self.body = (
            body
            or json.dumps(
                {"probabilities": {"registers": {"output_all": distribution}}}, indent=2
            ).encode()
        )
        self.job = job or {
            "id": "owned",
            "status": "completed",
            "type": "ionq.qasm3.v1",
            "backend": "simulator",
            "results": {
                FORMAT: {"id": "artifact", "format": FORMAT, "media_type": "application/json"}
            },
        }
        self.calls = []

    def request(self, method, url, **kw):
        self.calls.append((method, url, kw))
        assert url.startswith("https://api.ionq.co/v0.4/")
        assert kw["headers"]["Authorization"] == "apiKey offline"
        if method == "POST":
            assert url.endswith("/jobs")
            data = json.dumps({"id": "owned"}).encode()
        elif url.endswith("/jobs/owned"):
            data = json.dumps(self.job).encode()
        elif url.endswith("/jobs/owned/artifacts/artifact"):
            data = self.body
        elif method == "PUT" and url.endswith("/jobs/owned/status/cancel"):
            data = b"{}"
        else:
            raise AssertionError((method, url))
        response = requests.Response()
        response.status_code = 200
        response._content = data
        return response


def ex(transport, **kw):
    return IonQExecutor(
        IonQExecutorConfig(api_version="0.4", api_key="offline", **kw), session=transport
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("bits", ["00", "10", "01", "11"])
async def test_asymmetric_basis_wire_mapping_and_real_export(bits):
    circuit = Circuit().z(0).z(1)
    for q, bit in enumerate(bits):
        if bit == "1":
            circuit.x(q)
    # Qiskit's independent statevector uses the opposite display order.
    expected = {
        key[::-1]: value
        for key, value in Statevector.from_instruction(circuit.to_qiskit())
        .probabilities_dict()
        .items()
    }
    transport = Transport({bits: 1.0})
    result = await ex(transport).execute(circuit, shots=200)
    assert result.metadata["source_probabilities"] == expected
    assert result.counts == {bits: 200}
    payload = transport.calls[0][2]["json"]
    assert payload["type"] == "ionq.qasm3.v1"
    assert payload["backend"] == "simulator"
    assert "target" not in payload and "format" not in payload["input"]
    assert "OPENQASM 3.0" in payload["input"]["data"]
    assert "c[0] = measure q[0]" in payload["input"]["data"]
    assert "c[1] = measure q[1]" in payload["input"]["data"]
    for q, bit in enumerate(bits):
        assert (f"x q[{q}];" in payload["input"]["data"]) == (bit == "1")


@pytest.mark.asyncio
async def test_original_probabilities_and_bytes_survive_allocation():
    probabilities = {"00": 0.333333333333, "10": 0.333333333333, "01": 0.333333333334}
    transport = Transport(probabilities)
    result = await ex(transport).execute(Circuit().h(0).h(1), shots=2)
    assert sum(result.counts.values()) == 2
    assert result.probabilities != probabilities
    assert result.metadata["source_probabilities"] == probabilities
    assert result.metadata["counts_kind"] == "probability-derived"
    assert result.metadata["raw_shot_eligible"] is False
    artifact = result.metadata["result_artifact"]
    assert base64.b64decode(artifact["body_base64"]) == transport.body
    assert artifact["sha256"] == hashlib.sha256(transport.body).hexdigest()
    assert artifact["job_id"] == result.raw_result["id"] == "owned"
    assert artifact["format"] == FORMAT and artifact["id"] == "artifact"
    json.dumps(result.metadata)  # portable evidence envelope, not bytes objects


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "probabilities",
    [
        {"1": 1.0},
        {"ab": 1.0},
        {"00": -1.0, "11": 2.0},
        {"00": float("nan")},
        {"00": True},
        {"00": 0.9},
        {},
    ],
)
async def test_invalid_artifact_refuses_with_identity_and_original_bytes(probabilities):
    transport = Transport(probabilities)
    with pytest.raises(ValueError) as caught:
        await ex(transport).execute(Circuit().h(0).h(1))
    assert caught.value.remote_job["job_id"] == "owned"
    assert base64.b64decode(caught.value.result_artifact["body_base64"]) == transport.body
    assert sum(method == "POST" for method, _, _ in transport.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [{"id": "other"}, {"backend": "qpu.forte-1"}, {"type": "ionq.circuit.v1"}, {"results": {}}],
)
async def test_wrong_identity_or_format_never_fetches_another_result(change):
    transport = Transport({"00": 1.0})
    transport.job.update(change)
    with pytest.raises(ValueError):
        await ex(transport).execute(Circuit().z(0).z(1))
    assert len(transport.calls) == 2


@pytest.mark.parametrize(
    "kw",
    [
        {"api_version": "0.5"},
        {"api_version": "0.4", "base_url": "https://api.ionq.co/v0.3"},
        {"base_url": "https://api.ionq.co/v0.4"},
        {"api_version": "0.4", "target": "qpu.forte-1"},
        {"api_version": "0.4", "noise_model": "aria-1"},
    ],
)
def test_unqualified_routes_rejected(kw):
    with pytest.raises(ValueError):
        IonQExecutorConfig(**kw)


@pytest.mark.asyncio
async def test_sparse_wires_refused_before_submission():
    transport = Mock()
    with pytest.raises(ValueError, match="dense integer"):
        await ex(transport).execute(Circuit().x(1))
    transport.request.assert_not_called()


@pytest.mark.asyncio
async def test_started_status_and_timeout_cancels_owned_job():
    transport = Transport({"0": 1.0})
    transport.job["status"] = "started"
    with pytest.raises(TimeoutError) as caught:
        await ex(transport, timeout_seconds=0.03, poll_interval_seconds=0.001).execute(
            Circuit().x(0)
        )
    assert caught.value.remote_job["job_id"] == "owned"
    cancels = [(method, url) for method, url, _ in transport.calls if method == "PUT"]
    assert cancels == [("PUT", "https://api.ionq.co/v0.4/jobs/owned/status/cancel")]


@pytest.mark.asyncio
async def test_legacy_fallback_retains_original_response_without_flipping_bits():
    body = b'{ "data": {"histogram": {"1": 0.333333333333, "2": 0.666666666667}} }'
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url))
        response = requests.Response()
        response.status_code = 200
        response._content = (
            b'{"id":"legacy"}'
            if method == "POST"
            else body
            if url.endswith("/results")
            else b'{"status":"completed"}'
        )
        return response

    result = await IonQExecutor(
        IonQExecutorConfig(api_key="offline"), session=Mock(request=request)
    ).execute(Circuit().h(0).h(1), shots=2)
    assert result.metadata["source_probabilities"] == {"01": 0.333333333333, "10": 0.666666666667}
    assert result.metadata["wire_order"] == "legacy-msb-first-unqualified"
    assert result.probabilities != result.metadata["source_probabilities"]
    assert base64.b64decode(result.metadata["result_artifact"]["body_base64"]) == body
    assert result.raw_result == {"status": "completed"}
    assert result.metadata["raw_shot_eligible"] is False


@pytest.mark.asyncio
async def test_legacy_inline_retains_terminal_job_bytes():
    body = b'{"status":"completed","data":{"histogram":{"1":1.0}}}'

    def request(method, url, **kwargs):
        response = requests.Response()
        response.status_code = 200
        response._content = b'{"id":"inline"}' if method == "POST" else body
        return response

    result = await IonQExecutor(
        IonQExecutorConfig(api_key="offline"), session=Mock(request=request)
    ).execute(Circuit().z(0).z(1))
    assert result.counts == {"01": 1000}
    assert base64.b64decode(result.metadata["result_artifact"]["body_base64"]) == body


@pytest.mark.asyncio
async def test_input_hash_stable_for_repeated_exports():
    results = [await ex(Transport({"10": 1.0})).execute(Circuit().x(0).z(1)) for _ in range(2)]
    assert results[0].metadata["input_sha256"] == results[1].metadata["input_sha256"]


@pytest.mark.asyncio
async def test_malformed_json_retains_original_failed_artifact():
    transport = Transport({}, body=b"{ broken json")
    with pytest.raises(ValueError) as caught:
        await ex(transport).execute(Circuit().x(0))
    artifact = caught.value.result_artifact
    assert base64.b64decode(artifact["body_base64"]) == transport.body
    assert artifact["sha256"] == hashlib.sha256(transport.body).hexdigest()
    assert artifact["format"] == FORMAT and artifact["id"] == "artifact"
    assert caught.value.remote_job["job_id"] == artifact["job_id"] == "owned"


def test_factory_forwards_explicit_version():
    from marqov.executors.factory import ExecutorFactory

    instance = ExecutorFactory._create_ionq_executor(
        "simulator", {"api_version": "0.4", "api_key": "offline"}
    )
    assert instance.config.api_version == "0.4"
    assert instance.config.base_url == "https://api.ionq.co/v0.4"


@pytest.mark.asyncio
@pytest.mark.parametrize("angle", [float("nan"), float("inf")])
async def test_nonfinite_rotations_refused_before_submission(angle):
    transport = Mock()
    with pytest.raises(ValueError):
        await ex(transport).execute(Circuit().rx(angle, 0))
    transport.request.assert_not_called()


@pytest.mark.asyncio
async def test_execution_snapshots_route_and_credentials_before_submit():
    transport = Transport({"10": 1.0})
    instance = ex(transport)
    original_request = transport.request

    def request(method, url, **kwargs):
        if method == "POST":
            instance.config.api_key = "changed"
            instance.config.base_url = "https://other.invalid"
            instance.config.target = "qpu.forte-1"
        return original_request(method, url, **kwargs)

    transport.request = request
    result = await instance.execute(Circuit().x(0).z(1))
    assert result.backend == "simulator"
    assert result.metadata["base_url"] == "https://api.ionq.co/v0.4"
    assert result.metadata["source_probabilities"] == {"10": 1.0}


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "canceled", "cancelled"])
async def test_terminal_failure_does_not_require_success_payload_fields(status):
    transport = Transport(
        {}, job={"id": "owned", "status": status, "failure": {"error": "vendor reason"}}
    )
    with pytest.raises(RuntimeError, match=f"owned {status}:.*vendor reason") as caught:
        await ex(transport).execute(Circuit().x(0))
    assert caught.value.remote_job["job_id"] == "owned"
    assert len(transport.calls) == 2


@pytest.mark.asyncio
async def test_canceled_job_without_failure_has_useful_status_message():
    transport = Transport({}, job={"id": "owned", "status": "canceled"})
    with pytest.raises(RuntimeError, match="IonQ job owned canceled$"):
        await ex(transport).execute(Circuit().x(0))


@pytest.mark.asyncio
async def test_wrong_job_failure_is_not_trusted():
    transport = Transport(
        {}, job={"id": "other", "status": "failed", "failure": {"error": "untrusted"}}
    )
    with pytest.raises(ValueError, match="identity") as caught:
        await ex(transport).execute(Circuit().x(0))
    assert caught.value.remote_job["job_id"] == "owned"
    assert len(transport.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        [],
        None,
        5,
        {"probabilities": []},
        {"probabilities": None},
        {"probabilities": {"registers": []}},
        {"probabilities": {"registers": None}},
    ],
)
async def test_nonobject_artifact_retains_original_bytes_with_clear_error(payload):
    transport = Transport({}, body=json.dumps(payload).encode())
    with pytest.raises(ValueError, match="must be a JSON object") as caught:
        await ex(transport).execute(Circuit().x(0))
    assert caught.value.remote_job["job_id"] == "owned"
    artifact = caught.value.result_artifact
    assert base64.b64decode(artifact["body_base64"]) == transport.body
    assert artifact["sha256"] == hashlib.sha256(transport.body).hexdigest()
    assert len(transport.calls) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("results", [[], None, 5])
async def test_nonobject_results_descriptor_refused_before_artifact_fetch(results):
    transport = Transport({"1": 1.0})
    transport.job["results"] = results
    with pytest.raises(ValueError, match="supported probabilities-v2 artifact") as caught:
        await ex(transport).execute(Circuit().x(0))
    assert caught.value.remote_job["job_id"] == "owned"
    assert len(transport.calls) == 2
