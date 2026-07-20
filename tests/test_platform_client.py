"""Tests for marqov.platform.client.MarqovClient.

All tests mock the Transport layer — NO real network calls.

Mock source citations (shapes transcribed from real routes):
  - Submit response ``job_id``:
      platform/src/app/api/jobs/submit/route.ts:1100
      ``return finalize(200, { job_id: jobId })``
  - Backends response ``backends`` array + item shape:
      platform/src/app/api/backends/route.ts:151-165 (response envelope)
      platform/src/app/api/backends/route.ts:100-131 (camelCase item mapping)
  - platform_info mocked path ``/api/meta`` — §11 TBC assumption:
      No real endpoint exists in platform/src/app/api/; mocked response shape
      ``{ "api_version": "..." }`` is a §11 TBC assumption.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from marqov.platform.client import MarqovClient
from marqov.platform._models import Backend, PlatformInfo
from marqov.platform.errors import (
    AuthenticationError,
    PaidBackendNotSupportedYet,
)
from marqov.platform.job import Job


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(api_key: str = "marqey_test_abc123", **kwargs) -> MarqovClient:
    """Build a MarqovClient with a real key but no network interaction."""
    return MarqovClient(api_key=api_key, base_url="http://test.invalid", **kwargs)


def _patch_transport_request(client: MarqovClient, return_value: dict):
    """Patch client._transport.request to return *return_value*."""
    return patch.object(client._transport, "request", return_value=return_value)


# ---------------------------------------------------------------------------
# Construction / authentication
# ---------------------------------------------------------------------------


class TestConstruction:
    """MarqovClient construction and key resolution."""

    def test_explicit_key_accepted(self):
        """Explicit api_key= does not raise."""
        client = _make_client(api_key="marqey_test_explicit")
        assert client._transport is not None

    def test_env_key_used_when_no_arg(self, monkeypatch):
        """MARQOV_PLATFORM_KEY env var is used when no argument is supplied.

        Key resolution lives in Transport; verified here at the client level.
        """
        monkeypatch.setenv("MARQOV_PLATFORM_KEY", "marqey_test_from_env")
        client = MarqovClient(base_url="http://test.invalid")
        # Transport is built; if key was missing it would have raised.
        assert client._transport is not None

    def test_explicit_key_beats_env(self, monkeypatch):
        """Explicit argument takes priority over env var."""
        monkeypatch.setenv("MARQOV_PLATFORM_KEY", "marqey_test_from_env")
        client = MarqovClient(api_key="marqey_test_explicit", base_url="http://test.invalid")
        assert client._transport._session.headers["Authorization"] == "Bearer marqey_test_explicit"

    def test_missing_key_raises_authentication_error(self, monkeypatch):
        """No key arg + no env var → AuthenticationError.

        No disk writes: key must come from the argument or env only.
        """
        monkeypatch.delenv("MARQOV_PLATFORM_KEY", raising=False)
        monkeypatch.delenv("MARQOV_PLATFORM_URL", raising=False)
        with pytest.raises(AuthenticationError):
            MarqovClient(base_url="http://test.invalid")

    def test_no_disk_write(self, tmp_path, monkeypatch):
        """Building a client does not write any files."""
        monkeypatch.chdir(tmp_path)
        _make_client()
        files_written = list(tmp_path.iterdir())
        assert files_written == [], f"Unexpected files written: {files_written}"


# ---------------------------------------------------------------------------
# submit() — str program path
# ---------------------------------------------------------------------------


class TestSubmitStringProgram:
    """submit() with a str program."""

    def test_str_with_framework_posts_inline_code(self):
        """str program + framework → POST body has inline_code + framework.

        Source: platform/src/lib/schemas.ts:164-198 — ``submitJobSchema``
        """
        client = _make_client()
        server_response = {"job_id": "job-uuid-001"}

        with patch.object(client._transport, "request", return_value=server_response) as mock_req:
            job = client.submit("x = 1", backend="dwave-sim", framework="marqov")

        mock_req.assert_called_once()
        _, kwargs = mock_req.call_args
        body = kwargs["json"]
        assert body["inline_code"] == "x = 1"
        assert body["framework"] == "marqov"
        assert body["backend"] == "dwave-sim"
        assert body["params"]["shots"] == 1000  # default

    def test_str_with_framework_returns_job_with_server_id(self):
        """submit() returns a Job whose id matches the server response job_id.

        Source: platform/src/app/api/jobs/submit/route.ts:1100
        """
        client = _make_client()
        server_response = {"job_id": "job-uuid-001"}

        with _patch_transport_request(client, server_response):
            job = client.submit("x = 1", backend="dwave-sim", framework="marqov")

        assert isinstance(job, Job)
        assert job.id == "job-uuid-001"

    def test_str_without_framework_raises_value_error(self):
        """Submitting a str without framework= must raise ValueError."""
        client = _make_client()
        with pytest.raises(ValueError, match="framework is required"):
            client.submit("x = 1", backend="dwave-sim")

    def test_str_custom_shots_passed(self):
        """shots= is forwarded as params.shots in the request body."""
        client = _make_client()
        server_response = {"job_id": "job-uuid-002"}

        with patch.object(client._transport, "request", return_value=server_response) as mock_req:
            client.submit("x = 1", backend="sv1", framework="marqov", shots=500)

        _, kwargs = mock_req.call_args
        assert kwargs["json"]["params"]["shots"] == 500

    def test_sdk_version_included_in_body(self):
        """sdk_version is included in the submit body (matches marqov.__version__)."""
        import marqov
        client = _make_client()
        server_response = {"job_id": "job-uuid-003"}

        with patch.object(client._transport, "request", return_value=server_response) as mock_req:
            client.submit("x = 1", backend="sv1", framework="marqov")

        _, kwargs = mock_req.call_args
        assert kwargs["json"]["sdk_version"] == marqov.__version__

    def test_idempotent_write_used(self):
        """submit() calls transport.request with idempotent_write=True."""
        client = _make_client()
        server_response = {"job_id": "job-uuid-004"}

        with patch.object(client._transport, "request", return_value=server_response) as mock_req:
            client.submit("x = 1", backend="sv1", framework="marqov")

        _, kwargs = mock_req.call_args
        assert kwargs.get("idempotent_write") is True


# ---------------------------------------------------------------------------
# submit() — Circuit program path
# ---------------------------------------------------------------------------


class TestSubmitCircuitProgram:
    """submit() with a marqov.Circuit program.

    PROVISIONAL — §11 reconciliation item: the ``circuit`` body field and the
    circuit-submission wire contract are pending the platform's circuit-submission
    variant (spec §8.6 #1).  These tests verify the SDK builds the correct body
    shape; end-to-end server behaviour is verified (or marked BLOCKED) by the
    Task 5b staging smoke test.
    """

    def test_circuit_posts_circuit_field_not_inline_code(self):
        """Circuit → body has a 'circuit' field, NOT inline_code.

        The circuit rides as a separate body field alongside backend/params/
        sdk_version.  inline_code is executable server-side code and must NOT
        carry a JSON envelope for a Circuit submission.

        PROVISIONAL — exact wire contract pending platform §8.6 #1 variant.
        Requires qiskit (openqasm extra).
        """
        pytest.importorskip("qiskit")
        from marqov import Circuit

        client = _make_client()
        server_response = {"job_id": "job-uuid-circuit-001"}
        circuit = Circuit().h(0)

        with patch.object(client._transport, "request", return_value=server_response) as mock_req:
            job = client.submit(circuit, backend="dwave-sim")

        _, kwargs = mock_req.call_args
        body = kwargs["json"]

        # Circuit path: body["circuit"] carries the serialised circuit.
        assert "circuit" in body, "body must contain 'circuit' field for Circuit submissions"
        assert body["circuit"]["format"] == "qasm3"
        assert "payload" in body["circuit"]
        assert isinstance(body["circuit"]["payload"], str)
        assert len(body["circuit"]["payload"]) > 0

        # inline_code must NOT be set for Circuit submissions.
        assert "inline_code" not in body, (
            "inline_code must not be set for Circuit submissions; "
            "circuit rides as a separate 'circuit' body field"
        )

        # framework must NOT be in the body for Circuit submissions.
        assert "framework" not in body

    def test_circuit_returns_job_with_server_id(self):
        """submit(Circuit) returns Job with the server-assigned id."""
        pytest.importorskip("qiskit")
        from marqov import Circuit

        client = _make_client()
        server_response = {"job_id": "job-uuid-circuit-002"}

        with _patch_transport_request(client, server_response):
            job = client.submit(Circuit().h(0), backend="dwave-sim")

        assert isinstance(job, Job)
        assert job.id == "job-uuid-circuit-002"

    def test_circuit_with_framework_raises_value_error(self):
        """Passing framework= with a Circuit raises ValueError."""
        pytest.importorskip("qiskit")
        from marqov import Circuit

        client = _make_client()
        with pytest.raises(ValueError, match="framework"):
            client.submit(Circuit().h(0), backend="dwave-sim", framework="marqov")

    def test_circuit_qasm3_payload_is_valid_qasm(self):
        """The payload in body['circuit'] is a valid QASM 3 string.

        PROVISIONAL — exact wire contract pending platform §8.6 #1 variant.
        """
        pytest.importorskip("qiskit")
        from marqov import Circuit

        client = _make_client()
        server_response = {"job_id": "job-uuid-circuit-003"}
        circuit = Circuit().h(0).cnot(0, 1)

        with patch.object(client._transport, "request", return_value=server_response) as mock_req:
            client.submit(circuit, backend="dwave-sim")

        _, kwargs = mock_req.call_args
        payload = kwargs["json"]["circuit"]["payload"]
        # Should be valid QASM — contains "OPENQASM" or "qasm"
        assert "OPENQASM" in payload or "qasm" in payload.lower()

    def test_non_circuit_non_str_raises_type_error(self):
        """Passing an unsupported type raises TypeError."""
        client = _make_client()
        with pytest.raises(TypeError, match="int"):
            client.submit(123, backend="sv1")

    def test_none_program_raises_type_error(self):
        """Passing None as program raises TypeError."""
        client = _make_client()
        with pytest.raises(TypeError):
            client.submit(None, backend="sv1")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# submit() — paid backend propagation
# ---------------------------------------------------------------------------


class TestSubmitPaidBackend:
    """Paid backend errors propagate as PaidBackendNotSupportedYet."""

    def test_paid_backend_propagates(self):
        """PaidBackendNotSupportedYet from transport propagates unmodified.

        The client does NOT intercept paid-backend errors; the transport maps
        server ``analysis_required`` (422) to PaidBackendNotSupportedYet and
        it propagates directly.
        """
        client = _make_client()

        with patch.object(
            client._transport,
            "request",
            side_effect=PaidBackendNotSupportedYet(
                "Paid backends require a valid analysis_id",
                code="analysis_required",
                status=422,
            ),
        ):
            with pytest.raises(PaidBackendNotSupportedYet):
                client.submit("x = 1", backend="ibm-qpu", framework="marqov")


# ---------------------------------------------------------------------------
# job() — reconnect
# ---------------------------------------------------------------------------


class TestJobReconnect:
    """client.job(id) reconnects to an existing job."""

    def test_job_returns_job_with_correct_id(self):
        """job() returns a Job with the supplied id."""
        client = _make_client()
        job_id = "550e8400-e29b-41d4-a716-446655440000"
        job = client.job(job_id)
        assert isinstance(job, Job)
        assert job.id == job_id

    def test_job_shares_transport(self):
        """The reconnected Job uses the same transport as the client."""
        client = _make_client()
        job = client.job("some-job-id")
        assert job._transport is client._transport


# ---------------------------------------------------------------------------
# backends()
# ---------------------------------------------------------------------------


class TestBackends:
    """backends() maps the /api/backends payload to Backend dataclasses.

    Mock shape transcribed from:
      platform/src/app/api/backends/route.ts:100-131 (camelCase item mapping)
      platform/src/app/api/backends/route.ts:151-165 (response envelope)
    """

    # Minimal camelCase payload matching what the server returns.
    # Source: platform/src/app/api/backends/route.ts:16-40 + 100-131
    _RAW_BACKEND = {
        "id": "be-001",
        "slug": "dwave-sim",
        "name": "D-Wave Simulator",
        "provider": "dwave",
        "deviceType": "simulator",
        "providerTargetId": "Advantage_system4.1",
        "region": None,
        "qubitCount": 5000,
        "isAvailable": True,
        "status": "online",
        "pricing": {"taskFee": 0.0, "perShot": 0.0, "minimumCost": 0.0},
        "maxShots": 10000,
        "maxQubits": 5000,
        "description": "D-Wave Advantage simulator",
        "documentationUrl": None,
        "tags": ["free", "annealing"],
        "displayOrder": 1,
        "isRecommended": True,
        "isRetired": False,
        "queueTimeSeconds": None,
        "statusUpdatedAt": None,
        "supportedProgramTypes": ["qubo"],
    }

    def test_backends_returns_list_of_backend_objects(self):
        """backends() returns a list of Backend dataclasses."""
        client = _make_client()
        server_response = {
            "backends": [self._RAW_BACKEND],
            "updatedAt": "2026-07-20T00:00:00.000Z",
        }

        with _patch_transport_request(client, server_response):
            result = client.backends()

        assert len(result) == 1
        assert isinstance(result[0], Backend)

    def test_backends_maps_slug(self):
        """Backend.slug is set from the server slug field."""
        client = _make_client()
        server_response = {"backends": [self._RAW_BACKEND], "updatedAt": ""}

        with _patch_transport_request(client, server_response):
            result = client.backends()

        assert result[0].slug == "dwave-sim"

    def test_backends_maps_is_available(self):
        """Backend.is_available is set from isAvailable (camelCase)."""
        client = _make_client()
        server_response = {"backends": [self._RAW_BACKEND], "updatedAt": ""}

        with _patch_transport_request(client, server_response):
            result = client.backends()

        assert result[0].is_available is True

    def test_backends_maps_device_type(self):
        """Backend.device_type is set from deviceType (camelCase)."""
        client = _make_client()
        server_response = {"backends": [self._RAW_BACKEND], "updatedAt": ""}

        with _patch_transport_request(client, server_response):
            result = client.backends()

        assert result[0].device_type == "simulator"

    def test_backends_maps_supported_program_types(self):
        """Backend.supported_program_types is set from supportedProgramTypes."""
        client = _make_client()
        server_response = {"backends": [self._RAW_BACKEND], "updatedAt": ""}

        with _patch_transport_request(client, server_response):
            result = client.backends()

        assert result[0].supported_program_types == ["qubo"]

    def test_backends_extra_fields_captured(self):
        """Unknown server fields land in Backend.extra."""
        client = _make_client()
        raw = dict(self._RAW_BACKEND)
        raw["futureProp"] = "value42"
        server_response = {"backends": [raw], "updatedAt": ""}

        with _patch_transport_request(client, server_response):
            result = client.backends()

        assert result[0].extra.get("futureProp") == "value42"

    def test_backends_multiple(self):
        """Multiple backends in response map to multiple Backend objects."""
        client = _make_client()
        raw2 = dict(self._RAW_BACKEND)
        raw2["slug"] = "sv1"
        raw2["name"] = "StateVec Simulator"
        server_response = {
            "backends": [self._RAW_BACKEND, raw2],
            "updatedAt": "",
        }

        with _patch_transport_request(client, server_response):
            result = client.backends()

        assert len(result) == 2
        assert result[1].slug == "sv1"

    def test_backends_empty_list(self):
        """Empty backends array returns an empty list."""
        client = _make_client()
        with _patch_transport_request(client, {"backends": [], "updatedAt": ""}):
            result = client.backends()
        assert result == []


# ---------------------------------------------------------------------------
# platform_info()
# ---------------------------------------------------------------------------


class TestPlatformInfo:
    """platform_info() returns PlatformInfo (mocked /api/meta — §11 TBC).

    §11 TBC assumption: no real /api/meta endpoint exists in the platform.
    This test mocks the transport response and verifies the client builds
    PlatformInfo correctly.  See client.py docstring for the assumption.
    """

    def test_platform_info_returns_platform_info(self):
        """platform_info() returns a PlatformInfo instance."""
        client = _make_client()
        server_response = {"api_version": "1.0.0"}

        with _patch_transport_request(client, server_response):
            info = client.platform_info()

        assert isinstance(info, PlatformInfo)

    def test_platform_info_sdk_version_matches_package(self):
        """PlatformInfo.sdk_version matches marqov.__version__."""
        import marqov
        client = _make_client()

        with _patch_transport_request(client, {"api_version": "1.0.0"}):
            info = client.platform_info()

        assert info.sdk_version == marqov.__version__

    def test_platform_info_api_version_from_response(self):
        """PlatformInfo.api_version comes from the mocked response."""
        client = _make_client()

        with _patch_transport_request(client, {"api_version": "2.5.0"}):
            info = client.platform_info()

        assert info.api_version == "2.5.0"

    def test_platform_info_missing_api_version_falls_back(self):
        """api_version defaults to 'unknown' when absent from response."""
        client = _make_client()

        with _patch_transport_request(client, {}):
            info = client.platform_info()

        assert info.api_version == "unknown"

    def test_platform_info_calls_get_api_meta(self):
        """platform_info() calls GET /api/meta (§11 TBC path)."""
        client = _make_client()

        with patch.object(
            client._transport, "request", return_value={"api_version": "1.0.0"}
        ) as mock_req:
            client.platform_info()

        mock_req.assert_called_once_with("GET", "/api/meta")
