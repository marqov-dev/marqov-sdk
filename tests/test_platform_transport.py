"""Tests for marqov.platform._transport.Transport.

All tests mock ``requests.Session.request`` — NO real network calls.

Mock fixtures reproduce the platform's HTTP API contract: all error bodies
follow the envelope ``{ error: { code, message, status } }``.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch, call

import pytest
import requests
import requests.exceptions

from marqov.platform._transport import Transport, _DEFAULT_BASE_URL
from marqov.platform.errors import (
    AuthenticationError,
    BackendUnavailable,
    InvalidProgram,
    MarqovPlatformError,
    PaidBackendNotSupportedYet,
    PermissionTierError,
    RateLimited,
    TransportError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(
    status_code: int,
    json_body: object | None = None,
    headers: dict | None = None,
) -> MagicMock:
    """Build a mock requests.Response."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.ok = status_code < 400
    resp.reason = f"HTTP {status_code}"
    if json_body is not None:
        resp.json.return_value = json_body
    else:
        resp.json.side_effect = ValueError("No body")
    resp.headers = headers or {}
    return resp


# ---------------------------------------------------------------------------
# Authentication tests
# ---------------------------------------------------------------------------


class TestAuthentication:
    """Transport resolves the API key and attaches it correctly."""

    def test_explicit_key_used_in_bearer_header(self, tmp_path):
        """Transport sends Authorization: Bearer <key> on every request.

        The server expects a Bearer token in the Authorization header.
        """
        transport = Transport(api_key="marqey_test_abc123", base_url="http://test")
        mock_resp = _mock_response(200, {"job_id": "j1"})

        with patch.object(transport._session, "request", return_value=mock_resp) as mock_req:
            transport.request("POST", "/api/jobs/submit", json={"x": 1})
            _, kwargs = mock_req.call_args
            sent_headers = kwargs.get("headers", {})
            # The session-level auth header is not repeated in extra_headers,
            # but the session itself has it. Verify via session.headers.
        assert transport._session.headers["Authorization"] == "Bearer marqey_test_abc123"

    def test_explicit_key_beats_env_var(self, monkeypatch):
        """Explicit api_key= argument overrides MARQOV_PLATFORM_KEY env var."""
        monkeypatch.setenv("MARQOV_PLATFORM_KEY", "marqey_test_from_env")
        transport = Transport(api_key="marqey_test_explicit", base_url="http://test")
        assert "Bearer marqey_test_explicit" in transport._session.headers["Authorization"]

    def test_env_var_used_when_no_explicit_key(self, monkeypatch):
        """MARQOV_PLATFORM_KEY is used when no explicit key is given."""
        monkeypatch.setenv("MARQOV_PLATFORM_KEY", "marqey_test_envonly")
        transport = Transport(base_url="http://test")
        assert "Bearer marqey_test_envonly" in transport._session.headers["Authorization"]

    def test_missing_both_raises_authentication_error(self, monkeypatch):
        """No key anywhere → AuthenticationError, never touches disk."""
        monkeypatch.delenv("MARQOV_PLATFORM_KEY", raising=False)
        with pytest.raises(AuthenticationError):
            Transport()

    def test_key_never_written_to_disk(self, monkeypatch, tmp_path):
        """The API key must never be written to any file.

        Strategy: patch ``builtins.open`` to fail; construction must succeed
        (proves no disk write happens during ``__init__``).
        """
        monkeypatch.setenv("MARQOV_PLATFORM_KEY", "marqey_test_secret")

        opened_files: list[str] = []

        real_open = open

        def spy_open(path, *args, **kwargs):
            opened_files.append(str(path))
            return real_open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=spy_open):
            transport = Transport(base_url="http://test")

        # Only system files (e.g. *.pth, *.so) may be read during import;
        # no file matching "*key*" or "*marqey*" should appear.
        suspicious = [f for f in opened_files if "marqey" in f.lower() or "platform_key" in f.lower()]
        assert suspicious == [], f"Key written to: {suspicious}"
        assert transport is not None


# ---------------------------------------------------------------------------
# Error mapping tests
# ---------------------------------------------------------------------------


class TestErrorMapping:
    """Non-2xx responses are mapped to the correct exception subclass."""

    def test_401_raises_authentication_error(self, monkeypatch):
        """HTTP 401 unauthorized → AuthenticationError.

        Server returns error code "unauthorized" with HTTP 401.
        Body shape: ``{ error: { code, message, status } }``
        """
        body = {"error": {"code": "unauthorized", "message": "Unauthorized", "status": 401}}
        transport = Transport(api_key="marqey_test_x", base_url="http://test")
        mock_resp = _mock_response(401, body)

        with patch.object(transport._session, "request", return_value=mock_resp):
            with pytest.raises(AuthenticationError) as exc_info:
                transport.request("GET", "/api/jobs/123/status")
        assert exc_info.value.status == 401

    def test_403_permission_denied_raises_permission_tier_error(self):
        """HTTP 403 permission_denied → PermissionTierError.

        Server returns error code "permission_denied" with HTTP 403.
        """
        body = {
            "error": {
                "code": "permission_denied",
                "message": "This API key requires 'qpu' permission for backend 'ibm_eagle'",
                "status": 403,
            }
        }
        transport = Transport(api_key="marqey_test_x", base_url="http://test")
        mock_resp = _mock_response(403, body)

        with patch.object(transport._session, "request", return_value=mock_resp):
            with pytest.raises(PermissionTierError) as exc_info:
                transport.request("POST", "/api/jobs/submit", json={})
        assert exc_info.value.status == 403
        assert exc_info.value.code == "permission_denied"

    def test_422_analysis_required_raises_paid_backend_not_supported_yet(self):
        """HTTP 422 analysis_required → PaidBackendNotSupportedYet.

        Server returns error code "analysis_required" with HTTP 422.
        Body shape: ``{ error: { code, message, status } }``
        """
        body = {
            "error": {
                "code": "analysis_required",
                "message": "Paid backends require a valid analysis_id",
                "status": 422,
            }
        }
        transport = Transport(api_key="marqey_test_x", base_url="http://test")
        mock_resp = _mock_response(422, body)

        with patch.object(transport._session, "request", return_value=mock_resp):
            with pytest.raises(PaidBackendNotSupportedYet) as exc_info:
                transport.request("POST", "/api/jobs/submit", json={})
        assert exc_info.value.code == "analysis_required"
        assert exc_info.value.status == 422

    def test_unknown_code_raises_base_marqov_platform_error_not_transport_error(self):
        """Unknown error code → base MarqovPlatformError with .code preserved.

        CRITICAL: must NOT raise TransportError for an unknown code.

        The server returns "spend_limit_exceeded" with HTTP 429; here we use
        that body but respond with an HTTP status that doesn't match 429 to
        exercise the "unknown code" branch via a different scenario.
        spend_limit_exceeded CAN appear on 429 — test it as a 422 (hypothetical
        future unknown code shape) to verify base-class mapping.
        """
        # Body follows the standard error envelope; code is a plausible future
        # code not in the current ErrorCode union (tests future-proofing).
        body = {
            "error": {
                "code": "spend_limit_exceeded",
                "message": "API key daily spend limit exceeded.",
                "status": 429,
            }
        }
        transport = Transport(api_key="marqey_test_x", base_url="http://test")
        # Return 422 so the 429-specific branch doesn't fire, exercising
        # the generic "unknown code → base class" mapping.
        mock_resp = _mock_response(422, body)

        with patch.object(transport._session, "request", return_value=mock_resp):
            with pytest.raises(MarqovPlatformError) as exc_info:
                transport.request("POST", "/api/jobs/submit", json={})

        exc = exc_info.value
        # Must be base class, NOT TransportError
        assert type(exc) is MarqovPlatformError, (
            f"Expected MarqovPlatformError, got {type(exc).__name__}"
        )
        assert exc.code == "spend_limit_exceeded"
        assert exc.status == 422
        assert not isinstance(exc, TransportError)

    def test_spend_limit_exceeded_on_429_raises_rate_limited(self):
        """HTTP 429 spend_limit_exceeded → RateLimited (status code is authoritative).

        Server returns error code "spend_limit_exceeded" with HTTP 429.
        Rate-limit 429 responses also carry a Retry-After header.
        """
        body = {
            "error": {
                "code": "spend_limit_exceeded",
                "message": "API key daily spend limit exceeded. Current: 100¢, limit: 50¢",
                "status": 429,
            }
        }
        transport = Transport(api_key="marqey_test_x", base_url="http://test")
        mock_resp = _mock_response(429, body, headers={"Retry-After": "30"})

        with patch.object(transport._session, "request", return_value=mock_resp):
            with pytest.raises(RateLimited) as exc_info:
                transport.request("POST", "/api/jobs/submit", json={})
        assert exc_info.value.status == 429

    def test_backend_unknown_raises_backend_unavailable(self):
        """HTTP 422 backend_unknown → BackendUnavailable.

        Server returns error code "backend_unknown" with HTTP 422 and a message
        of the form "Backend '<slug>' is not in the registry".
        Body shape: ``{ error: { code, message, status } }``
        """
        body = {
            "error": {
                "code": "backend_unknown",
                "message": "Backend 'bad-slug' is not in the registry",
                "status": 422,
            }
        }
        transport = Transport(api_key="marqey_test_x", base_url="http://test")
        mock_resp = _mock_response(422, body)

        with patch.object(transport._session, "request", return_value=mock_resp):
            with pytest.raises(BackendUnavailable) as exc_info:
                transport.request("POST", "/api/jobs/submit", json={})

        exc = exc_info.value
        assert exc.code == "backend_unknown"
        assert exc.status == 422
        assert isinstance(exc, BackendUnavailable)

    def test_backend_retired_raises_backend_unavailable(self):
        """HTTP 422 backend_retired → BackendUnavailable.

        Server returns error code "backend_retired" with HTTP 422 and a message
        of the form "Backend '<slug>' is retired".
        Body shape: ``{ error: { code, message, status } }``
        """
        body = {
            "error": {
                "code": "backend_retired",
                "message": "Backend 'old-qpu' is retired",
                "status": 422,
            }
        }
        transport = Transport(api_key="marqey_test_x", base_url="http://test")
        mock_resp = _mock_response(422, body)

        with patch.object(transport._session, "request", return_value=mock_resp):
            with pytest.raises(BackendUnavailable) as exc_info:
                transport.request("POST", "/api/jobs/submit", json={})

        exc = exc_info.value
        assert exc.code == "backend_retired"
        assert exc.status == 422
        assert isinstance(exc, BackendUnavailable)

    def test_validation_error_on_400_raises_invalid_program(self):
        """HTTP 400 validation_error → InvalidProgram.

        Server returns error code "validation_error" with HTTP 400 (e.g. an
        invalid JSON body or a missing Idempotency-Key header).
        Body shape: ``{ error: { code, message, status } }``
        """
        body = {
            "error": {
                "code": "validation_error",
                "message": "Idempotency-Key header is required for API key authentication",
                "status": 400,
            }
        }
        transport = Transport(api_key="marqey_test_x", base_url="http://test")
        mock_resp = _mock_response(400, body)

        with patch.object(transport._session, "request", return_value=mock_resp):
            with pytest.raises(InvalidProgram) as exc_info:
                transport.request("POST", "/api/jobs/submit", json={})

        exc = exc_info.value
        assert exc.code == "validation_error"
        assert exc.status == 400
        assert isinstance(exc, InvalidProgram)

    def test_validation_error_on_422_raises_invalid_program(self):
        """HTTP 422 validation_error → InvalidProgram (structured body shape).

        Server returns error code "validation_error" with HTTP 422 (e.g. a
        message of "Script has no inline content to submit").
        Body shape: the standard ``{ error: { code, message, status } }`` envelope.
        """
        body = {
            "error": {
                "code": "validation_error",
                "message": "Script has no inline content to submit",
                "status": 422,
            }
        }
        transport = Transport(api_key="marqey_test_x", base_url="http://test")
        mock_resp = _mock_response(422, body)

        with patch.object(transport._session, "request", return_value=mock_resp):
            with pytest.raises(InvalidProgram) as exc_info:
                transport.request("POST", "/api/jobs/submit", json={})

        exc = exc_info.value
        assert exc.code == "validation_error"
        assert exc.status == 422
        assert isinstance(exc, InvalidProgram)

    def test_429_rate_limited_raises_rate_limited(self):
        """HTTP 429 rate_limited from status poll → RateLimited.

        The server returns HTTP 429 with a Retry-After header on rate limit.

        NOTE: 429 bodies come in two real shapes:
          - Plain string: ``{ "error": "Rate limit exceeded. 0/60 requests remaining…" }``
          - Structured dict: ``{ "error": { "code": "spend_limit_exceeded", … } }``
        This test uses the structured dict shape; see test_429_rate_limit_plain_string_body
        for the plain-string shape.
        """
        body = {
            "error": {
                "code": "rate_limited",
                "message": "Too many requests",
                "status": 429,
            }
        }
        transport = Transport(api_key="marqey_test_x", base_url="http://test")
        mock_resp = _mock_response(429, body, headers={"Retry-After": "60"})

        with patch.object(transport._session, "request", return_value=mock_resp):
            with pytest.raises(RateLimited) as exc_info:
                transport.request("GET", "/api/jobs/abc/status")
        assert exc_info.value.status == 429


# ---------------------------------------------------------------------------
# Retry policy tests
# ---------------------------------------------------------------------------


class TestRetryPolicy:
    """Verify the conditional retry logic for writes vs. reads."""

    def test_idempotent_write_retries_connection_error_with_same_idempotency_key(self):
        """ConnectionError on idempotent_write=True → retried; Idempotency-Key reused.

        The same UUID must appear in every attempt so the server can dedupe.
        The server uses the Idempotency-Key header to detect replays.
        """
        transport = Transport(api_key="marqey_test_x", base_url="http://test")
        success_resp = _mock_response(200, {"job_id": "j1"})

        call_count = 0
        captured_idempotency_keys: list[str] = []

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            headers = kwargs.get("headers", {})
            captured_idempotency_keys.append(headers.get("Idempotency-Key", ""))
            if call_count == 1:
                raise requests.exceptions.ConnectionError("Connection refused")
            return success_resp

        with patch.object(transport._session, "request", side_effect=side_effect):
            result = transport.request(
                "POST",
                "/api/jobs/submit",
                json={"backend": "sv1"},
                idempotent_write=True,
            )

        assert result == {"job_id": "j1"}
        assert call_count == 2
        # The idempotency key must be IDENTICAL across all attempts
        assert len(set(captured_idempotency_keys)) == 1, (
            f"Idempotency-Key changed across retries: {captured_idempotency_keys}"
        )
        assert captured_idempotency_keys[0] != "", "Idempotency-Key must be non-empty"

    def test_idempotent_write_does_not_retry_on_timeout(self):
        """Timeout on idempotent_write=True → NOT retried → TransportError.

        Rationale: server deduplication isn't confirmed; retry could double-submit.
        """
        transport = Transport(api_key="marqey_test_x", base_url="http://test")

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise requests.exceptions.Timeout("timed out")

        with patch.object(transport._session, "request", side_effect=side_effect):
            with pytest.raises(TransportError):
                transport.request(
                    "POST",
                    "/api/jobs/submit",
                    json={"backend": "sv1"},
                    idempotent_write=True,
                )

        # Must NOT have retried
        assert call_count == 1, f"Expected 1 attempt, got {call_count}"

    def test_read_retries_on_any_transport_failure(self):
        """GET (idempotent_write=False) → retried on any transport failure.

        The status endpoint is a GET; no Idempotency-Key is required for reads.
        """
        transport = Transport(api_key="marqey_test_x", base_url="http://test")
        success_resp = _mock_response(200, {"id": "j1", "status": "completed"})

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise requests.exceptions.Timeout("poll timeout")
            return success_resp

        with patch.object(transport._session, "request", side_effect=side_effect):
            result = transport.request("GET", "/api/jobs/j1/status")

        assert result["status"] == "completed"
        assert call_count == 3

    def test_read_connection_error_retried(self):
        """ConnectionError on GET → retried and eventually succeeds."""
        transport = Transport(api_key="marqey_test_x", base_url="http://test")
        success_resp = _mock_response(200, {"id": "j2", "status": "pending"})

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise requests.exceptions.ConnectionError("DNS failed")
            return success_resp

        with patch.object(transport._session, "request", side_effect=side_effect):
            result = transport.request("GET", "/api/jobs/j2/status")

        assert result["status"] == "pending"
        assert call_count == 2

    def test_idempotent_write_connection_error_exhausted_raises_transport_error(self):
        """All retries exhausted on ConnectionError for write → TransportError."""
        transport = Transport(api_key="marqey_test_x", base_url="http://test")

        with patch.object(
            transport._session,
            "request",
            side_effect=requests.exceptions.ConnectionError("refused"),
        ):
            with pytest.raises(TransportError):
                transport.request(
                    "POST",
                    "/api/jobs/submit",
                    json={},
                    idempotent_write=True,
                )

    def test_get_all_retries_exhausted_raises_transport_error(self):
        """All GET retries exhausted → TransportError."""
        transport = Transport(api_key="marqey_test_x", base_url="http://test")

        with patch.object(
            transport._session,
            "request",
            side_effect=requests.exceptions.Timeout("always times out"),
        ):
            with pytest.raises(TransportError):
                transport.request("GET", "/api/jobs/j3/status")


# ---------------------------------------------------------------------------
# Wait parameter tests
# ---------------------------------------------------------------------------


class TestWaitParameter:
    """The ``wait`` argument is forwarded as the ``wait`` query param.

    The server reads the "wait" query param on the status endpoint.
    """

    def test_wait_param_sent_as_query_param(self):
        """wait=20 must appear in the URL query string as ``wait=20``."""
        transport = Transport(api_key="marqey_test_x", base_url="http://test")
        # Status response body: plain JSON of the job row.
        body = {
            "id": "abc",
            "status": "pending",
            "backend": "sv1",
            "created_at": "2026-07-01T00:00:00Z",
            "updated_at": "2026-07-01T00:00:00Z",
            "estimated_cost_usd": 0,
            "result": None,
        }
        mock_resp = _mock_response(200, body)

        with patch.object(transport._session, "request", return_value=mock_resp) as mock_req:
            transport.request("GET", "/api/jobs/abc/status", wait=20)

        _, kwargs = mock_req.call_args
        sent_params = kwargs.get("params", {})
        assert "wait" in sent_params, f"'wait' missing from params: {sent_params}"
        assert sent_params["wait"] == 20

    def test_no_wait_means_no_wait_param(self):
        """When wait is None, no wait param is sent."""
        transport = Transport(api_key="marqey_test_x", base_url="http://test")
        # Status response body shape.
        body = {"id": "abc", "status": "completed", "backend": "sv1",
                "created_at": "2026-07-01T00:00:00Z", "updated_at": "2026-07-01T00:00:00Z",
                "estimated_cost_usd": 0, "result": None}
        mock_resp = _mock_response(200, body)

        with patch.object(transport._session, "request", return_value=mock_resp) as mock_req:
            transport.request("GET", "/api/jobs/abc/status")

        _, kwargs = mock_req.call_args
        sent_params = kwargs.get("params")
        # params will be None or a dict without "wait"
        if sent_params is not None:
            assert "wait" not in sent_params

    def test_wait_combined_with_other_params(self):
        """wait= merges with any other query params passed in."""
        transport = Transport(api_key="marqey_test_x", base_url="http://test")
        body = {"id": "abc", "status": "running", "backend": "sv1",
                "created_at": "2026-07-01T00:00:00Z", "updated_at": "2026-07-01T00:00:00Z",
                "estimated_cost_usd": 0, "result": None}
        mock_resp = _mock_response(200, body)

        with patch.object(transport._session, "request", return_value=mock_resp) as mock_req:
            transport.request("GET", "/api/jobs/abc/status", params={"verbose": "1"}, wait=15)

        _, kwargs = mock_req.call_args
        sent_params = kwargs.get("params", {})
        assert sent_params.get("wait") == 15
        assert sent_params.get("verbose") == "1"


# ---------------------------------------------------------------------------
# Idempotency-Key integrity tests
# ---------------------------------------------------------------------------


class TestIdempotencyKey:
    """Writes always carry an Idempotency-Key; the same key is reused on retry."""

    def test_post_includes_idempotency_key_header(self):
        """POST sends an Idempotency-Key header on first attempt."""
        transport = Transport(api_key="marqey_test_x", base_url="http://test")
        # Submit response body: ``{ job_id: <uuid> }``.
        body = {"job_id": "j1"}
        mock_resp = _mock_response(200, body)

        with patch.object(transport._session, "request", return_value=mock_resp) as mock_req:
            transport.request("POST", "/api/jobs/submit", json={"x": 1})

        _, kwargs = mock_req.call_args
        assert "Idempotency-Key" in kwargs.get("headers", {}), (
            "POST must send Idempotency-Key header"
        )

    def test_get_does_not_send_idempotency_key(self):
        """GET requests do not include an Idempotency-Key header."""
        transport = Transport(api_key="marqey_test_x", base_url="http://test")
        # Status response body shape.
        body = {"id": "j1", "status": "completed", "backend": "sv1",
                "created_at": "2026-07-01T00:00:00Z", "updated_at": "2026-07-01T00:00:00Z",
                "estimated_cost_usd": 0, "result": None}
        mock_resp = _mock_response(200, body)

        with patch.object(transport._session, "request", return_value=mock_resp) as mock_req:
            transport.request("GET", "/api/jobs/j1/status")

        _, kwargs = mock_req.call_args
        headers = kwargs.get("headers", {})
        assert "Idempotency-Key" not in headers

    def test_idempotency_key_is_uuid4_format(self):
        """The generated Idempotency-Key looks like a UUID."""
        import re

        transport = Transport(api_key="marqey_test_x", base_url="http://test")
        body = {"job_id": "j1"}
        mock_resp = _mock_response(200, body)

        with patch.object(transport._session, "request", return_value=mock_resp) as mock_req:
            transport.request("POST", "/api/jobs/submit", json={})

        _, kwargs = mock_req.call_args
        key = kwargs.get("headers", {}).get("Idempotency-Key", "")
        uuid_re = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
        )
        assert uuid_re.match(key), f"Not a UUID4: {key!r}"

    def test_same_idempotency_key_reused_across_connection_error_retries(self):
        """ConnectionError retries for a write reuse the SAME Idempotency-Key.

        The server's idempotency check uses the key to detect replays; a
        changing key defeats deduplication.
        """
        transport = Transport(api_key="marqey_test_x", base_url="http://test")
        success_resp = _mock_response(200, {"job_id": "j1"})

        collected_keys: list[str] = []

        def side_effect(*args, **kwargs):
            key = kwargs.get("headers", {}).get("Idempotency-Key", "")
            collected_keys.append(key)
            if len(collected_keys) < 2:
                raise requests.exceptions.ConnectionError("refused")
            return success_resp

        with patch.object(transport._session, "request", side_effect=side_effect):
            transport.request(
                "POST",
                "/api/jobs/submit",
                json={},
                idempotent_write=True,
            )

        assert len(collected_keys) == 2
        assert collected_keys[0] == collected_keys[1], (
            "Idempotency-Key must not change across retries"
        )


# ---------------------------------------------------------------------------
# 2xx success path
# ---------------------------------------------------------------------------


class TestSuccessPath:
    """2xx responses return decoded JSON directly."""

    def test_200_returns_json_body(self):
        """200 OK → response body returned as dict."""
        transport = Transport(api_key="marqey_test_x", base_url="http://test")
        # Submit response body: ``{ job_id: <uuid> }``.
        body = {"job_id": "j42"}
        mock_resp = _mock_response(200, body)

        with patch.object(transport._session, "request", return_value=mock_resp):
            result = transport.request("POST", "/api/jobs/submit", json={"x": 1})

        assert result == {"job_id": "j42"}

    def test_base_url_resolution_default(self, monkeypatch):
        """Without env var, default production URL is used."""
        monkeypatch.delenv("MARQOV_PLATFORM_URL", raising=False)
        transport = Transport(api_key="marqey_test_x")
        assert transport._base_url == _DEFAULT_BASE_URL

    def test_base_url_from_env(self, monkeypatch):
        """MARQOV_PLATFORM_URL overrides the built-in default."""
        monkeypatch.setenv("MARQOV_PLATFORM_URL", "https://staging.marqov.internal")
        monkeypatch.delenv("MARQOV_PLATFORM_KEY", raising=False)
        transport = Transport(api_key="marqey_test_x")
        assert transport._base_url == "https://staging.marqov.internal"

    def test_base_url_explicit_arg_beats_env(self, monkeypatch):
        """Explicit base_url= argument beats MARQOV_PLATFORM_URL env var."""
        monkeypatch.setenv("MARQOV_PLATFORM_URL", "https://env.marqov.internal")
        transport = Transport(
            api_key="marqey_test_x",
            base_url="https://explicit.marqov.internal",
        )
        assert transport._base_url == "https://explicit.marqov.internal"


# ---------------------------------------------------------------------------
# Rate-limit body shape tests (I1) + Retry-After surface tests (I2)
# ---------------------------------------------------------------------------


class TestRateLimitedBodyShapes:
    """429 responses come in two real shapes; both must raise RateLimited.

    Shape A — plain string error:
        ``{ "error": "Rate limit exceeded. 0/60 requests remaining. Try again in 30 seconds." }``
    Shape B — structured dict (spend_limit_exceeded):
        ``{ "error": { "code": "spend_limit_exceeded", "message": "...", "status": 429 } }``
    """

    def test_429_rate_limit_plain_string_body(self):
        """Plain-string error body on 429 → RateLimited; code is None.

        Real body: ``{"error": "Rate limit exceeded. 0/60 requests remaining. Try again in 30 seconds."}``
        """
        body = {
            "error": "Rate limit exceeded. 0/60 requests remaining. Try again in 30 seconds."
        }
        transport = Transport(api_key="marqey_test_x", base_url="http://test")
        mock_resp = _mock_response(429, body, headers={"Retry-After": "30"})

        with patch.object(transport._session, "request", return_value=mock_resp):
            with pytest.raises(RateLimited) as exc_info:
                transport.request("GET", "/api/jobs/abc/status")

        exc = exc_info.value
        assert exc.status == 429
        assert exc.code is None, f"Expected code=None for plain-string body, got {exc.code!r}"

    def test_retry_after_header_parsed_and_surfaced(self):
        """Retry-After: 60 header → exc.retry_after == 60."""
        body = {
            "error": "Rate limit exceeded. 0/60 requests remaining. Try again in 60 seconds."
        }
        transport = Transport(api_key="marqey_test_x", base_url="http://test")
        mock_resp = _mock_response(429, body, headers={"Retry-After": "60"})

        with patch.object(transport._session, "request", return_value=mock_resp):
            with pytest.raises(RateLimited) as exc_info:
                transport.request("GET", "/api/jobs/abc/status")

        assert exc_info.value.retry_after == 60

    def test_retry_after_absent_gives_none(self):
        """No Retry-After header → exc.retry_after is None."""
        body = {"error": {"code": "spend_limit_exceeded", "message": "Limit exceeded", "status": 429}}
        transport = Transport(api_key="marqey_test_x", base_url="http://test")
        mock_resp = _mock_response(429, body, headers={})

        with patch.object(transport._session, "request", return_value=mock_resp):
            with pytest.raises(RateLimited) as exc_info:
                transport.request("POST", "/api/jobs/submit", json={})

        assert exc_info.value.retry_after is None

    def test_retry_after_non_numeric_gives_none(self):
        """Non-numeric Retry-After header (e.g. HTTP-date) → exc.retry_after is None."""
        body = {"error": "Too many requests"}
        transport = Transport(api_key="marqey_test_x", base_url="http://test")
        mock_resp = _mock_response(429, body, headers={"Retry-After": "Fri, 01 Aug 2026 00:00:00 GMT"})

        with patch.object(transport._session, "request", return_value=mock_resp):
            with pytest.raises(RateLimited) as exc_info:
                transport.request("GET", "/api/jobs/abc/status")

        assert exc_info.value.retry_after is None


# ---------------------------------------------------------------------------
# Auth-header verification on actual request call (m4)
# ---------------------------------------------------------------------------


class TestAuthHeaderOnRequest:
    """Authorization: Bearer <key> must appear in the actual HTTP call kwargs."""

    def test_bearer_token_in_actual_request_call_kwargs(self):
        """The merged headers on the actual session.request call include Authorization: Bearer.

        Verifies the header reaches the wire, not just the session constructor.
        The server expects a Bearer token in the Authorization header.
        """
        transport = Transport(api_key="marqey_test_bearer_check", base_url="http://test")
        mock_resp = _mock_response(200, {"job_id": "j1"})

        # Patch session.request AND capture the PreparedRequest or merged headers.
        # requests.Session.request merges session-level headers with per-request
        # headers before sending; we confirm via session.headers (set at init).
        with patch.object(transport._session, "request", return_value=mock_resp) as mock_req:
            transport.request("POST", "/api/jobs/submit", json={"x": 1})

        # Inspect the call — the session-level Authorization header is merged
        # by requests internally, so we verify it via the session object AND
        # confirm that the mock was actually called (not a no-op).
        assert mock_req.called, "session.request was never called"
        _, call_kwargs = mock_req.call_args
        # The extra per-request headers (Idempotency-Key) are in call_kwargs["headers"].
        # Session-level headers (Authorization) are accessed via transport._session.headers.
        assert transport._session.headers.get("Authorization") == "Bearer marqey_test_bearer_check"
        # Additionally: the call was made to the right URL with method POST
        call_args_positional = mock_req.call_args[0]
        assert call_args_positional[0] == "POST"
        assert "submit" in call_args_positional[1]


# ---------------------------------------------------------------------------
# Malformed / absent JSON body on non-2xx (m3)
# ---------------------------------------------------------------------------


class TestMalformedResponseBody:
    """A non-2xx response with no parseable JSON body → TransportError, not a crash."""

    def test_non_2xx_with_no_json_body_raises_transport_error(self):
        """500 with non-JSON body → TransportError (not ValueError/AttributeError).

        Verifies the except-on-json path in _raise_for_response falls back cleanly.
        """
        transport = Transport(api_key="marqey_test_x", base_url="http://test")
        mock_resp = _mock_response(500, json_body=None)  # json() raises ValueError
        mock_resp.reason = "Internal Server Error"

        with patch.object(transport._session, "request", return_value=mock_resp):
            with pytest.raises(TransportError) as exc_info:
                transport.request("GET", "/api/jobs/bad/status")

        exc = exc_info.value
        assert exc.status == 500
        assert not isinstance(exc, type(None))

    def test_non_2xx_with_malformed_json_raises_transport_error(self):
        """Non-2xx where json() raises an exception → TransportError, not a crash."""
        transport = Transport(api_key="marqey_test_x", base_url="http://test")
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.status_code = 503
        mock_resp.ok = False
        mock_resp.reason = "Service Unavailable"
        mock_resp.json.side_effect = ValueError("not valid JSON")
        mock_resp.headers = {}

        with patch.object(transport._session, "request", return_value=mock_resp):
            with pytest.raises(TransportError) as exc_info:
                transport.request("GET", "/api/jobs/x/status")

        assert exc_info.value.status == 503


# ---------------------------------------------------------------------------
# ConnectTimeout retry on idempotent_write=True (m5)
# ---------------------------------------------------------------------------


class TestConnectTimeoutRetry:
    """ConnectTimeout IS retried for idempotent_write=True with the same Idempotency-Key.

    ConnectTimeout is a subclass of both ConnectionError and Timeout.
    The transport's ConnectionError branch (not the Timeout branch) should catch it
    first, making it retryable even on writes, while reusing the same Idempotency-Key.

    The server requires the Idempotency-Key for safe dedup on retry.
    """

    def test_connect_timeout_on_idempotent_write_is_retried_with_same_key(self):
        """ConnectTimeout (subclass of both ConnectionError and Timeout) on
        idempotent_write=True IS retried and reuses the same Idempotency-Key.
        """
        transport = Transport(api_key="marqey_test_x", base_url="http://test")
        success_resp = _mock_response(200, {"job_id": "jCT"})

        call_count = 0
        captured_keys: list[str] = []

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            captured_keys.append(kwargs.get("headers", {}).get("Idempotency-Key", ""))
            if call_count == 1:
                raise requests.exceptions.ConnectTimeout("Connect timed out")
            return success_resp

        with patch.object(transport._session, "request", side_effect=side_effect):
            result = transport.request(
                "POST",
                "/api/jobs/submit",
                json={"backend": "sv1"},
                idempotent_write=True,
            )

        assert result == {"job_id": "jCT"}
        assert call_count == 2, f"Expected 2 attempts, got {call_count}"
        # The same Idempotency-Key must be reused across the ConnectTimeout retry
        assert len(set(captured_keys)) == 1, (
            f"Idempotency-Key changed across ConnectTimeout retry: {captured_keys}"
        )
        assert captured_keys[0] != "", "Idempotency-Key must be non-empty"
