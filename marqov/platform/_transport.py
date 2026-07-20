"""HTTP transport layer for the Marqov Platform client.

Handles authentication, error mapping, conditional retry with idempotency-key
reuse, and long-poll parameter injection.

Source citations (real platform routes read before encoding any contract):
  - Auth/key format:
      marqov-platform/…/lib/auth/api-key.ts:40 — ``Bearer marqey_…``
  - Submit error codes / status codes:
      marqov-platform/…/app/api/jobs/submit/route.ts — ``analysis_required``
      (422), ``spend_limit_exceeded`` (429), ``permission_denied`` (403),
      ``unauthorized`` (401), 402 budget errors.
      ``backend_unknown`` (422): submit/route.ts:211-215
      ``backend_retired`` (422): submit/route.ts:228-232
      ``validation_error`` (400/422): submit/route.ts:105, 241, 494-500
      ``is_available==false`` (400, plain-string body): submit/route.ts:359-362
  - Status long-poll ``wait`` param name:
      marqov-platform/…/app/api/jobs/[id]/status/route.ts:70 —
      ``request.nextUrl.searchParams.get("wait")``.
  - Rate limit → 429 + ``Retry-After``:
      marqov-platform/…/app/api/jobs/[id]/status/route.ts:48-58 and
      marqov-platform/…/app/api/jobs/submit/route.ts:135-146.
"""

from __future__ import annotations

import os
import time
import typing
import uuid
from typing import Any

import requests
import requests.exceptions

from .errors import (
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
# Constants
# ---------------------------------------------------------------------------

#: Production API base URL — override with ``MARQOV_PLATFORM_URL`` env var or
#: the ``base_url`` constructor argument.
_DEFAULT_BASE_URL = "https://platform.marqov.com"

#: Maximum retry attempts for retryable failures.
_MAX_RETRIES = 3

#: Initial backoff (seconds) between retries; doubles each attempt.
_RETRY_BACKOFF_BASE = 0.5


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


class Transport:
    """Low-level HTTP transport for the Marqov Platform API.

    Manages a :class:`requests.Session` that attaches ``Authorization: Bearer
    <key>`` to every request.  All platform errors are mapped to the exception
    hierarchy defined in :mod:`marqov.platform.errors`.

    The API key is **never written to disk**.

    Args:
        api_key:  Marqov Platform API key (``marqey_live_…`` or
                  ``marqey_test_…``).  Falls back to the
                  ``MARQOV_PLATFORM_KEY`` environment variable when ``None``.
        base_url: Override the default production endpoint.  Falls back to
                  ``MARQOV_PLATFORM_URL`` env var, then the built-in default.
        timeout:  Per-request timeout in seconds (default 30 s).

    Raises:
        AuthenticationError: If no API key is found (neither argument nor env
            var).
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        # --- Resolve key (memory-only; never touches disk) ------------------
        resolved_key = api_key or os.environ.get("MARQOV_PLATFORM_KEY")
        if not resolved_key:
            raise AuthenticationError(
                "No Marqov Platform API key supplied.  "
                "Pass api_key= or set MARQOV_PLATFORM_KEY."
            )
        # --- Resolve base URL -----------------------------------------------
        resolved_url = (
            base_url
            or os.environ.get("MARQOV_PLATFORM_URL")
            or _DEFAULT_BASE_URL
        )
        self._base_url = resolved_url.rstrip("/")
        self._timeout = timeout

        # --- Build session --------------------------------------------------
        self._session = requests.Session()
        # Source: api-key.ts:40 — platform checks ``authHeader.startsWith("Bearer marqey_")``.
        self._session.headers.update({"Authorization": f"Bearer {resolved_key}"})

    @property
    def timeout(self) -> float:
        """Per-request timeout in seconds (read-only).

        Exposed so :class:`~marqov.platform.job.Job` can compute the
        ``wait`` long-poll budget: the server-side wait must be strictly
        shorter than this per-request timeout.
        """
        return self._timeout

    def request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: dict[str, Any] | None = None,
        idempotent_write: bool = False,
        wait: int | None = None,
    ) -> dict[str, Any]:
        """Execute an HTTP request against the platform API.

        For write requests an ``Idempotency-Key`` header is generated once per
        call and **reused across all retries** so the server can safely dedupe
        replays.

        Args:
            method:           HTTP method (``"GET"``, ``"POST"``, …).
            path:             URL path appended to ``base_url``
                              (e.g. ``"/api/jobs/submit"``).
            json:             Request body serialised as JSON.
            params:           URL query parameters.
            idempotent_write: When ``True`` the request is a write that is
                              safe to retry only on failures that provably never
                              reached the server (``ConnectionError`` — refused /
                              DNS).  ``Timeout`` / ``ReadTimeout`` are **not**
                              retried because the server may have processed the
                              request.  When ``False`` (reads / GETs) any
                              transport failure is retried.
            wait:             If given, appended as the ``wait`` query
                              parameter (long-poll seconds).  Name confirmed
                              against ``status/route.ts:70``.

        Returns:
            Decoded JSON response body as a plain ``dict``.

        Raises:
            AuthenticationError:         HTTP 401.
            PermissionTierError:         HTTP 403 / ``permission_denied``.
            PaidBackendNotSupportedYet:  HTTP 422 ``analysis_required``.
            RateLimited:                 HTTP 429 with ``Retry-After`` header.
            MarqovPlatformError:         Any other non-2xx with an error body
                                         (``code`` preserved; never coerced to
                                         ``TransportError``).
            TransportError:              Ambiguous network failure (e.g.
                                         ``Timeout`` on a write) or an HTTP
                                         error with no structured body.
        """
        url = self._base_url + path

        # Merge wait into query params.
        # Source: status/route.ts:70 — ``searchParams.get("wait")``
        merged_params: dict[str, Any] = dict(params or {})
        if wait is not None:
            merged_params["wait"] = wait

        # Generate one idempotency key per call; reuse across retries.
        idempotency_key: str | None = None
        if method.upper() not in ("GET", "HEAD", "OPTIONS"):
            idempotency_key = str(uuid.uuid4())

        extra_headers: dict[str, str] = {}
        if idempotency_key is not None:
            extra_headers["Idempotency-Key"] = idempotency_key

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            if attempt > 0:
                time.sleep(_RETRY_BACKOFF_BASE * (2 ** (attempt - 1)))

            try:
                resp = self._session.request(
                    method,
                    url,
                    json=json,
                    params=merged_params or None,
                    headers=extra_headers,
                    timeout=self._timeout,
                )
            except requests.exceptions.ConnectionError as exc:
                # Provably never reached the server — safe to retry regardless
                # of write/read mode.
                last_exc = exc
                continue
            except requests.exceptions.Timeout as exc:  # ReadTimeout is a subclass of Timeout; listed for documentation clarity
                # Ambiguous: the server may have received and processed the
                # request.  Retry only for idempotent reads.
                if idempotent_write:
                    raise TransportError(
                        f"Request timed out and was not retried ({exc!r})"
                    ) from exc
                # Reads: retryable
                last_exc = exc
                continue
            except requests.exceptions.RequestException as exc:
                # Other transport failure: treat like Timeout (ambiguous).
                if idempotent_write:
                    raise TransportError(f"Transport error: {exc!r}") from exc
                last_exc = exc
                continue

            # --- HTTP response received -------------------------------------
            if resp.ok:
                return resp.json()

            # Map non-2xx → exception
            return self._raise_for_response(resp)

        # All retries exhausted
        raise TransportError(
            f"Request failed after {_MAX_RETRIES} attempts: {last_exc!r}"
        ) from last_exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _raise_for_response(self, resp: requests.Response) -> typing.NoReturn:
        """Parse an error response and raise the appropriate exception.

        Guaranteed to raise — never returns normally.

        Error envelope shape (source: error-envelope.ts):
            ``{ "error": { "code": "...", "message": "...", "status": ... } }``
        """
        status = resp.status_code

        # --- Structured error body -----------------------------------------
        try:
            body = resp.json()
        except Exception:
            body = {}

        error_obj = body.get("error") if isinstance(body, dict) else None
        code: str | None = None
        message: str = resp.reason or f"HTTP {status}"

        if isinstance(error_obj, dict):
            code = error_obj.get("code")
            message = error_obj.get("message", message)
        elif isinstance(error_obj, str):
            message = error_obj

        # --- Error mapping (source: submit/route.ts + status/route.ts) -----

        # HTTP 401 → AuthenticationError
        # Source: require-auth.ts:70 — ``apiError("unauthorized", "Unauthorized", 401)``
        if status == 401 or code == "unauthorized":
            raise AuthenticationError(message, code=code, status=status)

        # HTTP 403 / permission_denied → PermissionTierError
        # Source: submit/route.ts:175 — ``apiError("permission_denied", …, 403)``
        if status == 403 or code == "permission_denied":
            raise PermissionTierError(message, code=code, status=status)

        # HTTP 422 analysis_required → PaidBackendNotSupportedYet
        # Source: submit/route.ts:611-617 — code "analysis_required", status 422
        if code == "analysis_required":
            raise PaidBackendNotSupportedYet(message, code=code, status=status)

        # HTTP 429 → RateLimited (when Retry-After present; always for 429)
        # Source: status/route.ts:48-58 + submit/route.ts:135-146 — 429 + Retry-After header
        if status == 429:
            _retry_after_raw = resp.headers.get("Retry-After")
            _retry_after: int | None = None
            if _retry_after_raw is not None:
                try:
                    _retry_after = int(_retry_after_raw)
                except (ValueError, TypeError):
                    _retry_after = None
            raise RateLimited(message, code=code, status=status, retry_after=_retry_after)

        # backend_unknown / backend_retired → BackendUnavailable
        # Source: submit/route.ts:211-215 — apiError("backend_unknown", …, 422)
        # Source: submit/route.ts:228-232 — apiError("backend_retired", …, 422)
        if code in ("backend_unknown", "backend_retired"):
            raise BackendUnavailable(message, code=code, status=status)

        # validation_error → InvalidProgram
        # Source: submit/route.ts:105 — apiError("validation_error", "Invalid JSON body", 400)
        # Source: submit/route.ts:241 — apiError("validation_error", "Idempotency-Key …", 400)
        # Source: submit/route.ts:494-500 — finalize(422, { error: { code: "validation_error", … } })
        if code == "validation_error":
            raise InvalidProgram(message, code=code, status=status)

        # All other structured errors → base MarqovPlatformError (code preserved)
        # NEVER coerce an unknown code to TransportError.
        if code is not None:
            raise MarqovPlatformError(message, code=code, status=status)

        # Unstructured non-2xx
        raise TransportError(
            f"Unexpected HTTP {status}: {message}", code=None, status=status
        )
