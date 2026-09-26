"""HTTP transport layer for the Marqov Platform client.

Handles authentication, error mapping, conditional retry with idempotency-key
reuse, and long-poll parameter injection.

Error codes, status codes, and param names below follow the platform's HTTP API contract.
"""

from __future__ import annotations

import datetime
import os
import time
import typing
import uuid
from email.utils import parsedate_to_datetime
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
_DEFAULT_BASE_URL = "https://app.marqov.ai"

#: Maximum retry attempts for retryable failures.
_MAX_RETRIES = 3

#: Initial backoff (seconds) between retries; doubles each attempt.
_RETRY_BACKOFF_BASE = 0.5

#: HTTP statuses retried for idempotent requests (GETs and idempotent writes).
#: These are transient server-side or gateway conditions, not client errors.
_RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504})

#: Substrings that identify a connect-phase failure when urllib3's exception
#: classes cannot be imported. Only used as a fallback: an unmatched message is
#: treated as ambiguous, which is the safe direction for a write.
_CONNECT_PHASE_MARKERS = (
    "failed to establish a new connection",
    "connection refused",
    "name or service not known",
    "nodename nor servname",
    "temporary failure in name resolution",
    "getaddrinfo failed",
    "connecttimeouterror",
    "newconnectionerror",
)


def _remaining_backoff_budget(attempt: int) -> float:
    """Total default backoff (seconds) still to come after `attempt` failed.

    ``attempt`` is zero-based, so after attempt 0 of 3 the transport would
    otherwise sleep ``_RETRY_BACKOFF_BASE`` and then ``2 * _RETRY_BACKOFF_BASE``.
    """
    return float(
        sum(
            _RETRY_BACKOFF_BASE * (2 ** (i - 1))
            for i in range(attempt + 1, _MAX_RETRIES)
        )
    )


def _is_connect_phase_failure(exc: BaseException) -> bool:
    """Report whether `exc` failed before the request bytes were sent.

    ``requests`` collapses connect-phase failures (refused, DNS, connect
    timeout) and post-send failures (the peer closing the socket, a reset
    mid-response) onto the same :class:`requests.exceptions.ConnectionError`,
    so the wrapped urllib3 cause is what distinguishes them. A connect-phase
    failure provably never reached the server; anything else is ambiguous.

    urllib3 is an install-time dependency of ``requests`` but is not declared
    by this package, so it is imported defensively and the exception's string
    form is used as a fallback.
    """
    # requests raises ConnectTimeout for a urllib3 ConnectTimeoutError that is
    # not a NewConnectionError, so the class itself is already conclusive.
    if isinstance(exc, requests.exceptions.ConnectTimeout):
        return True

    try:
        # NewConnectionError subclasses ConnectTimeoutError, so one target
        # covers both.
        from urllib3.exceptions import ConnectTimeoutError
    except ImportError:  # pragma: no cover - urllib3 ships with requests
        lowered = str(exc).lower()
        return any(marker in lowered for marker in _CONNECT_PHASE_MARKERS)

    # Walk the wrapped causes: requests passes the urllib3 error as an arg,
    # and a MaxRetryError carries the underlying failure on ``.reason``.
    # Implicit ``__context__`` is deliberately not followed: an exception raised
    # while handling an earlier connect-phase failure (for example a reset on a
    # later attempt) must not inherit its "never reached the server" verdict.
    seen: set[int] = set()
    pending: list[BaseException] = [exc]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, ConnectTimeoutError):
            return True
        candidates = (
            *getattr(current, "args", ()),
            getattr(current, "reason", None),
            current.__cause__,
        )
        for candidate in candidates:
            if isinstance(candidate, BaseException):
                pending.append(candidate)
    return False


def _parse_retry_after(raw: str) -> int | None:
    """Parse a ``Retry-After`` header value into whole seconds.

    RFC 7231 §7.1.3 allows two forms:
      - delta-seconds — a plain integer, e.g. ``"30"``.
      - HTTP-date — e.g. ``"Wed, 21 Oct 2026 07:28:00 GMT"``, converted to a
        delta against the current time and clamped to ``>= 0`` (a date in
        the past never yields a negative wait).

    Returns ``None`` if `raw` is neither a valid delta-seconds value nor a
    parseable HTTP-date.
    """
    try:
        return int(raw)
    except (ValueError, TypeError):
        pass

    try:
        target = parsedate_to_datetime(raw)
    except (ValueError, TypeError):
        return None
    # A malformed date without timezone info parses to a naive datetime;
    # without a timezone we can't compute a reliable delta against an
    # aware "now", so treat it as unparseable.
    if target is None or target.tzinfo is None:
        return None

    delta = target - datetime.datetime.now(datetime.timezone.utc)
    return max(0, int(delta.total_seconds()))


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
        # Server expects a Bearer token in the Authorization header.
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
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Execute an HTTP request against the platform API.

        For write requests an ``Idempotency-Key`` header is generated once per
        call and **reused across all retries** so the server can safely dedupe
        replays.

        Idempotent requests (GETs and ``idempotent_write=True`` writes) are
        also retried on the transient statuses 429, 502, 503 and 504, using the
        same doubling backoff.  A 429's ``Retry-After`` is honoured when it
        parses and asks for no longer than the remaining backoff budget.  Once
        attempts are exhausted the mapped exception for the last response is
        raised, so a rate limit still surfaces as ``RateLimited`` with its
        ``retry_after``.

        Args:
            method:           HTTP method (``"GET"``, ``"POST"``, …).
            path:             URL path appended to ``base_url``
                              (e.g. ``"/api/jobs/submit"``).
            json:             Request body serialised as JSON.
            params:           URL query parameters.
            idempotent_write: When ``True`` the request is a write that is
                              safe to retry only on failures that provably never
                              reached the server (connect-phase failures:
                              refused, DNS, connect timeout).  ``Timeout`` /
                              ``ReadTimeout`` and post-send connection failures
                              are **not** retried because the server may have
                              processed the request; the raised
                              ``TransportError`` carries the idempotency key so
                              the caller can reconcile.  It also opts the
                              request into the retryable-status policy below.
                              When ``False`` any transport failure is retried
                              for reads, while a write is never retried on a
                              retryable status.
            idempotency_key:  Caller-chosen ``Idempotency-Key`` for a write.
                              Sent verbatim and reused across retries.  When
                              omitted a fresh UUID4 is generated for this call.
                              Ignored for reads.
            wait:             If given, appended as the ``wait`` query
                              parameter (long-poll seconds).  The server reads
                              the ``"wait"`` query param on the status endpoint.

        Returns:
            Decoded JSON response body as a plain ``dict``.

        Raises:
            AuthenticationError:         HTTP 401.
            PermissionTierError:         HTTP 403 / ``permission_denied``.
            PaidBackendNotSupportedYet:  HTTP 422 ``analysis_required``.
            RateLimited:                 HTTP 429. ``retry_after`` is parsed
                                         from the ``Retry-After`` header when
                                         present — either delta-seconds or an
                                         RFC 7231 HTTP-date (converted to a
                                         delta, clamped to ``>= 0``) — else
                                         ``None``. The header is not required
                                         for this to raise.
            BackendUnavailable:          HTTP 422 ``backend_unknown`` /
                                         ``backend_retired``.
            InvalidProgram:              HTTP 400 or 422 ``validation_error``.
            MarqovPlatformError:         Any other non-2xx with an error body
                                         (``code`` preserved; never coerced to
                                         ``TransportError``).
            TransportError:              Ambiguous network failure (e.g.
                                         ``Timeout`` on a write) or an HTTP
                                         error with no structured body.
        """
        url = self._base_url + path

        # Merge wait into query params.
        # Server reads the "wait" query param on the status endpoint.
        merged_params: dict[str, Any] = dict(params or {})
        if wait is not None:
            merged_params["wait"] = wait

        # One idempotency key per call (the caller's, or a fresh UUID4); reused
        # across retries so the server can dedupe replays.
        if method.upper() in ("GET", "HEAD", "OPTIONS"):
            idempotency_key = None
        elif idempotency_key is None:
            idempotency_key = str(uuid.uuid4())

        extra_headers: dict[str, str] = {}
        if idempotency_key is not None:
            extra_headers["Idempotency-Key"] = idempotency_key

        # A GET/HEAD/OPTIONS carries no key and is idempotent by method; a
        # write is only replayable when the caller marked it as such.
        idempotent_request = idempotent_write or idempotency_key is None

        last_exc: Exception | None = None
        last_resp: requests.Response | None = None
        next_sleep: float | None = None
        for attempt in range(_MAX_RETRIES):
            if attempt > 0:
                default_backoff = _RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                time.sleep(next_sleep if next_sleep is not None else default_backoff)
            next_sleep = None

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
                # Connect-phase failures (refused, DNS, connect timeout)
                # provably never reached the server and stay freely retryable.
                # Everything else in this class (the peer closing the socket
                # after the body was written, a reset mid-response) is
                # ambiguous: the server may already hold the request.
                if idempotent_write and not _is_connect_phase_failure(exc):
                    raise TransportError(
                        "Connection failed after the request may have reached "
                        "the server and was not retried, to avoid a duplicate "
                        f"write ({exc!r})",
                        idempotency_key=idempotency_key,
                    ) from exc
                last_exc = exc
                # This attempt failed before any response, so an earlier
                # retryable response is no longer what the caller should see.
                last_resp = None
                continue
            except requests.exceptions.Timeout as exc:  # ReadTimeout is a subclass of Timeout; listed for documentation clarity
                # Ambiguous: the server may have received and processed the
                # request.  Retry only for idempotent reads.
                if idempotent_write:
                    raise TransportError(
                        f"Request timed out and was not retried ({exc!r})",
                        idempotency_key=idempotency_key,
                    ) from exc
                # Reads: retryable
                last_exc = exc
                last_resp = None
                continue
            except requests.exceptions.RequestException as exc:
                # Other transport failure: treat like Timeout (ambiguous).
                if idempotent_write:
                    raise TransportError(
                        f"Transport error: {exc!r}",
                        idempotency_key=idempotency_key,
                    ) from exc
                last_exc = exc
                last_resp = None
                continue

            # --- HTTP response received -------------------------------------
            if resp.ok:
                return resp.json()

            # Transient server-side statuses are retried for idempotent
            # requests only; a write that was not marked idempotent must not
            # be replayed even though the server answered.
            if (
                idempotent_request
                and resp.status_code in _RETRYABLE_STATUS_CODES
                and attempt < _MAX_RETRIES - 1
            ):
                last_resp = resp
                last_exc = None
                if resp.status_code == 429:
                    next_sleep = self._retry_after_sleep(resp, attempt)
                continue

            # Map non-2xx → exception
            return self._raise_for_response(resp)

        # All retries exhausted
        if last_resp is not None:
            # Raise the mapped exception for the final response so, for
            # example, RateLimited still surfaces with its retry_after.
            return self._raise_for_response(last_resp)
        raise TransportError(
            f"Request failed after {_MAX_RETRIES} attempts: {last_exc!r}"
        ) from last_exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _retry_after_sleep(resp: requests.Response, attempt: int) -> float | None:
        """Return the ``Retry-After`` wait to honour before the next attempt.

        The header is honoured only when it parses and asks for no more than
        the transport would otherwise spend on its remaining retries; a longer
        wait belongs to the caller, which still receives the parsed value on
        :class:`~marqov.platform.errors.RateLimited` once attempts run out.
        Returns ``None`` to fall back to the default doubling backoff.
        """
        raw = resp.headers.get("Retry-After")
        if raw is None:
            return None
        parsed = _parse_retry_after(raw)
        if parsed is None or parsed > _remaining_backoff_budget(attempt):
            return None
        return float(parsed)

    def _raise_for_response(self, resp: requests.Response) -> typing.NoReturn:
        """Parse an error response and raise the appropriate exception.

        Guaranteed to raise — never returns normally.

        Error responses use this JSON envelope:
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

        # --- Error mapping -------------------------------------------------

        # HTTP 401 → AuthenticationError
        # Server returns error code "unauthorized" with HTTP 401.
        if status == 401 or code == "unauthorized":
            raise AuthenticationError(message, code=code, status=status)

        # HTTP 403 / permission_denied → PermissionTierError
        # Server returns error code "permission_denied" with HTTP 403.
        if status == 403 or code == "permission_denied":
            raise PermissionTierError(message, code=code, status=status)

        # HTTP 422 analysis_required → PaidBackendNotSupportedYet
        # Server returns error code "analysis_required" with HTTP 422.
        if code == "analysis_required":
            raise PaidBackendNotSupportedYet(message, code=code, status=status)

        # HTTP 429 → RateLimited (when Retry-After present; always for 429)
        # Server returns HTTP 429 with a Retry-After header on rate limit.
        if status == 429:
            _retry_after_raw = resp.headers.get("Retry-After")
            _retry_after: int | None = None
            if _retry_after_raw is not None:
                _retry_after = _parse_retry_after(_retry_after_raw)
            raise RateLimited(message, code=code, status=status, retry_after=_retry_after)

        # backend_unknown / backend_retired → BackendUnavailable
        # Server returns error code "backend_unknown" with HTTP 422.
        # Server returns error code "backend_retired" with HTTP 422.
        if code in ("backend_unknown", "backend_retired"):
            raise BackendUnavailable(message, code=code, status=status)

        # validation_error → InvalidProgram
        # Server returns error code "validation_error" with HTTP 400 or 422.
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
