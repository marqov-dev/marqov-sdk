"""Exception hierarchy for the Marqov Platform client.

All exceptions inherit from :class:`MarqovPlatformError`.  Import them
directly::

    from marqov.platform.errors import AuthenticationError, JobFailed

These classes are the API-reference source; the docstrings below are the
authoritative descriptions.
"""

from __future__ import annotations


class MarqovPlatformError(Exception):
    """Base exception for all Marqov Platform client errors.

    Attributes:
        message: Human-readable description of what went wrong.
        code:    Optional machine-readable error code returned by the server
                 (e.g. ``"auth/token-expired"``).
        status:  Optional HTTP status code associated with the response.

    ``str()`` returns the message, and the code if present::

        raise MarqovPlatformError("bad key", code="auth/invalid-key", status=401)
        # str(e) → "bad key [auth/invalid-key]"
    """

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status

    def __str__(self) -> str:
        if self.code:
            return f"{self.message} [{self.code}]"
        return self.message


class AuthenticationError(MarqovPlatformError):
    """Raised when the API key is missing, invalid, or has been revoked.

    Check that ``MARQOV_API_KEY`` is set and that the key is still active
    in the Marqov dashboard.
    """


class PermissionTierError(MarqovPlatformError):
    """Raised when the caller's plan does not permit access to a resource.

    For example, attempting to use a QPU backend on a free-tier account.
    Upgrade your plan or contact support to gain access.
    """


class BackendUnavailable(MarqovPlatformError):
    """Raised when the requested backend is known but currently offline.

    This may be transient (maintenance window).  Retry with exponential
    back-off or choose an alternative backend.
    """


class PaidBackendNotSupportedYet(MarqovPlatformError):
    """Free-path guard raised in v1.0 when a paid backend is requested.

    In v1.0 the platform client routes all jobs through the free execution
    path.  Any attempt to target a paid QPU backend raises this error so
    callers fail fast rather than incurring unexpected charges.

    .. deprecated::
        This exception is **retained but deprecated** from v1.1 onward.
        Once paid backends are enabled the guard is removed from the hot
        path, but this class is **never deleted** so that existing
        ``except PaidBackendNotSupportedYet`` blocks continue to import
        and function correctly.  Removing it would be a silent breaking
        change for callers.
    """


class InvalidProgram(MarqovPlatformError):
    """Raised when the submitted quantum program is rejected by the server.

    Possible causes include unsupported gate sets, too many qubits, or a
    malformed QASM3 payload.  Inspect :attr:`message` for details from the
    server.
    """


class JobFailed(MarqovPlatformError):
    """Raised when a job reaches the ``FAILED`` terminal state.

    The server-side error reason (if available) is surfaced in
    :attr:`message`.  The original job ID, if known, may appear in
    :attr:`code`.
    """


class RateLimited(MarqovPlatformError):
    """Raised when the server returns HTTP 429 (Too Many Requests).

    Back off and retry.  The :attr:`status` attribute will be ``429``.
    """


class TransportError(MarqovPlatformError):
    """Raised for low-level network or HTTP-layer failures.

    This wraps connection timeouts, TLS errors, and unexpected non-2xx
    responses that do not map to a more specific error subclass.
    """
