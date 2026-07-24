"""Dataclass models for the Marqov Platform client.

All types here are plain :mod:`dataclasses` — no Pydantic dependency.
Field names for :class:`Backend` will be reconciled against the real
``/api/backends`` response shape in a later task; the fields listed here
are the authoritative first-pass set.

.. note::
    This module is private (``_models``).  Public re-exports are added in
    Task 6 via ``marqov/platform/__init__.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# JobStatus
# ---------------------------------------------------------------------------


class JobStatus(str, Enum):
    """Canonical job lifecycle states returned by the platform.

    Values are the lowercase strings used by the server API::

        JobStatus.COMPLETED == "completed"  # True — str-enum comparison

    **Comparison-only enum.**  Server status strings are kept as plain
    ``str`` throughout the client; ``JobStatus(raw_string)`` is **never
    called** at runtime because it raises :class:`ValueError` on any status
    value not listed here (e.g. future server additions).  Use
    :func:`is_terminal` for terminal-state checks on raw strings.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    DISPATCH_FAILED = "dispatch_failed"


def is_terminal(status: str) -> bool:
    """Return ``True`` if *status* represents a terminal job state.

    Works on plain ``str`` values directly from the server — never calls
    ``JobStatus(status)``, so unknown/future status strings return ``False``
    rather than raising::

        is_terminal("completed")  # True
        is_terminal("running")    # False
        is_terminal("archived")   # False — unknown, no exception

    Args:
        status: Raw job status string from the server API.

    Returns:
        ``True`` iff the status is one of the four server-side terminal
        states — ``completed``, ``failed``, ``cancelled``, or
        ``dispatch_failed``.
    """
    return status in {"completed", "failed", "cancelled", "dispatch_failed"}


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


@dataclass
class Backend:
    """A quantum backend available on the Marqov Platform.

    Field names mirror the ``/api/backends`` response; additional fields
    returned by the server are captured in :attr:`extra` and reconciled in
    a later task.

    Attributes:
        slug:                   Machine-readable backend identifier
                                (e.g. ``"sv1"``).
        name:                   Human-readable display name
                                (e.g. ``"StateVec Simulator"``).
        provider:               Backend provider name
                                (e.g. ``"marqov"``, ``"ibm"``, ``"aws"``).
        device_type:            Category of device
                                (e.g. ``"simulator"``, ``"qpu"``).
        status:                 Current operational status string from the
                                server (e.g. ``"online"``, ``"maintenance"``).
        is_available:           Whether jobs can currently be submitted.
        pricing:                Pricing metadata dict as returned by the
                                server.  Shape is server-defined.
        supported_program_types: List of accepted program type identifiers
                                (e.g. ``["qasm3"]``).
        extra:                  Overflow dict for any additional fields in
                                the server response not listed above.
    """

    slug: str
    name: str
    provider: str
    device_type: str
    status: str
    is_available: bool
    pricing: dict
    supported_program_types: list
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# PlatformInfo
# ---------------------------------------------------------------------------


@dataclass
class PlatformInfo:
    """Version metadata returned by the platform health/info endpoint.

    Attributes:
        sdk_version: Version of the ``marqov`` SDK on the client side.
        api_version: Version string of the server-side platform API.
    """

    sdk_version: str
    api_version: str


# ---------------------------------------------------------------------------
# PlatformResult
# ---------------------------------------------------------------------------


@dataclass
class PlatformResult:
    """Measurement result for a completed platform job.

    Wraps the raw JSON response body from the job-result endpoint.  The
    interface is intentionally consistent with
    :class:`marqov.executors.base.ExecutionResult` so SDK users see a
    familiar surface:

    * ``ExecutionResult.counts`` is a required field (always ``dict``).
    * ``PlatformResult.counts`` is an optional property (``None`` when the
      server response omits the key) to handle server variance gracefully.
    * Both ``ExecutionResult.probabilities`` and ``PlatformResult.probabilities``
      are ``@property`` (accessed without ``()``) — a uniform surface across
      result types.  ``PlatformResult.probabilities`` returns ``{}`` when
      ``counts`` is ``None`` (server variance) rather than raising.

    Attributes:
        raw: The unmodified JSON response dict from the server.
    """

    raw: dict

    @property
    def counts(self) -> dict[str, int] | None:
        """Measurement outcome counts, or ``None`` if not present in the result.

        Returns:
            Dict mapping bitstrings to integer counts
            (e.g. ``{"00": 512, "11": 488}``), or ``None`` if the server
            response did not include a ``"counts"`` key.
        """
        return self.raw.get("counts")

    @property
    def probabilities(self) -> dict[str, float]:
        """Probabilities derived from :attr:`counts`.

        A ``@property`` mirroring
        :attr:`marqov.executors.base.ExecutionResult.probabilities`; returns an
        empty dict when counts are absent or all-zero rather than raising.

        Returns:
            Dict mapping bitstrings to probabilities in ``[0.0, 1.0]``.
            Empty dict if ``counts`` is ``None`` or sums to zero.
        """
        c = self.counts
        if not c:
            return {}
        total = sum(c.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in c.items()}
