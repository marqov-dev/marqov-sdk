"""Job handle for the Marqov Platform client.

Provides :class:`Job` — a lightweight handle returned after submitting a
quantum job to the platform.  It supports status polling, blocking result
retrieval (with server long-poll + client-side exponential backoff), job
cancellation, and cost introspection.

Observable API contract notes:

  - Status endpoint response: the server returns a fixed set of fields
    (``id``, ``status``, ``backend``, ``created_at``, ``updated_at``,
    ``estimated_cost_usd``, ``result``) and does **NOT** include an
    ``error_message`` field.  On a ``failed`` terminal status the client
    raises :class:`~marqov.platform.errors.JobFailed` with the best
    available message from ``result`` if it contains an ``"error"`` key,
    otherwise a generic message.
  - Server ``wait`` param: the server accepts a ``wait`` query parameter
    (seconds) and caps it at 22 seconds.  Values below 0 are clamped to 0.
  - Terminal states: ``"completed"``, ``"failed"``, ``"cancelled"``,
    ``"dispatch_failed"``.
  - Cancel endpoint: **DOES NOT EXIST** as a user-facing route (§11 TBC).
      :meth:`Job.cancel` is implemented against the **mocked path**
      ``/api/jobs/{id}/cancel`` (POST).  Once the platform ships a real
      cancel endpoint this path must be updated.

.. note::
    ``dispatch_failed`` is one of the four terminal states (with ``completed``,
    ``failed``, ``cancelled``) recognised by
    :func:`~marqov.platform._models.is_terminal`.  The :meth:`Job.result`
    polling loop treats both ``failed`` and ``dispatch_failed`` as failures
    (raising :class:`~marqov.platform.errors.JobFailed`).
"""

from __future__ import annotations

import math
import random
import time
from typing import TYPE_CHECKING

from ._models import PlatformResult, is_terminal
from .errors import JobFailed

if TYPE_CHECKING:
    from ._transport import Transport

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The server caps the long-poll wait at 22 seconds.
_SERVER_MAX_WAIT_SECONDS = 22

#: Safety margin (seconds) subtracted from the transport's per-request
#: timeout when computing the ``wait`` param.  This leaves headroom for
#: auth, DB fetch, and network round-trip so the server response arrives
#: before the HTTP request itself times out.
_WAIT_MARGIN_SECONDS = 5

#: Exponential-backoff cap (seconds) between client polls.
_MAX_BACKOFF_SECONDS = 10.0


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------


class Job:
    """Handle for a submitted Marqov Platform job.

    Returned by the client after successfully submitting a job.  Provides
    non-blocking :meth:`status` queries, blocking :meth:`result` retrieval
    (using the server's long-poll ``wait`` parameter), job :meth:`cancel`,
    and :attr:`estimated_cost_usd` introspection.

    Args:
        transport:  Configured :class:`~marqov.platform._transport.Transport`
                    instance used to make API calls.
        job_id:     The UUID string of the submitted job.
        rng:        Optional :class:`random.Random` instance used for jitter
                    in exponential back-off.  Supply a seeded instance to make
                    back-off deterministic in tests.  Defaults to a fresh
                    ``random.Random()``.

    Example::

        job = client.submit(circuit)
        result = job.result(timeout=120.0)
        print(result.counts)
    """

    def __init__(
        self,
        transport: "Transport",
        job_id: str,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self._transport = transport
        self._job_id = job_id
        self.rng = rng if rng is not None else random.Random()
        # Cache the last status response to avoid re-fetching for
        # estimated_cost_usd after a terminal poll.
        self._last_status: dict | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def id(self) -> str:
        """The UUID of this job as a plain string.

        Returns:
            The job identifier assigned by the platform at submission time.
        """
        return self._job_id

    def status(self) -> str:
        """Fetch the current job status from the platform.

        Issues a single GET request to the status endpoint with no
        ``wait`` parameter (returns immediately).  The raw status string
        from the server is returned as-is — unknown or future status values
        do **not** raise.

        Returns:
            Raw status string from the server (e.g. ``"pending"``,
            ``"running"``, ``"completed"``, ``"failed"``, ``"cancelled"``,
            ``"dispatch_failed"``).

        Raises:
            :class:`~marqov.platform.errors.MarqovPlatformError`: Any
                non-2xx response that is not retried successfully.
        """
        resp = self._transport.request("GET", f"/api/jobs/{self._job_id}/status")
        self._last_status = resp
        return resp.get("status", "unknown")

    def result(
        self,
        timeout: float = 300.0,
        poll_interval: float = 2.0,
    ) -> PlatformResult:
        """Block until the job reaches a terminal state and return the result.

        Uses the server's ``wait`` long-poll parameter as the primary
        waiting mechanism; each request blocks on the server for up to
        ``wait`` seconds before returning the current (possibly
        non-terminal) status.  Between polls a client-side exponential
        back-off with jitter is applied, capped at
        :data:`_MAX_BACKOFF_SECONDS`.

        The ``wait`` value sent to the server is computed as::

            wait = min(
                _SERVER_MAX_WAIT_SECONDS,       # 22 s server cap
                remaining_budget,               # seconds left in timeout
                transport.timeout - _WAIT_MARGIN_SECONDS,  # per-request headroom
            )

        This guarantees the server-side hold is **strictly less than** the
        HTTP request timeout, so the transport cannot time out before the
        server responds.

        Args:
            timeout:       Overall wall-clock deadline in seconds.  If the
                           job has not reached a terminal state within this
                           time a :class:`TimeoutError` is raised.  The job
                           is **not** cancelled server-side on timeout.
            poll_interval: Starting interval (seconds) for client-side
                           exponential back-off between polls.  Doubles each
                           round, capped at :data:`_MAX_BACKOFF_SECONDS`.

        Returns:
            :class:`~marqov.platform._models.PlatformResult` wrapping the
            raw ``result`` field from the server response.

        Raises:
            :class:`~marqov.platform.errors.JobFailed`: The job reached the
                ``failed`` or ``dispatch_failed`` terminal state.  The error
                message is sourced from ``result["error"]`` in the response
                if present, otherwise a generic message is used (the server's
                ``error_message`` field is **not** returned by the status
                endpoint — see module docstring).
            :class:`TimeoutError`: The overall ``timeout`` elapsed before the
                job completed.  The job continues running server-side.
            :class:`~marqov.platform.errors.MarqovPlatformError`: Any
                non-2xx transport error.

        .. note::
            ``dispatch_failed`` is treated as a ``failed`` terminal state
            (raises :class:`~marqov.platform.errors.JobFailed`); it is one of
            the four states recognised by
            :func:`~marqov.platform._models.is_terminal`.
        """
        deadline = time.monotonic() + timeout
        backoff = poll_interval

        # Maximum ``wait`` budget based on the per-request transport timeout
        # (leave _WAIT_MARGIN_SECONDS headroom so the HTTP request does not
        # time out before the server responds).
        max_wait_per_request = max(
            0, int(self._transport.timeout - _WAIT_MARGIN_SECONDS)
        )

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Job {self._job_id} did not complete within {timeout}s"
                )

            # Compute server-side wait: must be < transport.timeout.
            # The server caps the "wait" param at 22 seconds.
            wait_seconds = int(
                min(
                    _SERVER_MAX_WAIT_SECONDS,
                    remaining,
                    max_wait_per_request,
                )
            )
            # Ensure wait is always strictly positive when budget allows
            wait_seconds = max(0, wait_seconds)

            resp = self._transport.request(
                "GET",
                f"/api/jobs/{self._job_id}/status",
                wait=wait_seconds if wait_seconds > 0 else None,
            )
            self._last_status = resp
            raw_status: str = resp.get("status", "unknown")

            if raw_status == "completed":
                return PlatformResult(raw=resp.get("result") or {})

            if raw_status in ("failed", "dispatch_failed"):
                # error_message is NOT returned by the status endpoint
                # (see module docstring).  Best-effort: check if result
                # contains an "error" key; fall back to a generic message.
                result_field = resp.get("result") or {}
                error_msg: str
                if isinstance(result_field, dict) and "error" in result_field:
                    error_msg = str(result_field["error"])
                else:
                    error_msg = (
                        f"Job {self._job_id} failed (status={raw_status!r}). "
                        "The server error_message is not returned by the status "
                        "endpoint; check the platform dashboard for details."
                    )
                raise JobFailed(error_msg, code=raw_status)

            if is_terminal(raw_status):
                # e.g. "cancelled" — not an error but not a result either
                raise JobFailed(
                    f"Job {self._job_id} ended with status {raw_status!r} "
                    "(no result produced).",
                    code=raw_status,
                )

            # Non-terminal: apply client-side backoff with jitter before
            # the next poll, but do not sleep past the deadline.
            remaining_after_poll = deadline - time.monotonic()
            if remaining_after_poll <= 0:
                raise TimeoutError(
                    f"Job {self._job_id} did not complete within {timeout}s"
                )

            jitter = self.rng.uniform(0.0, backoff * 0.25)
            sleep_time = min(backoff + jitter, _MAX_BACKOFF_SECONDS, remaining_after_poll)
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Exponential backoff (doubles each round, capped)
            backoff = min(backoff * 2.0, _MAX_BACKOFF_SECONDS)

    @property
    def estimated_cost_usd(self) -> float | None:
        """Estimated cost in USD from the most recent status response.

        Reads ``estimated_cost_usd`` from the last status response cached by
        :meth:`status` or :meth:`result`.  Returns ``None`` if no status has
        been fetched yet, or if the field was absent / ``null`` in the
        response (e.g. before the job reaches a terminal state or for free
        backends that return ``0.0``).

        Returns:
            Cost estimate in USD, or ``None`` if unavailable.

        .. note::
            ``0.0`` is a valid value for free backends — callers should
            distinguish ``None`` (unknown) from ``0.0`` (known zero cost).
        """
        if self._last_status is None:
            return None
        val = self._last_status.get("estimated_cost_usd")
        if val is None:
            return None
        return float(val)

    def cancel(self) -> None:
        """Request best-effort cancellation of this job.

        Issues a POST to ``/api/jobs/{id}/cancel`` and returns immediately
        without checking the resulting status.  Cancellation is best-effort:
        the server may have already transitioned the job to a terminal state,
        in which case this call is a no-op from the server's perspective.

        .. warning::
            **§11 TBC assumption**: The ``/api/jobs/{id}/cancel`` endpoint
            does **not** currently exist in the platform.  The only known
            job sub-routes are ``status`` and ``traces``; no user-facing
            cancel/revoke endpoint is present at this revision.  This method
            is implemented against the mocked path so that test-suite
            infrastructure and caller code can be written today.  When the
            platform ships a real cancel endpoint, update the path here (and
            remove this warning).

        Raises:
            :class:`~marqov.platform.errors.MarqovPlatformError`: If the
                cancel request fails with a non-2xx response.  Callers that
                want truly fire-and-forget behaviour should catch
                :class:`~marqov.platform.errors.MarqovPlatformError`.
        """
        self._transport.request(
            "POST",
            f"/api/jobs/{self._job_id}/cancel",
            idempotent_write=False,
        )
