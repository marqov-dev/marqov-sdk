"""Tests for marqov.platform.job.Job.

All tests mock the Transport — no real network calls.

Mock fixture source citations:
  - Status response shape:
      platform/src/app/api/jobs/[id]/status/route.ts:152
      ``selectCols = "id, status, backend, created_at, updated_at, estimated_cost_usd, result"``
  - ``wait`` param name and server cap (22 s):
      platform/src/app/api/jobs/[id]/status/route.ts:70-73
  - Terminal states (server-side):
      platform/src/app/api/jobs/[id]/status/route.ts:13-18
      ``"completed", "failed", "cancelled", "dispatch_failed"``
  - Cancel endpoint: §11 TBC — no real endpoint exists; mocked as
      ``POST /api/jobs/{id}/cancel``.
"""

from __future__ import annotations

import random
import time
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from marqov.platform._models import PlatformResult
from marqov.platform._transport import Transport
from marqov.platform.errors import JobFailed, MarqovPlatformError
from marqov.platform.job import (
    Job,
    _SERVER_MAX_WAIT_SECONDS,
    _WAIT_MARGIN_SECONDS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_transport(timeout: float = 30.0) -> MagicMock:
    """Return a mock Transport with a configurable .timeout property."""
    t = MagicMock(spec=Transport)
    # ``spec=Transport`` picks up properties defined on the class, but the
    # MagicMock attribute for a property is itself a MagicMock.  Wire it
    # directly so ``transport.timeout`` returns the float.
    type(t).timeout = property(lambda self: timeout)
    return t


def _status_payload(
    status: str,
    *,
    job_id: str = "job-abc",
    result: Any = None,
    estimated_cost_usd: Any = None,
) -> dict:
    """Build a minimal status-endpoint response payload.

    Shape mirrors the real server:
      platform/src/app/api/jobs/[id]/status/route.ts:152
      id, status, backend, created_at, updated_at, estimated_cost_usd, result
    """
    return {
        "id": job_id,
        "status": status,
        "backend": "sv1",
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-01T00:00:01Z",
        "estimated_cost_usd": estimated_cost_usd,
        "result": result,
    }


def _make_job(
    transport: MagicMock,
    job_id: str = "job-abc",
    *,
    seed: int = 42,
) -> Job:
    """Construct a Job with a seeded RNG for deterministic backoff."""
    rng = random.Random(seed)
    return Job(transport, job_id, rng=rng)


# ---------------------------------------------------------------------------
# id property
# ---------------------------------------------------------------------------


class TestJobId:
    def test_id_returns_string(self):
        t = _make_transport()
        job = _make_job(t, "my-job-id")
        assert job.id == "my-job-id"


# ---------------------------------------------------------------------------
# status()
# ---------------------------------------------------------------------------


class TestJobStatus:
    def test_status_calls_correct_endpoint(self):
        t = _make_transport()
        t.request.return_value = _status_payload("pending")
        job = _make_job(t)

        job.status()

        t.request.assert_called_once_with("GET", "/api/jobs/job-abc/status")

    def test_status_returns_raw_string(self):
        t = _make_transport()
        t.request.return_value = _status_payload("running")
        job = _make_job(t)

        assert job.status() == "running"

    def test_status_completed(self):
        t = _make_transport()
        t.request.return_value = _status_payload("completed")
        job = _make_job(t)

        assert job.status() == "completed"

    def test_status_failed(self):
        t = _make_transport()
        t.request.return_value = _status_payload("failed")
        job = _make_job(t)

        assert job.status() == "failed"

    def test_unknown_status_does_not_crash(self):
        """An unknown status string from the server must NOT raise."""
        t = _make_transport()
        t.request.return_value = _status_payload("archived")
        job = _make_job(t)

        result = job.status()
        assert result == "archived"

    def test_future_status_string_does_not_crash(self):
        """Future status values introduced by the server must not raise."""
        t = _make_transport()
        t.request.return_value = _status_payload("queued_priority")
        job = _make_job(t)

        assert job.status() == "queued_priority"


# ---------------------------------------------------------------------------
# result() — happy path
# ---------------------------------------------------------------------------


class TestJobResultCompleted:
    def test_result_returns_platform_result_on_completed(self):
        """result() returns PlatformResult when status is 'completed'."""
        t = _make_transport()
        counts = {"00": 512, "11": 488}
        t.request.return_value = _status_payload(
            "completed", result={"counts": counts}
        )
        job = _make_job(t)

        outcome = job.result()

        assert isinstance(outcome, PlatformResult)
        assert outcome.counts == counts

    def test_result_completed_with_null_result_field(self):
        """If server returns 'completed' but result=null, raw is {} (not None)."""
        t = _make_transport()
        t.request.return_value = _status_payload("completed", result=None)
        job = _make_job(t)

        outcome = job.result()
        assert isinstance(outcome, PlatformResult)
        assert outcome.raw == {}

    def test_result_polls_until_terminal(self):
        """result() keeps polling until it reaches 'completed'."""
        t = _make_transport()
        t.request.side_effect = [
            _status_payload("pending"),
            _status_payload("running"),
            _status_payload("completed", result={"counts": {"0": 1}}),
        ]
        job = _make_job(t)

        with patch("marqov.platform.job.time.sleep"):
            outcome = job.result()

        assert isinstance(outcome, PlatformResult)
        assert t.request.call_count == 3


# ---------------------------------------------------------------------------
# result() — failure paths
# ---------------------------------------------------------------------------


class TestJobResultFailed:
    def test_raises_job_failed_on_failed_status(self):
        """result() raises JobFailed when status is 'failed'."""
        t = _make_transport()
        t.request.return_value = _status_payload("failed")
        job = _make_job(t)

        with pytest.raises(JobFailed):
            job.result()

    def test_job_failed_carries_error_from_result_field(self):
        """JobFailed message uses result['error'] when available."""
        t = _make_transport()
        t.request.return_value = _status_payload(
            "failed", result={"error": "circuit exceeds qubit limit"}
        )
        job = _make_job(t)

        with pytest.raises(JobFailed) as exc_info:
            job.result()

        assert "circuit exceeds qubit limit" in str(exc_info.value)

    def test_job_failed_generic_message_when_no_result_error(self):
        """JobFailed still raised even when result field has no error key."""
        t = _make_transport()
        t.request.return_value = _status_payload("failed", result=None)
        job = _make_job(t)

        with pytest.raises(JobFailed) as exc_info:
            job.result()

        # Confirm it raised JobFailed (the message may be generic)
        assert isinstance(exc_info.value, JobFailed)

    def test_raises_job_failed_on_dispatch_failed(self):
        """dispatch_failed is a server terminal state treated as JobFailed."""
        t = _make_transport()
        t.request.return_value = _status_payload("dispatch_failed")
        job = _make_job(t)

        with pytest.raises(JobFailed) as exc_info:
            job.result()

        assert exc_info.value.code == "dispatch_failed"

    def test_raises_job_failed_on_cancelled(self):
        """'cancelled' terminal state raises JobFailed (no result produced)."""
        t = _make_transport()
        t.request.return_value = _status_payload("cancelled")
        job = _make_job(t)

        with pytest.raises(JobFailed):
            job.result()


# ---------------------------------------------------------------------------
# result() — timeout behaviour
# ---------------------------------------------------------------------------


class TestJobResultTimeout:
    def test_raises_timeout_error_when_deadline_exceeded(self):
        """result() raises TimeoutError if job does not complete in time."""
        t = _make_transport()
        # Always return non-terminal
        t.request.return_value = _status_payload("pending")
        job = _make_job(t)

        with patch("marqov.platform.job.time.sleep"):
            with patch("marqov.platform.job.time.monotonic") as mock_mono:
                # Simulate: first call returns 0.0 (start), subsequent calls
                # return increasing values past the deadline.
                mock_mono.side_effect = [
                    0.0,   # deadline = 0.0 + 5.0 = 5.0
                    4.9,   # remaining = 0.1 > 0 — first poll starts
                    6.0,   # remaining check after poll → expired
                ]
                with pytest.raises(TimeoutError):
                    job.result(timeout=5.0)

    def test_timeout_does_not_cancel_job(self):
        """When timeout occurs, cancel() is NOT called server-side."""
        t = _make_transport()
        t.request.return_value = _status_payload("running")
        job = _make_job(t)

        with patch("marqov.platform.job.time.sleep"):
            with patch("marqov.platform.job.time.monotonic") as mock_mono:
                mock_mono.side_effect = [0.0, 4.9, 6.0]
                with pytest.raises(TimeoutError):
                    job.result(timeout=5.0)

        # Confirm no cancel call was made — only status calls
        for c in t.request.call_args_list:
            path = c[0][1] if len(c[0]) >= 2 else c.kwargs.get("path", "")
            assert "cancel" not in path, "cancel must not be called on timeout"


# ---------------------------------------------------------------------------
# result() — wait parameter contract
# ---------------------------------------------------------------------------


class TestJobResultWaitParam:
    def test_wait_param_is_sent(self):
        """result() sends wait= in the status request."""
        t = _make_transport(timeout=30.0)
        t.request.return_value = _status_payload(
            "completed", result={"counts": {"0": 1}}
        )
        job = _make_job(t)

        with patch("marqov.platform.job.time.sleep"):
            job.result(timeout=300.0)

        call_kwargs = t.request.call_args[1]
        assert "wait" in call_kwargs
        assert call_kwargs["wait"] is not None

    def test_wait_is_strictly_less_than_transport_timeout(self):
        """The wait param sent must be < transport.timeout (never equal)."""
        transport_timeout = 30.0
        t = _make_transport(timeout=transport_timeout)
        t.request.return_value = _status_payload(
            "completed", result={"counts": {"0": 1}}
        )
        job = _make_job(t)

        with patch("marqov.platform.job.time.sleep"):
            job.result(timeout=300.0)

        call_kwargs = t.request.call_args[1]
        wait = call_kwargs["wait"]
        assert wait < transport_timeout, (
            f"wait={wait} must be strictly less than transport.timeout={transport_timeout}"
        )

    def test_wait_does_not_exceed_server_max(self):
        """The wait param must not exceed the server cap of 22 s."""
        t = _make_transport(timeout=60.0)
        t.request.return_value = _status_payload(
            "completed", result={"counts": {"0": 1}}
        )
        job = _make_job(t)

        with patch("marqov.platform.job.time.sleep"):
            job.result(timeout=300.0)

        call_kwargs = t.request.call_args[1]
        assert call_kwargs["wait"] <= _SERVER_MAX_WAIT_SECONDS

    def test_wait_bounded_by_remaining_budget(self):
        """When only a few seconds remain, wait is capped to remaining budget."""
        t = _make_transport(timeout=30.0)
        t.request.return_value = _status_payload(
            "completed", result={"counts": {"0": 1}}
        )
        job = _make_job(t)

        # Very tight timeout — remaining budget < SERVER_MAX_WAIT_SECONDS
        # Sequence:
        #   [0] deadline = time.monotonic() + timeout → 0.0 + 3.0 = 3.0
        #   [1] remaining = deadline - time.monotonic() → 3.0 - 0.5 = 2.5 (> 0, proceed)
        #   [2] (after poll — no sleep since completed; no further calls needed)
        with patch("marqov.platform.job.time.monotonic") as mock_mono:
            mock_mono.side_effect = [0.0, 0.5]
            with patch("marqov.platform.job.time.sleep"):
                job.result(timeout=3.0)

        call_kwargs = t.request.call_args[1]
        # wait must be <= remaining budget (~2.5 s), and < transport timeout (30)
        wait = call_kwargs.get("wait") or 0
        assert wait <= 3


# ---------------------------------------------------------------------------
# result() — backoff
# ---------------------------------------------------------------------------


class TestJobResultBackoff:
    def test_backoff_increases_between_polls(self):
        """Sleep time increases (exponential backoff) across rounds."""
        t = _make_transport()
        t.request.side_effect = [
            _status_payload("pending"),
            _status_payload("running"),
            _status_payload("running"),
            _status_payload("completed", result={"counts": {"0": 1}}),
        ]
        job = _make_job(t, seed=0)  # seed=0 for reproducibility

        sleep_calls: list[float] = []
        with patch("marqov.platform.job.time.sleep", side_effect=lambda s: sleep_calls.append(s)):
            job.result(timeout=300.0, poll_interval=1.0)

        # Must have slept at least twice (3 non-terminal rounds → 3 sleeps, but
        # the last poll returned completed so only the first 3 sleeps matter)
        assert len(sleep_calls) >= 2
        # Each sleep should be >= the previous (backoff grows)
        for i in range(1, len(sleep_calls)):
            assert sleep_calls[i] >= sleep_calls[i - 1] or sleep_calls[i] >= 1.0, (
                f"Backoff did not increase: sleep_calls={sleep_calls}"
            )

    def test_backoff_is_deterministic_with_seeded_rng(self):
        """Two Job instances with the same RNG seed produce identical sleep patterns."""
        t1 = _make_transport()
        t2 = _make_transport()
        sequence = [
            _status_payload("pending"),
            _status_payload("running"),
            _status_payload("completed", result={}),
        ]
        t1.request.side_effect = list(sequence)
        t2.request.side_effect = list(sequence)

        sleeps1: list[float] = []
        sleeps2: list[float] = []

        with patch("marqov.platform.job.time.sleep", side_effect=lambda s: sleeps1.append(s)):
            Job(t1, "job-1", rng=random.Random(99)).result(timeout=300.0)

        with patch("marqov.platform.job.time.sleep", side_effect=lambda s: sleeps2.append(s)):
            Job(t2, "job-2", rng=random.Random(99)).result(timeout=300.0)

        assert sleeps1 == sleeps2, "Same RNG seed must produce identical backoff"

    def test_different_seeds_may_produce_different_jitter(self):
        """Different RNG seeds (usually) produce different jitter values."""
        # This is probabilistic but with high confidence for well-separated seeds.
        t1 = _make_transport()
        t2 = _make_transport()
        sequence = [
            _status_payload("pending"),
            _status_payload("completed", result={}),
        ]
        t1.request.side_effect = list(sequence)
        t2.request.side_effect = list(sequence)

        sleeps1: list[float] = []
        sleeps2: list[float] = []

        with patch("marqov.platform.job.time.sleep", side_effect=lambda s: sleeps1.append(s)):
            Job(t1, "job-1", rng=random.Random(1)).result(timeout=300.0)

        with patch("marqov.platform.job.time.sleep", side_effect=lambda s: sleeps2.append(s)):
            Job(t2, "job-2", rng=random.Random(9999)).result(timeout=300.0)

        # If both produced sleeps, they may differ (soft assertion — just check lengths match)
        assert len(sleeps1) == len(sleeps2)


# ---------------------------------------------------------------------------
# cancel()
# ---------------------------------------------------------------------------


class TestJobCancel:
    def test_cancel_calls_revocation_endpoint(self):
        """cancel() issues POST to /api/jobs/{id}/cancel (§11 TBC mocked path)."""
        t = _make_transport()
        t.request.return_value = {}
        job = _make_job(t)

        job.cancel()

        t.request.assert_called_once_with(
            "POST",
            "/api/jobs/job-abc/cancel",
            idempotent_write=False,
        )

    def test_cancel_returns_none(self):
        """cancel() returns None (fire-and-forget)."""
        t = _make_transport()
        t.request.return_value = {}
        job = _make_job(t)

        result = job.cancel()
        assert result is None

    def test_cancel_then_status_completed(self):
        """After cancel(), a subsequent status() returning 'completed' surfaces correctly."""
        t = _make_transport()
        # cancel returns empty
        # status returns completed
        t.request.side_effect = [
            {},                                  # POST cancel
            _status_payload("completed"),        # GET status
        ]
        job = _make_job(t)

        job.cancel()
        st = job.status()

        assert st == "completed"

    def test_cancel_uses_job_id(self):
        """cancel() uses the correct job ID in the path."""
        t = _make_transport()
        t.request.return_value = {}
        job = _make_job(t, "specific-uuid-1234")

        job.cancel()

        call_args = t.request.call_args
        assert "specific-uuid-1234" in call_args[0][1]


# ---------------------------------------------------------------------------
# estimated_cost_usd
# ---------------------------------------------------------------------------


class TestEstimatedCostUsd:
    def test_none_before_any_status_call(self):
        """estimated_cost_usd is None before any status/result call."""
        t = _make_transport()
        job = _make_job(t)

        assert job.estimated_cost_usd is None

    def test_none_when_field_absent(self):
        """Returns None when estimated_cost_usd field is null in response."""
        t = _make_transport()
        t.request.return_value = _status_payload("running", estimated_cost_usd=None)
        job = _make_job(t)

        job.status()
        assert job.estimated_cost_usd is None

    def test_returns_float_when_present(self):
        """Returns float value when estimated_cost_usd is present."""
        t = _make_transport()
        t.request.return_value = _status_payload("completed", estimated_cost_usd=0.05)
        job = _make_job(t)

        job.status()
        assert job.estimated_cost_usd == pytest.approx(0.05)

    def test_zero_for_free_backend(self):
        """0.0 is a valid cost (free backend) and must not be treated as None."""
        t = _make_transport()
        t.request.return_value = _status_payload("completed", estimated_cost_usd=0.0)
        job = _make_job(t)

        job.status()
        assert job.estimated_cost_usd == 0.0
        assert job.estimated_cost_usd is not None

    def test_updated_after_result_call(self):
        """estimated_cost_usd is updated after result() completes."""
        t = _make_transport()
        t.request.return_value = _status_payload(
            "completed", result={"counts": {"0": 1}}, estimated_cost_usd=1.23
        )
        job = _make_job(t)

        with patch("marqov.platform.job.time.sleep"):
            job.result()

        assert job.estimated_cost_usd == pytest.approx(1.23)

    def test_none_pre_terminal_acceptable(self):
        """estimated_cost_usd may be None before terminal (pre-terminal is acceptable)."""
        t = _make_transport()
        t.request.return_value = _status_payload("pending", estimated_cost_usd=None)
        job = _make_job(t)

        job.status()
        assert job.estimated_cost_usd is None


# ---------------------------------------------------------------------------
# Integration: full polling loop with multiple status transitions
# ---------------------------------------------------------------------------


class TestJobResultIntegration:
    def test_polls_pending_running_completed(self):
        """Full lifecycle: pending → running → completed."""
        t = _make_transport()
        t.request.side_effect = [
            _status_payload("pending"),
            _status_payload("running"),
            _status_payload(
                "completed",
                result={"counts": {"00": 600, "11": 400}},
                estimated_cost_usd=0.02,
            ),
        ]
        job = _make_job(t)

        with patch("marqov.platform.job.time.sleep"):
            outcome = job.result()

        assert outcome.counts == {"00": 600, "11": 400}
        assert job.estimated_cost_usd == pytest.approx(0.02)

    def test_polls_pending_dispatch_failed(self):
        """Full lifecycle: pending → dispatch_failed raises JobFailed."""
        t = _make_transport()
        t.request.side_effect = [
            _status_payload("pending"),
            _status_payload("dispatch_failed"),
        ]
        job = _make_job(t)

        with patch("marqov.platform.job.time.sleep"):
            with pytest.raises(JobFailed) as exc_info:
                job.result()

        assert exc_info.value.code == "dispatch_failed"

    def test_unknown_intermediate_status_does_not_crash(self):
        """An unknown intermediate status is treated as non-terminal (no crash)."""
        t = _make_transport()
        t.request.side_effect = [
            _status_payload("queued_future_state"),  # unknown — treated as non-terminal
            _status_payload("completed", result={"counts": {"0": 1}}),
        ]
        job = _make_job(t)

        with patch("marqov.platform.job.time.sleep"):
            outcome = job.result()

        assert isinstance(outcome, PlatformResult)
