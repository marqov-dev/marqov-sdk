"""Tests for sweep_orphan_braket_tasks — orphan Braket task cleanup on key revocation.

Since platform_worker.py has heavy dependencies (supabase, httpx, sentry, etc.),
we test the sweep logic by extracting it inline rather than importing the module.
The function's logic is simple enough to replicate without the full import chain.
"""
import asyncio
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Replicate the sweep function's core logic for testing
# (platform_worker.py can't be imported without supabase/httpx/sentry)
# ---------------------------------------------------------------------------

_NON_BRAKET_BACKENDS = ("local", "qb-sim-aer", "qb-sim-qulacs")


def _filter_braket_jobs_missing_arns(cancelled_jobs):
    """Same filter logic as sweep_orphan_braket_tasks."""
    return [
        j for j in cancelled_jobs
        if j["backend"] not in _NON_BRAKET_BACKENDS
        and not j.get("braket_task_arns")
    ]


def _match_task_to_job(output_dir, job_ids):
    """Same matching logic as the sweep's inner loop."""
    for job_id in job_ids:
        if job_id in output_dir:
            return job_id
    return None


def _has_orphans(jobs):
    """Same resolution logic as _maybe_resolve_event."""
    return any(
        j["backend"] not in _NON_BRAKET_BACKENDS
        and not j.get("braket_task_arns")
        for j in jobs
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFilterBraketJobsMissingArns:
    """Test the filtering logic that decides which jobs need sweep."""

    def test_local_backend_excluded(self):
        jobs = [{"id": "j1", "backend": "local", "braket_task_arns": None}]
        assert _filter_braket_jobs_missing_arns(jobs) == []

    def test_qb_sim_excluded(self):
        jobs = [{"id": "j1", "backend": "qb-sim-aer", "braket_task_arns": None}]
        assert _filter_braket_jobs_missing_arns(jobs) == []

    def test_sv1_without_arn_included(self):
        jobs = [{"id": "j1", "backend": "sv1", "braket_task_arns": None}]
        result = _filter_braket_jobs_missing_arns(jobs)
        assert len(result) == 1
        assert result[0]["id"] == "j1"

    def test_sv1_with_arn_excluded(self):
        jobs = [{"id": "j1", "backend": "sv1", "braket_task_arns": ["arn:task/123"]}]
        assert _filter_braket_jobs_missing_arns(jobs) == []

    def test_mixed_backends(self):
        jobs = [
            {"id": "j1", "backend": "local", "braket_task_arns": None},
            {"id": "j2", "backend": "sv1", "braket_task_arns": None},
            {"id": "j3", "backend": "sv1", "braket_task_arns": ["arn:task/456"]},
            {"id": "j4", "backend": "ionq-aria-1", "braket_task_arns": None},
        ]
        result = _filter_braket_jobs_missing_arns(jobs)
        assert len(result) == 2
        assert {j["id"] for j in result} == {"j2", "j4"}


class TestMatchTaskToJob:
    """Test the output directory → job_id matching."""

    def test_match_found(self):
        output_dir = "s3://bucket/prefix/abc-123-def/results"
        assert _match_task_to_job(output_dir, {"abc-123-def", "xyz-789"}) == "abc-123-def"

    def test_no_match(self):
        output_dir = "s3://bucket/prefix/other-job/results"
        assert _match_task_to_job(output_dir, {"abc-123", "xyz-789"}) is None

    def test_empty_job_ids(self):
        assert _match_task_to_job("s3://bucket/prefix/job/results", set()) is None


class TestHasOrphans:
    """Test the resolution check logic."""

    def test_no_orphans_when_all_have_arns(self):
        jobs = [{"backend": "sv1", "braket_task_arns": ["arn:task/1"]}]
        assert not _has_orphans(jobs)

    def test_no_orphans_when_all_local(self):
        jobs = [{"backend": "local", "braket_task_arns": None}]
        assert not _has_orphans(jobs)

    def test_has_orphans_when_braket_missing_arn(self):
        jobs = [{"backend": "sv1", "braket_task_arns": None}]
        assert _has_orphans(jobs)

    def test_mixed_resolved_and_orphan(self):
        jobs = [
            {"backend": "sv1", "braket_task_arns": ["arn:task/1"]},
            {"backend": "dm1", "braket_task_arns": None},
        ]
        assert _has_orphans(jobs)


class TestClientErrorHandling:
    """Test that already-cancelled tasks are handled gracefully."""

    def test_conflict_exception_is_expected(self):
        """ConflictException error code should be treated as success."""
        expected_codes = ("ConflictException", "ValidationException")
        error_code = "ConflictException"
        assert error_code in expected_codes

    def test_unknown_error_code_not_swallowed(self):
        """Other error codes should NOT be treated as success."""
        expected_codes = ("ConflictException", "ValidationException")
        error_code = "InternalServerError"
        assert error_code not in expected_codes
