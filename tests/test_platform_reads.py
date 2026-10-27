"""Tests for MarqovClient's read-only job and project listings (marqov-sdk#292).

``recent_jobs``, ``projects`` and ``project_jobs`` go through the real
:class:`~marqov.platform._transport.Transport`; only
``requests.Session.request`` is mocked, so the transport's error mapping,
retries and deadline handling are exercised as they run in production.

Response shapes follow the Platform's REST API reference: ``GET /api/jobs``
returns ``{"data": [...], "next_cursor": ...}``, ``GET /api/projects`` returns
``{"projects": [...]}``, and a project's linked jobs are the ``job_run_id`` of
the result blocks in its documents.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from marqov.platform import JobSummary, MarqovClient, Project
from marqov.platform.errors import MarqovPlatformError, RateLimited, TransportError

PROJECT = "11111111-1111-4111-8111-111111111111"
WORKSPACE = "22222222-2222-4222-8222-222222222222"
REPORT = "33333333-3333-4333-8333-333333333333"
JOB_A = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
JOB_B = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
JOB_C = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"


def _response(status: int, body: object | None = None, headers: dict | None = None) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status
    resp.ok = status < 400
    resp.reason = f"HTTP {status}"
    if body is None:
        resp.json.side_effect = ValueError("not JSON")
    else:
        resp.json.return_value = body
    resp.headers = headers or {}
    return resp


def _client() -> MarqovClient:
    return MarqovClient(api_key="marqey_test_reads", base_url="http://test.invalid")


def _job_row(job_id: str, status: str = "running", **extra: object) -> dict:
    return {
        "id": job_id,
        "status": status,
        "backend": "sv1",
        "created_at": "2026-10-26T10:00:00Z",
        "updated_at": "2026-10-26T10:05:00Z",
        "completed_at": None,
        "estimated_cost_usd": 0.0,
        "actual_cost_usd": None,
        **extra,
    }


def _route(routes: dict[str, object]):
    """Session.request side effect answering by URL path (query excluded)."""

    def answer(method, url, **kwargs):
        path = url.removeprefix("http://test.invalid")
        body = routes[path]
        return body if isinstance(body, MagicMock) else _response(200, body)

    return answer


# ---------------------------------------------------------------------------
# recent_jobs
# ---------------------------------------------------------------------------


class TestRecentJobs:
    def test_lists_team_jobs_newest_first_with_limit(self):
        client = _client()
        body = {"data": [_job_row(JOB_A), _job_row(JOB_B, "completed")], "next_cursor": None}
        with patch.object(client._transport._session, "request", return_value=_response(200, body)) as req:
            jobs = client.recent_jobs(limit=2)
        method, url = req.call_args.args
        assert (method, url) == ("GET", "http://test.invalid/api/jobs")
        assert req.call_args.kwargs["params"] == {"limit": 2}
        assert [j.id for j in jobs] == [JOB_A, JOB_B]
        assert jobs[1] == JobSummary(
            id=JOB_B,
            status="completed",
            backend="sv1",
            created_at="2026-10-26T10:00:00Z",
            updated_at="2026-10-26T10:05:00Z",
            estimated_cost_usd=0.0,
            actual_cost_usd=None,
            hold_reason=None,
        )

    def test_result_and_error_bodies_are_not_kept(self):
        """A summary carries no result payload or error text (they are never displayed)."""
        client = _client()
        row = _job_row(JOB_A, "failed", result={"counts": {"0": 1}}, error_message="Traceback ...")
        with patch.object(client._transport._session, "request",
                          return_value=_response(200, {"data": [row], "next_cursor": None})):
            (job,) = client.recent_jobs()
        assert not hasattr(job, "result")
        assert not hasattr(job, "error_message")
        assert "Traceback" not in repr(job)

    @pytest.mark.parametrize("limit", [0, 101, True, 2.0])
    def test_limit_outside_the_documented_range_is_refused_before_sending(self, limit):
        client = _client()
        with patch.object(client._transport._session, "request") as req:
            with pytest.raises(ValueError, match="limit"):
                client.recent_jobs(limit=limit)
        req.assert_not_called()

    def test_non_json_success_body_raises_transport_error(self):
        client = _client()
        with patch.object(client._transport._session, "request", return_value=_response(200)):
            with pytest.raises(TransportError):
                client.recent_jobs()

    def test_unexpected_shape_raises_invalid_response(self):
        client = _client()
        with patch.object(client._transport._session, "request",
                          return_value=_response(200, {"data": {"id": JOB_A}})):
            with pytest.raises(MarqovPlatformError) as info:
                client.recent_jobs()
        assert info.value.code == "invalid_response"

    def test_429_surfaces_rate_limited_with_retry_after(self):
        client = _client()
        limited = _response(429, {"error": "Too many requests"}, headers={"Retry-After": "7"})
        with patch.object(client._transport._session, "request", return_value=limited), \
                patch("marqov.platform._transport.time.sleep"):
            with pytest.raises(RateLimited) as info:
                client.recent_jobs()
        assert info.value.retry_after == 7

    def test_deadline_stops_retries(self):
        """A passed deadline means one attempt and no backoff, as the transport documents."""
        client = _client()
        unavailable = _response(503, {"error": {"code": "unavailable", "message": "down", "status": 503}})
        with patch.object(client._transport._session, "request", return_value=unavailable) as req, \
                patch("marqov.platform._transport.time.sleep") as sleep:
            with pytest.raises(MarqovPlatformError):
                client.recent_jobs(deadline=time.monotonic() - 1)
        assert req.call_count == 1
        sleep.assert_not_called()


# ---------------------------------------------------------------------------
# projects
# ---------------------------------------------------------------------------


class TestProjects:
    def test_lists_projects(self):
        client = _client()
        body = {"projects": [{
            "id": PROJECT, "name": "Bell study", "description": "", "team_id": "t",
            "user_id": "u", "created_at": "2026-10-01T00:00:00Z", "updated_at": "2026-10-02T00:00:00Z",
        }]}
        with patch.object(client._transport._session, "request", return_value=_response(200, body)) as req:
            projects = client.projects()
        assert req.call_args.args == ("GET", "http://test.invalid/api/projects")
        assert projects == [Project(id=PROJECT, name="Bell study", description="",
                                    created_at="2026-10-01T00:00:00Z",
                                    updated_at="2026-10-02T00:00:00Z")]

    def test_non_json_success_body_raises_transport_error(self):
        client = _client()
        with patch.object(client._transport._session, "request", return_value=_response(200)):
            with pytest.raises(TransportError):
                client.projects()

    def test_429_surfaces_rate_limited_with_retry_after(self):
        client = _client()
        limited = _response(429, {"error": "slow down"}, headers={"Retry-After": "12"})
        with patch.object(client._transport._session, "request", return_value=limited), \
                patch("marqov.platform._transport.time.sleep"):
            with pytest.raises(RateLimited) as info:
                client.projects()
        assert info.value.retry_after == 12

    def test_deadline_stops_retries(self):
        client = _client()
        with patch.object(client._transport._session, "request",
                          side_effect=requests.exceptions.ConnectionError("refused")) as req, \
                patch("marqov.platform._transport.time.sleep") as sleep:
            with pytest.raises(TransportError):
                client.projects(deadline=time.monotonic() - 1)
        assert req.call_count == 1
        sleep.assert_not_called()

    def test_row_without_an_id_raises_invalid_response(self):
        client = _client()
        with patch.object(client._transport._session, "request",
                          return_value=_response(200, {"projects": [{"name": "x"}]})):
            with pytest.raises(MarqovPlatformError) as info:
                client.projects()
        assert info.value.code == "invalid_response"


# ---------------------------------------------------------------------------
# project_jobs
# ---------------------------------------------------------------------------


def _documents(*doc_ids: str) -> dict:
    return {"documents": [{"id": d, "document_type": "workspace"} for d in doc_ids],
            "total": len(doc_ids)}


def _document(doc_id: str, *job_ids: str | None, project: str = PROJECT) -> dict:
    blocks = [{"id": f"m-{doc_id}", "block_type": "markdown", "job_run_id": None}]
    blocks += [{"id": f"b{i}-{doc_id}", "block_type": "result", "job_run_id": j}
               for i, j in enumerate(job_ids)]
    return {"document": {"id": doc_id, "project_id": project}, "blocks": blocks, "sections": []}


class TestProjectJobs:
    def test_reads_result_blocks_across_documents_in_order_without_duplicates(self):
        client = _client()
        routes = {
            f"/api/projects/{PROJECT}/documents": _documents(WORKSPACE, REPORT),
            f"/api/projects/{PROJECT}/documents/{WORKSPACE}": _document(WORKSPACE, JOB_B, JOB_A),
            f"/api/projects/{PROJECT}/documents/{REPORT}": _document(REPORT, JOB_A, JOB_C),
            f"/api/jobs/{JOB_A}/status": _job_row(JOB_A, "completed", result={"counts": {}}),
            f"/api/jobs/{JOB_B}/status": _job_row(JOB_B, "pending", hold_reason="Scheduled window"),
            f"/api/jobs/{JOB_C}/status": _job_row(JOB_C, "failed"),
        }
        with patch.object(client._transport._session, "request", side_effect=_route(routes)) as req:
            jobs = client.project_jobs(PROJECT)
        assert [(j.id, j.status) for j in jobs] == [
            (JOB_B, "pending"), (JOB_A, "completed"), (JOB_C, "failed"),
        ]
        assert jobs[0].hold_reason == "Scheduled window"
        # Active documents only: archived ones are excluded server-side.
        list_call = req.call_args_list[0]
        assert list_call.kwargs["params"] == {"archived": "exclude", "limit": 100}

    def test_limit_stops_reading_documents_and_statuses(self):
        client = _client()
        routes = {
            f"/api/projects/{PROJECT}/documents": _documents(WORKSPACE, REPORT),
            f"/api/projects/{PROJECT}/documents/{WORKSPACE}": _document(WORKSPACE, JOB_A, JOB_B),
            f"/api/jobs/{JOB_A}/status": _job_row(JOB_A),
        }
        with patch.object(client._transport._session, "request", side_effect=_route(routes)) as req:
            jobs = client.project_jobs(PROJECT, limit=1)
        assert [j.id for j in jobs] == [JOB_A]
        paths = [c.args[1].removeprefix("http://test.invalid") for c in req.call_args_list]
        assert paths == [
            f"/api/projects/{PROJECT}/documents",
            f"/api/projects/{PROJECT}/documents/{WORKSPACE}",
            f"/api/jobs/{JOB_A}/status",
        ]

    def test_max_documents_bounds_the_documents_listed(self):
        client = _client()
        routes = {
            f"/api/projects/{PROJECT}/documents": _documents(WORKSPACE),
            f"/api/projects/{PROJECT}/documents/{WORKSPACE}": _document(WORKSPACE),
        }
        with patch.object(client._transport._session, "request", side_effect=_route(routes)) as req:
            client.project_jobs(PROJECT, max_documents=10)
        assert req.call_args_list[0].kwargs["params"] == {"archived": "exclude", "limit": 10}

    def test_max_documents_holds_even_if_the_server_returns_more(self):
        client = _client()
        routes = {
            f"/api/projects/{PROJECT}/documents": _documents(WORKSPACE, REPORT),
            f"/api/projects/{PROJECT}/documents/{WORKSPACE}": _document(WORKSPACE),
        }
        with patch.object(client._transport._session, "request", side_effect=_route(routes)) as req:
            assert client.project_jobs(PROJECT, max_documents=1) == []
        assert req.call_count == 2  # the list, then one document; REPORT is never read

    @pytest.mark.parametrize("value", [0, 101, True])
    def test_max_documents_outside_the_documented_range_is_refused(self, value):
        client = _client()
        with patch.object(client._transport._session, "request") as req:
            with pytest.raises(ValueError, match="max_documents"):
                client.project_jobs(PROJECT, max_documents=value)
        req.assert_not_called()

    def test_project_with_no_linked_jobs_is_empty_not_team_jobs(self):
        client = _client()
        routes = {
            f"/api/projects/{PROJECT}/documents": _documents(WORKSPACE),
            f"/api/projects/{PROJECT}/documents/{WORKSPACE}": _document(WORKSPACE),
        }
        with patch.object(client._transport._session, "request", side_effect=_route(routes)):
            assert client.project_jobs(PROJECT) == []

    def test_project_id_must_be_a_uuid(self):
        client = _client()
        with patch.object(client._transport._session, "request") as req:
            with pytest.raises(ValueError, match="project_id"):
                client.project_jobs("../jobs")
        req.assert_not_called()

    def test_document_from_another_project_is_refused(self):
        client = _client()
        routes = {
            f"/api/projects/{PROJECT}/documents": _documents(WORKSPACE),
            f"/api/projects/{PROJECT}/documents/{WORKSPACE}": _document(
                WORKSPACE, JOB_A, project="99999999-9999-4999-8999-999999999999"),
        }
        with patch.object(client._transport._session, "request", side_effect=_route(routes)):
            with pytest.raises(MarqovPlatformError) as info:
                client.project_jobs(PROJECT)
        assert info.value.code == "invalid_response"

    def test_status_for_a_different_job_is_refused(self):
        client = _client()
        routes = {
            f"/api/projects/{PROJECT}/documents": _documents(WORKSPACE),
            f"/api/projects/{PROJECT}/documents/{WORKSPACE}": _document(WORKSPACE, JOB_A),
            f"/api/jobs/{JOB_A}/status": _job_row(JOB_B),
        }
        with patch.object(client._transport._session, "request", side_effect=_route(routes)):
            with pytest.raises(MarqovPlatformError) as info:
                client.project_jobs(PROJECT)
        assert info.value.code == "invalid_response"

    def test_non_json_success_body_raises_transport_error(self):
        client = _client()
        routes = {f"/api/projects/{PROJECT}/documents": _response(200)}
        with patch.object(client._transport._session, "request", side_effect=_route(routes)):
            with pytest.raises(TransportError):
                client.project_jobs(PROJECT)

    def test_429_on_a_status_read_surfaces_rate_limited(self):
        client = _client()
        routes = {
            f"/api/projects/{PROJECT}/documents": _documents(WORKSPACE),
            f"/api/projects/{PROJECT}/documents/{WORKSPACE}": _document(WORKSPACE, JOB_A),
            f"/api/jobs/{JOB_A}/status": _response(429, {"error": "slow"}, headers={"Retry-After": "3"}),
        }
        with patch.object(client._transport._session, "request", side_effect=_route(routes)), \
                patch("marqov.platform._transport.time.sleep"):
            with pytest.raises(RateLimited) as info:
                client.project_jobs(PROJECT)
        assert info.value.retry_after == 3

    def test_deadline_is_passed_to_every_read(self):
        client = _client()
        deadline = time.monotonic() + 60
        routes = {
            f"/api/projects/{PROJECT}/documents": _documents(WORKSPACE),
            f"/api/projects/{PROJECT}/documents/{WORKSPACE}": _document(WORKSPACE, JOB_A),
            f"/api/jobs/{JOB_A}/status": _job_row(JOB_A),
        }
        with patch.object(client._transport, "request",
                          side_effect=lambda method, path, **kw: routes[path]) as req:
            client.project_jobs(PROJECT, deadline=deadline)
        assert [c.kwargs.get("deadline") for c in req.call_args_list] == [deadline] * 3
