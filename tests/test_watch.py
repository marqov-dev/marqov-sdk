"""Read-only watch contract, project scope, backoff and CLI composition."""

import json
import tomllib
from importlib.resources import files
from unittest.mock import MagicMock, Mock

import pytest
from click.testing import CliRunner

from marqov.platform.cli import main
from marqov.platform.watch import (
    MAX_BYTES,
    Poller,
    Reader,
    WatchError,
    project_job,
    resolve_key,
    save_snapshot,
    select_project,
    viewer_binary,
)

PROJECT = "00000000-0000-0000-0000-000000000001"
DOC = "00000000-0000-0000-0000-000000000002"
JOB = "00000000-0000-0000-0000-000000000003"
OTHER = "00000000-0000-0000-0000-000000000004"
P = {"id": PROJECT, "name": "Conference demo"}


def reader_with(routes):
    reader = Reader("test-secret", "https://app.marqov.ai")
    reader.get = Mock(side_effect=lambda path: routes[path])
    return reader


def routes():
    return {
        "/api/projects": {"projects": [P]},
        f"/api/projects/{PROJECT}/documents": {
            "documents": [{"id": DOC, "document_type": "workspace"}]
        },
        f"/api/projects/{PROJECT}/documents/{DOC}": {
            "document": {"project_id": PROJECT},
            "blocks": [{"job_run_id": JOB}, {"job_run_id": JOB}, {"content": "private script"}],
        },
        f"/api/jobs/{JOB}/status": {"id": JOB, "status": "running", "backend": "local"},
    }


def test_existing_auth_precedence_and_no_secret_in_missing_auth_error(monkeypatch, tmp_path):
    key = tmp_path / "key"
    key.write_text("file-secret\n")
    monkeypatch.setenv("MARQOV_PLATFORM_KEY", "sdk-secret")
    monkeypatch.setenv("MARQOV_API_KEY", "legacy-secret")
    assert resolve_key(key) == "sdk-secret"
    monkeypatch.delenv("MARQOV_PLATFORM_KEY")
    assert resolve_key(key) == "legacy-secret"
    monkeypatch.delenv("MARQOV_API_KEY")
    assert resolve_key(key) == "file-secret"


def test_project_jobs_are_only_explicit_workspace_links_and_deduplicated():
    reader = reader_with(routes())
    snapshot = reader.capture(P, 5)
    assert [j["workflow_id"] for j in snapshot["jobs"]] == [JOB]
    assert snapshot["project_name"] == P["name"]
    assert reader.get.call_count == 3
    assert all("/api/jobs?" not in call.args[0] for call in reader.get.call_args_list)
    assert "private script" not in json.dumps(snapshot)


def test_project_rejects_unlinked_explicit_job_and_wrong_document_identity():
    reader = reader_with(routes())
    with pytest.raises(WatchError, match="not linked"):
        reader.capture(P, 5, OTHER)
    data = routes()
    data[f"/api/projects/{PROJECT}/documents/{DOC}"]["document"]["project_id"] = OTHER
    with pytest.raises(WatchError, match="identity"):
        reader_with(data).capture(P, 5)


def test_empty_project_does_not_substitute_team_jobs():
    data = routes()
    data[f"/api/projects/{PROJECT}/documents/{DOC}"]["blocks"] = []
    snapshot = reader_with(data).capture(P, 5)
    assert snapshot["jobs"] == []
    assert "No jobs linked" in snapshot["message"]


def test_status_identity_mismatch_is_rejected():
    data = routes()
    data[f"/api/jobs/{JOB}/status"]["id"] = OTHER
    with pytest.raises(WatchError, match="identity"):
        reader_with(data).capture(P, 5)


def test_projection_redacts_payloads_and_terminal_controls():
    job = project_job(
        {
            "id": JOB,
            "status": "cancelling",
            "backend": "local\x1b[2J",
            "result": {"secret": "private results"},
            "error_message": "private error",
            "actual_cost_usd": float("nan"),
        }
    )
    assert job["cancellation"] == "requested"
    text = json.dumps(job)
    assert "private" not in text and "\\u001b" not in text and "NaN" not in text
    assert job["provider_job_id"] is None


def test_failure_keeps_last_observation_stale_and_honors_backoff(tmp_path):
    reader = reader_with(routes())
    snapshot = reader.capture(P, 5)
    path = tmp_path / "snapshot.json"
    save_snapshot(path, snapshot)
    poller = Poller(reader, P, 5, None, path, snapshot, 10)
    reader.capture = Mock(side_effect=WatchError("Read failed: HTTP 429", 429, 90))
    assert poller.refresh()
    retained = json.loads(path.read_text())
    assert retained["jobs"] == snapshot["jobs"]
    assert retained["observed_at"] == snapshot["observed_at"]
    assert retained["freshness"] == "stale" and poller.delay == 90
    reader.capture = Mock(return_value=snapshot)
    assert poller.refresh()
    assert json.loads(path.read_text())["freshness"] == "fresh"
    assert poller.delay == poller.interval


def test_auth_failure_stops_polling_and_late_capture_does_not_recreate_snapshot(tmp_path):
    reader = reader_with(routes())
    snapshot = reader.capture(P, 5)
    path = tmp_path / "snapshot.json"
    poller = Poller(reader, P, 5, None, path, snapshot, 10)
    reader.capture = Mock(side_effect=WatchError("Read failed: HTTP 401", 401))
    assert not poller.refresh()
    assert "refresh stopped" in json.loads(path.read_text())["message"]
    path.unlink()
    reader.capture = Mock(side_effect=lambda *args: poller.stop.set() or snapshot)
    assert not poller.refresh()
    assert not path.exists()


def test_reader_restricts_methods_paths_redirects_and_response_size():
    session = MagicMock()
    response = session.get.return_value.__enter__.return_value
    reader = Reader("secret", "https://app.marqov.ai", session)
    response.status_code = 302
    response.headers = {}
    with pytest.raises(WatchError, match="302"):
        reader.get("/api/projects")
    assert session.get.call_args.kwargs["allow_redirects"] is False
    with pytest.raises(WatchError, match="endpoint"):
        reader.get("/api/jobs/submit")
    response.status_code = 200
    response.iter_content.return_value = [b"x" * (MAX_BYTES + 1)]
    with pytest.raises(WatchError, match="1 MiB"):
        reader.get("/api/projects")


def test_cli_preserves_local_commands_and_explicit_project_selection(monkeypatch):
    runner = CliRunner()
    help_result = runner.invoke(main, ["--help"])
    assert help_result.exit_code == 0
    for command in ("watch", "run", "worker", "status", "list"):
        assert command in help_result.output
    monkeypatch.setattr("marqov.platform.watch.resolve_key", lambda _: "secret")
    monkeypatch.setattr("marqov.platform.watch.Reader", lambda *args: reader_with(routes()))
    result = runner.invoke(main, ["watch", "--project", P["name"], "--once"])
    assert result.exit_code == 0, result.output
    snapshot = json.loads(result.output)
    assert snapshot["jobs"][0]["workflow_id"] == JOB
    assert "secret" not in result.output
    result = runner.invoke(main, ["watch", "--once"])
    assert result.exit_code != 0 and "--project" in result.output


def test_ambiguous_project_and_invalid_viewer_are_clear_errors(tmp_path):
    with pytest.raises(Exception, match="ambiguous"):
        select_project([P, {"id": OTHER, "name": P["name"]}], P["name"], False)
    with pytest.raises(Exception, match="executable"):
        viewer_binary(tmp_path / "missing")


def test_packaged_viewer_sources_and_selection_scope_are_present():
    source = files("marqov.platform").joinpath("_watch_ui")
    assert (source / "Cargo.lock").is_file()
    assert (source / "src/main.rs").is_file()
    manifest = tomllib.loads((source / "Cargo.toml").read_text())
    assert manifest["package"]["name"] == "marqov-watch-ui"


def test_bad_job_payload_is_a_read_error_not_a_fresh_empty_view():
    reader = reader_with({"/api/jobs?limit=5": {"data": ["invalid"]}})
    with pytest.raises(WatchError, match="job-list"):
        reader.capture(None, 5)


def test_snapshot_write_failure_stops_worker_instead_of_leaving_a_fresh_view(monkeypatch, tmp_path):
    reader = reader_with(routes())
    snapshot = reader.capture(P, 5)
    poller = Poller(reader, P, 5, None, tmp_path / "snapshot.json", snapshot, 10)
    poller.delay = 0

    def full_disk(*args):
        raise OSError("private filesystem information")

    monkeypatch.setattr("marqov.platform.watch.save_snapshot", full_disk)
    poller.run()
    assert poller.stop.is_set()
    assert "OSError" in poller.fatal and "private" not in poller.fatal
