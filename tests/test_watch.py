"""Tests for ``marqov watch``, the read-only Platform job view (marqov-sdk#292).

The view is pure Python: a plain-text redraw that needs no extra dependency,
and a Rich live table when the ``[watch]`` extra is installed. A fake client
stands in for :class:`~marqov.platform.MarqovClient`; no network is used.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pytest
from click.testing import CliRunner

from marqov.cli import main as cli
from marqov.platform import JobSummary, Project
from marqov.platform import watch as watch_mod
from marqov.platform.errors import AuthenticationError, RateLimited, TransportError

NOW = datetime(2026, 10, 26, 12, 0, 0, tzinfo=timezone.utc)
PROJECT = Project(id="11111111-1111-4111-8111-111111111111", name="Bell study",
                  description="", created_at=None, updated_at=None)
OTHER = Project(id="44444444-4444-4444-8444-444444444444", name="QAOA sweep",
                description="", created_at=None, updated_at=None)


def _job(n: int, status: str, **kw) -> JobSummary:
    return JobSummary(
        id=f"{n:08d}-0000-4000-8000-000000000000",
        status=status,
        backend=kw.pop("backend", "sv1"),
        created_at="2026-10-26T11:00:00Z",
        updated_at="2026-10-26T11:59:00Z",
        estimated_cost_usd=kw.pop("estimated_cost_usd", 0.01),
        actual_cost_usd=kw.pop("actual_cost_usd", None),
        hold_reason=kw.pop("hold_reason", None),
    )


# marqov-sdk#261's eight-state demo journey, as job summaries.
DEMO_STATES = ["queued", "running", "needs_review", "checking_execution",
               "running", "cancelled", "completed", "failed"]
DEMO_JOBS = [_job(i + 1, s) for i, s in enumerate(DEMO_STATES)]


class FakeClient:
    """Stands in for MarqovClient; each listing answers from a queue of outcomes."""

    base_url = "https://app.marqov.ai"

    def __init__(self, *, recent=(), linked=(), projects=(PROJECT, OTHER)):
        self.recent = list(recent)
        self.linked = list(linked)
        self._projects = list(projects)
        self.calls: list[tuple] = []
        self.closed = False

    @staticmethod
    def _next(queue):
        outcome = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def projects(self, *, deadline=None):
        self.calls.append(("projects",))
        return self._projects

    def recent_jobs(self, *, limit=20, deadline=None):
        self.calls.append(("recent_jobs", limit, deadline))
        return self._next(self.recent)[:limit]

    def project_jobs(self, project_id, *, limit=10, max_documents=100, deadline=None):
        self.calls.append(("project_jobs", project_id, limit, deadline, max_documents))
        return self._next(self.linked)[:limit]

    def close(self):
        self.closed = True


@pytest.fixture
def frozen_now(monkeypatch):
    monkeypatch.setattr(watch_mod, "_utcnow", lambda: NOW)


@pytest.fixture
def fake(monkeypatch):
    """Install a FakeClient for the CLI; tests set its outcomes."""
    client = FakeClient(recent=[[_job(1, "running"), _job(2, "completed")]],
                        linked=[[_job(3, "pending", hold_reason="Scheduled window")]])
    monkeypatch.setattr(watch_mod, "_make_client", lambda: client)
    return client


# ---------------------------------------------------------------------------
# --once: the documented JSON snapshot
# ---------------------------------------------------------------------------


class TestOnce:
    def test_team_snapshot_matches_the_documented_shape(self, fake, frozen_now):
        result = CliRunner().invoke(cli, ["watch", "--all", "--once", "--limit", "2"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == {
            "schema_version": 1,
            "source": "platform-api",
            "platform_url": "https://app.marqov.ai",
            "scope": {"kind": "team", "project_id": None, "project_name": None},
            "observed_at": "2026-10-26T12:00:00Z",
            "freshness": "fresh",
            "stale": None,
            "jobs": [
                {
                    "id": "00000001-0000-4000-8000-000000000000",
                    "status": "running",
                    "backend": "sv1",
                    "created_at": "2026-10-26T11:00:00Z",
                    "updated_at": "2026-10-26T11:59:00Z",
                    "estimated_cost_usd": 0.01,
                    "actual_cost_usd": None,
                    "hold_reason": None,
                },
                {
                    "id": "00000002-0000-4000-8000-000000000000",
                    "status": "completed",
                    "backend": "sv1",
                    "created_at": "2026-10-26T11:00:00Z",
                    "updated_at": "2026-10-26T11:59:00Z",
                    "estimated_cost_usd": 0.01,
                    "actual_cost_usd": None,
                    "hold_reason": None,
                },
            ],
        }
        assert ("recent_jobs", 2) == fake.calls[-1][:2]
        assert fake.closed

    def test_project_selected_by_name_ignoring_case(self, fake, frozen_now):
        result = CliRunner().invoke(cli, ["watch", "--project", "bell STUDY", "--once"])
        assert result.exit_code == 0, result.output
        snapshot = json.loads(result.output)
        assert snapshot["scope"] == {"kind": "project", "project_id": PROJECT.id,
                                     "project_name": "Bell study"}
        assert [j["status"] for j in snapshot["jobs"]] == ["pending"]
        assert fake.calls[-1][:2] == ("project_jobs", PROJECT.id)

    def test_project_selected_by_id(self, fake, frozen_now):
        result = CliRunner().invoke(cli, ["watch", "--project", OTHER.id, "--once"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["scope"]["project_name"] == "QAOA sweep"

    def test_unknown_or_ambiguous_project_is_a_usage_error(self, fake):
        fake._projects = [PROJECT, Project(id="55555555-5555-4555-8555-555555555555",
                                           name="Bell Study", description="",
                                           created_at=None, updated_at=None)]
        for name in ("Nope", "bell study"):
            result = CliRunner().invoke(cli, ["watch", "--project", name, "--once"])
            assert result.exit_code == 2
            assert "--project" in result.output

    def test_project_and_all_together_are_refused(self, fake):
        result = CliRunner().invoke(cli, ["watch", "--project", "Bell study", "--all", "--once"])
        assert result.exit_code == 2

    def test_no_scope_without_a_terminal_is_refused(self, fake):
        result = CliRunner().invoke(cli, ["watch", "--once"])
        assert result.exit_code == 2
        assert "--project" in result.output and "--all" in result.output

    def test_no_scope_with_stdout_redirected_is_refused_not_prompted(self, fake, monkeypatch):
        """A menu on stdout would corrupt `--once > snap.json` or a piped live view."""
        monkeypatch.setattr(watch_mod, "_is_terminal", lambda stream: stream is sys.stdin)
        result = CliRunner().invoke(cli, ["watch", "--once"])
        assert result.exit_code == 2
        assert "Recent team jobs" not in result.output
        assert fake.calls == []

    def test_platform_error_exits_nonzero_without_a_traceback(self, fake):
        fake.recent = [TransportError("Server returned a non-JSON response")]
        result = CliRunner().invoke(cli, ["watch", "--all", "--once"])
        assert result.exit_code == 1
        assert "Traceback" not in result.output
        assert fake.closed

    def test_once_does_not_need_rich(self, fake, frozen_now, monkeypatch):
        monkeypatch.setitem(sys.modules, "rich", None)  # `import rich` now fails
        result = CliRunner().invoke(cli, ["watch", "--all", "--once"])
        assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# Live refresh: plain-text tier (no Rich)
# ---------------------------------------------------------------------------


class TestPlainRefresh:
    def test_two_refreshes_keep_the_last_good_view_and_mark_it_stale(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "rich", None)
        times = iter([NOW, NOW + timedelta(seconds=10)])
        monkeypatch.setattr(watch_mod, "_utcnow", lambda: next(times))
        client = FakeClient(recent=[
            [_job(1, "running"), _job(2, "completed")],
            RateLimited("rate limited", status=429, retry_after=30),
        ])
        watcher = watch_mod.Watcher(client, project=None, limit=10, interval=10)
        out = io.StringIO()
        sleeps: list[float] = []
        watch_mod.run_plain(watcher, out=out, sleep=sleeps.append, refreshes=2, clear=False)

        first, second = out.getvalue().split(watch_mod.FRAME_SEPARATOR)[:2]
        assert "FRESH" in first and "STALE" not in first
        assert "00000001-0000-4000-8000-000000000000" in first
        assert "running" in first and "completed" in first

        assert "STALE" in second
        assert "RateLimited" in second and "HTTP 429" in second
        assert "retry in 30s" in second
        # The last good rows stay on screen, with their original capture time.
        assert "00000001-0000-4000-8000-000000000000" in second
        assert "captured 12:00:00 UTC" in second
        assert sleeps == [10]

    def test_auth_failure_stops_refreshing(self):
        client = FakeClient(recent=[[_job(1, "running")], AuthenticationError("bad key", status=401)])
        watcher = watch_mod.Watcher(client, project=None, limit=10, interval=5)
        out = io.StringIO()
        with pytest.raises(watch_mod.RefreshStopped):
            watch_mod.run_plain(watcher, out=out, sleep=lambda s: None, refreshes=5, clear=False)
        assert out.getvalue().count(watch_mod.FRAME_SEPARATOR) == 2
        assert "refresh stopped" in out.getvalue()
        assert len([c for c in client.calls if c[0] == "recent_jobs"]) == 2

    def test_error_text_from_the_server_is_not_shown(self, monkeypatch):
        monkeypatch.setattr(watch_mod, "_utcnow", lambda: NOW)
        client = FakeClient(recent=[[_job(1, "running")],
                                    TransportError("<html>secret body marqey_live_x</html>", status=502)])
        watcher = watch_mod.Watcher(client, project=None, limit=10, interval=5)
        watcher.refresh()
        watcher.refresh()
        text = watch_mod.render_plain(watcher.snapshot, now=NOW)
        assert "TransportError" in text and "HTTP 502" in text
        assert "secret" not in text and "marqey" not in text

    def test_terminal_controls_in_remote_strings_are_removed(self):
        job = _job(1, "running", backend="sv1\x1b[2J\x1b]52;c;aGk=\x07", hold_reason="wait\rX")
        snapshot = watch_mod.snapshot_for([job], project=None, platform_url="https://app.marqov.ai",
                                          observed_at=NOW)
        text = watch_mod.render_plain(snapshot, now=NOW)
        assert "\x1b" not in text and "\x07" not in text and "\r" not in text

    def test_unencodable_names_do_not_crash_a_legacy_code_page_stream(self):
        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding="cp1252")
        watch_mod._tolerate_unencodable(stream)
        project = Project(id=PROJECT.id, name="Квантовый study", description="",
                          created_at=None, updated_at=None)
        snapshot = watch_mod.snapshot_for([], project=project, platform_url="https://app.marqov.ai",
                                          observed_at=NOW)
        stream.write(watch_mod.render_plain(snapshot, now=NOW))
        stream.flush()
        assert b"? study" in raw.getvalue() or b"?????????" in raw.getvalue()

    def test_footer_does_not_claim_resources_were_released(self):
        snapshot = watch_mod.snapshot_for([_job(1, "completed")], project=None,
                                          platform_url="https://app.marqov.ai", observed_at=NOW)
        text = watch_mod.render_plain(snapshot, now=NOW)
        assert "read only" in text
        assert "does not confirm" in text and "released" in text

    def test_empty_project_says_so(self):
        snapshot = watch_mod.snapshot_for([], project=PROJECT, platform_url="https://app.marqov.ai",
                                          observed_at=NOW)
        text = watch_mod.render_plain(snapshot, now=NOW)
        assert "Bell study" in text
        assert "No jobs are linked to this project" in text


# ---------------------------------------------------------------------------
# Pacing and backoff
# ---------------------------------------------------------------------------


class TestPacing:
    def test_success_waits_the_refresh_interval(self):
        watcher = watch_mod.Watcher(FakeClient(recent=[[]]), project=None, limit=10, interval=7)
        watcher.refresh()
        assert watcher.delay == 7

    def test_project_view_bounds_and_paces_its_fan_out(self):
        """A project refresh is at most 1 + MAX_DOCUMENTS + limit requests, paced to ~30/min."""
        client = FakeClient(linked=[[]])
        watcher = watch_mod.Watcher(client, project=PROJECT, limit=10, interval=5)
        watcher.refresh()
        assert client.calls[-1][4] == watch_mod.MAX_DOCUMENTS == 10
        worst_case = 1 + watch_mod.MAX_DOCUMENTS + 10
        assert watcher.delay == 2 * worst_case
        assert worst_case / watcher.delay * 60 <= 30

    def test_failures_back_off_and_success_resets(self):
        error = TransportError("down", status=503)
        client = FakeClient(recent=[[], error, error, error, []])
        watcher = watch_mod.Watcher(client, project=None, limit=10, interval=10)
        delays = []
        for _ in range(5):
            watcher.refresh()
            delays.append(watcher.delay)
        assert delays == [10, 20, 40, 80, 10]

    def test_backoff_is_capped(self):
        error = TransportError("down", status=503)
        watcher = watch_mod.Watcher(FakeClient(recent=[[], error]), project=None, limit=10, interval=60)
        for _ in range(11):
            watcher.refresh()
        assert watcher.delay == watch_mod.MAX_BACKOFF_SECONDS

    def test_retry_after_is_honoured_when_longer(self):
        client = FakeClient(recent=[[], RateLimited("slow", status=429, retry_after=90)])
        watcher = watch_mod.Watcher(client, project=None, limit=10, interval=10)
        watcher.refresh()
        watcher.refresh()
        assert watcher.delay == 90

    def test_huge_retry_after_is_capped(self):
        client = FakeClient(recent=[[], RateLimited("slow", status=429, retry_after=10**12)])
        watcher = watch_mod.Watcher(client, project=None, limit=10, interval=10)
        watcher.refresh()
        watcher.refresh()
        assert watcher.delay == watch_mod.MAX_RETRY_AFTER_SECONDS

    def test_first_capture_failure_is_raised_not_shown_as_stale(self):
        client = FakeClient(recent=[TransportError("down", status=503)])
        watcher = watch_mod.Watcher(client, project=None, limit=10, interval=10)
        with pytest.raises(TransportError):
            watcher.refresh()
        assert watcher.snapshot is None

    def test_each_refresh_is_bounded_by_a_deadline(self, monkeypatch):
        monkeypatch.setattr(watch_mod.time, "monotonic", lambda: 1000.0)
        client = FakeClient(recent=[[]])
        watch_mod.Watcher(client, project=None, limit=3, interval=10).refresh()
        assert client.calls[-1] == ("recent_jobs", 3, 1000.0 + watch_mod.REFRESH_DEADLINE_SECONDS)


# ---------------------------------------------------------------------------
# Rich tier
# ---------------------------------------------------------------------------


class TestRich:
    def test_live_table_shows_every_demo_state_and_the_stale_header(self, monkeypatch):
        rich_console = pytest.importorskip("rich.console")
        monkeypatch.setattr(watch_mod, "_utcnow", lambda: NOW)
        client = FakeClient(recent=[DEMO_JOBS, TransportError("down", status=503)])
        watcher = watch_mod.Watcher(client, project=None, limit=10, interval=10)
        watcher.refresh()
        watcher.refresh()
        console = rich_console.Console(record=True, width=140, color_system=None)
        console.print(watch_mod.render_rich(watcher.snapshot, now=NOW + timedelta(seconds=20)))
        text = console.export_text()
        rows = [line for line in text.splitlines() if "-0000-4000-8000-" in line]
        assert len(rows) == len(DEMO_JOBS)
        for job, row in zip(DEMO_JOBS, rows):
            assert job.id in row and f" {job.status} " in row
        assert "STALE" in text
        assert "TransportError" in text and "HTTP 503" in text
        assert "captured 12:00:00 UTC" in text

    def test_run_rich_updates_the_live_table_until_refresh_stops(self, monkeypatch):
        rich_console = pytest.importorskip("rich.console")
        monkeypatch.setattr(watch_mod, "_utcnow", lambda: NOW)
        client = FakeClient(recent=[DEMO_JOBS, TransportError("down", status=503),
                                    AuthenticationError("bad key", status=401)])
        watcher = watch_mod.Watcher(client, project=None, limit=10, interval=10)
        out = io.StringIO()
        console = rich_console.Console(file=out, force_terminal=True, width=140, color_system=None)
        sleeps: list[float] = []
        with pytest.raises(watch_mod.RefreshStopped):
            watch_mod.run_rich(watcher, sleep=sleeps.append, console=console)
        assert sleeps == [10, 20]
        assert "refresh stopped" in out.getvalue()
        assert "needs_review" in out.getvalue()

    def test_remote_strings_cannot_inject_rich_markup(self):
        rich_console = pytest.importorskip("rich.console")
        job = _job(1, "running", backend="[bold]sv1[/bold][blink]")
        snapshot = watch_mod.snapshot_for([job], project=None, platform_url="https://app.marqov.ai",
                                          observed_at=NOW)
        console = rich_console.Console(record=True, width=160, color_system=None)
        console.print(watch_mod.render_rich(snapshot, now=NOW))
        assert "[bold]sv1[/bold][blink]" in console.export_text()


# ---------------------------------------------------------------------------
# CLI wiring and the core/platform boundary
# ---------------------------------------------------------------------------


class TestCliWiring:
    def test_live_view_without_rich_draws_plain_frames(self, fake, monkeypatch):
        monkeypatch.setitem(sys.modules, "rich", None)
        monkeypatch.setattr(watch_mod.time, "sleep", lambda s: None)
        fake.recent = [[_job(1, "running")], AuthenticationError("bad key", status=401)]
        result = CliRunner().invoke(cli, ["watch", "--all"])
        assert result.exit_code == 1
        assert result.output.count(watch_mod.FRAME_SEPARATOR) == 2
        assert "FRESH" in result.output and "STALE" in result.output
        assert "Refresh stopped" in result.output

    def test_terminal_with_rich_uses_the_live_table(self, fake, monkeypatch):
        pytest.importorskip("rich")
        chosen = []
        monkeypatch.setattr(watch_mod, "_is_terminal", lambda stream: True)
        monkeypatch.setattr(watch_mod, "run_rich", lambda watcher: chosen.append("rich"))
        monkeypatch.setattr(watch_mod, "run_plain", lambda watcher, **kw: chosen.append("plain"))
        result = CliRunner().invoke(cli, ["watch", "--all"])
        assert result.exit_code == 0, result.output
        assert chosen == ["rich"]

    def test_watch_is_listed_in_the_marqov_cli(self):
        result = CliRunner().invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "watch" in result.output

    def test_watch_help_describes_the_options(self):
        result = CliRunner().invoke(cli, ["watch", "--help"])
        assert result.exit_code == 0
        for option in ("--project", "--all", "--limit", "--refresh", "--once"):
            assert option in result.output

    def test_local_commands_do_not_load_the_platform_client(self):
        """`marqov list --help` must not import marqov.platform: commands load on use."""
        code = (
            "import sys\n"
            "from click.testing import CliRunner\n"
            "from marqov.cli import main\n"
            "r = CliRunner().invoke(main, ['list', '--help'])\n"
            "assert r.exit_code == 0, r.output\n"
            "print('marqov.platform' in sys.modules)\n"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        assert out.stdout.strip() == "False"
