"""``marqov watch``: a read-only terminal view of Marqov Platform jobs.

Shows a team's recent jobs, or the jobs linked to one project, refreshed on a
paced interval.  The plain-text tier needs nothing beyond the SDK's own
dependencies; with the ``[watch]`` extra installed (Rich), a terminal gets a
live, coloured table instead.

Design rules (carried over from the Ratatui prototype in marqov-sdk#261):

- Read only: nothing here submits, retries or cancels a job.
- Paced, bounded refresh: one refresh at a time, each with a deadline; a
  project view slows down for its request fan-out; failures back off.
- A failed refresh keeps the last good snapshot on screen, marked STALE, with
  its original capture time.  An authentication or permission failure stops
  refreshing.
- Freshness and provenance are always shown, and a finished status is never
  presented as proof that provider resources were released.
- Response bodies, error text from the server and credentials are never
  printed: failures are described by exception type, HTTP status and code.

The command is registered with the core CLI through the ``marqov.cli``
entry-point group, so ``marqov.cli`` never imports this package.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from typing import IO, Any, Callable, Protocol
from urllib.parse import urlparse

import click

from ._models import JobSummary, Project
from .client import MarqovClient
from .errors import AuthenticationError, MarqovPlatformError, PermissionTierError

SCHEMA_VERSION = 1
MIN_REFRESH_SECONDS = 5.0
MAX_BACKOFF_SECONDS = 300.0
# A server's Retry-After is honoured up to this long.
MAX_RETRY_AFTER_SECONDS = 3600.0
# A project view considers its most recently active documents only, so one
# refresh is at most 1 + MAX_DOCUMENTS + limit requests.
MAX_DOCUMENTS = 10
# Upper bound on one refresh's retries; each request also has CLIENT_TIMEOUT_SECONDS.
REFRESH_DEADLINE_SECONDS = 20.0
CLIENT_TIMEOUT_SECONDS = 10.0
FRAME_SEPARATOR = "\n" + "=" * 72 + "\n"
FOOTER = (
    "This view is read only. Status as recorded by the Marqov Platform when captured; a finished "
    "status does not confirm that provider resources were released."
)


class RefreshStopped(Exception):
    """Refreshing stopped after an authentication or permission failure."""


class _Client(Protocol):
    @property
    def base_url(self) -> str: ...

    def projects(self, *, deadline: float | None = None) -> list[Project]: ...

    def recent_jobs(self, *, limit: int = 20, deadline: float | None = None) -> list[JobSummary]: ...

    def project_jobs(
        self,
        project_id: str,
        *,
        limit: int = 10,
        max_documents: int = 100,
        deadline: float | None = None,
    ) -> list[JobSummary]: ...


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _make_client() -> MarqovClient:
    return MarqovClient(timeout=CLIENT_TIMEOUT_SECONDS)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _describe(exc: MarqovPlatformError) -> str:
    """Name a failure without its message, which may carry server text."""
    text = type(exc).__name__
    if exc.status is not None:
        text += f" (HTTP {exc.status})"
    if exc.code:
        text += f" [{_clean(exc.code, 60)}]"
    return text


def _clean(value: Any, limit: int = 80) -> str:
    """Printable characters only, bounded: remote text cannot drive the terminal."""
    return "".join(c for c in str(value) if c.isprintable())[:limit]


# ---------------------------------------------------------------------------
# Snapshot: the one model both renderers and --once read
# ---------------------------------------------------------------------------


def snapshot_for(
    jobs: list[JobSummary],
    *,
    project: Project | None,
    platform_url: str,
    observed_at: datetime,
) -> dict[str, Any]:
    """Build a fresh snapshot (the ``--once`` JSON shape, ``schema_version`` 1)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "source": "platform-api",
        "platform_url": platform_url,
        "scope": {
            "kind": "project" if project else "team",
            "project_id": project.id if project else None,
            "project_name": project.name if project else None,
        },
        "observed_at": _iso(observed_at),
        "freshness": "fresh",
        "stale": None,
        "jobs": [
            {
                "id": job.id,
                "status": job.status,
                "backend": job.backend,
                "created_at": job.created_at,
                "updated_at": job.updated_at,
                "estimated_cost_usd": job.estimated_cost_usd,
                "actual_cost_usd": job.actual_cost_usd,
                "hold_reason": job.hold_reason,
            }
            for job in jobs
        ],
    }


class Watcher:
    """Refresh state: the last good snapshot, staleness, and the next delay.

    Args:
        client:   A :class:`MarqovClient` (or anything with its read methods).
        project:  The project to watch, or ``None`` for recent team jobs.
        limit:    Maximum number of jobs shown.
        interval: Seconds between successful refreshes.  A project view uses
                  at least ``2 * (1 + MAX_DOCUMENTS + limit)`` seconds: two
                  seconds per request of its worst case, so it stays at or
                  under 30 requests a minute.
    """

    def __init__(
        self, client: _Client, *, project: Project | None, limit: int, interval: float
    ) -> None:
        self.client = client
        self.project = project
        self.limit = limit
        self.interval = max(interval, 2 * (1 + MAX_DOCUMENTS + limit)) if project else interval
        self.delay = self.interval
        self.failures = 0
        self.stopped = False
        self.snapshot: dict[str, Any] | None = None
        self.refreshed_at: datetime | None = None

    def refresh(self) -> None:
        """Take one capture; on failure keep the last snapshot and mark it stale.

        Raises:
            MarqovPlatformError: The very first capture failed (there is no
                last good snapshot to show).
        """
        deadline = time.monotonic() + REFRESH_DEADLINE_SECONDS
        try:
            if self.project is None:
                jobs = self.client.recent_jobs(limit=self.limit, deadline=deadline)
            else:
                jobs = self.client.project_jobs(
                    self.project.id,
                    limit=self.limit,
                    max_documents=MAX_DOCUMENTS,
                    deadline=deadline,
                )
        except MarqovPlatformError as exc:
            if self.snapshot is None:
                raise
            # Times are taken once the reads end: retries can make a refresh long.
            now = self.refreshed_at = _utcnow()
            self.failures += 1
            self.stopped = isinstance(exc, (AuthenticationError, PermissionTierError))
            backoff = min(MAX_BACKOFF_SECONDS, self.interval * 2 ** min(self.failures, 16))
            retry_after = min(MAX_RETRY_AFTER_SECONDS, float(getattr(exc, "retry_after", None) or 0))
            self.delay = max(retry_after, backoff)
            previous = self.snapshot.get("stale") or {}
            self.snapshot = {
                **self.snapshot,
                "freshness": "stale",
                "stale": {
                    "since": previous.get("since", _iso(now)),
                    "reason": _describe(exc),
                    "retry_in_seconds": None if self.stopped else self.delay,
                    "refresh_stopped": self.stopped,
                },
            }
            return
        now = self.refreshed_at = _utcnow()
        self.failures = 0
        self.delay = self.interval
        self.snapshot = snapshot_for(
            jobs, project=self.project, platform_url=self.client.base_url, observed_at=now
        )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_COLUMNS = ("JOB", "STATUS", "BACKEND", "UPDATED", "EST. USD", "HOLD REASON")


def _cells(job: dict[str, Any]) -> tuple[str, ...]:
    cost = job["estimated_cost_usd"]
    return (
        _clean(job["id"], 36),
        _clean(job["status"], 40),
        _clean(job["backend"] or "-", 40),
        _clean(job["updated_at"] or "-", 25),
        "-" if cost is None else f"{cost:.2f}",
        _clean(job["hold_reason"] or "", 80),
    )


def _title(snapshot: dict[str, Any]) -> str:
    host = urlparse(snapshot["platform_url"]).netloc or snapshot["platform_url"]
    scope = snapshot["scope"]
    what = (
        f"Project: {_clean(scope['project_name'])}" if scope["kind"] == "project"
        else "Recent team jobs"
    )
    return f"Marqov Platform · {_clean(host)} · {what}"


def _freshness(snapshot: dict[str, Any], now: datetime) -> str:
    observed = datetime.strptime(snapshot["observed_at"], "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )
    age = max(0, int((now - observed).total_seconds()))
    captured = f"captured {observed:%H:%M:%S} UTC ({age}s ago)"
    stale = snapshot["stale"]
    if stale is None:
        return f"FRESH · {captured}"
    if stale["refresh_stopped"]:
        then = "refresh stopped; check MARQOV_PLATFORM_KEY and its permissions"
    else:
        then = f"retry in {stale['retry_in_seconds']:g}s"
    return f"STALE · showing data {captured} · refresh failed: {stale['reason']} · {then}"


def _empty(snapshot: dict[str, Any]) -> str:
    if snapshot["scope"]["kind"] == "project":
        return "No jobs are linked to this project (result blocks in its documents)."
    return "No jobs yet."


def render_plain(snapshot: dict[str, Any], *, now: datetime) -> str:
    """Render a snapshot as plain text (no colour, no dependency)."""
    lines = [_title(snapshot), _freshness(snapshot, now), ""]
    rows = [_cells(job) for job in snapshot["jobs"]]
    if rows:
        widths = [max(len(r[i]) for r in [_COLUMNS, *rows]) for i in range(len(_COLUMNS))]
        for row in [_COLUMNS, tuple("-" * w for w in widths), *rows]:
            lines.append("  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip())
    else:
        lines.append(_empty(snapshot))
    lines += ["", FOOTER]
    return "\n".join(lines)


_STATUS_STYLES = {
    "completed": "green",
    "failed": "red",
    "dispatch_failed": "red",
    "running": "cyan",
    "pending": "yellow",
    "queued": "yellow",
    "scheduled": "yellow",
    "cancelling": "magenta",
    "cancelled": "magenta",
}


def render_rich(snapshot: dict[str, Any], *, now: datetime) -> Any:
    """Render a snapshot as a Rich renderable (needs the ``[watch]`` extra).

    Every remote string goes in as :class:`rich.text.Text`, never as markup.
    """
    from rich.console import Group
    from rich.table import Table
    from rich.text import Text

    stale = snapshot["stale"] is not None
    header = Text(_freshness(snapshot, now), style="bold white on red" if stale else "green")
    parts: list[Any] = [Text(_title(snapshot), style="bold"), header]
    if snapshot["jobs"]:
        table = Table(*_COLUMNS, expand=False, header_style="bold", style="dim" if stale else "")
        for job in snapshot["jobs"]:
            cells = [Text(c) for c in _cells(job)]
            cells[1].stylize(_STATUS_STYLES.get(job["status"], ""))
            table.add_row(*cells)
        parts.append(table)
    else:
        parts.append(Text(_empty(snapshot)))
    parts.append(Text(FOOTER, style="dim"))
    return Group(*parts)


def _is_terminal(stream: Any) -> bool:
    return bool(getattr(stream, "isatty", lambda: False)())


def _tolerate_unencodable(stream: Any) -> None:
    """Print ``?`` for characters the stream cannot encode instead of crashing.

    A redirected stdout on Windows uses the locale code page (cp1252, say),
    which cannot encode most non-Latin project names.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(errors="replace")


def _rich_available() -> bool:
    try:
        import rich.live  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# Refresh loops
# ---------------------------------------------------------------------------


def _stop_if_needed(watcher: Watcher) -> None:
    if watcher.stopped:
        raise RefreshStopped(
            "Refresh stopped after an authentication or permission failure; fix the key and rerun."
        )


def run_plain(
    watcher: Watcher,
    *,
    out: IO[str],
    sleep: Callable[[float], None] = time.sleep,
    refreshes: int | None = None,
    clear: bool,
) -> None:
    """Refresh and redraw as plain text until interrupted (or ``refreshes`` frames).

    With ``clear`` each frame replaces the last (a terminal); otherwise frames
    are appended, each followed by :data:`FRAME_SEPARATOR` (a pipe or a log).
    """
    count = 0
    while True:
        watcher.refresh()
        count += 1
        assert watcher.snapshot is not None and watcher.refreshed_at is not None
        if clear:
            click.clear()
        out.write(render_plain(watcher.snapshot, now=watcher.refreshed_at) + FRAME_SEPARATOR)
        out.flush()
        _stop_if_needed(watcher)
        if refreshes is not None and count >= refreshes:
            return
        sleep(watcher.delay)


def run_rich(
    watcher: Watcher, *, sleep: Callable[[float], None] = time.sleep, console: Any = None
) -> None:
    """Refresh into a Rich live table until interrupted (``console``: Rich's default)."""
    from rich.live import Live

    watcher.refresh()
    assert watcher.snapshot is not None and watcher.refreshed_at is not None
    with Live(render_rich(watcher.snapshot, now=watcher.refreshed_at), auto_refresh=False,
              console=console) as live:
        while True:
            _stop_if_needed(watcher)
            sleep(watcher.delay)
            watcher.refresh()
            live.update(render_rich(watcher.snapshot, now=watcher.refreshed_at), refresh=True)


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


def _select(projects: list[Project], wanted: str) -> Project:
    matches = [p for p in projects if p.id == wanted.lower()]
    if not matches:
        matches = [p for p in projects if p.name.casefold() == wanted.casefold()]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise click.UsageError(
            f"--project {wanted!r} matches {len(matches)} projects; pass the project ID instead."
        )
    raise click.UsageError(f"No project matches --project {wanted!r} (an ID or exact name).")


def _prompt(projects: list[Project]) -> Project | None:
    click.echo("0  Recent team jobs")
    for number, project in enumerate(projects, 1):
        click.echo(f"{number}  {_clean(project.name)}  ({project.id})")
    choice = click.prompt("Watch", type=click.IntRange(0, len(projects)))
    return projects[choice - 1] if choice else None


@click.command()
@click.option("--project", help="Project ID or exact name (case-insensitive) to watch.")
@click.option("--all", "all_jobs", is_flag=True, help="Watch the team's recent jobs.")
@click.option("--limit", type=click.IntRange(1, 20), default=10, show_default=True,
              help="Maximum number of jobs shown.")
@click.option("--refresh", "interval", type=click.FloatRange(MIN_REFRESH_SECONDS, 3600),
              default=10.0, show_default=True, help="Seconds between refreshes.")
@click.option("--once", is_flag=True, help="Print one JSON snapshot to stdout and exit.")
def watch(project: str | None, all_jobs: bool, limit: int, interval: float, once: bool) -> None:
    """Watch Marqov Platform jobs in the terminal (read only).

    Uses MARQOV_PLATFORM_KEY, and MARQOV_PLATFORM_URL when set. Without
    --project or --all, a terminal is offered a list of projects to choose from.
    """
    if project and all_jobs:
        raise click.UsageError("Choose either --project or --all, not both.")
    # The menu is written to stdout, so it must be a terminal too: a menu in
    # `--once > snap.json` would corrupt the JSON.
    interactive = _is_terminal(sys.stdin) and _is_terminal(sys.stdout)
    if not project and not all_jobs and not interactive:
        raise click.UsageError("Choose --project ID|NAME or --all when not in a terminal.")
    try:
        client = _make_client()
    except AuthenticationError:
        raise click.ClickException(
            "No Marqov Platform API key: set MARQOV_PLATFORM_KEY."
        ) from None
    try:
        selected: Project | None = None
        if project:
            selected = _select(client.projects(), project)
        elif not all_jobs:
            selected = _prompt(client.projects())
        watcher = Watcher(client, project=selected, limit=limit, interval=interval)
        if once:
            watcher.refresh()
            click.echo(json.dumps(watcher.snapshot, indent=2))
        elif _rich_available() and _is_terminal(sys.stdout):
            run_rich(watcher)
        else:
            _tolerate_unencodable(sys.stdout)
            run_plain(watcher, out=sys.stdout, clear=_is_terminal(sys.stdout))
    except KeyboardInterrupt:
        pass
    except RefreshStopped as exc:
        raise click.ClickException(str(exc)) from None
    except MarqovPlatformError as exc:
        raise click.ClickException(f"Platform read failed: {_describe(exc)}") from None
    finally:
        client.close()
