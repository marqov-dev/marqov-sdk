"""Read-only, project-aware Platform dashboard; network I/O stays outside Ratatui."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from typing import Any
from datetime import datetime, timezone
from urllib.parse import urlparse

import click
import requests

MAX_BYTES = 1_048_576
UUID = r"[0-9a-fA-F-]{36}"


class WatchError(Exception):
    """Sanitized error: response bodies and credentials never reach the screen."""

    def __init__(self, message: str, status: int = 0, retry_after: float = 0):
        super().__init__(message)
        self.status, self.retry_after = status, retry_after


def cache_dir() -> Path:
    path = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "marqov/watch"
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def resolve_key(key_file: Path | None) -> str:
    key = os.environ.get("MARQOV_PLATFORM_KEY") or os.environ.get("MARQOV_API_KEY")
    if not key:
        path = key_file or Path.home() / ".config/marqov/api-key"
        try:
            key = path.read_text().strip()
        except OSError:
            raise click.ClickException(
                "No API key. Set MARQOV_PLATFORM_KEY or use --key-file with your existing key."
            ) from None
    if not key or "\n" in key or "\r" in key:
        raise click.ClickException("The API key is empty or malformed.")
    return key


def clean(value: Any, limit: int = 200) -> str:
    """Keep remote strings printable, bounded and unable to inject terminal controls."""
    return "".join(c for c in str(value) if c.isprintable())[:limit]


def objects(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise WatchError(f"Unsupported {label} response.")
    return value


class Reader:
    def __init__(self, key: str, base: str, session: requests.Session | None = None):
        parsed = urlparse(base)
        if (
            parsed.scheme != "https"
            or parsed.netloc not in {"app.marqov.ai", "staging.marqov.ai"}
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise click.ClickException("Use https://app.marqov.ai or https://staging.marqov.ai.")
        self.base = base.rstrip("/")
        self.session = session or requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {key}", "Accept": "application/json"}
        )
        self.calls = 0

    def get(self, path: str) -> dict[str, Any]:
        pattern = rf"/api/(projects|jobs\?limit=([1-9]|10)|jobs/{UUID}/status|projects/{UUID}/documents(/{UUID})?)"
        if re.fullmatch(pattern, path) is None:
            raise WatchError("Unsupported read endpoint.")
        self.calls += 1
        try:
            with self.session.get(
                self.base + path, timeout=(3, 5), allow_redirects=False, stream=True
            ) as response:
                if response.status_code != 200:
                    retry = response.headers.get("Retry-After", "")
                    delay = min(3600, int(retry)) if retry.isdigit() else 0
                    raise WatchError(
                        f"Read failed: HTTP {response.status_code}", response.status_code, delay
                    )
                body = bytearray()
                for chunk in response.iter_content(65536):
                    body.extend(chunk)
                    if len(body) > MAX_BYTES:
                        raise WatchError("Response exceeds the 1 MiB limit.")
                result = json.loads(body)
                if not isinstance(result, dict):
                    raise WatchError("Unsupported API response.")
                return result
        except WatchError:
            raise
        except (requests.RequestException, OSError, ValueError) as error:
            raise WatchError(f"Read unavailable ({type(error).__name__}).") from None

    def projects(self) -> list[dict[str, str]]:
        rows = objects(self.get("/api/projects").get("projects"), "project-list")
        return [
            {"id": row["id"], "name": clean(row.get("name", "Untitled project"))}
            for row in rows
            if isinstance(row, dict) and re.fullmatch(UUID, str(row.get("id", "")))
        ]

    def capture(
        self, project: dict[str, str] | None, limit: int, job_id: str | None = None
    ) -> dict[str, Any]:
        if project is None and job_id is None:
            rows = objects(self.get(f"/api/jobs?limit={limit}").get("data"), "job-list")[:limit]
        else:
            ids = [job_id] if job_id else []
            if project is not None:
                documents = objects(
                    self.get(f"/api/projects/{project['id']}/documents").get("documents"),
                    "document-list",
                )
                workspaces = [d for d in documents if d.get("document_type") == "workspace"]
                if len(workspaces) > 1:
                    raise WatchError("Project has multiple workspaces; cannot select one safely.")
                linked: list[str] = []
                if workspaces:
                    doc = self.get(f"/api/projects/{project['id']}/documents/{workspaces[0]['id']}")
                    if doc.get("document", {}).get("project_id") != project["id"]:
                        raise WatchError("Workspace identity mismatch.")
                    blocks = objects(doc.get("blocks"), "workspace")
                    linked = list(
                        dict.fromkeys(
                            str(b["job_run_id"])
                            for b in blocks
                            if re.fullmatch(UUID, str(b.get("job_run_id", "")))
                        )
                    )
                if job_id and job_id not in linked:
                    raise WatchError("This job is not linked to the selected project's workspace.")
                if not job_id:
                    ids = linked[:limit]
            rows = []
            for identity in ids:
                status = self.get(f"/api/jobs/{identity}/status")
                if status.get("id") != identity:
                    raise WatchError("Job status identity mismatch.")
                rows.append(status)
        jobs = [project_job(row) for row in rows]
        return {
            "schema_version": 1,
            "source": "platform-api",
            "project_name": project["name"] if project else "All team jobs",
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "freshness": "fresh",
            "message": (
                "No jobs linked in this project's workspace."
                if project and not jobs
                else "FRESH AT CAPTURE · automatic refresh · read only"
            ),
            "jobs": jobs,
        }


def project_job(row: dict[str, Any]) -> dict[str, Any]:
    identity = row.get("id")
    if not re.fullmatch(UUID, str(identity)):
        raise WatchError("Invalid job identity.")
    raw_state = row.get("status", "unknown")
    state = (
        raw_state
        if isinstance(raw_state, str) and re.fullmatch(r"[a-z_]{1,40}", raw_state)
        else "unknown"
    )
    reason = clean(row.get("hold_reason") or "", 160)
    if state in {"failed", "dispatch_failed"}:
        reason = "Platform reports failure; details remain in the web record."
    details = [
        f"backend={clean(row.get('backend', 'unknown'), 80)}",
        "trace not fetched; scheduler identity not exposed by these read endpoints",
    ]
    for name in ("estimated_cost_usd", "actual_cost_usd"):
        value = row.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            details.append(f"{name}={value}")
    if row.get("updated_at"):
        details.append(f"record_updated_at={clean(row['updated_at'], 40)}")
    return {
        "workflow_id": identity,
        "provider_job_id": None,
        "slurm_job_id": None,
        "state": state,
        "reason": reason,
        "provenance": "; ".join(details),
        "cancellation": (
            "confirmed"
            if state == "cancelled"
            else "requested"
            if state == "cancelling"
            else "none"
        ),
    }


def save_snapshot(path: Path, snapshot: dict[str, Any]) -> None:
    temporary = path.with_suffix(".new")
    with open(temporary, "w", opener=lambda p, f: os.open(p, f, 0o600)) as stream:
        json.dump(snapshot, stream)
    temporary.replace(path)


class Poller:
    def __init__(
        self,
        reader: Reader,
        project: dict[str, str] | None,
        limit: int,
        job_id: str | None,
        path: Path,
        snapshot: dict[str, Any],
        interval: float,
    ):
        self.reader, self.project, self.limit, self.job_id = reader, project, limit, job_id
        self.path, self.snapshot = path, snapshot
        # At most ~30 requests/minute even for the project status fan-out.
        self.interval = max(interval, 2 * (limit + 2)) if project else interval
        self.delay = self.interval
        self.stop = threading.Event()
        self.failures = 0
        self.fatal: str | None = None

    def refresh(self) -> bool:
        try:
            observed = self.reader.capture(self.project, self.limit, self.job_id)
            if self.stop.is_set():
                return False
            self.snapshot = observed
            self.failures = 0
            self.delay = self.interval
        except WatchError as error:
            if self.stop.is_set():
                return False
            self.failures += 1
            self.delay = max(
                error.retry_after, min(120, self.interval * 2 ** min(self.failures, 5))
            )
            self.snapshot = {
                **self.snapshot,
                "freshness": "stale",
                "message": f"STALE · {error} · retry in {self.delay:g}s",
            }
            if error.status in {401, 403}:
                self.snapshot["message"] = (
                    f"STALE · {error} · refresh stopped; check authentication"
                )
                save_snapshot(self.path, self.snapshot)
                return False
        save_snapshot(self.path, self.snapshot)
        return True

    def run(self) -> None:
        while not self.stop.wait(self.delay):
            try:
                if not self.refresh():
                    return
            except (OSError, ValueError, TypeError, KeyError) as error:
                self.fatal = (
                    f"Refresh stopped ({type(error).__name__}); last view is no longer current."
                )
                self.stop.set()
                return


def viewer_binary(explicit: Path | None) -> Path:
    if explicit:
        if not explicit.is_file() or not os.access(explicit, os.X_OK):
            raise click.ClickException("The selected viewer is not executable.")
        return explicit.resolve()
    source = Path(__file__).with_name("_watch_ui")
    fingerprint = hashlib.sha256(
        b"".join(p.read_bytes() for p in sorted(source.rglob("*")) if p.is_file())
    ).hexdigest()[:16]
    cache = cache_dir() / fingerprint
    binary = cache / "marqov-watch-ui"
    if binary.is_file():
        return binary
    cargo = shutil.which("cargo")
    if cargo is None:
        raise click.ClickException("First launch needs Rust/Cargo installed, or --viewer PATH.")
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    click.echo("Preparing terminal viewer (first launch only)…", err=True)
    environment = {
        k: v for k, v in os.environ.items() if k not in {"MARQOV_PLATFORM_KEY", "MARQOV_API_KEY"}
    }
    environment.update(CARGO_BUILD_JOBS="1", CARGO_TARGET_DIR=str(cache / "target"))
    process = subprocess.Popen(
        [cargo, "build", "--locked", "--manifest-path", str(source / "Cargo.toml")],
        env=environment,
        start_new_session=True,
    )
    try:
        if process.wait(timeout=240) != 0:
            raise click.ClickException("Could not build the terminal viewer.")
        shutil.copy2(cache / "target/debug/marqov-watch-ui", binary)
    except subprocess.TimeoutExpired:
        raise click.ClickException("Viewer build exceeded four minutes; stopped.") from None
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
    return binary


def select_project(
    projects: list[dict[str, str]], name: str | None, all_jobs: bool
) -> dict[str, str] | None:
    if all_jobs:
        if name:
            raise click.UsageError("Choose either --project or --all.")
        return None
    if name:
        matches = [
            p for p in projects if name == p["id"] or name.casefold() == p["name"].casefold()
        ]
        if len(matches) != 1:
            raise click.ClickException(
                "Project not found or name is ambiguous; use --list-projects and its ID."
            )
        return matches[0]
    if not click.get_text_stream("stdin").isatty():
        raise click.UsageError("Choose --project ID or --all in non-interactive sessions.")
    click.echo("0  All team jobs")
    for index, project in enumerate(projects, 1):
        click.echo(f"{index}  {project['name']}  ({project['id']})")
    choice = click.prompt("Watch project", type=click.IntRange(0, len(projects)))
    return projects[choice - 1] if choice else None


@click.command()
@click.option("--project", help="Project ID or exact name; omit to choose interactively.")
@click.option(
    "--all", "all_jobs", is_flag=True, help="Watch recent team jobs, without a project filter."
)
@click.option("--list-projects", is_flag=True, help="List projects and exit.")
@click.option(
    "--job",
    "job_id",
    help="Watch one existing job ID; project membership is checked when selected.",
)
@click.option("--refresh", "interval", type=click.FloatRange(5, 300), default=10, show_default=True)
@click.option("--limit", type=click.IntRange(1, 10), default=5, show_default=True)
@click.option("--key-file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--url", envvar="MARQOV_PLATFORM_URL", default="https://app.marqov.ai", show_default=True
)
@click.option(
    "--viewer",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    envvar="MARQOV_WATCH_BIN",
)
@click.option("--once", is_flag=True, help="Print one sanitized JSON snapshot and exit.")
@click.option(
    "--duration",
    type=click.FloatRange(1, 3600),
    help="Optional automatic exit after this many seconds.",
)
def watch(
    project: str | None,
    all_jobs: bool,
    list_projects: bool,
    job_id: str | None,
    interval: float,
    limit: int,
    key_file: Path | None,
    url: str,
    viewer: Path | None,
    once: bool,
    duration: float | None,
) -> None:
    """Inspect existing Platform jobs in a read-only, automatically refreshed dashboard."""
    if job_id and re.fullmatch(UUID, job_id) is None:
        raise click.UsageError("--job must be a UUID.")
    reader = Reader(resolve_key(key_file), url)
    try:
        projects = reader.projects()
        if list_projects:
            for item in projects:
                click.echo(f"{item['id']}  {item['name']}")
            return
        selected = select_project(projects, project, all_jobs)
        snapshot = reader.capture(selected, limit, job_id)
        if once:
            snapshot["message"] = "FRESH AT CAPTURE · single observation"
            click.echo(json.dumps(snapshot, indent=2))
            return
        if (
            not click.get_text_stream("stdin").isatty()
            or not click.get_text_stream("stdout").isatty()
        ):
            raise click.ClickException("Interactive watch needs a terminal; use --once for JSON.")
        if os.name != "posix":
            raise click.ClickException(
                "Interactive watch currently supports macOS/Linux; use --once for JSON."
            )
        binary = viewer_binary(viewer)
        with tempfile.TemporaryDirectory(prefix="session-", dir=cache_dir()) as directory:
            path = Path(directory) / "snapshot.json"
            save_snapshot(path, snapshot)
            poller = Poller(reader, selected, limit, job_id, path, snapshot, interval)
            worker = threading.Thread(target=poller.run, daemon=True)
            child = None
            import termios
            import sys

            before = termios.tcgetattr(sys.stdin.fileno())
            try:
                environment = {
                    k: v
                    for k, v in os.environ.items()
                    if k not in {"MARQOV_PLATFORM_KEY", "MARQOV_API_KEY"}
                }
                child = subprocess.Popen([str(binary), "--snapshot", str(path)], env=environment)
                worker.start()
                deadline = time.monotonic() + duration if duration else float("inf")
                while child.poll() is None and time.monotonic() < deadline:
                    if poller.fatal:
                        raise click.ClickException(poller.fatal)
                    time.sleep(0.1)
                if child.poll() is not None and child.returncode:
                    raise click.ClickException("Terminal viewer exited with an error.")
            except KeyboardInterrupt:
                pass
            finally:
                poller.stop.set()
                if child is not None and child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=2)
                if worker.ident is not None:
                    worker.join(timeout=1)
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, before)
                click.echo("\x1b[?1049l\x1b[?25h\x1b[0m", nl=False)
    except WatchError as error:
        raise click.ClickException(str(error)) from None
    finally:
        reader.session.close()
