# Watch Platform jobs in your terminal

`marqov watch` is a read-only view of your team's jobs on the Marqov Platform,
refreshed on an interval. It cannot submit, retry or cancel a job.

```sh
export MARQOV_PLATFORM_KEY=marqey_live_...
marqov watch --all                      # the team's recent jobs
marqov watch --project "Bell study"     # jobs linked to one project
marqov watch --project PROJECT_UUID --limit 5 --refresh 30
marqov watch --all --once               # one JSON snapshot, then exit
```

`MARQOV_PLATFORM_URL` selects another platform URL, as it does for
`MarqovClient`. When both stdin and stdout are a terminal, running `marqov watch`
with neither `--project` nor `--all` lists your projects to choose from;
otherwise (a pipe, a redirect such as `--once > snap.json`, or a script) one of
the two is required.

> **Live-server caveat:** like the rest of `marqov.platform`, this command is
> tested against a mocked transport and has not yet been verified against the
> hosted API.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--project ID\|NAME` | | A project UUID, or its exact name ignoring case. A name shared by several projects is refused; use the ID. |
| `--all` | | The team's most recent jobs (`GET /api/jobs`). |
| `--limit N` | 10 | Jobs shown, 1 to 20. |
| `--refresh SECONDS` | 10 | Seconds between refreshes, at least 5. |
| `--once` | | Print one JSON snapshot to stdout and exit. |

## What a project shows

The Platform has no endpoint listing a project's jobs. A job belongs to a
project's view when a result block in one of the project's active documents
(Workspaces and Reports) links it, so the view reads the project's documents
(`GET /api/projects/{id}/documents` and each document's blocks), then each
linked job's status. Only the 10 most recently active documents are read, so a
job linked only from an older document does not appear. Jobs appear in
document order (most recent activity first), then block order. A project with
no linked jobs shows an empty view; it never falls back to unrelated team jobs.

## Refresh, pacing and staleness

- One refresh runs at a time. Each one has a 20 second deadline for its
  retries, and each request a 10 second timeout.
- A project refresh makes at most `1 + 10 + limit` requests (the document
  list, up to 10 documents, and one status per job), so a project view
  refreshes no faster than every `2 × (11 + limit)` seconds (42 seconds at the
  default limit). That keeps it at or under 30 requests a minute.
- When a refresh fails, the last good view stays on screen marked **STALE**,
  with its original capture time and the failure's type and HTTP status.
  Retries back off (doubling, at most 5 minutes) and wait at least as long as
  a `Retry-After` the server sends, up to an hour.
- After an authentication or permission failure (HTTP 401 or 403) refreshing
  stops and the command exits with an error; fix the key and run it again.
- If the very first read fails, the command exits with an error instead.

The header always shows the platform host, the scope, and when the data was
captured. A status is what the Platform recorded at capture time: a finished
status does not confirm that provider resources were released. Server error
messages, results, scripts and the API key are never printed.

## Plain text and Rich

Without extras the view is plain text: in a terminal each refresh redraws the
screen; when stdout is not a terminal, each frame is appended, followed by a
line of `=` characters. This tier uses no POSIX-only terminal APIs (the
redraw is `click.clear()`), so it is meant to work on Windows too; it has been
tested on macOS. Characters the output encoding cannot represent (a
non-Latin project name on a Windows code page, say) print as `?`. For a live, coloured table, install the extra:

```sh
pip install "marqov[watch]"
```

When Rich is installed and stdout is a terminal, the live table is used.
Status names are always shown as text, not by colour alone. Press Ctrl+C to
quit.

## The `--once` snapshot

`--once` prints one JSON object (`schema_version` 1) and exits, for scripts:

```json
{
  "schema_version": 1,
  "source": "platform-api",
  "platform_url": "https://app.marqov.ai",
  "scope": {"kind": "project", "project_id": "<uuid>", "project_name": "Bell study"},
  "observed_at": "2026-10-26T12:00:00Z",
  "freshness": "fresh",
  "stale": null,
  "jobs": [
    {
      "id": "<uuid>",
      "status": "running",
      "backend": "sv1",
      "created_at": "2026-10-26T11:00:00Z",
      "updated_at": "2026-10-26T11:59:00Z",
      "estimated_cost_usd": 0.01,
      "actual_cost_usd": null,
      "hold_reason": null
    }
  ]
}
```

`scope.kind` is `"team"` for `--all`, with `project_id` and `project_name`
null. `--once` always reports `"freshness": "fresh"`: a failed read exits with
an error instead of printing a snapshot. `actual_cost_usd` is filled for team
listings only; a project's jobs come from status reads, which do not carry it.
The reverse holds for `hold_reason`: status reads carry it, but team job-list
rows do not, so it is always null with `--all`.
The shape is pinned by `tests/test_watch.py::TestOnce::test_team_snapshot_matches_the_documented_shape`.

## Origins

The design comes from a Ratatui (Rust) prototype in marqov-sdk#261, which was
not merged because it needed a Rust toolchain at first launch. This version
keeps its rules (read only, paced refresh, stale marking, provenance) in pure
Python. The new client methods it uses are in the
[API reference](api-reference.md).
