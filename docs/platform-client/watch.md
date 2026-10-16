# Watch Platform workflows in your terminal

After installing this SDK revision, run:

```sh
marqov watch
```

Choose a project from the numbered list, or choose **All team jobs**. Use the
arrow keys to select a workflow, `d` to toggle details, and `q`, Escape or Ctrl+C
to quit. Status labels remain visible alongside their colours.

The command uses `MARQOV_PLATFORM_KEY`, then the legacy `MARQOV_API_KEY`, then
your existing `~/.config/marqov/api-key` file. `--key-file PATH` selects another
file when neither environment variable is set. It does not create credentials
or require a second login. `MARQOV_PLATFORM_URL` selects production or staging.

```sh
marqov watch --list-projects
marqov watch --project "My experiment"
marqov watch --project PROJECT_UUID --refresh 10 --limit 5
marqov watch --all
marqov watch --all --job JOB_UUID
```

Project names must match exactly (ignoring case); use an ID if names repeat.
Project views show jobs explicitly linked in that project's Workspace, up to
the selected limit (default 5, maximum 10). A project with no linked jobs stays
empty; it never substitutes unrelated team jobs. `--all` shows recent team jobs.
With both `--project` and `--job`, the job must be linked to that project.

## Refresh and scope

Status refreshes automatically. The default interval is 10 seconds, with a
minimum of 5 seconds. Project views increase the interval when needed to keep
their document and job reads to roughly 30 requests per minute or fewer.
Requests do not overlap. `r` rereads the latest local observation; it does not
send an extra API request.

If a read fails, the last observation stays visible with a **STALE** warning.
Retries back off, honour numeric `Retry-After` values, and stop after an
authentication or permission failure. Quit and restart after fixing credentials.
The capture timestamp describes when Marqov's record was fetched, not independent
evidence that a provider is still running.

This is read-only. It cannot submit, retry or cancel jobs. It shows workflow
identity/state, backend, available recorded costs, hold reasons and update times.
Raw results, scripts and error bodies are not displayed. These read endpoints do
not establish provider/Slurm identity; unknown identities remain unknown.

For scripts or a time-limited demonstration:

```sh
marqov watch --project PROJECT_UUID --once       # sanitized JSON, then exit
marqov watch --project PROJECT_UUID --duration 120
```

Non-interactive sessions must choose `--project` or `--all`; use `--once` without
a terminal. The interactive viewer currently supports macOS and Linux.

## First launch

The viewer is built from the bundled, locked Rust source on its first interactive
launch. Install Rust/Cargo first (verified with Rust 1.94.1). The single-job build
has a four-minute deadline and happens before the dashboard claims the terminal.
Later launches use the cached executable. `--viewer PATH` or `MARQOV_WATCH_BIN`
can select an existing compatible viewer, including a prebuilt executable.
`--once` and `--list-projects` do not need Rust.

Private temporary observations and compiled files live under
`${XDG_CACHE_HOME:-~/.cache}/marqov/watch`. Session observations are removed on
normal exit. API keys stay in memory and are not passed to the viewer or compiler.
