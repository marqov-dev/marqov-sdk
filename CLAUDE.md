# Marqov SDK: guidance for coding agents

Conventions for working in this repo. Setup and architecture details live in
[CONTRIBUTING.md](CONTRIBUTING.md); this file covers what an automated or
AI-assisted contributor needs to work here without supervision.

## Environment and tests

- Never use system Python. Use the project venv (`.venv/bin/python`); see
  CONTRIBUTING.md, Development Setup.
- Run the full suite with `.venv/bin/python -m pytest` before considering any
  change done. All tests must pass; do not skip or xfail a failing test to
  get to green.
- Ruff and mypy carry some pre-existing findings. Do not fix them in a PR
  scoped to something else, but do keep your own changed lines clean.

## Scope and style

- Keep diffs scoped to the issue or task at hand. Adjacent cleanups belong in
  their own PR.
- Match the surrounding code's style, naming, and comment density.
- Plain conventional commit messages (`fix(qutip): ...`, `docs: ...`,
  `test(boundary): ...`); no attribution footers or co-author lines.

## Documentation is a contract

Prose claims in docs/, README, and docstrings must be traceable to code.
Before writing a claim (an API's behavior, a default value, a test citation,
an install command), verify it against the source. If a doc cites a pytest
node ID, run `pytest --collect-only` on it. Several tests pin doc/code sync
(provider lists, the fallback-message wording); expect CI to fail if prose
and code drift apart.

## What the hosted platform uses from this SDK

Do not assume the hosted Marqov Platform runs this SDK's code. It runs users'
work with its own code. Before ranking a change as important "for hosted
users" or "for the platform release", check which side runs it.

- **Used by hosted users:** writing programs (`marqov.circuits.Circuit` and its
  OpenQASM export, the `@task` and `@workflow` decorators as source text) and
  the `marqov.platform` client (`MarqovClient`, `Job`, `marqov watch`).
  `client.submit(Circuit)` sends `to_openqasm(version=3)`; `submit_native()`
  sends source that the platform compiles and runs with the SDK version in
  its own images, which can be older than the user's.
- **Not used by the hosted platform:** the Temporal workflow engine in
  `marqov/workflows/` (`temporal_workflow.py`, `activity.py`, the worker CLI),
  the provider executors in `marqov/executors/`, `ExecutorFactory`, and
  `MarqovDevice` / `get_device()`. These run where someone operates the SDK
  themselves, with their own Temporal service or provider accounts. A defect
  there is real for those users and does not affect a hosted job.
- **Limits differ.** The hosted platform enforces its own limits (for example
  on task result size). A limit in this repository says nothing about the
  hosted one. See `docs/platform-client/native-workflows.md`.

The platform's maintainers keep a dated map of how work runs in production.
If a task depends on how the hosted product behaves and that map is not
available to you, say so and ask, instead of inferring it from this
repository.

## Architecture boundary

Core SDK code (everything under `marqov/` except `marqov/platform/`) must
never import `marqov.platform`: the platform client is a consumer of the
core, never a dependency of it. This includes type-only imports under
`if TYPE_CHECKING:`. The rule is enforced by
`tests/test_platform_package.py::TestCoreDoesNotImportPlatform` and
documented in `docs/design/platform-client-boundary.md`.
