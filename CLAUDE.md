# Marqov SDK — guidance for coding agents

Conventions for working in this repo. Setup and architecture details live in
[CONTRIBUTING.md](CONTRIBUTING.md); this file covers what an automated or
AI-assisted contributor needs to work here without supervision.

## Environment and tests

- Never use system Python. Use the project venv (`.venv/bin/python`); see
  CONTRIBUTING.md §Development Setup.
- Run the full suite with `.venv/bin/python -m pytest` before considering any
  change done. All tests must pass; do not skip or xfail a failing test to
  get to green.
- Ruff and mypy carry some pre-existing findings. Do not fix them in a PR
  scoped to something else — but do keep your own changed lines clean.

## Scope and style

- Keep diffs scoped to the issue or task at hand. Adjacent cleanups belong in
  their own PR.
- Match the surrounding code's style, naming, and comment density.
- Plain conventional commit messages (`fix(qutip): …`, `docs: …`,
  `test(boundary): …`); no attribution footers or co-author lines.

## Documentation is a contract

Prose claims — in docs/, README, and docstrings — must be traceable to code.
Before writing a claim (an API's behavior, a default value, a test citation,
an install command), verify it against the source. If a doc cites a pytest
node ID, run `pytest --collect-only` on it. Several tests pin doc/code sync
(provider lists, the fallback-message wording); expect CI to fail if prose
and code drift apart.

## Architecture boundary

Core SDK code (everything under `marqov/` except `marqov/platform/`) must
never import `marqov.platform` — the platform client is a consumer of the
core, never a dependency of it. This includes type-only imports under
`if TYPE_CHECKING:`. The rule is enforced by
`tests/test_platform_package.py::TestCoreDoesNotImportPlatform` and
documented in `docs/design/platform-client-boundary.md`.
