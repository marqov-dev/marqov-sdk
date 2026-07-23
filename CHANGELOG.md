# Changelog

All notable changes to the `marqov` SDK are documented here. This project follows
[Semantic Versioning](https://semver.org/). While on `0.x`, the public API may
still change between minor versions; `1.0.0` is reserved for the first API-stable
release.

## [Unreleased]

### Changed
- **`@task` bodies now execute in an isolated subprocess** with a deny-by-default,
  scrubbed environment — the task process inherits only an explicit allowlist of
  environment variables plus any credentials a host passes for that job, never the
  ambient process environment. Task results are forwarded without being
  deserialized in the parent worker process.
- **Serialization is deferred from decoration to dispatch.** `@task` no longer
  `cloudpickle`-serializes the decorated function when the decorator runs; it does
  so lazily at graph build. This resolves the `RecursionError` that could occur
  when decorating a local/closure task with a large set of heavy backends imported
  (see 0.2.0 *Known limitations*), and avoids serializing tasks that are never
  dispatched.

### Behavioral change
- **Closure-at-dispatch:** because a task is now serialized at dispatch rather than
  at decoration, its closure captures referenced variables **as of dispatch**, not
  as of import/decoration time. This is identical for deterministic code; a task
  that closes over a value mutated between decoration and dispatch will now observe
  the dispatch-time value.

### Added
- **`MARQOV_SCRUB_ALLOWLIST` environment variable (host↔SDK interface):** a host
  runner may set this (comma-separated variable names) in the worker process to
  define the allowlist the SDK uses when building a task's subprocess environment.
  When unset, the SDK falls back to a minimal, secret-free default and logs a
  warning.

## [0.2.0] — 2026-06-29

First public release on PyPI.

### Added
- Public release of the `marqov` SDK: build quantum circuits and run them across
  multiple hardware backends (AWS Braket, IBM Quantum, Azure Quantum, IonQ,
  Rigetti, Quantinuum, and local simulation) behind one API.
- Circuit interop helpers (import/export with Qiskit, Cirq, PennyLane, pytket,
  and pyQuil) and workflow/task decorators for composing multi-step programs.

### Changed
- The circuit IR / transpilation dependency is now the published
  [`marqov-quantumflow`](https://pypi.org/project/marqov-quantumflow/) package
  instead of a git URL, which is what makes `pip install marqov` possible.
- The package version is single-sourced from `marqov/__init__.py` (`__version__`)
  via hatchling, so the distribution metadata, `marqov.__version__`, and
  `marqov --version` always agree.

### Known limitations
- `@task`/`@workflow` serialize the decorated function with `cloudpickle` at
  decoration time. When the function is a local/closure (not module-level) and a
  large set of heavy backends is imported in the same process, this can overflow
  into a `RecursionError`. Module-level task functions are unaffected. A fix that
  defers/removes the function serialization is planned.

[0.2.0]: https://pypi.org/project/marqov/0.2.0/
