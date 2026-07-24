# Changelog

All notable changes to the `marqov` SDK are documented here. This project follows
[Semantic Versioning](https://semver.org/). While on `0.x`, the public API may
still change between minor versions; `1.0.0` is reserved for the first API-stable
release.

## [Unreleased]

### Changed
- **`@task` bodies now execute in an isolated subprocess** with a minimal,
  allowlisted environment — the task process inherits only an explicit allowlist
  of environment variables plus any values a host passes for that job, never the
  ambient process environment. Task results are forwarded by the parent worker
  without being deserialized in-process.

### Added
- **`MARQOV_SCRUB_ALLOWLIST` environment variable (host↔SDK interface):** a host
  runner may set this (comma-separated variable names) in the worker process to
  define the allowlist the SDK uses when building a task's subprocess environment.
  When unset, the SDK falls back to a minimal, secret-free default and logs a warning.

## [0.3.0] — 2026-07-20

### Added

- **`marqov.platform` — hosted platform client (v1.0).** New optional
  subpackage that submits quantum jobs to the Marqov Platform without
  touching the rest of the SDK.  Importing `marqov` has no cost from this
  subpackage — it is only loaded on explicit `from marqov.platform import ...`.

  Public surface:
  - `MarqovClient(api_key, *, base_url, timeout)` — explicit client over the
    platform REST API.  Key resolves from `MARQOV_PLATFORM_KEY` if not
    supplied; never written to disk.
  - `Job` — async job handle with `.result()` (blocking poll with server
    long-poll + client exponential back-off), `.status()`, `.cancel()`,
    `.estimated_cost_usd`, and `.id`.
  - `backends()` — list available platform backends.
  - `platform_info()` — runtime SDK version + platform API version.
  - `JobStatus` — `str`-enum of lifecycle states; `is_terminal()` utility for
    safe raw-string checks.
  - `Backend`, `PlatformResult` — dataclass result types.
  - Full error hierarchy: `MarqovPlatformError` → `AuthenticationError`,
    `PermissionTierError`, `PaidBackendNotSupportedYet`, `BackendUnavailable`,
    `InvalidProgram`, `JobFailed`, `RateLimited`, `TransportError`.

  `marqov.platform`'s public API follows **semver** — no breaking changes
  within a major version.

- **QASM 3 circuit wire format.** `Circuit` objects submitted via
  `client.submit(circuit, ...)` are serialised as OpenQASM 3 (via
  `circuit.to_openqasm(version=3)`), verified to round-trip losslessly across
  the full gate set.  Circuit submission requires a forthcoming platform-side
  change and is not yet active on the server.

### Changed

- **`requests` is now a core dependency** (previously only pulled in
  transitively via the `ionq` and `all` extras).  This means `pip install marqov`
  now installs `requests` unconditionally.  Downstream users who pinned the
  extras to exclude `requests` will pick it up automatically on upgrade.

### Notes

- v1.0 of the platform client supports **free backends** (e.g. `dwave-sim`).
  Paid QPU backends are coming in a future update.
- `PaidBackendNotSupportedYet` is the free-path guard for v1.0.  It is
  retained (never deleted) after paid backends are enabled so that existing
  `except PaidBackendNotSupportedYet` blocks continue to work.
- The platform client is unit- and contract-verified against a mocked
  transport.  Live-server verification is pending the staging environment.

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
