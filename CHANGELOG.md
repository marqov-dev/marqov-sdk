# Changelog

All notable changes to the `marqov` SDK are documented here. This project follows
[Semantic Versioning](https://semver.org/). While on `0.x`, the public API may
still change between minor versions; `1.0.0` is reserved for the first API-stable
release.

## [0.4.0] — Unreleased

### Added

- **Verbatim compilation on `MarqovDevice`.** `MarqovDevice.run(circuit,
  verbatim=True)` now wraps Braket circuits in a verbatim box (mirroring
  `BraketExecutor`), so the provider compiler executes the gates exactly as
  given. Required for randomized benchmarking on Rigetti — without it, the
  compiler folds Clifford-plus-inverse sequences to identity and survival is
  flat at every sequence length. Requires native gates only (1Q: Rx/Rz, 2Q:
  CZ/XY) and raises `ValueError` otherwise. Addresses the verbatim gap
  identified in the device/executor parity audit (marqov-sdk#66).

- **`marqov.qutip.record` — open-system-dynamics result capture.** New optional
  helper that serialises a QuTiP solver `Result` (`mesolve`/`sesolve`/`mcsolve`)
  into the platform's `open-system-dynamics` v1 stdout-JSON contract with a
  single call: `record(result, ["sigma_z", "sigma_x"])`. Test-only qutip
  dependency — `record.py` duck-types the `Result` and never imports qutip at
  runtime. Enforces the contract's guardrails so a hand-built dict can't get
  them wrong: density-matrix `.states` are refused on stdout (offload by
  reference), unseeded `mcsolve` runs are rejected as non-reproducible (seeds
  emitted as strings to survive JSON float precision), and non-finite / genuinely
  complex expectation values are rejected rather than silently coerced.

- **Device execution windows on `DeviceStatus`.** `BraketExecutor.get_status()`
  now serializes a Braket device's advertised execution windows to
  `DeviceStatus.execution_windows` — a portable
  `[{"executionDay", "windowStartHour", "windowEndHour"}]` list (UTC). Fail-safe
  (returns `None` on any error, never blanking availability); `[]` (reported none)
  stays distinct from `None` (unknown); `is_device_available()` is unchanged.
  (marqov-sdk#70)

### Fixed

- **`MarqovDevice.run` is now event-loop-safe on real QPUs.** Braket's
  `AwsQuantumTask.result()` polls via `run_until_complete()`, which raised
  *"This event loop is already running"* when `run()` was called from within a
  running event loop (e.g. an async experiment runner driving a real QPU). The
  blocking Braket call is now offloaded to a worker thread that has its own
  event loop when a running loop is detected. Simulators were unaffected.
  (marqov-sdk#67)

## [0.3.1] — 2026-07-25

### Fixed
- **Platform client default endpoint.** `MarqovClient` defaulted to a
  `platform.marqov.com` host that does not resolve — so callers who didn't set
  `base_url` or `MARQOV_PLATFORM_URL` hit a dead host. The default is now
  `https://app.marqov.ai`. Explicit overrides were unaffected.

## [0.3.0] — 2026-07-24

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

- **`CudaqExecutor` — NVIDIA CUDA-Q backend** (`marqov[cudaq]`, Linux-only wheels).
  Adds three backend slugs: `cudaq-cpu` (CPU statevector, the default — no GPU
  required), `cudaq-gpu` (GPU statevector via CUDA-Q's `nvidia` target, scaling
  past the CPU/SV1 simulators), and `cudaq-iqm` (direct IQM Resonance — a
  lower-latency route than IQM-through-Braket).

- **`MARQOV_SCRUB_ALLOWLIST` (host↔SDK interface).** A host runner may set this
  (comma-separated variable names) in the worker process to define the allowlist
  the SDK uses when building a `@task`'s subprocess environment. When unset, the
  SDK falls back to a minimal, secret-free default and logs a warning.

### Changed

- **`requests` is now a core dependency** (previously only pulled in
  transitively via the `ionq` and `all` extras).  This means `pip install marqov`
  now installs `requests` unconditionally.  Downstream users who pinned the
  extras to exclude `requests` will pick it up automatically on upgrade.

- **`@task` bodies now run in an isolated subprocess** with a minimal, allowlisted
  environment (defense-in-depth: a task runs with only what it needs).
  **What this means for you:** if your `@task` reads an environment variable that
  was set in the calling process, it will **no longer see it** unless the host
  passes it through explicitly (via `MARQOV_SCRUB_ALLOWLIST`, or provider
  credentials the host injects for the job). Pure-compute tasks are unaffected.

### Fixed

- **`@task` no longer serializes at import/decoration time.** Function
  serialization is deferred (and cached) to workflow-graph build. This resolves
  the `RecursionError` that could fire when decorating a local/closure `@task`
  with many heavy backends imported (see 0.2.0 *Known limitations*), and stops
  serializing tasks that are never dispatched.

- **IonQ Direct is no longer listed twice** by `get_supported_providers()` — the
  provider list now returns a single, de-duplicated IonQ Direct entry.

### Behavioral change

- **Closure-at-dispatch.** A `@task` is now serialized at dispatch (graph build)
  rather than at import (decoration), so its closure captures referenced variables
  **as of dispatch**. **What this means for you:** if your task closes over a
  variable whose value changes between decoration and dispatch, the task now
  observes the **dispatch-time** value. Deterministic code behaves identically.

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
