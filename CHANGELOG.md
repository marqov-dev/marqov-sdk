# Changelog

All notable changes to the `marqov` SDK are documented here. This project follows
[Semantic Versioning](https://semver.org/). While on `0.x`, the public API may
still change between minor versions; `1.0.0` is reserved for the first API-stable
release.

## [0.5.1] — 2026-08-23

### Added

- **`QiliSDKExecutor` — a new `Qilimanjaro` provider running on `qilisdk`'s
  local simulators.** Wraps `QiliSim` (their own C++ simulator, ships in the
  base `qilisdk` package) or `QutipBackend` (pure-Python reference sim, via
  `qilisdk`'s own `qutip` extra) behind the same `BaseExecutor` interface as
  every other provider — no cloud account or SpeQtrum credentials required.
  Marqov's canonical gate set (`H`/`X`/`Y`/`Z`/`S`/`T`/`Rx`/`Ry`/`Rz`/`CNOT`/
  `CZ`/`SWAP`) translates 1:1 onto `qilisdk.digital` gate constructors; results
  come back through `qilisdk`'s `SamplingReadoutResult.samples`. First step of
  the Qilimanjaro integration — real-hardware/SpeQtrum access is a follow-up
  once beta credentials are available.

  `qilisdk` is installed separately (`pip install qilisdk`), not via a
  `marqov[...]` extra, the same treatment as Quantum Brilliance's `qristal`:
  `qilisdk` requires `numpy>=2.3` (macOS) / `>=2.4.1` (elsewhere), which only
  overlaps marqov's own `numpy<2.4` core pin in the macOS `2.3.x` window. A
  formal `marqov[qilisdk]` extra was tried and reverted — even scoped with a
  `sys_platform == "darwin"` marker, its mere presence in
  `[project.optional-dependencies]` made the *entire* `uv.lock` unresolvable
  (uv locks the union of all extras together, so one platform-incompatible
  extra poisons the whole project, not just itself). (marqov-sdk#104)

- **`QiliSDKExecutor.execute_analog()` — analog/annealing mode on qilisdk's local simulators.**
  Runs Qilimanjaro's actual hardware differentiator (fluxonium quantum annealing) locally, via
  `qilisdk.analog.Hamiltonian`/`Schedule` and the `AnalogEvolution` functional — no SpeQtrum
  account needed here either. Deliberately not part of the shared `BaseExecutor` contract:
  analog Hamiltonians have no Marqov-canonical representation the way digital gate circuits do
  (`qilisdk`'s own program type — `Hamiltonian`/`Schedule` — is passed straight through), and no
  other Marqov backend has an analog capability yet to unify against. A survey of how Braket,
  Azure Quantum, CUDA-Q, and classical heterogeneous-compute systems (Kubernetes, MLIR, XLA, Ray,
  Slurm) all handle this same digital/analog split confirmed the pattern: shared job-lifecycle
  contract, paradigm-native program types — never one flat representation for both.

- **`normalized_fidelity` and related application-level benchmarking
  metrics**, in `marqov/benchmarking/`. The Lubinski et al. / QED-C metric
  `max(0, (F_backend − F_uniform) / (1 − F_uniform))` normalizes out the
  trivial uniform-noise baseline, alongside new `classical_fidelity`
  (Bhattacharyya fidelity) and `fidelity_with_uniform` helpers. Where the
  existing SPAM tooling measures per-qubit readout error, this measures how
  faithfully a whole application's output distribution survives on real
  hardware. Ported from Open QBench (Apache-2.0), attributed in the module.
  (marqov-sdk#59)

- **`CUNQAExecutor` — a new `CUNQA` provider running distributed circuits
  across vQPUs on a real Slurm cluster.** CUNQA (CESGA, Apache-2.0) is a
  distributed quantum-computing emulator: vQPUs run as Slurm tasks, with
  real classical/quantum communication channels between them (Phase 1 of
  this integration, not yet wired in here — this first phase is
  correctness-only, N≈4, no failover/reproducibility/scaling logic). The
  executor launches a family of vQPUs via `qraise`, splits requested shots
  evenly across them, gathers and merges results, and always tears the
  family down with `qdrop` — including on failure or cancellation, so a
  crashed or timed-out run doesn't strand a Slurm allocation. Since
  Marqov's `Circuit` IR has no measurement concept at all, this executor
  is responsible for adding its own (`add_measure_all`), the same pattern
  `RigettiExecutor` already established.

  Verified end to end against a live AWS ParallelCluster Slurm cluster: a
  real no-comm QPE circuit recovering the exact expected phase bin across
  4 real vQPUs. Not on PyPI at all (no wheel; build from source, see
  `CESGA-Quantum-Spain/cunqa`) and not a `marqov[...]` extra — CUNQA's
  exact `qiskit==1.2.4` pin would downgrade the whole project's lockfile
  the same way `qilisdk`'s numpy floor does; install `qiskit==1.2.4`
  separately in the environment where CUNQA is built. (marqov-sdk#106)

### Fixed

- **Local backend format conversion ignored stray credentials.**
  `_to_backend_format()` was missing the local-backend short-circuit that
  `_get_provider_device()` and `run()` both already had, so `backend="local"`
  with a leftover `ibm_token`/`azure_subscription_id` in params converted the
  circuit to Qiskit while the device build stayed on Braket's
  `LocalSimulator`, which doesn't accept it. (marqov-sdk#102)

- **`CUNQAExecutor` could hang indefinitely on a live cluster.** Real
  CUNQA's `qraise()` blocks internally — polling `squeue` until the Slurm
  job is `RUNNING` with every vQPU registered — with no timeout of its
  own, unlike the fake client this executor was originally tested
  against. If a job never reaches `RUNNING` (a stuck node, insufficient
  capacity), the call never returned, and the executor's own
  `startup_timeout_s` only ever guarded the poll loop *after* `qraise`
  returned — too late. The vQPU family name is now generated up front
  (passed through to `qraise`'s own `family=` kwarg) instead of read from
  its return value, and the `qraise` call itself is wrapped in
  `asyncio.wait_for(timeout=cfg.startup_timeout_s)`, so teardown/`qdrop`
  can still be attempted even if `qraise` itself never returns.
  (marqov-sdk#109)

### Internal

- **`uv.lock` regenerated across all extras** — the `cudaq` extra was declared
  in `pyproject.toml` but never locked, so `uv sync --extra cudaq --frozen`
  hit the network instead of resolving from the lock. Two gaps hit during the
  0.4.1 release are now documented in `RELEASING.md`: a bare release-branch
  push gets no CI run (open a PR against `main` instead), and the `pypi`
  GitHub Environment's required-reviewer gate silently waits on approval.
  (marqov-sdk#100)

- **Four unexercised release-process footguns closed in `RELEASING.md`/
  `release.yml`** — none had bitten across 0.2.0–0.4.1, which is evidence
  the failure modes hadn't happened yet, not that the logic was correct.
  `publish-pypi` no longer trusts the tag ref alone (a `workflow_dispatch`
  against an existing tag would otherwise publish to real PyPI);
  `publish-testpypi` now tolerates a repeated dry-run at the same version.
  (marqov-sdk#108)

## [0.4.1] — 2026-08-14

### Fixed

- **Verbatim compilation no longer allow-lists an explicit `Measure`.** 0.4.0's
  allowed native set was `{Rx, Rz, CZ, XY, Measure}` and both the error message and
  the changelog advertised "plus Measure". Braket rejects it: `add_verbatim_box`
  raises *"cannot measure a subcircuit inside a verbatim box"*. So a caller passing an
  explicit measurement with `verbatim=True` cleared our validator and then hit an
  opaque Braket error instead of our actionable one. `Measure` is now excluded in both
  paths (`MarqovDevice.run` and `BraketExecutor.execute`), and the error names the
  offending instruction. Braket applies measurement implicitly via `shots`, so an
  explicit `Measure` was never needed — for randomized benchmarking in particular.
  (marqov-sdk#66)

- **`marqov.qutip.record` emitted `mcsolve` seeds that did not reproduce the run.**
  Seeds were serialised as `SeedSequence.entropy` alone. Every trajectory of a
  single `mcsolve` call is a *spawned child* of one root `SeedSequence`: they all
  share `entropy` and differ only by `spawn_key`. So the recorded seeds were
  identical to one another, and a replay collapsed the Monte Carlo ensemble to a
  single trajectory repeated `ntraj` times — measured **0/20** reproduction on runs
  containing a collapse.

  Seeds are now serialised as the full `SeedSequence.state` (numpy's own round-trip
  contract), one JSON object per trajectory:

  ```json
  {"entropy": "179663937626102255308327548008974693620",
   "spawn_key": [3], "pool_size": 4, "n_children_spawned": 0}
  ```

  `entropy` stays a string (128-bit; would lose precision in JS/`jsonb` float64) and
  `spawn_key` is a list (JSON has no tuple). New public helper
  **`marqov.qutip.record.seed_from_json`** rebuilds a `SeedSequence` for replay, so
  callers do not reimplement those conversions. Reproduction is now **20/20** on runs
  with collapses.

  **Breaking, deliberately:** the `seeds` field changes from `list[str]` to
  `list[object]`. Records written by **0.4.0 cannot be replayed** — their seeds were
  never sufficient to reproduce the run, so nothing correct is lost. Re-record any
  affected `mcsolve` runs. The platform stores this field in flexible `jsonb` and does
  not destructure it.

  This was masked by a test whose collapse rate was so low that most runs had no
  quantum jumps at all — a degenerate seed set reproduces a jump-free run trivially.
  With the old fixture (`0.1 * sigmam()`, `ntraj=4`), 57/60 runs were jump-free and
  passed vacuously; the 3 that jumped failed 3/3, which read as CI flake. The test now
  uses `0.4 * sigmam()`/`ntraj=8`, loops until a trajectory actually collapses, and
  asserts exact reproduction — plus a one-line guard that the seeds are distinct from
  each other, which is what would have caught this immediately.

- **`RateLimited.retry_after` no longer drops HTTP-date `Retry-After` headers.**
  Only the integer delta-seconds form was parsed; the RFC 7231 HTTP-date form
  (`Wed, 21 Oct 2026 07:28:00 GMT`) silently became `None`, so a caller doing
  `time.sleep(e.retry_after)` on the documented "parsed when present" contract
  crashed with `TypeError`. Both header forms now parse (dates convert to a
  delta against the current time, clamped to `>= 0`); `None` is reserved for
  absent or unparseable values. The transport docstring's `Raises` block also
  gained the two mapped exceptions it omitted (`BackendUnavailable`,
  `InvalidProgram`). (marqov-sdk#86)

- **`create_executor`'s unsupported-provider error no longer omits providers.**
  The message was a hand-written list that had drifted from the registry:
  Quantinuum was wired up but missing from it, so a caller with a case typo was
  told a supported provider did not exist. The message is now built from
  `get_supported_providers()` at raise time, the package docstring and README
  backend table are synced to all nine providers, and new tests pin each of
  those surfaces to the registry so the next provider added without updating
  them fails CI. (marqov-sdk#85)

- **Quantum Brilliance without qristal now fails with instructions, not a bare
  `ModuleNotFoundError`.** `qristal` is not distributed on PyPI, so no
  `marqov[...]` extra can exist for it, but nothing said so:
  `create_executor(provider="Quantum Brilliance")` succeeded and `execute()`
  then died deep inside the simulation path. The import failure now raises an
  actionable `ImportError` pointing at Quantum Brilliance's source-build and
  Docker install paths, and the factory docstring and README backend table
  document the external requirement. (marqov-sdk#84)

### Documentation

- **Every prose claim in the docs and docstrings was traced to code, and the
  drift fixed.** Two passes (doc files, then all inline docstrings) plus a
  post-merge review of the result: the broken quickstart examples, the
  `marqov[qiskit]` vs `marqov[ibm]` extra mix-up, stale provider lists, a test
  citation in `error-handling.md` that did not collect, the `job.cancel()`
  contract (it raises on any non-2xx; the server route is still provisional),
  `PlatformResult.raw`'s description, and `ibm.py`'s advertised-but-unread
  `**kwargs` options. (marqov-sdk#81, marqov-sdk#87, marqov-sdk#88,
  marqov-sdk#89, marqov-sdk#90)

- **The mcsolve replay contract is now stated precisely.** Bit-exact
  reproduction (`max|Δ| == 0.0`) holds under `options={"map": "serial"}`,
  which is also qutip's own default. Under an explicit parallel map,
  seed-to-trajectory assignment is stable but summation order is not: replay
  reproduces to floating-point tolerance (measured deviations of 1 ULP on a
  minority of collapsing runs). A new tolerance-based test covers the parallel
  path. (marqov-sdk#83)

### Internal

- **The core-to-platform import boundary is now enforced by an AST scan.** The
  previous regex guard missed submodule, relative, aliased, and dynamic import
  forms, scanned an incomplete hand-maintained path list, and could pass
  vacuously if a path was renamed. The scan now parses every file under
  `marqov/` except `marqov/platform/`, resolves relative imports against each
  file's package, treats type-only (`if TYPE_CHECKING:`) imports as
  violations, and asserts a minimum scan size so a moved tree fails loudly.
  (marqov-sdk#82)

## [0.4.0] — 2026-08-11

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
