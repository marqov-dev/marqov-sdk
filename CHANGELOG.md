# Changelog

All notable changes to the `marqov` SDK are documented here. This project follows
[Semantic Versioning](https://semver.org/). While on `0.x`, the public API may
still change between minor versions; `1.0.0` is reserved for the first API-stable
release.

## [Unreleased]

### Added

- `marqov watch`: a read-only Ratatui dashboard with existing API-key authentication,
  project selection, automatic bounded refresh, stale-data retention and retry backoff.
  The viewer builds from bundled locked Rust source on first launch; JSON and project
  listing modes work without Rust.

## [0.10.1] - 2026-10-21

A patch release of fixes found after 0.10.0. The platform client keeps its
error contract on unreadable responses and honours the `result()` deadline,
workflow task scratch directories are removed and kept out of the shared temp
root, `NoiseModel.realistic_device` reproduces the T2 it is given, and the
CUNQA executor stops over-counting shots and validates its inputs. The README
on PyPI is current again.

### Upgrading

These are fixes, but they can change what existing code observes.

- `NoiseModel.realistic_device` produces different simulated results: the
  phase-damping rate no longer double counts T1, so coherence now decays at
  the requested T2. It raises `ValueError` for non-positive `t1`, `t2` or
  `gate_time` and for `t2 > 2 * t1`.
- `Job.result()` reports a transport failure after its deadline as
  `TimeoutError`, and stops retrying once the deadline passes. A response
  below HTTP 400 whose body is not JSON raises `TransportError` instead of a
  raw `requests` decode error; `submit_native()` and `submit_script()` still
  report it as `submission_outcome_unknown` with the idempotency key.
- Workflow task workdirs are deleted when the task finishes (set
  `MARQOV_KEEP_TASK_WORKDIR=1` to keep them), and the task's process group is
  killed when the task process exits, so a background process a task leaves
  running no longer outlives it. Workdirs now sit one level down, in
  `$TMPDIR/marqov_worker_*/marqov_task_*`: a host cleanup script that globbed
  `$TMPDIR/marqov_task_*` for directories a killed worker left behind should
  also sweep `marqov_worker_*` directories whose worker process has exited.
- `CUNQAExecutor` raises `ValueError` for `n_qpus` below 1 or not an integer,
  for `shots` below 1 or not an integer, and for a `startup_timeout_s` or
  `poll_interval_s` that is not a finite number above 0, before any CUNQA
  call. The two timing fields are stored as float.
  `startup_timeout_s` is one budget across `qraise` and vQPU registration, so
  slow starts time out sooner, and `execution_time_ms` no longer includes
  teardown.
- Calling an `async def` `@task` outside a workflow still returns a coroutine
  that must be awaited. This is now documented; a future minor release may
  change how direct invocation behaves.

### Fixed

- The platform client raises `TransportError` for a non-JSON response below
  HTTP 400 and never retries it, clamps a negative `Retry-After` to 0, gains
  `close()` and context-manager support on `MarqovClient` and the transport,
  and bounds each `Job.result()` poll and retry by the remaining deadline.
  (marqov-sdk#277, closes marqov-sdk#150)

- `execute_task` removes each task's workdir after the task finishes, on
  success, failure and cancellation, off the event loop; task workdirs live in
  a private per-process parent (mode 0700) instead of the shared temp root.
  (marqov-sdk#278, closes marqov-sdk#144)

- `NoiseModel.realistic_device` derives phase damping from the pure dephasing
  rate `1/T_phi = 1/T2 - 1/(2*T1)` and validates its inputs.
  (marqov-sdk#279, closes marqov-sdk#158)

- `LocalExecutor` records the seed in `metadata["reproducibility"]`, with the
  same keys as the other executors' records. (marqov-sdk#276, closes
  marqov-sdk#78)

- `LocalExecutor` looks up the package versions for that record once per
  process instead of on every call, which had made a small-circuit execute
  about 40 times slower, and records `None` for a package without
  distribution metadata instead of raising. (marqov-sdk#284)

- `execute_task` keeps heartbeating until the task workdir has been removed,
  so a slow removal cannot trip the activity's heartbeat timeout after the
  task succeeded, and a cancellation that arrives while the heartbeat stops
  now propagates. (marqov-sdk#285)

- `CUNQAExecutor` uses only the requested number of vQPUs (extra
  registrations multiplied the shots), validates `n_qpus` and `shots`, keeps a
  `TimeoutError` raised by `qraise` itself, drops a family whose `qraise`
  finishes after the timeout, and lets `asyncio.run()` return promptly after a
  timeout. NumPy integers are accepted for both values. (marqov-sdk#281,
  closes marqov-sdk#138)

- `CUNQAExecutor` validates `startup_timeout_s` and `poll_interval_s` (a zero
  poll interval busy-polled vQPU registration), and `ExecutorFactory` accepts
  the same NumPy values for `n_qpus` and the timing fields as
  `CUNQAExecutorConfig` does. (marqov-sdk#286)

- The H2 VQE benchmarks build the documented UCC ansatz, which reaches the
  exact ground energy (the previous circuit stopped 1.23 Ha short), take an
  optional `seed`, and the raw-SDK scripts use the correct bit order and
  ground-energy constant. Benchmark code is not in the wheel.
  (marqov-sdk#280, closes marqov-sdk#160)

### Documentation

- The README, which is the PyPI project page, no longer names 0.7.0 as the
  current release and uses absolute links so they work on PyPI.
  (marqov-sdk#274)

- The preflight module no longer claims mandatory server-side budget
  enforcement; it states that `preflight()` performs no budget comparison.
  (marqov-sdk#275, closes marqov-sdk#80)

- The `@task` docstring and README document that an async task called outside
  a workflow must be awaited, and the test suite fails on a never-awaited
  coroutine. (marqov-sdk#282, closes marqov-sdk#103)

## [0.10.0] — 2026-10-20

A correctness release. A review of the 0.8.1 to 0.9.1 changes found
twenty-one defects, and a second pass closed nine more open issues in the
same vein. The common threads: remote executors keep the provider job
identity on every exit after submission; provenance records report what was
observed rather than what was requested; options and settings an executor
cannot apply now fail loudly instead of being ignored; and workflow task
arguments travel intact. A dependency ceiling on `qiskit` works around an
upstream break on Linux.

### Upgrading

These changes can alter the behaviour of existing code; the entries below
give the detail.

- Options an entry point does not apply now raise `TypeError` instead of
  being ignored: `MarqovDevice.run()` accepts `verbatim` and
  `disable_qubit_rewiring` on the cloud Braket path and nothing on the
  local, `marqov-sim`, IBM and Azure paths; `BraketExecutor.execute()`
  accepts the same two; `LocalExecutor`, `CudaqExecutor`, `RigettiExecutor`,
  `IonQExecutor`, `AzureQuantumExecutor` and `CUNQAExecutor` accept none;
  `QuantinuumExecutor` accepts only the options pytket's `process_circuits`
  reads (`max_cost`, `group`, `pytketpass`, `request_options` and per-call
  `options` now raise; use the matching `QuantinuumExecutorConfig` fields).
- `IBMExecutorConfig` raises `ValueError` at construction, including through
  `ExecutorFactory`, for `resilience_level` other than 1 or
  `poll_interval_seconds` other than 2.0; SamplerV2 cannot apply either.
  Stored backend configs carrying other values must be edited.
- `IBMExecutor.execute()` no longer wraps every post-submission failure in
  `IBMExecutionError`. A result the SDK refuses after retrieving it raises
  `ValueError` (shot accounting) or `NotImplementedError` (several classical
  registers), each naming the job id. Code that re-polls on
  `IBMExecutionError` should treat those two types as final.
- `AzureQuantumExecutor.is_device_available()` returns False for Azure's
  `Unavailable` (it was True); `get_status()` reports `maintenance` instead
  of `offline` for an unreadable status, a missing target or an unknown
  vocabulary value, and logs a warning naming a target the workspace does
  not have.
- `PreflightResult.dry_run_passed` is `bool | None` and `preflight()` returns
  `None`, meaning no dry run was attempted; `if result.dry_run_passed:` is
  now falsy by default. The no-cost warning text changed.
- Workflow task arguments that are not JSON (a `Circuit`, an array, a
  dataclass, a namedtuple, a dict with non-string keys, a set, a `datetime`)
  now reach the task as the original objects instead of JSON-coerced forms;
  a task result dict with non-string keys comes back from `run()` with its
  keys intact. A task result hidden inside a set, frozenset, object attribute
  or non-string-keyed dict, and any argument or node input over 3 MiB, now
  fail at dispatch with `TypeError` or `ValueError` naming the task.
  `WorkflowDispatch.run()` and `run_with_ids()` return decoded objects, not
  the `{"__cloudpickle__": ...}` envelope.
- `ExecutorFactory.get_supported_providers()` and `is_provider_supported()`
  now include `"CUNQA"` and `create_executor` routes it; a CUNQA config with
  a wrongly typed value or a `seed` key raises `ValueError`.
- `QBRemoteExecutor.get_status()` and `SpeQtrumExecutor.get_status()` return a
  `DeviceStatus` with status `maintenance` instead of raising;
  `SpeQtrumExecutor` raises `ImportError` at construction without `qilisdk`.
- Every extra that declares `qiskit` caps it below 2.6; the `[ibm]` and `[all]`
  extras require `qiskit-ibm-runtime>=0.21.0`. Environments that pin
  `qiskit>=2.6` cannot resolve this release until the cap is lifted.
- `Circuit.to_openqasm()` and `to_qiskit()` always name the register `q`.

### Fixed

- SPAM calibration registers wires through each target before preparation,
  preserving absolute bit positions and preventing false 0.5 readout fidelity
  on noiseless qubits above zero. Correction accepts mixed-width bitstrings
  independent of insertion order, combines equivalent padded outcomes and
  reports matrix-width mismatches clearly. (marqov-sdk#159)

- IBM execution releases the caller at `timeout_seconds` instead of holding the
  script until the queued job finishes, and attaches `remote_job` identity with
  a note to every error after submission, including cancellation. Multi-register
  circuits and shot-accounting mismatches raise their own error types naming
  the job instead of `IBMExecutionError`. A timeout still never cancels the IBM
  job. (marqov-sdk#260)

- Quantum Brilliance remote and SpeQtrum execution keep the job identity when
  the awaiting task is cancelled during submission, so callers know whether a
  job was accepted and avoid double submission. `get_status()` on both returns
  a `maintenance` status on transport failure, as the other executors do, and
  SpeQtrum raises a clear `ImportError` at construction when `qilisdk` is
  missing. (marqov-sdk#258)

- Circuit export and import: QASM 2 `swap` round-trips through
  `from_openqasm`, `from_braket` accepts terminal measurements as the other
  importers do, `to_pyquil` works for Braket-imported gates and raises
  `NotImplementedError` for gates with no Quil equivalent, and `to_openqasm`
  always names the register `q`, so repeated exports are byte-identical and
  reproducibility hashes are stable across calls. (marqov-sdk#259)

- Qristal simulation records the noise model and an `inputs_sha256` hash in the
  reproducibility record, so noiseless and noisy runs, and runs with different
  noise models, no longer produce identical records. NumPy-typed noise inputs
  are accepted. (marqov-sdk#255)

- QiliSDK validates sampler counts against the requested shots and the circuit
  width, raising `RuntimeError` on mismatch, and records the observed shot total
  and `bit_order` in the reproducibility record. (marqov-sdk#256)

- Azure Qiskit runs report `cost_estimate` and provider execution time from the
  job details; previously both were silently lost. Unknown provider timing is
  `metadata["quantum_time_ms"] = None` with wall time in `execution_time_ms`.
  The Cirq error mapping that could never match was removed. (marqov-sdk#257)

- Braket retains the task identity (`remote_job` with the task ARN and a note)
  on provider errors, result-processing errors and cancellation during
  submission, matching the Azure and Rigetti executors. (marqov-sdk#252)

- IonQ on the legacy v0.3 route attaches `remote_job` to errors, logs the job id
  and sends a best-effort cancel when a wait is interrupted or the job reports
  an unrecognised status, and raises the intended `RuntimeError` for a failed
  job with `"failure": null`. (marqov-sdk#254)

- Azure, Braket and Rigetti log a warning naming the job as soon as a blocking
  result wait is interrupted, so the job id is recorded even when
  `asyncio.run()` turns the cancellation into a bare `KeyboardInterrupt`.
  (marqov-sdk#251)

- `WorkflowDispatch.run()` and `run_with_ids()` return decoded task outputs
  instead of the base64 cloudpickle envelope, and NumPy scalar outputs are
  converted to Python values before serialisation. (marqov-sdk#253)

- `IBMExecutor`, the Qristal `SimulationExecutor` and the platform client accept
  NumPy integer shot counts. `LightningExecutor` now raises `NotImplementedError`
  for an unsupported device name before constructing it. (marqov-sdk#250)

- CUDA-Q execution raises `RuntimeError` when the sampled shot total differs
  from the request, and its result carries `vendor`, `engine`, `access_path`
  and a `reproducibility` record. (marqov-sdk#248)

- `ExecutorFactory.get_supported_providers()` and `is_provider_supported()` now
  include Alice & Bob, and the README provider table lists it. (marqov-sdk#249)

- `MarqovDevice.run()` and every executor's `execute()` raise `TypeError`
  for options they do not apply, naming the entry point, the option and the
  supported set; `verbatim=True` can no longer vanish on a path that does not
  implement it. (marqov-sdk#271, closes marqov-sdk#77)

- `IBMExecutorConfig` rejects `resilience_level` and `poll_interval_seconds`
  values the SamplerV2 path cannot apply instead of silently ignoring them.
  (marqov-sdk#268, closes marqov-sdk#79)

- Azure `is_device_available()` no longer reports an `Unavailable` target as
  available, `get_status()` reports an unreadable status as `maintenance`
  rather than `offline`, and a target missing from the workspace is logged.
  (marqov-sdk#267, closes marqov-sdk#132)

- `preflight()` no longer reports a dry run it never attempted or implies a
  budget check; `experiment_config` key names are logged at debug level,
  never values. (marqov-sdk#270, closes marqov-sdk#163)

- Workflow task arguments that are not JSON travel in the same cloudpickle
  envelope as results, with NumPy scalars unboxed, so the documented
  `Circuit`-argument example dispatches; each argument and each node's input
  are capped at 3 MiB before submission; a task result the worker cannot
  resolve is refused at dispatch; non-string dict keys survive in both
  directions. (marqov-sdk#272, closes marqov-sdk#142)

- `ExecutorFactory` routes the `"CUNQA"` provider and validates its config
  values. (marqov-sdk#269, closes marqov-sdk#139)

### Changed

- The lockfile (`uv.lock`, dev and CI only) upgrades urllib3, pyjwt, pillow,
  oauthlib, anyio and pyasn1 past their advisories; the cryptography and
  setuptools advisories remain open, pinned by azure-quantum and
  amazon-braket. Nothing a PyPI user installs changes. (marqov-sdk#266,
  marqov-sdk#186)

- The `[ibm]` and `[all]` extras require `qiskit-ibm-runtime>=0.21.0`, the first
  release that ships `SamplerV2`, which `IBMExecutor` imports. (marqov-sdk#247)

- Every extra that declares `qiskit` now caps it below 2.6. qiskit 2.6.0 breaks
  `pennylane-lightning` 0.45.0 on Linux x86_64 (every `lightning.qubit`
  construction fails with a static TLS allocation error once qiskit is loaded
  first). Lift the cap when pennylane-lightning or qiskit ships a fix for the
  TLS allocation. (marqov-sdk#262)

### Documentation

- Corrected 0.9.x documentation drift: the marqov-quantumflow pin in
  `docs/braket-native-labels.md`, the versions the IBM qualification receipt
  records, the Rigetti `cancel` docstring, and the squash-merge note in
  `RELEASING.md`. The four timeouts docs now say where the job id is logged
  when a wait is interrupted. (marqov-sdk#250, marqov-sdk#251)

- The Python 3.13 qualification record is regenerated from a `main` commit
  (the previous Linux record cited a pull-request merge ref that no branch
  reaches), and the per-profile test counts in `docs/python-compatibility.md`
  match it again. (marqov-sdk#263)

- `SECURITY.md` lists the supported minor (pinned by a test), CONTRIBUTING
  documents `DeviceStatus.execution_windows` and the current quilc/qvm flags,
  the platform-client docs describe the paid-backend error and `backends()`
  ordering as the code behaves, and `@task(executor=...)` is documented as
  reserved with `provider_env` not wired for dispatched tasks.
  (marqov-sdk#269, marqov-sdk#272, closes marqov-sdk#167, marqov-sdk#151,
  marqov-sdk#145)

## [0.9.1] — 2026-10-10

### Changed

- Quantum Brilliance simulation now returns `result.backend == "qb-sim-noisy-aer"`
  whenever Aer runs, including noise-enabled requests for statevector,
  tensor-network or GPU simulators. Previously the result retained the requested
  backend label, or used the unregistered `qb-sim-noisy` label for direct Aer
  requests. Consumers should use `metadata.requested_simulator` for the requested
  simulator and `metadata.simulator` for the simulator that ran.
- Noise-enabled simulation requests now use Aer's SDK limit of 28 qubits:
  requests above 28 that previously passed a tensor-network limit are rejected
  before execution, while requests for `cudaq:dm` are no longer capped at 14
  when routed to Aer. This validation limit does not establish available memory
  or native Aer qualification.
- NumPy numeric parameters and qubit labels normalized by `Circuit.to_dict()`
  can now pass QB remote payload validation as Python numeric values; NumPy
  integer qubit labels also pass IonQ v0.4's dense-wire validation. Other
  provider checks still apply, and execution requires an explicit request.

### Fixed

- Noisy Quantum Brilliance simulation validates Aer’s actual qubit limit
  before native execution, reports the canonical `qb-sim-noisy-aer` backend,
  and retains both actual and requested simulator metadata. Existing Aer noise
  routing, input configuration and noiseless backend selection are preserved.
  (marqov-sdk#129)

- CI rejects dependency metadata/lockfile drift and unregistered pytest markers.
  All live integration modules carry the integration marker; environment guards
  still control execution. Lint and Python 3.13 all-extras checks are advisory,
  separate from required Python 3.12 and CUDA-Q gates. (marqov-sdk#166)

- Circuit dictionaries normalize NumPy integer and float16/float32/float64
  parameters and qubit labels to JSON-native values, enabling ordinary JSON
  round trips without mutating the source circuit. Python numeric output and
  existing Lightning circuit hashes are unchanged. Symbolic and extended-precision
  serialization remain separate decisions. (marqov-sdk#194, numeric slice)

### Added

- Independent installed-wheel Python 3.13 qualification and a user-facing
  compatibility matrix with exact dependency records. Seventeen Linux profiles
  feed required CI; Rigetti, pyQuil and the all-extras bundle retain visible
  installation restrictions. Native GPU, live provider and deployed-runtime
  qualification remain separate.

## [0.9.0] — 2026-10-08

### Fixed

- Local QuantumFlow sampling accepts floating-point overshoot/undershoot near
  unit probability mass without changing the simulated state. Nonfinite or
  materially unnormalized states are rejected explicitly. Requires the
  corrected marqov-quantumflow 1.0.2 dependency. (marqov-sdk#233)

- IonQ v0.4 reports terminal vendor failures before validating success-only
  fields, while still checking job identity. Malformed result objects raise
  clear validation errors with job identity and original artifact bytes retained.

- Rigetti measures actual exported physical wires, including sparse labels,
  and records their ordered mapping in `metadata.measured_qubits`. Incomplete,
  wrong-width or nonbinary readout is rejected instead of returned as successful
  counts. Active-wire result width is unchanged. (marqov-sdk#229, marqov-sdk#230)

- IonQ retains original probability result artifacts and labels Hamilton counts
  as probability-derived, ineligible for raw-shot inference. An explicit v0.4
  ideal-simulator QASM3 route validates job/artifact identity and uses the
  documented q0-leftmost register order; legacy defaults and ordering remain.

- Rigetti execution uses non-joining blocking workers and retains native QPU
  handles for best-effort cancellation on timeout or caller cancellation.
  Exceptions preserve job and processor identity; uncertain submission is never
  replayed. QVM/run-only backends expose no native cancellation handle.
  (marqov-sdk#134, Rigetti slice)

- Azure Cirq execution retains its native job handle, requests best-effort
  cancellation after an interrupted result wait, and preserves structured
  recovery context. Native and IonQ result conversion and no-timeout vendor
  polling defaults are preserved. (marqov-sdk#134, Cirq slice)

- Azure and all extras now install the vendor's Qiskit/Cirq adapter dependencies,
  including Q# support and the supported Cirq range. Credential-free adapter
  imports are checked in CI and installed-wheel release verification. (marqov-sdk#223)

- Azure Qiskit result waits request best-effort cancellation on their own job
  after timeout or caller cancellation. Vendor polling receives the configured
  budget, workers do not delay event-loop shutdown, and exceptions retain
  structured job/workspace recovery context. In-flight submission acceptance
  remains explicit and unknown. (marqov-sdk#134, Qiskit slice)

- Braket result waits cancel their own task on timeout or caller cancellation.
  Configured timeouts also bound vendor polling; blocking result/cleanup calls
  use a per-execution pool so event-loop shutdown does not join them. Cleanup
  waits at most one second and is best effort. (marqov-sdk#134, Braket slice)

- IonQ submission workers retain per-call job ownership after caller cancellation
  and request cancellation once if a job ID arrives later. Uncertain acceptance
  and failed cleanup are reported without replaying the POST. (marqov-sdk#216)

- Release publication requires verification of the same built artifacts against
  the installed wheel, including fork ownership and separate QiliSDK coverage.
  Tag releases must descend from main. (marqov-sdk#165)

- Direct Qristal simulation configs again accept the documented
  `cudaq:custatevec_fp32` target, retaining its 28-qubit limit. Hardware and
  unknown IDs remain refused locally; GPU execution remains unqualified.

- Workflow graphs reject missing dependencies before planning and duplicate task
  IDs before mutation. Generated task IDs use the full UUID to reduce collision
  risk. (marqov-sdk#143)

- Workflow returns reject task proxies from another graph, and execution planning
  rejects missing output IDs before dispatch. Missing completed outputs fail
  Temporal workflows with a non-retryable application error. (marqov-sdk#212)

- Decorator and workflow graph tests now fail normally on regressions instead of
  hiding failures behind a blanket expected-failure marker. (marqov-sdk#146)

- Both development benchmark harnesses retain measured zero quantum times in
  summaries and distinguish them from missing timing data. Saved result
  timestamps include the UTC offset; result filenames keep their existing shape.
  (marqov-sdk#164)

- Braket auxiliary metadata failures no longer discard completed results. Timing
  reads documented simulator result metadata, preserves real zero durations, and
  reports unknown execution/queue duration as None. Azure polling config
  documentation states that the interval is currently unused. (marqov-sdk#135)

- Explicit native Braket export preserves sparse source labels in the Rigetti
  Rx/Rz/CZ/XY basis. Both submission paths share validation and support opt-in
  label preservation with verbatim and disabled rewiring; the executor now
  forwards explicitly supplied rewiring options. Ordinary logical conversion
  stays unchanged and hardware mapping remains unqualified. (marqov-sdk#207)

- Braket executor results retain count-origin evidence, requested and observed
  shot totals, and provider measured-wire order in additive provenance metadata.
  Probability-derived and unknown counts are explicitly ineligible for raw-shot
  analysis; physical mapping remains unqualified. (marqov-sdk#205)

- Braket probability-only results conserve requested shots through MarqovDevice,
  matching the executor's shared count allocation. Nonempty provider counts
  remain unchanged. (marqov-sdk#66)

- pytket export accepts gates supported by its Qiskit converter, including
  imported Braket V and ISwap circuits. Converter failures retain the documented
  `NotImplementedError` with the original message and cause. (marqov-sdk#155)

- OpenQASM imports detect version headers after leading comments and raise the
  documented `ValueError` for parse failures, retaining the detected version and
  original parser exception. (marqov-sdk#156)

- PennyLane circuit imports accept NumPy integer wire labels and normalize them
  to Python integers, while rejecting boolean and non-integer labels. Nested
  decomposition failures now retain the actual failing gate or exception. (marqov-sdk#157)

- Braket XY imports use their own angle instead of a previous instruction's
  arguments, and dictionary reconstruction now restores XY with its existing
  QuantumFlow parameter convention and qubit order. Requires the corrected
  marqov-quantumflow 1.0.1 dependency. Re-import earlier XY circuits from their
  Braket source: existing records cannot reveal an angle already lost during
  the old import. (marqov-sdk#195)

- Braket circuit imports reject instruction-level controls and non-default
  powers before conversion instead of silently losing them. Explicit gates
  such as CNot and unmodified imports remain supported. (marqov-sdk#192)

- Circuit dictionary reconstruction restores the standard gates emitted by
  the importers, including the 18 Braket gate types previously omitted.
  Unsupported gates and incorrect qubit/parameter counts now raise ValueError
  naming the gate and index instead of silently changing the circuit.
  Serialized output and parameter conventions are unchanged. (marqov-sdk#153)

- PyQuil imports reject gate modifiers before mapping or SWAP recognition,
  preventing loss of DAGGER, CONTROLLED and FORKED semantics. Unmodified
  canonical imports remain supported. (marqov-sdk#193)

- CUDA-Q executions serialize process-global target and seed state, temporarily
  apply configured IQM credentials with environment restoration, and retain the
  vendor SampleResult. Unpolled remote status reports maintenance rather than
  claiming online. (marqov-sdk#136)

- IBM execution retains each submitted job ID on result errors and timeouts, checks shot accounting and records executor provenance and reproducibility hashes. Tokens are excluded from configuration repr. SamplerV2 is qualified locally with Aer; live cloud and hosted qualification remain separate.

- Quantum Brilliance simulation supports current Qristal sessions without `init()` and their flat result map, preserving legacy session support. Counts are checked against shots, and results record the actual engine and reproducibility hashes. Native QPP checks pass; full SDK and hosted qualification remain separate.

### Added

- Platform client saved-script helpers upload source, request analysis explicitly,
  and submit an ordinary saved-script job using the existing server contracts.
  Analysis and submission remain separate caller actions; existing native
  workflow and inline submission APIs are unchanged. (marqov-sdk#234)

- PennyLane Lightning CPU execution with seed and numerical configuration in
  reproducibility records. GPU and tensor paths require separate device
  qualification. (marqov-sdk#184)

- Alice & Bob executor with real local cat-qubit models, native Qiskit initialization/delay preservation, canonical counts and reproducibility records. Explicit direct remote mode requires provider credentials; live remote and hosted execution remain unqualified.

### Changed

- Explicit SpeQtrum and Quantum Brilliance remote executors and factory routes
  are available, with retained job IDs and bounded polling. Lightning adds
  GPU execution paths and an explicitly unseeded tensor contract, subject to
  device qualification probes. Local QB simulation rejects hardware and unknown
  target IDs. Live remote, GPU and hosted execution remain unqualified.
  (marqov-sdk#191)

- QiliSDK local digital and analog simulator calls run off the event loop so
  concurrent coroutines and timeouts can progress. Cancelling the caller does
  not stop an already-running simulator thread. (marqov-sdk#137)

- Qilimanjaro digital and analog results now record vendor, framework, actual
  engine, access path, compute provider and software/seed/count provenance.
  Correct installation guidance for QiliSDK 0.3.0 and its QuTiP dependencies.

## [0.8.1] — 2026-09-26

### Fixed

- `MarqovDevice.run` on the IBM branch no longer crashes with an
  `AttributeError`: count extraction now calls the same implementation
  `IBMExecutor` uses, so it resolves the classical register by capability and
  returns bitstrings in the SDK's qubit-0-leftmost convention. A result with
  no resolvable register raises instead of returning `{}`. (marqov-sdk#161)
- `MarqovDevice.run` on the Azure branch returns integer counts keyed by
  bitstrings instead of `get_results()`'s normalised probabilities keyed by
  display strings such as `'[0]'`. Bit order matches `AzureQuantumExecutor`.
  Output data formats other than `microsoft.quantum-results.v1` and
  `microsoft.quantum-results.v2` raise a `ValueError` naming the format.
  (marqov-sdk#161)

- The CLI's `list` command no longer shadows the `list` builtin. The command
  function is renamed to `list_workflows` internally (the command itself is
  still invoked as `marqov list`); previously the shadowed builtin broke
  every `marqov run ... --arg key=value` invocation. See marqov-sdk#148.

- The Temporal task activity now drains the child process's stderr while the
  child runs, so a task that writes more than the pipe buffer holds (verbose
  provider logging, progress output) can no longer block the child in `write()`
  and stall the activity until its timeout. Only the last 32 KiB are kept, and
  the crash message says how many earlier bytes were dropped. A failing
  heartbeat loop is now logged at error level instead of being discarded
  silently. (marqov-sdk#140)

- The task activity validates the child process's result envelope instead of
  forwarding it verbatim. A task that writes a `result.json` naming another
  node, carrying unexpected keys, or containing content that cannot be parsed
  as JSON (invalid JSON, non-UTF-8 bytes, or nesting too deep to decode) now
  fails the activity with a non-retryable error, so one task can no longer
  overwrite a sibling's result. Child-supplied text quoted in those errors is
  truncated, keeping the failure message within Temporal's payload limit. The
  activity still never deserializes the result value. (marqov-sdk#141)
- Malformed activity payloads raise a Temporal `ApplicationError` in the job
  workflow rather than a bare `json.JSONDecodeError`, so a bad payload fails
  the workflow instead of retrying its workflow task indefinitely.
  (marqov-sdk#141)

- `QuantinuumExecutor.execute()` measures the converted circuit before
  submitting it. Marqov's `Circuit` IR carries no measurements, so the
  submitted pytket circuit had zero classical bits and every run returned
  `{"": shots}`. Measurements are added only when the converted circuit has
  no classical bits, and qubit 0 stays the leftmost bit of the counts keys.
  (marqov-sdk#130)
- `QuantinuumExecutorConfig.poll_interval_seconds` is documented as currently
  unused, matching `IBMExecutorConfig`: `execute()` blocks on the backend's
  `get_result()` rather than polling. (marqov-sdk#130)

- **IBM connections work again on current `qiskit-ibm-runtime`.** The default
  channel is now `ibm_quantum_platform` (IBM retired `ibm_quantum`, and the
  library rejects it before authenticating), and `instance` is optional
  instead of defaulting to the dead `ibm-q/open/main`: when omitted, the
  service auto-discovers the instance from the token, so a CRN is only needed
  when a token maps to more than one instance. A single normaliser shared by
  `IBMExecutorConfig`, `ExecutorFactory` and `MarqovDevice` translates stored
  legacy values rather than forwarding them: `channel="ibm_quantum"` becomes
  `ibm_quantum_platform` and a `hub/group/project` instance is dropped, each
  with a `DeprecationWarning`, and `""` now behaves like an absent key on every
  path. Pinned by `tests/test_ibm_channel.py`. (marqov-sdk#115, marqov-sdk#168)

- IonQ job polling is bounded. An unrecognized job status (for example
  `deleted`) now raises a `RuntimeError` naming the status and job id instead
  of polling forever, `IonQExecutorConfig.timeout_seconds` defaults to one hour
  instead of no timeout, and a timed-out or cancelled wait issues a best-effort
  cancel for the submitted IonQ job. Pass `timeout_seconds=None` to keep the
  previous unbounded behaviour. See
  [marqov-sdk#133](https://github.com/marqov-dev/marqov-sdk/issues/133).

- **Azure Cirq execution path:** `AzureQuantumExecutor` no longer treats the
  value returned by `AzureQuantumService.run()` as a job handle. That call
  already blocks and returns a `cirq.Result`, so every Cirq run previously
  failed with `AttributeError: 'ResultDict' object has no attribute 'results'`.
  The Cirq histogram conversion also dropped its extra reversal: cirq folds
  histograms big-endian, which is already Marqov's qubit-0-leftmost
  convention, so the reversal was inverting correct bitstrings. Cirq runs
  report `job_id` as `None`, since `run()` exposes no job id. The Qiskit path
  is unchanged. Both paths are now covered by tests that drive the real
  executor with fake services returning genuine framework result objects.
  (marqov-sdk#131)

- The platform transport no longer retries a write after a connection failure
  that happened once the request had been sent, so a submit or cancel cannot be
  double-submitted. Only connect-phase failures (refused, DNS, connect timeout)
  are still retried; an ambiguous failure raises `TransportError` with the
  idempotency key on its `idempotency_key` attribute. Idempotent requests now also retry the
  transient statuses 429, 502, 503 and 504, honouring a `Retry-After` that fits
  the remaining backoff budget and raising the mapped exception for the last
  response when attempts run out. `Job.cancel()` now uses the idempotent-write
  policy. See [error handling](docs/platform-client/error-handling.md).
  (marqov-sdk#149)

### Documentation

- The IonQ histogram bit-order claim in `_histogram_to_counts` is marked
  unverified and cites IonQ's Direct API guide, which documents little-endian
  histogram keys. The current conversion is unchanged and now pinned by an
  asymmetric-bitstring test, so any future change of convention is deliberate.

## [0.8.0] — 2026-09-25

### Added

- `MarqovClient.managed_runtimes(team_id)` and `MarqovClient.submit_native(...)`
  run Python `@task`/`@workflow` programs on the hosted managed native runtime
  through its versioned submission contract.
  - `team_id` and `cap_cents` are required.
  - Exactly one of `script_id` or `source` is required.
  - The runtime must match discovery exactly; no backend or runtime is
    substituted.
  - The request is validated locally before sending.
  - A caller-supplied `idempotency_key` is honoured across retries.
  - The admission receipt's structure and input hash are verified, plus its
    source hash when the source is known locally (inline source or
    `source_sha256`).
  - Any failure after the request may have been sent exposes the effective
    key as `exc.idempotency_key`, so an ambiguous submission can be resolved
    safely.
  
  See [Native workflows on the hosted platform](docs/platform-client/native-workflows.md).

### Changed

- Platform-client documentation now states what the hosted API does not
  currently accept: `submit()` for native or paid programs, `Circuit`
  submission, API-key `job.cancel()` and `platform_info()`. Existing behaviour of
  `submit()`, `Job`, `PlatformResult`, workflow capture and executors is
  unchanged.

## [0.7.1] — 2026-09-13

### Added

- Per-call `seed` support for QiliSDKExecutor digital and analog QiliSim runs.
  Seeded calls use a fresh single-threaded simulator and record the seed and
  thread count in result metadata. Invalid seeds, seeded QuTiP calls and other
  unsupported options fail explicitly. See [seed support](docs/qilisdk.md).

### Fixed

- QiliSDK digital and analog execution reject unsupported keyword options
  instead of silently discarding them. Supported QiliSim seeds are honoured;
  seeded QuTiP calls fail explicitly.

## [0.7.0] — 2026-09-12

Published on [PyPI](https://pypi.org/project/marqov/0.7.0/). See the
[upgrade guide](docs/releases/0.7.0.md) and
[release record](https://github.com/marqov-dev/marqov-sdk/releases/tag/v0.7.0).

### Changed

- AWS Braket is now an optional `marqov[braket]` dependency, also included in
  `marqov[all]`, with a security floor of `amazon-braket-sdk>=1.117.0`.
  Existing Braket users must select that extra when installing or upgrading.
  This includes `MarqovDevice`'s `local`/`marqov-sim` simulator and Braket circuit
  conversions. Their execution semantics are unchanged; missing installations
  now report the required extra.
- Workflow serialization declares `cloudpickle>=2.2.1` directly. A core-only
  compiler/task environment can pin its own compatible serialization version
  without inheriting Braket's job serialization dependency constraint.

### Documentation

- Added the [QuTiP guide](docs/qutip.md) covering local installation,
  copy-and-run examples, result recording, seed replay, saved states and
  troubleshooting. This guide was added after the 0.6.1 release.

## [0.6.1] — 2026-09-08

### Added

- Optional `marqov[qutip]` installation includes QuTiP 5 dependencies
  (`>=5.3.0,<6.0.0`) for solver use with the existing `marqov.qutip.record` helper.
  QuTiP is also included in `marqov[all]`; the base SDK dependencies are unchanged.

- A runnable local QuTiP decay example and a clean-install wheel smoke check
  verify optional installation and recorded observables against analytic decay.

## [0.6.0] — 2026-09-08

### Added

- `WorkflowDispatch.capture()` exposes the existing graph and a process-local
  return template for execution adapters, preserving names, plain containers,
  constants and repeated task-result references. `WorkflowCapture.resolve_output`
  reconstructs that value from completed node results without submitting work.
  See [workflow capture](docs/workflow-capture.md) for supported forms and codec
  responsibilities. Legacy `run()`/`start()` result transport is unchanged.

### Fixed

- **Executor result normalization:** IBM now selects classical registers by their
  count-reading capability and normalizes bitstrings to qubit 0 leftmost. Results
  containing multiple measured registers raise `NotImplementedError` instead of
  silently returning only one register. Braket probability fallback and IonQ
  histogram conversion now use shared largest-remainder allocation so counts
  sum to the requested shots, including unnormalized histograms. Asymmetric
  result fixtures cover bit order. (marqov-sdk#114)

- Task dependency extraction now follows nested list/tuple/dictionary values,
  matching argument serialization, and deduplicates predecessors in encounter
  order. Cyclic argument containers raise an explicit error.
- **Behavior correction:** using an unresolved task proxy as a boolean now raises
  `TypeError` instead of silently selecting the true branch during graph build.
  Dynamic branching on task results requires an execution mechanism; Python
  truthiness cannot implement it at capture time.

- **`Circuit.from_qiskit`/`from_cirq`/`from_pyquil` silently discarded
  mid-circuit `measure`, `reset`, and `delay` instructions.** A terminal
  measurement (nothing after it touches its qubit or reads its classical bit)
  is genuinely implicit under this SDK's gate-only, `shots`-based circuit
  model and is still silently skipped. A **mid-circuit** measurement,
  `reset`, or `delay` is never implicit — dropping one silently changed what
  the imported circuit computed, with no warning. All three importers now
  raise a clear error naming the instruction and qubit instead.

  **Breaking, deliberately:** any circuit containing a mid-circuit
  measurement, `reset`, or `delay` that previously imported "successfully"
  (silently as a different, wrong circuit) now raises `ValueError`
  (`from_qiskit`/`from_cirq`) or `NotImplementedError` (`from_pyquil`)
  instead. There is no opt-out flag: the old behavior was never a
  correctness guarantee to preserve, it was a silent miscomputation.
  (marqov-sdk#76)

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
  (marqov-sdk#107)

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
