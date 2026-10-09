# Quantum Brilliance qualification

The SDK's existing SimulationExecutor is reused. Current Qristal constructs an
initialized session and returns a flat MapVectorBoolInt. Older releases have
init() and a nested result map. Both layouts are supported; counts must account
for every requested shot. Results identify the actual engine, including Aer
when the existing noise path selects it. State-vector extraction is refused if
the installed build does not expose the required API.

For current noisy execution, the SDK resolves Aer before validating its
28-qubit limit or importing the native runtime. Results use the registered
`qb-sim-noisy-aer` backend and record the original target in
`metadata.requested_simulator`; `metadata.simulator` and `metadata.engine`
identify Aer. Input configuration is retained. SDK boundary tests cover this
resolution and early refusal with a vendor-shaped fake; they do not qualify
native Aer noise simulation in any of the images recorded below.

The exact SDK native execution function source ran successfully against the
retained source-built Qristal QPP image on Linux/amd64: asymmetric `10` and `01`
and a Bell circuit (32 shots each). The native engine was real, not mocked.
`qristal_native_functions.py` records how that narrow check was executed.

This is **native-function qualification**, not qualification of an installed
SDK package or hosted execution: the retained Qristal image uses Python 3.10,
while the current SDK requires Python 3.12. The proof loads the exact two native
function definitions with minimal supporting types; it does not import the full
SDK or claim that its package requirements were satisfied. A Python 3.12 Qristal
binding remains required for full in-process SDK qualification. GPU, tensor,
commercial noise and hosted admission are separate unqualified paths.

Retained local image: `qristal-qpp-attribution:20260914`; image index:
`sha256:03a2db140fdb579f3d6376700c36016af2bd3ffa139aeb5282439498a9a4aa2f`.
No image was published or deployed by this SDK change. The module does not expose
a package version; the record leaves it unknown instead of inventing one.

## Full installed SDK qualification (2026-10-03)

The Python 3.12 binding subsequently compiled from the exact matching Core source
`a5c3e5fa544c07d538974d3a289b19652d483848`, reusing the retained C++ libraries.
The complete current SDK was installed in the resulting image. All three cases
then passed through `SimulationExecutor.execute()` with actual Marqov circuits,
without AST extraction or support-type substitution. This resolves the Python
3.12 blocker for local QPP execution.

The record `qristal-python312-sdk.json` binds the image and source identities.
The earlier Python 3.10 native-function evidence above remains historical.
The image is a local qualification builder, contains compiler tooling, and is
not a deployment candidate. No managed/hosted, GPU or commercial noise claim
follows from this result. Runtime hardening, dependency locking and hosted
admission remain necessary before deployment.

The saved Dockerfile/CMake recipe consumes a staged context containing the exact
Core `src/python` directory at `source/python`, the SDK's `marqov` directory at
`sdk-marqov`, plus its `pyproject.toml` and `README.md` under the names in the recipe.
Pinned retained images are available locally; the staged input source IDs and
executed image digests are recorded separately. Dependency resolution was not
locked before this qualification build; exact installed versions are retained in
`qristal-python312-packages.json`. Do not claim the recipe alone reproduces the
recorded image byte-for-byte.

## Relocated compiler runtime integration

A packaged native runtime can keep its libraries and catalogue under immutable
/opt/qristal, outside tenant /work scratch. SimulationConfig accepts an explicit
remote_backend_database_path for that catalogue; omitted configuration preserves
the vendor default. This setting does not choose a provider or change the engine.
The runtime must separately initialize XACC's supported explicit root path before
Qristal import. The platform packaging qualification is separate from this SDK PR.

The full current SDK source at 2d5ca74 was subsequently installed in a separate
qualification image on the partner compiler's Python 3.12.14 environment. All
existing dependencies remained installed and `uv pip check` passed. The explicit
catalogue-path setting returned 10:32, 01:32 and Bell00:16/11:16 through the
actual SimulationExecutor, with engine=qpp and shot/provenance records.
qristal-relocated-sdk.json binds that evidence. This is an unpublished source
package qualification, not hosted execution or release publication. The saved
Dockerfile consumes a local staged sdk-source context and the internal native
full candidate; it is not a standalone deployed-image reproduction claim.

## Explicit direct QB circuit API

`QBRemoteExecutor(QBRemoteConfig(endpoint=..., target=..., account=..., token=...))`
uses the pinned Qristal `a5c3e5fa544c07d538974d3a289b19652d483848` QDK circuit
wire schema. Factory selection requires `provider="Quantum Brilliance"` and
`access_path="remote"`; omitted/local access retains the existing QPP route.
This adapter never calls the native hardware session, whose HTTP wrapper retries
uncertain POSTs and disables certificate verification.

The supported workload is deliberately narrow: QB-QDK2-CZ native Rx/Ry/CZ,
finite numeric angles, zero initial state, ascending measurement slots, exact
sampling (1–100000 shots, 1–28 qubits, at most 10000 gates). Unknown gates/models,
noise, seeds and extra options fail before contact. These are adapter bounds,
not assertions of a particular device's capacity. Endpoint/target/model/account
must be connected-qualified by the caller. The circuit response API does not
independently echo account, target or submitted payload; retained bindings and
payload hashes are local request provenance, not a provider attestation.

One authenticated POST uses verified HTTPS, no redirects/retries/ambient proxy
or credentials. The caller supplies an already valid reservation token;
this adapter does not reserve hardware or send reservation writes. Any uncertain
POST raises acceptance_unknown and must not be resubmitted automatically.
Known IDs remain in immutable `last_job` for GET-only `readback(job)` across
restarts. Null data/HTTP 425 means pending; exact binary per-shot rows are required
for success. Errors/timeouts retain the known ID. No cancellation/status endpoint
or provider idempotency guarantee is claimed: cancel returns False and device
status raises NotImplementedError. Poll/request/byte limits are finite; DNS and
OS blocking may leave a daemon one-send request in flight after the caller
deadline; the watchdog bounds caller wait without claiming request cancellation.
An uncertain write must still be reconciled and never replayed. Local simulation
accepts catalogue simulator IDs and the documented `cudaq:custatevec_fp32`
single-precision variant ([Qristal backend reference](https://qristal.readthedocs.io/en/stable/rst/backends.html)).
Hardware and unknown IDs are refused before session creation, so arbitrary
backend strings cannot bypass the explicit remote transport.

This is a direct SDK adapter with local fixture evidence only. No QB hardware
job, credential custody, managed funding or connected qualification is claimed.
