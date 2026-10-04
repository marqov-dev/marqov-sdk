# Quantum Brilliance qualification

The SDK's existing SimulationExecutor is reused. Current Qristal constructs an
initialized session and returns a flat MapVectorBoolInt. Older releases have
init() and a nested result map. Both layouts are supported; counts must account
for every requested shot. Results identify the actual engine, including Aer
when the existing noise path selects it. State-vector extraction is refused if
the installed build does not expose the required API.

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
