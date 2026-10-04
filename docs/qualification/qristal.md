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
