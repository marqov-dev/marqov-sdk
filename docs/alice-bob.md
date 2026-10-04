# Alice & Bob executor

Install in a dedicated environment with `pip install 'marqov[alice-bob]'`.
The qualified vendor version uses Qiskit 1.x; keep this environment separate
from IBM Runtime installations that require Qiskit 2.x.

`AliceBobExecutor()` runs Alice & Bob's `EMU:40Q:LOGICAL_NOISELESS` model locally.
It returns counts in Marqov's qubit-0-leftmost convention and records the exact
backend, model parameters, package versions, seed and circuit/count hashes.
Local model execution is not execution on a physical Alice & Bob QPU.

```python
from marqov.circuits import Circuit
from marqov.executors import AliceBobExecutor

result = await AliceBobExecutor().execute(
    Circuit().x(0).cz(0, 1), shots=32, seed=7
)
assert result.counts == {"10": 32}
```

Use `execute_native(qiskit_circuit, shots=...)` for cat-qubit workloads that
need Qiskit initialization or physical delays. These operations are passed
through the vendor transpiler rather than converted through Marqov's gate-only
circuit representation. Require one classical register, full terminal
measurement and the mapping q[i] to c[i]. Unsupported measurement layouts are
refused before submission.

Direct remote execution requires explicit `mode="remote"`, a provider API key
and an available backend name. It never falls back to a local model. This path
is not yet qualified against a live account, and does not implement hosted
Marqov credential isolation or managed billing. Do not put the key in hosted
runner input. Local seeds and result timeouts are refused where the vendor
cannot support them. If result retrieval fails after submission,
`AliceBobExecutionError.job_id` identifies the submitted job: inspect it before
retrying. The executor does not automatically resubmit or promise cancellation.

Qualification: real vendor local basis-state counts and a native
initialization/delay workload pass. Focused checks also cover refused classical
mappings, secret redaction and retained job identity after an unresolved result.
