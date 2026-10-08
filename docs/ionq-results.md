# IonQ API routes and probability provenance

The default `IonQExecutorConfig` preserves the legacy v0.3/OpenQASM 2 route.
Its historical decimal-index formatting is unchanged and explicitly labelled
`legacy-msb-first-unqualified`. Do not infer a physical wire mapping from that
label: the current [integer-key guide](https://docs.ionq.com/guides/direct-api-submission)
describes qubit i as the integer bit with weight 2^i. Resolving legacy ordering
requires a separate compatibility decision and connected asymmetric controls.

## Explicit v0.4 ideal simulator route

```python
from marqov import Circuit
from marqov.executors import IonQExecutor, IonQExecutorConfig

executor = IonQExecutor(IonQExecutorConfig(api_version="0.4", api_key="your-key"))
result = await executor.execute(Circuit().x(0).rz(0.0, 1), shots=200)
```

The factory also forwards `api_version`. The API version selects the default
endpoint. Supplying a known `/v0.3` or `/v0.4` endpoint that disagrees with the
version raises; changing only the URL does not migrate the protocol.

This opt-in route supports the ideal `simulator` only, with no noise model.
QPU/noisy execution is refused. Inputs must use dense integer wire labels from
zero and concrete gates that convert to the canonical Qiskit builder basis:
H/X/Y/Z/S/T/Rx/Ry/Rz/CX/CZ/SWAP. Empty circuits, sparse labels, unresolved
parameters and other converted operations are refused before submission.

The SDK emits [OpenQASM 3](https://docs.ionq.com/api-reference/v0.4/openqasm3)
as `type=ionq.qasm3.v1`, `backend=simulator`, with individual final measurements
from q[i] into c[i]. Registers are named deterministically. It recognizes the
v0.4 `started` polling status and retrieves only the completed job's
`ionq.result.probabilities.json.v2` descriptor through the job-scoped
[artifact endpoint](https://docs.ionq.com/api-reference/v0.4/jobs/get-job-artifact).
The terminal ID, backend and input type must match the submission. Descriptor
format/media type/ID, probability values, normalization, bitstring width and
alphabet are validated. Other formats are refused rather than guessed. Result
retrieval shares the post-submission completion budget.

The selected `output_all` register represents final qubit state. IonQ's QASM3
contract explicitly places q[0] at the leftmost character; this differs from
legacy decimal integer interpretation. Named classical registers are preserved
in the artifact but are not merged or used to infer wire order. Offline
asymmetric X(q0)/X(q1) controls check this mapping against an independent local
statevector. No live provider qualification is established by these tests;
connected controls are still required before relying on account/provider behavior.

## Result contract

`raw_result` remains the terminal job dictionary. Metadata adds:

- `job_id`, `target`, `base_url`, `api_version` and `input_type`.
- `input_sha256`: SHA-256 of the UTF-8 submitted QASM3 source.
- `num_qubits`, `wire_labels`, `wire_order="q0-leftmost"`,
  `result_register="output_all"` and explicit offline-only ordering evidence.
- `source_probabilities`: original probability values, before count allocation.
- `counts_kind="probability-derived"`, `counts_allocation="hamilton"`,
  `raw_shot_eligible=False`, and `ideal_shots_ignored=True`.
- `result_artifact`: parsed `payload`, `body_base64`, `sha256`, `source`,
  descriptor `format`/`id`, and `job_id`.

`body_base64` encodes the exact HTTP response entity bytes exposed by
`requests.Response.content` (after requests' content decoding), not reserialized
JSON or compressed wire bytes. `sha256` hashes those bytes. No credentials or
request authorization headers are retained. Artifact requests use the configured
API endpoint and descriptor ID, never an arbitrary URL from a job response.

IonQ ignores requested shots on the ideal simulator. `counts` sum to the
requested allocation total but are deterministic Hamilton allocations from
probabilities, not observed shots. `ExecutionResult.probabilities` retains its
existing meaning: fractions calculated from allocated counts. For ideal
comparisons use `source_probabilities` or parse the original artifact. Do not
use these counts for raw-shot inference, sampling uncertainty or shot ordering.

Legacy results also retain their original parsed probability response and
available response bytes/hash in `result_artifact`, including inline terminal-job
responses. Minimal injected legacy transports without `content` retain the
payload but explicitly report unavailable bytes/hash as `None`; the v0.4 route
requires original bytes. Legacy counts and the probabilities property are unchanged.

Known-ID v0.4 interruptions carry `remote_job` context. Validation/readback
errors also retain `result_artifact`, including malformed JSON bytes when
available. Submission keeps the existing per-call late-ID cancellation behavior,
without replay. Cancellation remains best effort and does not establish terminal
provider state, recovery authorization or settlement. This route is an offline-
verified protocol candidate, not permission to submit a provider job.
