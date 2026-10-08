# Rigetti interruption and job ownership

`RigettiExecutorConfig.timeout_seconds` is an optional positive, finite budget
covering client initialization, compilation, submission and result retrieval.
`None` adds no SDK deadline; the configured pyQuil compiler and execution
timeouts still apply. Circuit conversion happens before this budget starts.
Connection settings are captured when the executor is constructed. Construct a
new executor to change them; the overall budget is read for each execution.

For a native pyQuil `QPU`, the executor submits with `QPU.execute`, retains the
response and waits with `QPU.get_result`. An interrupted result wait requests
`QPU.cancel` using that same QPU object and response, preserving the submitting
client, processor and execution options. Cleanup waits at most one additional
second. pyQuil cancellation applies to jobs that have not started; a completed
request does not establish terminal provider state.

`TimeoutError` and caller `CancelledError` retain their types. They expose a
`remote_job` dictionary with provider, requested and actual processor IDs,
`as_qvm`, `job_id`, `submission_status`, `phase` and `cancellation_supported`.
Backend/result conversion failures are wrapped in `RuntimeError` with the same
context. Successful result metadata includes these fields and preserves counts
and the raw pyQuil result.

Before submission, status is `not_submitted`. During interrupted submission it
is `unknown`, with no job ID: a worker can finish submitting after the caller
returns. The SDK neither recovers a late handle nor replays submission. Once a
response is received, status is `submitted` and the native ID is available.
An interruption during compilation prevents subsequent submission by that call.

`await executor.cancel(job_id)` supports owned active handles and the last 32
interrupted handles retained by that executor instance. It returns `True` when
the cancellation request finishes, without confirming provider state. Unknown,
evicted, completed or other-instance handles return `False`; request failures
and requests exceeding one second also return `False`. Retained handles include
compiled programs, so the cache is bounded. QVM and compatible run-only injected
clients retain their existing `run` path and offer no native cancellation handle.

Blocking calls use a per-execution pool whose shutdown does not join workers.
This keeps event-loop shutdown from delaying interruption, but does not stop an
already running vendor call or guarantee immediate interpreter exit. Provider
timeouts and external reconciliation remain necessary. Timeout or cancellation
alone must not authorize resubmission or establish settlement.

The offline tests exercise real pyQuil QPU submission/cancellation primitives
with network boundaries replaced. They cover exact job ownership, concurrent
execution, interruption in compilation/submission/result phases, bounded handle
retention and failed or slow cleanup. They do not qualify live QCS execution.
