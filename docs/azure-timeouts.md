# Azure Qiskit interruption and recovery

The Qiskit execution path retains the native job returned by `backend.run()`.
`timeout_seconds` starts after that submission returns, is forwarded to
`job.result(timeout=...)`, and also bounds the asynchronous result wait. A
configured value must be finite and positive; `None` preserves vendor defaults.
`poll_interval_seconds` remains unused: the vendor owns its polling schedule.

A result timeout or caller cancellation makes one best-effort `job.cancel()`
request on that invocation's handle. Concurrent submissions cannot substitute
the last tracked job. Cleanup waits at most one additional second; failed,
slow or repeatedly interrupted cleanup preserves the original exception type
and message. A warning reports unconfirmed cleanup. A returned cancellation
request does not establish the job's terminal provider state.

Exceptions after a known submission expose a `remote_job` dictionary containing
`provider`, `framework`, `job_id`, `backend`, `subscription_id`, `resource_group`,
`workspace_name`, `location` and `submission_status="submitted"`. The same fields
are included in successful result metadata. The Qiskit workspace route is
captured at executor construction and used for both workspace/backend creation
and each invocation's recovery context; create a new executor to change that
route. Timeout budgets are still read per invocation. This captures the configured
workspace route; it does not attest the Azure credential's principal or prove
billing account ownership. Credentials are not included. The exception keeps
its original type, including `TimeoutError` and `asyncio.CancelledError`, and
has a recovery note naming the job and workspace.

Errors or cancellation while `backend.run()` is in flight carry the same
context with `job_id=None` and `submission_status="unknown"`. The blocking call
may still finish remotely after its caller leaves. No automatic replay or
late-job recovery is implemented in this slice. The workspace identity is
context for reconciliation, not evidence that submission failed.

Submission, result and auxiliary metadata use a per-execution pool that closes
without joining active calls. Result/cleanup workers therefore do not delay
`asyncio.run()` shutdown. Workspace/backend initialization still uses the
existing default executor before submission. Vendor polling can overshoot its
budget while sleeping or doing network I/O; this is not a hard process deadline.
Python interpreter shutdown can still wait for live worker threads, and repeated
interruptions can accumulate workers until their calls finish. Use finite
budgets when bounded vendor polling is required.

Counts and bit ordering, raw results, cost/timing extraction and successful
wall-time reporting are preserved. Neither an interrupted wait nor a cancellation
request authorizes releasing financial holds, settling as no contact, or
resubmitting a job. Platform recovery must confirm provider state separately.

This is the Qiskit portion of marqov-sdk#134. The Cirq path still calls
`service.run()` without retaining a job handle and needs separate work. IBM's
preserve-job-on-timeout policy is unchanged. Tests use offline provider-shaped
jobs; connected cancellation and deployed runtime compatibility remain
unqualified.
