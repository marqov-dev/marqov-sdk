# Braket timeout and cancellation handling

`BraketExecutorConfig.timeout_seconds` bounds the result wait after Braket
returns a submitted task. When configured, the executor also passes that budget
as the vendor's `poll_timeout_seconds`, avoiding its default five-day poll.
`poll_interval_seconds` is forwarded and capped at the timeout so a polling
sleep cannot exceed the whole budget. Both values must be finite and positive
when a timeout is configured. With `timeout_seconds=None`, vendor polling
defaults are unchanged, including the potentially five-day polling budget.
Configure a finite timeout when bounded polling is required.

On result timeout or caller cancellation, the executor makes one best-effort
cancellation request for that invocation's task ARN. Another concurrent call's
last-task tracking cannot replace this ID. Cleanup is given at most one second
of additional waiting. Failure, a slow request, or repeated caller cancellation
preserves the original `TimeoutError` or `CancelledError`; a warning reports
unconfirmed cleanup. The exception note includes the task ARN. A completed
cancellation request is not proof that the provider stopped execution.

Blocking submission, result and cached-metadata calls use a pool owned by the
execution. A second worker permits cleanup while result retrieval is blocked.
The pool closes without joining active calls, so `asyncio.run()` does not join
these result or cleanup workers. The synchronous Braket poller's event loop is
closed when that worker finishes. Queued work is cancelled when the pool closes.

This is not a hard wall-clock or process deadline. Blocking network calls may
outlive the polling budget, and Python interpreter shutdown can still wait for
threads. Failed cancellation with no configured timeout can leave a worker
polling for the vendor's full budget; repeated interruptions can accumulate
active pools until their calls finish. A lost response or process termination
can prevent cancellation.
Device initialization and submission precede the result deadline; cancellation
while submission is in flight can leave acceptance unresolved. Do not assume
that a failed or interrupted submission is safe to replay.

Counts, raw results, provenance, metadata timing and no-timeout defaults are
preserved for successful runs. A vendor result of `None` is treated as failure
or polling timeout, rather than as a successful result.

Tests exercise real execute paths with blocked provider-shaped fakes and the
installed Braket poller with an injected session. No AWS requests are made.
Connected-provider cancellation remains unqualified. This covers the Braket
result-wait portion of marqov-sdk#134; other adapters and submission ownership
need their own work.
