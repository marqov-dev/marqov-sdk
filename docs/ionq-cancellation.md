# IonQ submission cancellation

`IonQExecutor.execute()` submits one POST and polls its returned job ID. Its
`timeout_seconds` applies to polling after submission; it is not a deadline
for the whole HTTP submission.

If the caller cancels during submission, it receives `CancelledError` promptly.
The submission worker retains ownership of that call. If its original response
supplies a usable job ID, cleanup makes one best-effort cancellation request for
that ID using the submission's original endpoint and authentication. It never
uses another concurrent call's ID and never replays the POST. Cancellation after
the worker returns, before the coroutine resumes, also schedules cleanup.

A cancellation exception note and warning describe unresolved acceptance. If
submission finishes without a usable ID, a warning states that acceptance is
unknown. Failed cancellation or unavailable cleanup scheduling logs the known
job ID and warns that the job may still be running. A successful HTTP response
to a cancellation request does not prove the provider stopped execution.
Do not automatically resubmit an interrupted or uncertain submission.

Cancellation while polling retains the existing best-effort cancellation
behavior. Normal successful execution and result metadata are unchanged.
`_current_job_id` remains private compatibility tracking; it is not authoritative
ownership for concurrent calls.

Both submission and cleanup use the existing 30-second HTTP timeout. This bounds
socket waits, not total wall time: DNS, continuing response traffic, an injected
transport or thread-pool contention can take longer. The caller does not wait
for submission cleanup, but `asyncio.run()` and interpreter shutdown may wait
for outstanding default-executor threads. Process termination or a lost response
can prevent cleanup. No guaranteed remote cancellation or hard process deadline
is claimed; broader executor lifetime work is tracked separately in marqov-sdk#134.

Regression tests use a blocking injected transport and exercise late job IDs,
concurrent success, repeated cancellation, missing IDs, request failures and
cancellation between worker completion and coroutine resumption. They make no
provider calls. Connected IonQ cancellation remains separate qualification work.
