# Saved-script execution

The hosted platform already supports saving a circuit in Scripts, analysing its
execution options, submitting a job and collecting its result in Jobs. These SDK
helpers expose that sequence to a Python coordinator. Run remains the code IDE;
Scripts remains the library; Jobs tracks submissions; Projects holds Workspaces
and Reports. This does not introduce another orchestration UI.

The wire sequence was exercised by the marqov-hpc feedback and QPE studies using
a separate HTTP adapter. This SDK change has contract tests, not a claim of
additional live runs or Slurm execution. The existing generic submit() method
retains its documented hosted limitations.

## Choose an execution destination deliberately

1. Save supported source with save_script(). Retain its script_id.
2. Read script_execution_options(). Confirm the team, saved-source SHA256 and
   intended destination. Discovery is advisory, not an admission reservation.
3. Call analyse_script() with the same content, backend and shot count. Inspect
   can_run, checks and cost. Stop on refusals or unacceptable cost; select any
   warning IDs explicitly after reviewing them.
4. Re-read execution options if source may have changed. Compare source_sha256
   with SHA256 of the expected UTF-8 source. The analysis content_hash response
   is a djb2 hint and must not be compared to SHA256.
5. Persist the exact submission parameters and an operation key, then call
   submit_script() with the returned analysis UUID. Persist job.id immediately.
6. Poll job.result(), or reconnect using client.job(saved_job_id) after restart.
   Check result metadata for source identity and measurement bit ordering before
   calculating scientific outputs.

The helpers return discovery and analysis dictionaries unchanged so callers can
inspect platform fields without waiting for a new SDK release. They do not
choose a backend, authorize cost or accept warnings. A free-route analysis ID
currently supplies traceability; it does not enforce an immutable source pin.

## When the outcome is unknown

Use a different durable operation key for each upload, analysis and submission.
When no key is supplied, the transport generates one per call. The client does
not persist keys, request bodies or job IDs to disk; durable workflow state is
the coordinator's responsibility.

An ambiguous post-send network failure is not automatically retried. A failed
submission response, including a missing or malformed job ID, retains the key
on TransportError.idempotency_key. Hold that operation and investigate; a new
key could create a second job. The current saved-script platform path does not
provide a general operation lookup or guaranteed recovery for every response-loss
window. Replaying a cached response is not a proof of atomic admission recovery.
These helpers do not implement the byte-exact durable journal used by the HPC
experiment coordinator.

Once a job ID is known, reconnecting and polling uses ordinary GET requests.
Polling timeout does not cancel execution. API-key cancellation remains
unsupported; see the API reference for the browser-session cancellation limits.

## Evidence

The existing free hosted runs and their precise limitations are recorded in
[marqov-hpc feedback](https://github.com/marqov-dev/marqov-hpc/tree/main/experiments/workflow-feedback)
and [QPE](https://github.com/marqov-dev/marqov-hpc/tree/main/experiments/workflow-qpe).
They establish the saved-script HTTP path with the experiment adapter, not
qualification of these new SDK wrappers, a paid provider or an HPC scheduler.
