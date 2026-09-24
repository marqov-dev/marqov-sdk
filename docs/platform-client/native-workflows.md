# Running native workflows on the hosted platform

`MarqovClient.submit_native()` runs a Python `@task`/`@workflow` program on
Marqov's managed native runtime. It uses the platform's versioned managed-native
submission (`marqov.public-submission/v1`). This is also the request the Marqov
app's **Run** page sends.

Your program runs on the platform's own pinned `marqov` version. Your locally
installed version only affects the client calls on this page.

## What you need

| | |
|---|---|
| API key | `simulate` permission or higher, for the paying team (`MARQOV_PLATFORM_KEY`) |
| `team_id` | The UUID of the team that owns and pays for the run. Required: the platform does not infer it from the key |
| Program | A script saved in that team (`script_id`) **or** the source text (`source`) — exactly one |
| `entrypoint` | The name of the `@workflow` (or task) function to call |
| `cap_cents` | Your spending cap for the run, in cents, covering compilation and tasks. Required; there is no default |
| Enabled runtime | Managed native execution must be enabled for the team. `client.managed_runtimes(team_id)` shows what is enabled |

## Complete example: discover → submit → wait → read

The saved script (`MARQOV_SCRIPT_ID`) contains:

```python
from marqov import task, workflow


@task(timeout=60)
def alice(seed: int):
    values = [seed, seed + 1, seed + 2]
    return {"stage": "alice", "seed": seed, "values": values, "total": sum(values)}


@workflow
def native_canary(seed: int = 7):
    return {"schema_version": 1, "alice": alice(seed)}
```

The client program ([`examples/platform_native_workflow.py`](../../examples/platform_native_workflow.py)):

<!-- example: examples/platform_native_workflow.py (checked verbatim by tests) -->
```python
"""Run a saved @workflow on Marqov's managed native runtime.

Discover the team's runtime -> submit -> wait -> read the result.

Needs a Marqov API key with ``simulate`` permission or higher, the UUID of the
team that pays for the run, and a script saved in that team whose source
defines the workflow named below. Set:

    MARQOV_PLATFORM_KEY   API key (marqey_live_... / marqey_test_...)
    MARQOV_TEAM_ID        team UUID
    MARQOV_SCRIPT_ID      saved script UUID
"""

import os
import uuid

from marqov.platform import MarqovClient

client = MarqovClient()  # reads MARQOV_PLATFORM_KEY
team_id = os.environ["MARQOV_TEAM_ID"]
script_id = os.environ["MARQOV_SCRIPT_ID"]

# 1. Discover. Only runtimes enabled for this team are listed; an empty list
#    means managed native execution is not enabled for it.
runtime = next(
    (r for r in client.managed_runtimes(team_id) if r["programming_model"] == "native_workflow"),
    None,
)
if runtime is None:
    raise SystemExit("Managed native workflows are not enabled for this team.")
print(f"{runtime['backend']}: cap between {runtime['min_cap_cents']} and "
      f"{runtime['max_cap_cents']} cents")

# 2. Submit. The cap is your decision: the most this run may spend, in cents,
#    covering compilation and tasks. Keep the idempotency key: resubmitting with
#    the same key returns the original admission instead of running twice.
cap_cents = 100
idempotency_key = str(uuid.uuid4())
job = client.submit_native(
    team_id=team_id,
    script_id=script_id,
    entrypoint="native_canary",   # the @workflow function in the saved script
    kwargs={"seed": 7},
    cap_cents=cap_cents,
    idempotency_key=idempotency_key,
)
print("Admitted job", job.id)

# 3. Wait. Raises JobFailed if the job fails and TimeoutError if it is still
#    running when the timeout expires (it keeps running server-side).
result = job.result(timeout=600.0)

# 4. Read. A workflow result is the platform's managed-result projection: one
#    entry per declared output, with its JSON value when it can be displayed.
for output in result.raw["outputs"]:
    display = output["display"]
    if display["state"] == "available":
        print(output["task_key"], display["value"])
    else:
        print(output["task_key"], "value not displayable:", display["reason"])
```

## What `submit_native()` does

1. **Validates and serialises locally.** Nothing is sent if this fails
   (`ValueError` / `TypeError`). It checks:
   - UUIDs;
   - exactly one of `script_id` / `source`;
   - an identifier entry point;
   - `cap_cents` from 1 to 10 000;
   - arguments that are JSON values: `str` dictionary keys, finite numbers,
     nesting depth limited, integers of at most 4300 digits;
   - source and input of at most 1 MiB each, and a body of at most 4 MiB.
   
   Arguments are sent as JSON text, so large integers arrive exactly.
2. **Discovers the team's runtimes.** It requires one matching `backend`
   (default `"marqov-sim"`) **and** `programming_model` (default
   `"native_workflow"`), and `cap_cents` must be inside that runtime's range.
   It never substitutes another backend or runtime. If no runtime matches, it
   raises `MarqovPlatformError` with `code="runtime_not_enabled"` and submits
   nothing.
3. **Submits** with an `Idempotency-Key`: yours, or a fresh UUID for each call.
   The same key is reused on the client's own retries.
4. **Verifies the admission receipt.** The receipt must be for exactly this
   source and input. If it cannot be confirmed, the call raises
   `invalid_receipt` or `receipt_mismatch` rather than returning a job. Retry
   with the same `idempotency_key` to confirm.

Admission means the job is funded and queued, not that it has finished. The
returned `Job` is the ordinary job handle; `status()` and `result()` work
unchanged.

### Refusals

| Code | HTTP | Meaning / what to do |
|---|---|---|
| `runtime_not_enabled` | — | No matching runtime is enabled for the team (raised by the client before submitting) |
| `validation_error` | 400 / 422 | The platform refused the request shape, or `cap_cents` is outside the runtime's range |
| `spend_limit_exceeded` | 402 | Available funding, or a team or key spending limit, cannot cover `cap_cents` |
| `permission_denied` | 403 | The key's permission is below `simulate` |
| `not_found` | 404 | The team or saved script is not visible to this key |
| `idempotency_conflict` | 409 | The key was already used with different material; use a new key |
| `analysis_mismatch` | 409 | `source_sha256` no longer matches the saved script |
| `rate_limited` | 429 | Submission rate or the team's daily quota was reached (`RateLimited`, with `retry_after` when given) |
| `managed_execution_unavailable` | 503 | Not enabled for the team and runtime, or admission could not be confirmed; retry with the **same** `idempotency_key` |

## Current limitations

- **No result helper yet.** Read values from `result.raw["outputs"]` as shown
  above. `result.counts` is `None` for a workflow result.
- **`submit()` does not run native programs.** `client.submit(str, ...)` sends
  the generic submission body, which the hosted API refuses for `@task`/`@workflow`
  and plain Python source (`422 execution_unavailable`). Use `submit_native()`.
- **`Circuit` submission is not accepted.** `client.submit(Circuit, ...)` sends a
  `circuit` field the hosted API does not accept, and the request is refused
  (400).
- **`job.cancel()` does not work with an API key.** The hosted cancellation
  endpoint accepts only signed-in browser sessions, so an API key receives
  `403 api_key_not_supported`. Cancel from the job page in the Marqov app; this
  covers funded jobs that have not yet reached a provider.
- **`client.platform_info()` has no endpoint.** The hosted API does not serve
  `/api/meta`.
- **One runtime backend.** Managed native execution currently runs on
  `marqov-sim`, for teams where it is enabled.
