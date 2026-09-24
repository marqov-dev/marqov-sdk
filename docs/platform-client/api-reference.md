# API Reference — `marqov.platform`

`marqov.platform` is an optional subpackage.  It is only loaded when you
import it explicitly; `import marqov` has no cost from this package.

```python
from marqov.platform import MarqovClient
from marqov.platform import Job, JobStatus, Backend, PlatformResult
from marqov.platform import MarqovPlatformError, AuthenticationError, JobFailed
```

> **Live-server caveat:** This API is unit- and contract-verified against a
> mocked transport. These examples are not yet verified against a live server —
> live verification is pending the staging environment.

---

## `MarqovClient`

High-level client for the Marqov Platform API.

```python
client = MarqovClient(
    api_key: str | None = None,
    *,
    base_url: str | None = None,
    timeout: float = 30.0,
)
```

**Key resolution:** `api_key` takes precedence over `MARQOV_PLATFORM_KEY`.
If neither is set, `AuthenticationError` is raised immediately.

**Base URL resolution:** `base_url` takes precedence over `MARQOV_PLATFORM_URL`,
then the built-in production endpoint (`https://app.marqov.ai`).

**`timeout`** is the per-request HTTP timeout in seconds (not the overall
`result()` poll timeout — those are separate).

---

### `client.submit()`

```python
job: Job = client.submit(
    program: str | Circuit,
    *,
    backend: str,
    shots: int = 1000,
    framework: str | None = None,
)
```

Submit a quantum program to the platform and return a `Job` handle.

> **Not currently accepted by the hosted API.** This method sends the generic
> submission body. The hosted API refuses it for `@task`/`@workflow` and plain
> Python source (`422 execution_unavailable`), for inline code sent to a paid
> backend (`422 script_required`, which surfaces as `MarqovPlatformError`
> rather than `PaidBackendNotSupportedYet`), and for a `Circuit` (`400`).
> For native programs use [`client.submit_native()`](#clientsubmit_native).

**`program`:**

- `str` — treated as inline executable code. `framework` is **required**;
  omitting it raises `ValueError`.
- `Circuit` — serialised as OpenQASM 3 in a `circuit` field. `framework`
  must **not** be supplied; passing it raises `ValueError`. The hosted API does
  not accept the `circuit` field and refuses the request (`400`).

**`backend`:** backend slug (e.g. `"marqov-sim"`). The client does not check
the backend; the platform decides whether the program can run there.

**`shots`:** number of measurement shots. Default: `1000`.

**`framework`:** framework identifier for string programs (e.g. `"marqov"`).

**Raises:**
- `ValueError` — `program` is `str` but `framework` was omitted, or `program`
  is a `Circuit` and `framework` was supplied.
- `TypeError` — `program` is neither `str` nor `Circuit`.
- `PaidBackendNotSupportedYet` — HTTP 422 `analysis_required`.
- `AuthenticationError` — HTTP 401.
- `MarqovPlatformError` — any other non-2xx platform error.

---

### `client.managed_runtimes()`

```python
runtimes: list[dict] = client.managed_runtimes(team_id: str)
```

List the managed native runtimes enabled for a team
(`GET /api/jobs/managed-runtimes`). Each entry is
`{"backend", "programming_model", "min_cap_cents", "max_cap_cents"}`. An empty
list means managed native execution is not enabled for the team. Discovery is
advisory: it reserves and authorises nothing.

**Raises:** `ValueError` if `team_id` is not a UUID (nothing is sent);
`MarqovPlatformError` with `code="invalid_response"` on an unexpected
response shape, or the server's error (e.g. `404` for a team the key cannot see).

---

### `client.submit_native()`

```python
job: Job = client.submit_native(
    *,
    team_id: str,
    entrypoint: str,
    cap_cents: int,
    script_id: str | None = None,       # exactly one of script_id / source
    source: str | None = None,
    args: Sequence = (),
    kwargs: Mapping[str, Any] | None = None,
    programming_model: str = "native_workflow",   # or "single_task"
    backend: str = "marqov-sim",
    source_sha256: str | None = None,
    idempotency_key: str | None = None,
)
```

Run a Python `@task`/`@workflow` program on the managed native runtime using
the versioned submission `marqov.public-submission/v1`.

- `team_id` and `cap_cents` are required and have no defaults.
- It validates locally, then requires a discovered runtime matching `backend`
  **and** `programming_model` (never substituting another).
- It submits with your `idempotency_key` (or a fresh UUID for each call) and
  checks the admission receipt: its structure and input hash always, and its
  source hash when the client holds the source (inline `source`, or `script_id`
  with `source_sha256`). For a `script_id` without `source_sha256` the client
  cannot independently verify the source content.
- Any failure after the request may have been sent carries the effective key as
  `exc.idempotency_key` (see [When the outcome is unknown](native-workflows.md#when-the-outcome-is-unknown)).
- The returned `Job` is polled and read as usual. A workflow's
  `result().raw` is the platform's `marqov.managed-result/v1` projection.

Full guide, the complete example and refusal codes:
[Native workflows on the hosted platform](native-workflows.md).

**Raises:**
- `ValueError` / `TypeError` — invalid request (nothing sent), or `cap_cents`
  outside the runtime's range.
- `MarqovPlatformError` — `runtime_not_enabled`, `invalid_receipt`,
  `receipt_mismatch`, or the server's refusal code (subclasses such as
  `InvalidProgram`, `PermissionTierError` and `RateLimited` apply as usual).

---

### `client.job()`

```python
job: Job = client.job(job_id: str)
```

Reconnect to an existing job by UUID. Use this to resume polling a job that
was submitted in a previous process or session.

---

### `client.backends()`

```python
backends: list[Backend] = client.backends()
```

Fetch the list of available quantum backends from the platform.

Returns `Backend` instances ordered by the server's display order.

**Raises:** `MarqovPlatformError` on any non-2xx response.

---

### `client.platform_info()`

```python
info: PlatformInfo = client.platform_info()
```

Return version metadata about the SDK and the platform API.

> **Not available against the hosted API.** This method requests `/api/meta`,
> which the hosted API does not serve.

**Returns:** `PlatformInfo` with `sdk_version` (the installed `marqov` version)
and `api_version` (from the server response).

---

## `Job`

A handle for a submitted platform job. Returned by `client.submit()` or
`client.job()`.

---

### `job.id`

```python
job.id -> str
```

The UUID assigned to this job by the platform at submission time.

---

### `job.status()`

```python
status: str = job.status()
```

Fetch the current job status from the platform (single GET, returns immediately).

Returns the raw status string as sent by the server. Unknown or future status
values are returned as-is rather than raising. Use `JobStatus` constants for
comparison:

```python
from marqov.platform import JobStatus

if job.status() == JobStatus.COMPLETED:
    ...
```

---

### `job.result()`

```python
result: PlatformResult = job.result(
    timeout: float = 300.0,
    poll_interval: float = 2.0,
)
```

Block until the job reaches a terminal state, then return the result.

Uses the server's long-poll `wait` parameter as the primary waiting mechanism;
client-side exponential back-off with jitter is applied between polls, capped
at 10 seconds.

**`timeout`:** overall wall-clock deadline in seconds. If the job has not
completed in time, `TimeoutError` is raised. **The job continues running
server-side on timeout** — it is not cancelled automatically.

**`poll_interval`:** starting interval for client-side back-off (seconds).
Doubles each round, capped at 10 seconds.

**Raises:**
- `JobFailed` — job reached `failed` or `dispatch_failed` (or `cancelled`) terminal state.
- `TimeoutError` — deadline elapsed before the job completed.
- `MarqovPlatformError` — transport error during polling.

---

### `job.estimated_cost_usd`

```python
job.estimated_cost_usd -> float | None
```

The platform's cost estimate in USD, read from the most recent status response
cached by `status()` or `result()`. Returns `None` if no status has been
fetched yet, or if the field was absent in the response.

`0.0` is a valid value for free backends — it is distinct from `None`
(not yet fetched).

---

### `job.cancel()`

```python
job.cancel() -> None
```

Send a best-effort cancellation request. Returns immediately without
confirming the outcome.

> **Does not work with an API key.** The hosted cancellation endpoint accepts
> only signed-in browser sessions; an API key receives
> `403 api_key_not_supported` (raised as `PermissionTierError`). Cancel from the
> job page in the Marqov app, for funded jobs that have not yet reached a
> provider.

**Raises:** `MarqovPlatformError` if the request itself fails. For fire-and-
forget behaviour, catch `MarqovPlatformError`.

---

## `JobStatus`

```python
from marqov.platform import JobStatus
```

A `str`-enum of known job lifecycle states for use in comparisons.

| Constant | Value |
|----------|-------|
| `JobStatus.PENDING` | `"pending"` |
| `JobStatus.RUNNING` | `"running"` |
| `JobStatus.COMPLETED` | `"completed"` |
| `JobStatus.FAILED` | `"failed"` |
| `JobStatus.CANCELLING` | `"cancelling"` |
| `JobStatus.CANCELLED` | `"cancelled"` |
| `JobStatus.DISPATCH_FAILED` | `"dispatch_failed"` |

`job.status()` returns a raw `str` — the server may return values not listed
here. Use `is_terminal()` for safe terminal-state checks on raw strings:

```python
from marqov.platform import is_terminal

if is_terminal(job.status()):
    print("Job is done")
```

`is_terminal()` returns `True` for `completed`, `failed`, `cancelled`, and
`dispatch_failed`. Unknown status strings return `False` rather than raising.

---

## `Backend`

```python
from marqov.platform import Backend
```

A dataclass representing a quantum backend available on the platform.

| Field | Type | Description |
|-------|------|-------------|
| `slug` | `str` | Machine-readable identifier (e.g. `"dwave-sim"`) |
| `name` | `str` | Human-readable display name |
| `provider` | `str` | Provider name (e.g. `"marqov"`, `"ibm"`, `"aws"`) |
| `device_type` | `str` | Category: `"simulator"` or `"qpu"` |
| `status` | `str` | Operational status (e.g. `"online"`, `"maintenance"`) |
| `is_available` | `bool` | Whether jobs can currently be submitted |
| `pricing` | `dict` | Pricing metadata (shape is server-defined) |
| `supported_program_types` | `list` | Accepted program types (e.g. `["qasm3"]`) |
| `extra` | `dict` | Additional server fields not mapped above |

---

## `PlatformResult`

```python
from marqov.platform import PlatformResult
```

Wraps the raw result field from the job status response.

| Member | Type | Description |
|--------|------|-------------|
| `raw` | `dict` | The unmodified result dict from the server |
| `counts` | `dict[str, int] \| None` | Measurement outcome counts, or `None` if absent in the response |
| `probabilities` | `dict[str, float]` | Probabilities derived from `counts`; empty dict if counts are absent or sum to zero |

Both `counts` and `probabilities` are `@property` accessors on
`PlatformResult` (no parentheses) — `counts` reads from `raw`, and
`probabilities` is derived from `counts` each time it's read. This differs
from `marqov.executors.base.ExecutionResult`, where `counts` is a plain
field (always present, never `None`) and only `probabilities` is a
`@property`; `PlatformResult.counts` is a property specifically so it can
be `None` when the server response omits it.

---

## Error classes

All inherit from `MarqovPlatformError`. Import any from `marqov.platform`:

| Class | Trigger |
|-------|---------|
| `MarqovPlatformError` | Base for all platform errors; `.message`, `.code`, `.status` |
| `AuthenticationError` | Missing/invalid/revoked API key |
| `PermissionTierError` | Plan tier does not permit the resource |
| `PaidBackendNotSupportedYet` | Paid backend requested in v1.0 (free-path only) |
| `BackendUnavailable` | Backend offline or unrecognised |
| `InvalidProgram` | Server rejected the program (bad gates, syntax, etc.) |
| `JobFailed` | Job reached a failure terminal state |
| `RateLimited` | HTTP 429; `.retry_after` (int seconds or `None`) |
| `TransportError` | Low-level network or unstructured HTTP failure |

`PaidBackendNotSupportedYet` is retained (never deleted) after paid backends
are enabled in a future update, so that existing `except PaidBackendNotSupportedYet`
blocks continue to import and function correctly.

See the [error-handling guide](error-handling.md) for retry advice and
`RateLimited.retry_after` usage.
