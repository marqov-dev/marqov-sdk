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

**`program`:**

- `str` — treated as inline executable code. `framework` is **required**;
  omitting it raises `ValueError`.
- `Circuit` — serialised as OpenQASM 3. `framework` must **not** be supplied;
  passing it raises `ValueError`. Circuit submission requires a forthcoming
  platform-side change and is not yet active on the server.

**`backend`:** backend slug (e.g. `"dwave-sim"`). In v1.0, only free backends
are supported. Paid backends raise `PaidBackendNotSupportedYet`.

**`shots`:** number of measurement shots. Default: `1000`.

**`framework`:** framework identifier for string programs (e.g. `"marqov"`).

**Raises:**
- `ValueError` — `program` is `str` but `framework` was omitted, or `program`
  is a `Circuit` and `framework` was supplied.
- `TypeError` — `program` is neither `str` nor `Circuit`.
- `PaidBackendNotSupportedYet` — backend requires pre-run cost analysis (v1.0).
- `AuthenticationError` — HTTP 401.
- `MarqovPlatformError` — any other non-2xx platform error.

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

> **Note:** The platform endpoint backing this method is not yet confirmed.
> It is implemented against a provisional path. When the platform ships a
> confirmed info/health endpoint, this method will be updated.

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

> **Note:** The platform cancel endpoint is not yet confirmed. This method
> is implemented against a provisional path and will be updated when the
> platform ships a confirmed cancellation endpoint.

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
