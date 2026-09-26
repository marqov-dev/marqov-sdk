# Error Handling — Marqov Platform Client

All exceptions raised by `marqov.platform` inherit from `MarqovPlatformError`.
They carry three attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `message` | `str` | Human-readable description |
| `code` | `str \| None` | Machine-readable server error code (e.g. `"auth/token-expired"`) |
| `status` | `int \| None` | HTTP status code from the response |

`str(error)` returns `"message [code]"` if a code is present, or just the
message.

---

## Error taxonomy

| Exception | When it is raised | Retry? |
|-----------|-------------------|--------|
| `AuthenticationError` | API key is missing, invalid, or has been revoked — raised at construction if no key is found, or on HTTP 401 | No — fix the key first |
| `PermissionTierError` | Your plan does not permit access to the requested resource (HTTP 403) | No — upgrade plan or contact support |
| `PaidBackendNotSupportedYet` | The backend requires a pre-run cost analysis (paid backends); v1.0 only routes free backends | No — choose a free backend (e.g. `dwave-sim`) |
| `BackendUnavailable` | The requested backend is known but currently offline or unrecognised | Maybe — retry with back-off or choose another backend |
| `InvalidProgram` | The submitted program was rejected by the server (bad QASM, unsupported gates, too many qubits, malformed body) | No — fix the program |
| `JobFailed` | The job reached the `failed` or `dispatch_failed` terminal state | Depends — inspect `message`, but see caveat below |
| `RateLimited` | HTTP 429 — too many requests | Yes — back off by `retry_after` seconds |
| `TransportError` | Low-level network failure: connection refused, TLS error, or unrecognised non-2xx with no structured error body | Yes for reads; conditional for writes (see below) |
| `MarqovPlatformError` | Any other structured server error not mapped to a subclass above | Depends on `code` |

---

## `JobFailed.message` is best-effort, not a guaranteed server reason

The status endpoint's response does **not** include an `error_message`
field — the server only ever returns `id`, `status`, `backend`,
`created_at`, `updated_at`, `estimated_cost_usd`, and `result`. When a job
reaches `failed` or `dispatch_failed`, the client builds `.message` as
follows:

- If `result` is present and contains an `"error"` key, that value becomes
  `.message`.
- Otherwise `.message` is a generic fallback string (e.g. *"Job \<id\> failed
  (status='failed'). The server error_message is not returned by the status
  endpoint; check the platform dashboard for details."*) — this behaviour is
  pinned by `tests/test_platform_job.py::TestJobResultFailed::test_job_failed_generic_message_when_no_result_error`.

Do not assume `.message` always carries the server's actual failure reason —
check the platform dashboard for full detail when it doesn't.

---

## Conditional retry for writes

The client's retry policy splits network failures on a write request (e.g.
`submit()`, `cancel()`) by the phase in which they happened:

- **Connect phase** (connection refused, DNS failure, connect timeout): the
  request bytes never left the client, so it provably never reached the
  server and is retried freely, up to 3 attempts with a doubling backoff. The
  same `Idempotency-Key` header is reused across every attempt. Pinned by
  `tests/test_platform_transport.py::TestConnectionErrorPhaseSplit::test_connect_phase_connection_error_on_write_is_retried`.

- **Ambiguous** (a read timeout, or the connection dropping once the request
  may already have reached the server, which `requests` reports as a
  `ConnectionError` too): the server may have received and processed the
  request. The client raises
  `TransportError` rather than retrying, so a write cannot be double-submitted.
  The error's `idempotency_key` attribute (also shown in its message) holds
  the `Idempotency-Key` that was sent, so you can reconcile against the
  platform dashboard, and you can recover by reconnecting
  with `client.job(job_id)` if you know the ID. Pinned by
  `tests/test_platform_transport.py::TestConnectionErrorPhaseSplit::test_ambiguous_connection_error_on_write_is_not_retried`.

A retry that the client does perform is only safe against duplication if the
server de-duplicates on `Idempotency-Key`. That is a requirement the client
places on the server, not something the client can guarantee: what the client
guarantees is that it never changes the key between attempts of one call, and
that it never replays a write after an ambiguous failure.

Read requests (status polls, `backends()`) are always safe to retry on any
transport failure, in either phase. Pinned by
`tests/test_platform_transport.py::TestConnectionErrorPhaseSplit::test_ambiguous_connection_error_on_read_is_retried`
and
`tests/test_platform_transport.py::TestConnectionErrorPhaseSplit::test_connect_phase_connection_error_on_read_is_retried`.

### Retried HTTP statuses

Once the server has answered, the client retries only the transient statuses
**429, 502, 503 and 504**, and only for idempotent requests: GETs, and writes
the client marks as idempotent (`submit()`, `cancel()`). Every other non-2xx
raises on the first response. Pinned by
`tests/test_platform_transport.py::TestRetryableStatusCodes::test_503_then_200_on_get_returns_body`,
`tests/test_platform_transport.py::TestRetryableStatusCodes::test_502_and_504_retried_for_idempotent_write`
and
`tests/test_platform_transport.py::TestRetryableStatusCodes::test_500_is_not_retried`.

When the attempts run out, the exception raised is the one mapped from the
last response, not a generic `TransportError`, so a 503 still surfaces as
`BackendUnavailable` and a 429 as `RateLimited` with its `retry_after`. Pinned
by
`tests/test_platform_transport.py::TestRetryableStatusCodes::test_503_every_attempt_raises_mapped_exception`
and
`tests/test_platform_transport.py::TestRetryableStatusCodes::test_429_retry_after_respected_and_surfaced_when_exhausted`.
If a later attempt failed before any response arrived, that transport failure
is what you get instead, not the earlier status. Pinned by
`tests/test_platform_transport.py::TestRetryableStatusCodes::test_retryable_status_then_transport_failures_raises_transport_error`.

A 429's `Retry-After` header is honoured as the wait before the next attempt
only when it parses and asks for no longer than the backoff the client would
otherwise spend on its remaining attempts. A longer wait is left to you: the
client falls back to its own backoff and hands you the parsed value on
`RateLimited.retry_after`. Pinned by
`tests/test_platform_transport.py::TestRetryableStatusCodes::test_429_retry_after_beyond_budget_falls_back_to_default_backoff`.

---

## Timeout behaviour of `result()`

`job.result(timeout=300.0)` raises `TimeoutError` if the job has not reached
a terminal state within the specified wall-clock budget.

**The job continues running on the server.** The client does not cancel the
job when a timeout occurs. Reconnect to it later:

```python
try:
    result = job.result(timeout=60.0)
except TimeoutError:
    job_id = job.id
    # ... later, in the same or a different process:
    result = client.job(job_id).result(timeout=300.0)
```

---

## Cancellation behaviour

`job.cancel()` sends a cancellation request to the server. It does **not**
swallow failures: any non-2xx response raises `MarqovPlatformError`, same as
any other API call.

> **The cancel endpoint is not yet confirmed on the platform.** `cancel()` is
> implemented against a provisional path (see `job.cancel()` in
> [`api-reference.md`](api-reference.md)) — until the platform ships a
> confirmed cancellation route, calling it against the real service will
> typically raise `MarqovPlatformError` rather than cancel the job.

This matters for cleanup code. A pattern like:

```python
try:
    result = job.result(timeout=120.0)
finally:
    job.cancel()
```

lets a raised `MarqovPlatformError` from `cancel()` escape the `finally`
block and mask whatever exception (if any) was already propagating. If you
want fire-and-forget cleanup, catch it explicitly:

```python
finally:
    try:
        job.cancel()
    except MarqovPlatformError:
        pass  # best-effort — cancellation isn't guaranteed to reach the server
```

If cancellation does succeed server-side, the job ends in the `cancelled`
state. When `result()` polls and encounters `cancelled`, it raises
`JobFailed` (the job reached a terminal state that produced no result).

---

## Unknown server error codes

If the server returns an error code not listed above (e.g. from a newer
server version), the client raises the base `MarqovPlatformError` with the
code and message intact — it never crashes on an unknown code or coerces it
to `TransportError`. Your `except MarqovPlatformError` block will catch it.

---

## Import paths

```python
# All error classes are importable directly from marqov.platform:
from marqov.platform import (
    MarqovPlatformError,
    AuthenticationError,
    PermissionTierError,
    PaidBackendNotSupportedYet,
    BackendUnavailable,
    InvalidProgram,
    JobFailed,
    RateLimited,
    TransportError,
)
```

---

## Full example

```python
import time
from marqov.platform import MarqovClient
from marqov.platform import (
    AuthenticationError,
    BackendUnavailable,
    InvalidProgram,
    JobFailed,
    PaidBackendNotSupportedYet,
    RateLimited,
    TransportError,
)

client = MarqovClient()  # reads MARQOV_PLATFORM_KEY

script = "from marqov import task; ..."

try:
    job = client.submit(script, backend="dwave-sim", framework="marqov")
except AuthenticationError as e:
    print("Auth failed:", e)
except PaidBackendNotSupportedYet:
    print("Paid backends require a future update — use dwave-sim for now")
except BackendUnavailable as e:
    print("Backend offline:", e)
except InvalidProgram as e:
    print("Program rejected:", e.message)
except RateLimited as e:
    wait = e.retry_after or 10
    print(f"Rate limited. Waiting {wait}s before retry")
    time.sleep(wait)
    # retry...
except TransportError as e:
    print("Network error:", e)
else:
    try:
        result = job.result(timeout=300.0)
        print(result.counts)
    except JobFailed as e:
        print("Job failed:", e.message, "code:", e.code)
    except TimeoutError:
        print(f"Timed out. Reconnect with client.job({job.id!r})")
```
