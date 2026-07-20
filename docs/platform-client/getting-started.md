# Getting Started with the Marqov Platform Client

`marqov.platform` is an **optional** subpackage that lets you submit jobs to
the hosted Marqov Platform from your existing Python code.  It is independent
of the rest of the SDK — you can use it whether or not you use `@task`,
`@workflow`, or any executor.

> **Live-server caveat:** The examples below are not yet verified against a live
> server — live verification is pending the staging environment. They are
> unit-tested against a mocked transport and are correct against the specified
> API contract.

> **v1.0 scope:** v1.0 supports **free backends** (e.g. `dwave-sim`).
> Paid QPU backends and `Circuit` submission are coming in a future update.

---

## 1. Install

`marqov.platform` ships inside the `marqov` package — no extra install step:

```bash
pip install marqov
```

`requests` (a core dependency as of `marqov` 0.3.0) is the only new
transitive dependency.

---

## 2. Obtain an API key

Sign in to the Marqov Platform dashboard and generate an API key.  Keys
follow the format `marqey_live_…` (production) or `marqey_test_…` (sandbox).

Set the key as an environment variable so it stays out of source code:

```bash
export MARQOV_PLATFORM_KEY="marqey_live_your_key_here"
```

Alternatively, pass it directly when constructing the client (see below).  The
client **never writes the key to disk**.

---

## 3. Create a client

```python
from marqov.platform import MarqovClient

# Key is read from MARQOV_PLATFORM_KEY automatically
client = MarqovClient()

# Or pass it explicitly
client = MarqovClient(api_key="marqey_live_your_key_here")
```

`AuthenticationError` is raised immediately if no key can be resolved.

**Constructor options:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `api_key` | `None` (reads `MARQOV_PLATFORM_KEY`) | Your platform API key |
| `base_url` | production endpoint (reads `MARQOV_PLATFORM_URL`) | Override the API base URL |
| `timeout` | `30.0` | Per-request HTTP timeout in seconds |

---

## 4. Check available backends

```python
backends = client.backends()
for b in backends:
    print(b.slug, b.name, "available:", b.is_available)
```

Free backends (e.g. `dwave-sim`) have zero pricing.  Attempting to submit to a
paid backend in v1.0 raises `PaidBackendNotSupportedYet` — paid backend support
is coming in a future update.

---

## 5. Submit a raw script

Pass a Python string as the program together with a `framework` identifier and
a backend slug:

```python
script = """
from marqov import task

@task
async def bell(shots):
    from marqov.circuits import Circuit
    from marqov.executors import LocalExecutor
    result = await LocalExecutor().execute(
        Circuit().h(0).cnot(0, 1), shots=shots
    )
    return result.counts

bell(1000)
"""

job = client.submit(script, backend="dwave-sim", framework="marqov", shots=1000)
print("Submitted job:", job.id)
```

`framework` is **required** when the program is a string — it tells the
platform how to interpret your code.  Omitting it raises `ValueError`.

---

## 6. Poll for results

```python
# Blocking: waits up to 5 minutes (default timeout)
result = job.result(timeout=300.0)

print(result.counts)       # e.g. {"00": 507, "11": 493}
print(result.probabilities) # e.g. {"00": 0.507, "11": 0.493}
```

The polling loop uses the server's long-poll mechanism: each request blocks
server-side for up to ~22 seconds before returning the current status.
Between polls, client-side exponential back-off with jitter is applied, capped
at 10 seconds.

**If the timeout elapses, `TimeoutError` is raised — the job continues running
on the server.** You can reconnect to it later (see below).

---

## 7. Non-blocking status checks

```python
status = job.status()   # "pending" | "running" | "completed" | "failed" | ...
print(status)
```

`status()` issues a single GET request and returns immediately with the raw
status string.

---

## 8. Reconnect to a job from a previous session

```python
job = client.job("550e8400-e29b-41d4-a716-446655440000")
result = job.result(timeout=60.0)
```

This lets you reconnect to a job that was submitted in a different process or
script by passing its UUID.

---

## 9. Estimated cost

For free backends this is `0.0`:

```python
result = job.result()
print(job.estimated_cost_usd)   # 0.0 for free backends; None before first status fetch
```

`estimated_cost_usd` is read from the most recent status response.  It
returns `None` if no status has been fetched yet.  Note that `0.0` (known
zero cost) is distinct from `None` (not yet fetched).

---

## 10. Circuit submission (coming in a future update)

Submitting a `marqov.Circuit` object directly is supported in the client code
but is **not yet active on the server** — the platform-side circuit-submission
variant is under development.  Do not rely on this path against a live server
until it is announced.

When it ships, the call will look like this (no `framework` argument — the
circuit self-describes its format as OpenQASM 3):

```python
from marqov.circuits import Circuit
from marqov.platform import MarqovClient

# Coming in a future update — not yet live
client = MarqovClient()
circuit = Circuit().h(0).cnot(0, 1)
job = client.submit(circuit, backend="dwave-sim", shots=1000)
result = job.result()
print(result.counts)
```

Passing `framework=` with a `Circuit` raises `ValueError` immediately (the
circuit already identifies its format).

---

## 11. Error handling

All platform errors inherit from `MarqovPlatformError`.  See the
[error-handling guide](error-handling.md) for the full taxonomy and retry
advice.

```python
from marqov.platform import MarqovClient
from marqov.platform import AuthenticationError, JobFailed, RateLimited

client = MarqovClient()

try:
    job = client.submit(script, backend="dwave-sim", framework="marqov")
    result = job.result(timeout=120.0)
except AuthenticationError:
    print("Check your MARQOV_PLATFORM_KEY")
except JobFailed as e:
    print("Job failed:", e.message)
except RateLimited as e:
    retry_in = e.retry_after  # seconds, or None
    print(f"Rate limited. Retry after {retry_in}s")
except TimeoutError:
    print("Timed out — job is still running server-side")
```
