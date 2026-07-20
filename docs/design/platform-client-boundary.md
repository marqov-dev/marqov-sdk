# Platform Client Boundary — Design Rationale

This document explains why the SDK/platform boundary sits where it does and why
it must be actively preserved.

---

## The core principle: open-core, not open-wrapper

The Marqov SDK is an **honest standalone resource**. A user who installs it,
builds circuits, and runs them on their own provider accounts gets full value
from the package — no Marqov account, no API key, no network call home.

The hosted Marqov Platform is an **opt-in value-add**. It adds managed backend
credentials, spend controls, persisted job history, execution traces, and team
features on top of the SDK. It is built *on* the SDK; the SDK is not built *for*
the platform.

This is the open-core model. The SDK's value does not diminish if you never use
the platform. The platform's value compounds because it builds on a solid,
independently useful foundation — not because the SDK is artificially incomplete
without it.

---

## What stays in the SDK (always free, always standalone)

- Building and composing `Circuit` objects.
- Running circuits locally (`LocalExecutor`).
- Submitting directly to QPU providers on **your own** accounts
  (`ExecutorFactory` + AWS Braket, IBM Quantum, IonQ, Rigetti, Quantinuum, Azure).
- Framework interop: export to Qiskit, Cirq, PennyLane, pytket, pyQuil; import
  from the same.
- `@task` / `@workflow` decorators; self-hosting a Temporal worker.

None of this ever requires a Marqov account. None of it should grow an implicit
dependency on platform internals.

---

## What lives in `marqov.platform` (the optional boundary)

`marqov.platform` is a deliberately optional subpackage. Importing `marqov` has
**zero cost** from it: the platform client only loads when the caller writes
`from marqov.platform import MarqovClient`.

Inside `marqov.platform`:

- `MarqovClient` — an explicit, opt-in client over the platform REST API.
- `Job` — a handle for an async, server-side job.
- Error classes — the full server error taxonomy.
- `backends()`, `platform_info()` — platform-side metadata.

**Nothing in the core SDK (`marqov/circuits/`, `marqov/executors/`,
`marqov/workflows/`) imports from `marqov/platform/`.** The dependency only runs
one way: platform → SDK.

---

## Why an explicit client, not a device or backend

The `ExecutorFactory` already routes to many backends using a uniform `.run()`
interface. A natural instinct is to add a `"marqov-platform/sv1"` backend and
route it through `ExecutorFactory`. We deliberately chose not to do this.

Platform jobs are **fundamentally different** from local or direct-provider
jobs:

- They are **asynchronous** — the result is not available at call time.
- They are **persisted** — the job lives in a database with an ID.
- They are **billed** — cost estimation and spend controls are first-class
  concerns, not afterthoughts.

Routing through `ExecutorFactory` would hide these differences behind a uniform
interface that was designed for synchronous, direct-to-provider, no-billing
execution. The explicit `MarqovClient` + `Job` pattern (modelled on IBM Qiskit
Runtime and the OpenAI SDK) makes the async/persisted/billed nature visible and
deliberate. The "two mental models" (`.run()` local vs `client.submit()` hosted)
are not a flaw — they reflect the fact that these are genuinely different things.

---

## The two-direction rule

Every change to the SDK or the platform client should be checked against this
rule:

1. **Core SDK changes must not require a platform account.** If a new feature
   in `marqov/circuits/` or `marqov/executors/` only makes sense with a
   `MARQOV_PLATFORM_KEY`, it belongs in `marqov.platform`, not in core.

2. **`marqov.platform` must not leak into the core.** No import of
   `marqov.platform` anywhere in `marqov/circuits/`, `marqov/executors/`, or
   `marqov/workflows/`. The platform sub-package is a *consumer* of the core,
   not a dependency of it.

Both directions are checked by the existing test suite's import-isolation tests.

---

## The dependency direction in practice

```
marqov/circuits/        <── standalone IR; no network
marqov/executors/       <── BYOK, direct-to-provider
marqov/workflows/       <── @task/@workflow; self-hosted Temporal
         ↑
marqov/platform/        (imports core SDK; never imported by core)
         |
   MarqovClient → hosted platform REST API
```

`marqov.platform._transport` is the only place that holds the platform's base
URL, API key, and HTTP session. All network calls flow through it. Adding new
platform features means adding to or extending this subpackage, not touching
the core.

---

## What "eroding the boundary" looks like

Concrete anti-patterns to watch for during code review:

- Adding `from marqov.platform import ...` to any file in `marqov/circuits/`,
  `marqov/executors/`, or `marqov/workflows/`.
- Making `pip install marqov` pull in a platform credential (even as optional).
- Designing a new `Circuit` or executor feature that only works if
  `MARQOV_PLATFORM_KEY` is set.
- Routing `ExecutorFactory` through the platform REST API implicitly (without
  the user writing `from marqov.platform import MarqovClient`).
- Importing `requests` in core SDK files for platform calls (core already uses
  it for IonQ, but platform calls belong in `_transport.py`).

---

## Summary

| Core SDK | `marqov.platform` |
|----------|-------------------|
| No Marqov account needed | Explicit opt-in via `MarqovClient` |
| Bring your own provider keys | API key for the hosted platform |
| Synchronous executors | Async jobs with persistent IDs |
| No billing | Cost estimation, spend controls (future) |
| Always free | Free backends now; paid backends in a future update |

The boundary is the product's promise to its users: **the SDK is genuinely
useful standalone**. Keeping the boundary clean keeps that promise.
