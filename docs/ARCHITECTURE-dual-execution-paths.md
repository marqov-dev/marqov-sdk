# Why the SDK has two execution paths

> Investigated 2026-06-17 (cross-repo: marqov-sdk + marqov-platform). This note
> exists because the duplication looks like an accident from inside the SDK alone
> — it isn't. One path's only real caller lives in another repository.

The SDK ships **two parallel ways to run a circuit on a backend**. They are
**intentional and serve different consumers**, and per the platform's
`docs-internal/MULTI_CLOUD_EXECUTOR_DESIGN.md` they should **not** be merged.

| Path | Module | Interface | Purpose | Real caller |
|------|--------|-----------|---------|-------------|
| **MarqovDevice** | `marqov/device.py` (`get_device`, `MarqovDevice.run`) | **Synchronous** `run(circuit, shots)` | A thin, Pythonic wrapper that mirrors vendor SDKs, meant to be called from **user-written scripts** | The **marqov-platform worker** injects it into user/playground/inline scripts (`marqov.get_device(params).run(...)`) and runs them in a subprocess. This is why `get_device` has **zero callers inside the SDK repo** — its consumer lives in `marqov-platform`. |
| **ExecutorFactory** | `marqov/executors/` (`BaseExecutor`, `ExecutorFactory.create_executor`) | **Async** `await execute(circuit, shots)` | A provider-string-dispatched executor registry for **platform infrastructure** | Live QPU status polling, QB simulation jobs, and Temporal workflow activities (all async). The SDK's own README quick-start also uses `LocalExecutor().execute(...)`. |

## Key facts (with provenance)

- **Both shipped together** in the SDK's initial commit (`4628d5b`, 2026-06-03).
  In the originating platform repo, `ExecutorFactory` was designed first
  (`26958c4`, 2026-01-06, "Multi-Cloud Foundation"); `MarqovDevice` was added
  later (`c5bcaec`, 2026-03-08, "device router for backend abstraction") as the
  synchronous user-script path.
- The SDK was extracted from `Github/marqov-platform`. A **vendored copy** still
  lives at `marqov-platform/sdk/marqov/`; changes must be made in **both** repos
  until the PyPI cutover (blocked on the `quantumflow` VCS dependency).
- `MarqovDevice` was originally **Braket-native** (implicit `else` → Braket) and
  was deliberately made vendor-neutral in commit `8259451`. That conversion left
  the residual debt addressed by the refactor plan below.

## Consequence for "proper abstraction"

"Unify the two paths" is **off the table** — they are a deliberate sync (user
scripts) vs. async (infrastructure) split. `ExecutorFactory` is already cleanly
abstracted (explicit `provider` dispatch, no privileged fall-through). **All the
remaining abstraction debt is inside `device.py`**, where three methods
(`_get_provider_device`, `run`, `_to_backend_format`) each independently
re-sniff `is_braket()`/`is_ibm()` and can disagree — including a latent crash for
`{"backend": "local", "ibm_token": ...}` (Qiskit circuit fed to a Braket
`LocalSimulator`).

The fix is specified in
[`docs/superpowers/plans/2026-06-12-provider-registry-refactor.md`](superpowers/plans/2026-06-12-provider-registry-refactor.md):
resolve the provider **once** via a new `detect_provider(backend, params)` in
`backends.py` and have all three dispatch sites branch on the resolved string.
That plan rejects a full adapter-class registry as YAGNI at four providers.
