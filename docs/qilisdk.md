# Reproducible sampling with QiliSim

The Marqov QiliSDK adapter accepts a per-call `seed` on both digital and analog
execution. Requires Marqov 0.7.1 or later.

```python
from marqov import Circuit
from marqov.executors import QiliSDKExecutor

executor = QiliSDKExecutor()
circuit = Circuit().h(0).cnot(0, 1)
first = await executor.execute(circuit, shots=1000, seed=42)
second = await executor.execute(circuit, shots=1000, seed=42)
assert first.counts == second.counts
assert first.metadata["seed"] == 42

# Given a QiliSDK Schedule:
# result = await executor.execute_analog(schedule, shots=1000, seed=42)
```

A seeded call constructs a fresh QiliSim with
`ExecutionConfig(seed=seed, num_threads=1)`. The fresh simulator prevents earlier
calls from advancing its random stream. Single-threaded execution makes the
thread setting explicit and can be slower than unseeded execution on larger
problems. Result metadata records both `seed` and `num_threads`.

Repeatability requires identical inputs, shots, solver settings and the same
software/platform stack. Identical samples across QiliSDK releases, CPU
architectures or numerical-library versions are not promised. Different seeds
can legitimately produce the same counts.

- Seeds are Python integers in `[0, 2**31 - 1]`, matching the tested native
  configuration range; zero is supported. Booleans, strings and floats are rejected.
- Omitting the seed, or passing `None`, uses the existing unseeded backend and
  does not add seed metadata. A seeded call does not replace that backend.
- `simulator="qutip"` rejects a seed with `ValueError`: its adapter backend does
  not expose the same execution configuration. Its ordinary unseeded path remains
  supported.
- Other extra options, including `num_threads`, still raise `TypeError`. They
  are never silently ignored.

Validation uses real QiliSDK 0.3.0 on CPython 3.12/macOS arm64: digital and analog
runs at seeds 0, 42 and 2147483647 replay across intervening calls and match a
separately constructed QiliSim with the same configuration. These tests use
multi-outcome sampling, not only deterministic basis states. Existing adapter
checks also exercise unseeded QiliSim and QuTiP execution.

Qilimanjaro documents the constructor configuration in its
[QiliSim backend guide](https://qilimanjaro-tech.github.io/qilisdk/en/0.2.0/modules/backends/backends_qilisim.html).
Marqov implements the per-call reset and option validation described here.

## Explicit direct SpeQtrum execution

The separate `SpeQtrumExecutor` requires pinned `qilisdk==0.3.0`, an explicit
`device_code`, `username`, and API key. Factory configuration uses
`provider="Qilimanjaro", access_path="speqtrum"`; omitting `access_path` retains
local simulator routing. Unknown paths and incomplete remote configuration fail.
No credentials are read from the environment or shared keyring.

The client authenticates before sending, disables redirects/retries, and never
replays `/execute` after a 401 or uncertain response. Digital gate circuits and
QiliSDK analog schedules use sampling readout. Results retain canonical QiliSDK
bit order and require exact shot accounting. Results are decoded only through an inert sampling-only schema.

`executor.last_job` is an immutable `SpeQtrumJob` record for `readback(record)`
without another submission. Retain it when a known job has an unresolved result.
An acceptance-unknown exception without a job ID must not trigger a new submit.
The pinned API exposes no cancellation or idempotency guarantee: `cancel()`
returns False. Terminal timeout/error/cancellation never produces success.

Discovery supplies a device code, but its model has no numeric ID. Job readback
supplies `device_id`; the adapter checks its continuity within polling and records
it as provenance. This does not independently prove a code-to-numeric-ID mapping.
That mapping remains a connected qualification requirement. Poll/request and
response limits are finite; DNS and operating-system blocking are not a hard
wall-clock deadline guarantee.

This is an unqualified direct SDK adapter. No provider job, managed hosted
credential custody, funding authorization or remote hardware qualification is
claimed. Connected qualification needs an explicit account/key/device and bounded
cost authorization. The existing local CPU qualification remains separate.
