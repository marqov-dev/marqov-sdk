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
