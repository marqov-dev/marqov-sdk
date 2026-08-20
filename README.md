# Marqov SDK

Orchestration engine for hybrid quantum-classical workflows.

Run a Bell state immediately — no credentials, no infrastructure:

```python
import asyncio
from marqov.circuits import Circuit
from marqov.executors import LocalExecutor

async def main():
    result = await LocalExecutor().execute(
        Circuit().h(0).cnot(0, 1), shots=1000
    )
    print(result.counts)  # {'00': ~500, '11': ~500}

asyncio.run(main())
```

Scale to parallel workflows across any backend:

```python
from marqov import task, workflow, bell_state
from marqov.executors import LocalExecutor

@task
async def measure(shots):
    result = await LocalExecutor().execute(bell_state(), shots=shots)
    return result.counts

@workflow
def multi_shot_study(shot_counts):
    return [measure(n) for n in shot_counts]  # all run in parallel

dispatch = multi_shot_study([100, 500, 1000, 5000])
# dispatch.run(client) — needs a Temporal worker
```

Independent tasks execute in parallel automatically. Marqov handles scheduling, retries, and result collection across any supported backend. Run your own Temporal worker (see `marqov/workflows/`) — or skip the infrastructure entirely with the hosted [Marqov Platform](https://marqov.ai).

---

## Installation

```bash
pip install marqov
```

With backend-specific extras:

```bash
# IBM Quantum
pip install "marqov[ibm]"

# All extras
pip install "marqov[all]"
```

For local development:

```bash
git clone https://github.com/marqov-dev/marqov-sdk
cd marqov-sdk
pip install -e ".[all,dev]"
pytest tests/ -v
```

---

## Cloud Executors

Swap in a cloud backend when you're ready to run on hardware — on **your own
provider accounts**, no Marqov account needed:

```python
import asyncio
from marqov.circuits import Circuit
from marqov.executors import ExecutorFactory

async def main():
    circuit = Circuit().h(0).cnot(0, 1)

    executor = ExecutorFactory.create_executor("sv1", {
        "provider": "AWS Braket",
        "device_arn": "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
        "s3_bucket": "my-bucket",
        "s3_prefix": "jobs",
    })
    result = await executor.execute(circuit, shots=1000)
    print(result.counts)

asyncio.run(main())
```

Or run directly on IonQ hardware via the native REST API (no AWS account needed):

```python
executor = ExecutorFactory.create_executor("qpu.aria-1", {
    "provider": "IonQ Direct",
    "api_key": "your-ionq-api-key",  # or set IONQ_API_KEY
})
result = await executor.execute(circuit, shots=1000)
```

Or run on Rigetti QPUs (or the local QVM, no cloud account needed) via Rigetti QCS:

```python
executor = ExecutorFactory.create_executor("2q-qvm", {
    "provider": "Rigetti QCS",
})
result = await executor.execute(circuit, shots=1000)
```

---

## Supported Backends

| Backend | Status |
|---|---|
| Local (QuantumFlow simulator) | Available |
| AWS Braket | Available |
| IBM Quantum | Available |
| Azure Quantum | Available |
| IonQ Direct | Available |
| Rigetti QCS | Available |
| Quantinuum | Available |
| Quantum Brilliance | Available — requires `qristal` installed separately (not on PyPI, no `marqov[...]` extra); build from source or use the Docker image: https://qristal.readthedocs.io/ |
| CUDA-Q | Available — not in `[all]` (GPU-heavy); install separately with `pip install "marqov[cudaq]"` |
| Qilimanjaro (qilisdk local simulators) | Available — not in `[all]`; install separately with `pip install "marqov[qilisdk]"` |

---

## Circuit Interop

`Circuit` is a backend-agnostic abstraction that converts to any supported framework's native format:

```python
from marqov.circuits import Circuit

circuit = Circuit().h(0).cnot(0, 1)

circuit.to_qiskit()   # qiskit.QuantumCircuit
circuit.to_braket()   # braket.circuits.Circuit
circuit.to_cirq()     # cirq.Circuit
circuit.to_pyquil()   # pyquil.Program  (requires pip install marqov[pyquil])
```

Import from other formats:

```python
circuit = Circuit.from_qiskit(qiskit_circuit)
circuit = Circuit.from_cirq(cirq_circuit)
circuit = Circuit.from_pennylane(tape)
circuit = Circuit.from_pyquil(pyquil_program)  # requires pip install marqov[pyquil]
```

---

## Using the hosted platform (`marqov.platform`)

The SDK runs fully standalone — everything above needs no Marqov account.

If you want managed backend credentials, persistent job history, execution
traces, and spend controls without running your own infrastructure, the Marqov
Platform is an opt-in value-add.

`marqov.platform` is an **optional import** — loading `marqov` never loads the
platform client. It is only activated when you import it explicitly.

> **Live-server caveat:** The examples below are not yet verified against a live
> server — live verification is pending our staging environment.

> **v1.0 scope:** v1.0 supports **free backends** (e.g. `dwave-sim`).
> Paid backends and `Circuit` submission are coming in a future update.

### Quickstart

**1. Set your API key** (get one from the Marqov Platform dashboard):

```bash
export MARQOV_PLATFORM_KEY="marqey_live_your_key_here"
```

**2. Submit a script and poll for results:**

```python
from marqov.platform import MarqovClient

# Key is read from MARQOV_PLATFORM_KEY automatically
client = MarqovClient()

script = """
import asyncio
from marqov import task

@task
async def bell(shots):
    from marqov.circuits import Circuit
    from marqov.executors import LocalExecutor
    result = await LocalExecutor().execute(
        Circuit().h(0).cnot(0, 1), shots=shots
    )
    return result.counts

# An async @task called outside a @workflow isn't awaited automatically
# (see marqov/workflows/decorators.py for details) — drive it with
# asyncio.run() rather than calling bell(1000) bare.
asyncio.run(bell(1000))
"""

job = client.submit(script, backend="dwave-sim", framework="marqov", shots=1000)
print("Job ID:", job.id)

# Block until complete (up to 5 minutes by default)
result = job.result(timeout=300.0)
print(result.counts)       # e.g. {'00': 507, '11': 493}
print(result.probabilities) # e.g. {'00': 0.507, '11': 0.493}
```

**3. Check available backends:**

```python
for b in client.backends():
    print(b.slug, b.name, "available:", b.is_available)
```

**4. Reconnect to a job from a previous session:**

```python
job = client.job("550e8400-e29b-41d4-a716-446655440000")
result = job.result(timeout=60.0)
```

### Error handling

All platform errors inherit from `MarqovPlatformError`:

```python
from marqov.platform import AuthenticationError, JobFailed, RateLimited

try:
    job = client.submit(script, backend="dwave-sim", framework="marqov")
    result = job.result(timeout=120.0)
except AuthenticationError:
    print("Check your MARQOV_PLATFORM_KEY")
except JobFailed as e:
    print("Job failed:", e.message)
except RateLimited as e:
    print(f"Rate limited — retry after {e.retry_after}s")
except TimeoutError:
    print("Timed out — job is still running server-side")
```

For the full error taxonomy and retry guidance see
[`docs/platform-client/error-handling.md`](docs/platform-client/error-handling.md).

### Platform documentation

- [Getting started](docs/platform-client/getting-started.md)
- [Error handling](docs/platform-client/error-handling.md)
- [API reference](docs/platform-client/api-reference.md)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the executor interface spec, canonical gate set, factory registration steps, and local QVM setup for Rigetti development.

The bounty issues that were open through [unitaryHACK 2026](https://unitaryhack.dev) — have been claimed, but follow the [issues page](https://github.com/marqov-dev/marqov-sdk/issues) as we will be looking at ongoing and rolling issue bounties to support and encourage community participation.

## Authors

This project was created by **David Ryan** ([@ddri](https://github.com/ddri)), with contributions from the [community](https://github.com/marqov-dev/marqov-sdk/graphs/contributors).

## License

[Apache 2.0](LICENSE)
