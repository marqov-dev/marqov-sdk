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
# Use the Marqov platform or run your own: see marqov/workflows/
```

Independent tasks execute in parallel automatically. Marqov handles scheduling, retries, and result collection across any supported backend.

## Installation

```bash
pip install "git+https://github.com/marqov-dev/marqov-sdk.git"
```

With backend-specific extras:

```bash
# IBM Quantum
pip install "marqov[qiskit] @ git+https://github.com/marqov-dev/marqov-sdk.git"

# All extras
pip install "marqov[all] @ git+https://github.com/marqov-dev/marqov-sdk.git"
```

For local development:

```bash
git clone https://github.com/marqov-dev/marqov-sdk
cd marqov-sdk
pip install -e ".[all,dev]"
pytest tests/ -v
```

## Cloud Executors

Swap in a cloud backend when you're ready to run on hardware:

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

## Supported Backends

| Backend | Status |
|---|---|
| Local (QuantumFlow simulator) | ✅ Available |
| AWS Braket | ✅ Available |
| IBM Quantum | ✅ Available |
| Azure Quantum | ✅ Available |
| IonQ Direct | ✅ Available |
| Rigetti QCS | [🔧 Open issue #2](https://github.com/marqov-dev/marqov-sdk/issues/2) |
| Quantinuum | [🔧 Open issue #6](https://github.com/marqov-dev/marqov-sdk/issues/6) |

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
```

## Marqov Platform

The SDK runs standalone, but the [Marqov platform](https://marqov.ai) removes the infrastructure overhead, with an integrated Temporal worker, job tracking and cost visibility built in, and one-click access to every supported QPU. Scripts written against the SDK run unchanged on the platform via `MarqovDevice`, so the platform handles backend routing, retries, and result storage. In private beta (but early teams are granted QPU credits).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the executor interface spec, canonical gate set, factory registration steps, and local QVM setup for Rigetti development.

Bounty issues are open through [unitaryHACK 2026](https://unitaryhack.dev) — see the [issues page](https://github.com/marqov-dev/marqov-sdk/issues) for what's available.

## License

[Apache 2.0](LICENSE)
