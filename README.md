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

Independent tasks execute in parallel automatically. Marqov handles scheduling, retries, and result collection across any supported backend. Run your own Temporal worker (see `marqov/workflows/`). Hosted execution is provided separately by the [Marqov Platform](https://marqov.ai). SDK installation alone does not enable managed execution.

---

## Installation

SDK 0.7.0 is published. See the [changelog](CHANGELOG.md) and
[0.7.0 upgrade guide](docs/releases/0.7.0.md), especially if you use AWS Braket.

```bash
pip install marqov
```

With framework- or backend-specific extras:

```bash
# AWS Braket
pip install "marqov[braket]"

# IBM Quantum
pip install "marqov[ibm]"

# QuTiP solvers and Marqov's result-recording helper
pip install "marqov[qutip]"

# Combine selected frameworks
pip install "marqov[qutip,qiskit]"

# Broad framework bundle
pip install "marqov[all]"
```

AWS Braket is an optional provider dependency. Existing Braket installations
should use `marqov[braket]` (or `marqov[all]`) when upgrading. Core workflows,
`LocalExecutor` and `Circuit.simulate()` do not require it. `MarqovDevice` with
`local` or `marqov-sim` uses Braket's local simulator and needs the extra, as do
Braket circuit conversions. Hosted compiler/task environments can pin
their serialization version without inheriting Braket's separate job serializer
constraint.

See the [QuTiP guide](docs/qutip.md) for a copy-and-run simulation, recorded
observables, seed replay and saved-state handling. In a source checkout, run:

```bash
python examples/qutip_decay.py
```

The example prints JSON and runs locally without an account. Installing the
wheel does not install the examples directory or enable managed execution.

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
provider accounts**, no Marqov account needed. The AWS example below requires
`pip install "marqov[braket]"` and your AWS credentials:

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
| Qilimanjaro (qilisdk local simulators — digital `execute()` and analog `execute_analog()`) | Available — not on PyPI as a `marqov[...]` extra (like Quantum Brilliance): `qilisdk`'s numpy floor is incompatible with marqov's own numpy ceiling outside a narrow macOS overlap window. Install separately: `pip install qilisdk`. |
| CESGA CUNQA (distributed-QC emulator, Slurm-based) | Available — not on PyPI at all (no wheel; build from source, see `CESGA-Quantum-Spain/cunqa`) and not a `marqov[...]` extra: CUNQA's exact `qiskit==1.2.4` pin would downgrade the whole project's lockfile if included in `[project.optional-dependencies]`, same class of problem `qilisdk` had. Install `qiskit==1.2.4` separately in the environment where CUNQA is built. |

---

For QiliSim sampling seeds and repeatability limits, see [QiliSDK seed support](docs/qilisdk.md).

## Circuit Interop

`Circuit` is a backend-agnostic abstraction that converts to any supported framework's native format:

```python
from marqov.circuits import Circuit

circuit = Circuit().h(0).cnot(0, 1)

circuit.to_qiskit()   # qiskit.QuantumCircuit
circuit.to_braket()   # braket.circuits.Circuit (requires marqov[braket])
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

> **What works today.** Managed native `@task`/`@workflow` programs run
> through `client.submit_native()`; `client.backends()`, `client.job()`,
> `job.status()` and `job.result()` work for any job. The request shapes are
> tested against the hosted API's contract; authenticated live verification is
> pending. See [Current limitations](docs/platform-client/native-workflows.md#current-limitations)
> — in particular `client.submit()`, `Circuit` submission, API-key
> `job.cancel()` and `platform_info()` do not currently work against the
> hosted API.

### Quickstart

**1. Set your API key and team** (both from the Marqov app):

```bash
export MARQOV_PLATFORM_KEY="marqey_live_your_key_here"
export MARQOV_TEAM_ID="your-team-uuid"
```

**2. Run a saved workflow: discover → submit → wait → read:**

```python
import os
from marqov.platform import MarqovClient

client = MarqovClient()                     # reads MARQOV_PLATFORM_KEY
team_id = os.environ["MARQOV_TEAM_ID"]

print(client.managed_runtimes(team_id))     # [] means not enabled for this team

job = client.submit_native(
    team_id=team_id,
    script_id="your-saved-script-uuid",     # or source="...python..."
    entrypoint="native_canary",             # the @workflow function to call
    kwargs={"seed": 7},
    cap_cents=100,                          # your spending cap for this run; required
)
result = job.result(timeout=600.0)
for output in result.raw["outputs"]:
    print(output["task_key"], output["display"].get("value"))
```

The complete example, with the saved script and idempotent retry, is in
[`docs/platform-client/native-workflows.md`](docs/platform-client/native-workflows.md)
and [`examples/platform_native_workflow.py`](examples/platform_native_workflow.py).

**3. Check available backends:**

```python
for b in client.backends():
    print(b.slug, b.name, "available:", b.is_available)
```

Listing a backend (or `is_available`) is the catalogue; it does not mean a
particular team or program can execute there. That is decided when you submit.

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
    job = client.submit_native(team_id=team_id, script_id=script_id,
                               entrypoint="native_canary", cap_cents=100)
    result = job.result(timeout=600.0)
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
- [Native workflows on the hosted platform](docs/platform-client/native-workflows.md)
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
