# Contributing to Marqov SDK

Thank you for contributing. This guide covers everything you need to add a new
executor, circuit converter, or other contribution to the Marqov SDK.

## Development Setup

```bash
git clone https://github.com/marqov-dev/marqov-sdk
cd marqov-sdk
python3.12 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[all,dev]"
pytest tests/ -v
```

## §1 — Canonical Gate Set

All executor and circuit converter contributions must support the following
gates and raise `NotImplementedError` for anything outside this set. This list
is derived from `marqov/circuits.py` (`_QISKIT_GATE_MAP`) and represents what
the SDK can round-trip today.

| Category      | Gates                        |
|---------------|------------------------------|
| Single-qubit  | H, X, Y, Z, S, T            |
| Rotation      | Rx(θ), Ry(θ), Rz(θ)         |
| Two-qubit     | CNOT/CX, CZ, SWAP            |

## §2 — Executor Interface

All executors inherit from `BaseExecutor` in `marqov/executors/base.py`.

### Required methods

**`async execute(circuit: Circuit, shots: int = 1000, **kwargs) -> ExecutionResult`**

Submit the circuit to the backend and return results. `ExecutionResult` fields
you must populate:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `counts` | `dict[str, int]` | Yes | Measurement outcomes e.g. `{"00": 512, "11": 488}` |
| `backend` | `str` | Yes | Name or identifier of the backend |
| `execution_time_ms` | `float` | Yes | Wall time in milliseconds |
| `shots` | `int` | Yes | Number of shots executed |
| `raw_result` | `Any` | No | Provider-specific result object, for debugging |
| `metadata` | `dict` | No | Additional provider metadata |

**`async cancel(job_id: str) -> bool`**

Cancel a running job. Return `True` if successful, `False` otherwise. If the
provider does not support cancellation, the default `BaseExecutor` implementation
returns `False` — do not override it unless the provider supports cancellation.

`job_id` is the provider's job or task identifier. Callers obtain it from
`ExecutionResult.metadata` (e.g. `result.metadata["task_arn"]` for Braket).
Executors that support cancellation should store the active job ID as instance
state (e.g. `self._current_job_id`) and include it in `ExecutionResult.metadata`
under a documented key. Job polling (queued → running → completed) happens
internally inside `execute()` and is not related to this method.

**`async get_status() -> DeviceStatus`**

Return the QPU's **operational availability** — whether the device is currently
accepting new job submissions. This is **device-level status**, not job-level
status. Job polling (queued → running → completed) is handled internally inside
`execute()` and is not exposed via `get_status()`.

`DeviceStatus` fields:

| Field | Type | Values |
|-------|------|--------|
| `status` | `str` | `"online"`, `"offline"`, `"maintenance"` |
| `queue_depth` | `int \| None` | Number of queued tasks, or `None` if unknown |
| `queue_time_seconds` | `int \| None` | Estimated queue wait, or `None` if unknown |

The default `BaseExecutor.get_status()` returns `DeviceStatus.always_online()`.
Cloud backends should override this to query the provider's device status endpoint.

## §3 — Adding a New Executor

1. Create `marqov/executors/<name>.py` with a config dataclass and executor class:

```python
from dataclasses import dataclass
from typing import Any
from marqov.executors.base import BaseExecutor, DeviceStatus, ExecutionResult
from marqov.circuits import Circuit

@dataclass
class MyProviderExecutorConfig:
    api_key: str
    device_name: str
    shots: int = 1000

class MyProviderExecutor(BaseExecutor):
    def __init__(self, config: MyProviderExecutorConfig) -> None:
        self.config = config

    async def execute(self, circuit: Circuit, shots: int = 1000, **kwargs: Any) -> ExecutionResult:
        # convert circuit, submit, poll, return ExecutionResult
        ...

    async def get_status(self) -> DeviceStatus:
        # query provider device status endpoint
        ...
```

2. Register in `marqov/executors/factory.py`:
   - Add import at top: `from marqov.executors.<name> import MyProviderExecutor, MyProviderExecutorConfig`
   - Add branch in `create_executor()`:
     ```python
     if provider == "My Provider":
         return cls._create_myprovider_executor(backend_slug, backend_config)
     ```
   - Add `_create_myprovider_executor()` classmethod following the pattern of `_create_ibm_executor()`
   - Add `"My Provider"` to the list in `get_supported_providers()`

3. Add the provider package to `pyproject.toml` as an optional dependency:
   ```toml
   [project.optional-dependencies]
   myprovider = ["my-provider-sdk>=1.0.0"]
   ```

4. Export from `marqov/executors/__init__.py`.

## §4 — Local QVM Setup (Rigetti executor development)

The Rigetti QCS executor tests run against a local QVM instance. QVM requires
`quilc` running alongside it.

```bash
docker pull rigetti/quilc
docker pull rigetti/qvm
docker run -d -p 5555:5555 rigetti/quilc -server
docker run -d -p 5000:5000 rigetti/qvm -server
```

Verify:
```bash
python -c "from pyquil import get_qc; qc = get_qc('2q-qvm'); print(qc)"
```

Expected output: `<QVM 2q-qvm>` or similar. If you see a connection error,
check both containers are running with `docker ps`.

## §5 — Running Benchmarks

`benchmarks/suite.py` runs a fixed set of reference circuits — Bell, 3-qubit
GHZ, and a deterministic depth-5 random circuit — against any configured
executor and prints a comparison table. It works out of the box with
`LocalExecutor`, so no credentials are required:

```bash
python benchmarks/suite.py --executor local --shots 1000 --seed 1234
```

Output format — one row per (backend × circuit) combination:

| backend | circuit   | shots | exec_time_ms | top_3_outcomes                       |
|---------|-----------|-------|--------------|--------------------------------------|
| local   | bell      | 1000  | 0.4          | {"11": 502, "00": 498}               |
| local   | ghz       | 1000  | 0.4          | {"111": 524, "000": 476}             |
| local   | random_d5 | 1000  | 0.4          | {"010": 262, "100": 254, "110": 243} |

Columns:
- `backend`: executor name
- `circuit`: circuit name (`bell`, `ghz`, `random_d5`)
- `shots`: number of shots
- `exec_time_ms`: wall time in milliseconds (varies per run and machine)
- `top_3_outcomes`: counts of the 3 most frequent measurement outcomes, ordered
  by count descending

The measurement outcomes above are reproducible for a fixed `--seed` (it seeds
both the random circuit and the local sampler); `exec_time_ms` is wall time and
will differ on your machine.

**Scope.** The suite compares execution time and outcome distribution only.
Shot-fidelity and queue-overhead metrics are out of scope for this harness;
device-level queue status is available separately via `BaseExecutor.get_status()`.

**Error handling.** If a backend errors on any circuit, the suite skips that
*entire* backend (it emits no partial rows for it), logs the backend and the
failing circuit to stderr, and continues with the next backend — it never
aborts. If every backend fails the command exits non-zero, so CI can flag a
fully broken run.

**Benchmarking other backends.** The CLI only constructs the zero-credential
`local` executor, but `run_suite()` is executor-agnostic — pass a mapping of
name → `BaseExecutor` to benchmark cloud backends programmatically:

```python
import asyncio
from benchmarks.suite import run_suite, format_table
from marqov.executors import ExecutorFactory, LocalExecutor

executors = {
    "local": LocalExecutor(),
    "sv1": ExecutorFactory.create_executor("sv1", braket_config),
}
rows = asyncio.run(run_suite(executors, shots=1000))
print(format_table(rows))
```
