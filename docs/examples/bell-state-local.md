# Bell state with local execution

This example creates a two-qubit Bell state and samples it with the local
QuantumFlow simulator. It does not require cloud credentials or a queue on a
quantum device.

## Install

```bash
pip install -e .
```

## Run

```python
import asyncio

from marqov.circuits import Circuit
from marqov.executors import LocalExecutor


async def main() -> None:
    circuit = Circuit().h(0).cnot(0, 1)
    result = await LocalExecutor().execute(circuit, shots=1000)

    print("backend:", result.backend)
    print("counts:", result.counts)
    print("probabilities:", result.probabilities)


if __name__ == "__main__":
    asyncio.run(main())
```

A typical output is split between `00` and `11`, because the Hadamard gate creates
a superposition and the CNOT gate entangles the second qubit with the first:

```text
backend: local
counts: {'00': 505, '11': 495}
probabilities: {'00': 0.505, '11': 0.495}
```

The exact counts vary because the local executor samples measurement results.
The probabilities should stay close to 50/50 as the number of shots increases.

## Runnable script

The same example is available as [`examples/bell_state_local.py`](../../examples/bell_state_local.py).
