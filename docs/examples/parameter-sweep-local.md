# Local parameter sweep

This example runs the same one-qubit rotation circuit over several angles with
the local executor. It is a small offline version of the workflow pattern used
for parameter scans: build many related circuits, execute them independently,
then compare the measured probabilities.

## Run

```python
import asyncio
from math import pi

from marqov.circuits import Circuit
from marqov.executors import LocalExecutor, LocalExecutorConfig


async def measure_angle(angle: float) -> tuple[float, dict[str, float]]:
    circuit = Circuit().ry(angle, 0)
    executor = LocalExecutor(LocalExecutorConfig(seed=42))
    result = await executor.execute(circuit, shots=1000)
    return angle, result.probabilities


async def main() -> None:
    angles = [0.0, pi / 6, pi / 4, pi / 3, pi / 2]
    results = await asyncio.gather(*(measure_angle(angle) for angle in angles))

    for angle, probabilities in results:
        print(f"angle={angle:.3f}", probabilities)


if __name__ == "__main__":
    asyncio.run(main())
```

The probability of measuring `1` increases as the `ry` angle moves away from
zero. In a larger experiment this loop could be replaced by `@task` and
`@workflow` dispatch so each measurement can be scheduled independently.

## Runnable script

The same example is available as
[`examples/parameter_sweep_local.py`](../../examples/parameter_sweep_local.py).
