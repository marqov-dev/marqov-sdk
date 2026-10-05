"""Run a small local parameter sweep with Marqov."""

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
