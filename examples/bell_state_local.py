"""Run a Bell-state circuit on Marqov's local executor."""

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
