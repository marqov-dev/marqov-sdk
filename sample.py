import asyncio

from marqov.circuits import bell_state
from marqov.executors import RigettiExecutor, RigettiExecutorConfig


async def main():
    config = RigettiExecutorConfig(
        quantum_processor_id="2q-qvm",
        as_qvm=True,
    )

    executor = RigettiExecutor(config)

    result = await executor.execute(
        bell_state(),
        shots=1000,
    )

    print(result.counts)


asyncio.run(main())