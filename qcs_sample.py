import asyncio
import os
from pprint import pprint
from dotenv import load_dotenv

load_dotenv()
from marqov.circuits import bell_state
from marqov.executors import RigettiExecutor, RigettiExecutorConfig


async def main():
    config = RigettiExecutorConfig(
        quantum_processor_id="Cepheus-1-108Q",  # Your QCS processor
        as_qvm=False,
        timeout_seconds=30,
        refresh_token=os.environ.get("QCS_REFRESH_TOKEN"),
        issuer=os.environ.get("QCS_API_ISSUER"),
        client_id=os.environ.get("QCS_API_CLIENT_ID"),
    )

    executor = RigettiExecutor(config)

    print("Checking processor status...")
    status = await executor.get_status()
    print(f"Status: {status.status}")

    if status.status != "online":
        print("Processor is not available.")
        return

    print("Submitting job to QCS...")

    result = await executor.execute(
        bell_state(),
        shots=10,
    )

    print("\nCounts:")
    pprint(result.counts)

    print("\nBackend:")
    print(result.backend)

    print("\nExecution Time (ms):")
    print(round(result.execution_time_ms, 2))

    print("\nMetadata:")
    pprint(result.metadata)


if __name__ == "__main__":
    asyncio.run(main())