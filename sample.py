import asyncio
from pprint import pprint

from marqov.circuits import bell_state
from marqov.executors import RigettiExecutor, RigettiExecutorConfig


async def run_bell_state():
    print("=" * 60)
    print("1. Creating Rigetti Executor")
    print("=" * 60)

    config = RigettiExecutorConfig(
        quantum_processor_id="2q-qvm",
        as_qvm=True,
        poll_interval_seconds=0.2,
        timeout_seconds=30,
        qvm_url="http://127.0.0.1:5000",
        quilc_url="tcp://127.0.0.1:5555",
    )

    executor = RigettiExecutor(config)

    print("\nChecking device status...")

    status = await executor.get_status()

    print(f"Status       : {status.status}")
    print(f"Queue Depth  : {status.queue_depth}")
    print(f"Queue Time   : {status.queue_time_seconds}")

    if status.status != "online":
        print("\nQVM/quilc is not running.")
        print("Start them first:")
        print("  qvm -S")
        print("  quilc -S")
        return

    print("\nExecuting Bell State Circuit...")

    try:
        result = await executor.execute(
            bell_state(),
            shots=1000,
            compile_timeout=10,
        )

        print("\nExecution Successful")
        print("-" * 40)

        print("Counts:")
        pprint(result.counts)

        print("\nBackend:")
        print(result.backend)

        print("\nShots:")
        print(result.shots)

        print("\nExecution Time (ms):")
        print(round(result.execution_time_ms, 2))

        print("\nMetadata:")
        pprint(result.metadata)

        print("\nRaw Result Type:")
        print(type(result.raw_result).__name__)

    except TimeoutError as e:
        print(f"Execution timed out: {e}")

    except Exception as e:
        print(f"Execution failed: {e}")


async def demonstrate_cancellation():
    print("\n" + "=" * 60)
    print("2. Demonstrating Job Cancellation")
    print("=" * 60)

    config = RigettiExecutorConfig(
        quantum_processor_id="2q-qvm",
        as_qvm=True,
        timeout_seconds=120,
    )

    executor = RigettiExecutor(config)

    task = asyncio.create_task(
        executor.execute(
            bell_state(),
            shots=1_000_000,
        )
    )

    await asyncio.sleep(0.1)

    job_id = executor._current_job_id

    if job_id:
        cancelled = await executor.cancel(job_id)
        print(f"Cancellation requested: {cancelled}")
    else:
        print("No active job ID available yet.")

    try:
        await task
    except asyncio.CancelledError:
        print("Task cancelled.")
    except Exception as e:
        print(f"Execution stopped: {e}")


async def main():
    await run_bell_state()

    # Uncomment if you want to test cancellation
    # await demonstrate_cancellation()


if __name__ == "__main__":
    asyncio.run(main())