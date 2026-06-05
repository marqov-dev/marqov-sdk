"""Integration tests for RigettiExecutor against local Docker QVM + quilc."""

from __future__ import annotations

import pytest

from marqov.circuits import bell_state
from marqov.executors.rigetti import RigettiExecutor, RigettiExecutorConfig


@pytest.mark.asyncio
async def test_bell_state_on_local_qvm() -> None:
    """Run Bell state end-to-end through pyQuil and local QVM."""
    pytest.importorskip("pyquil")

    executor = RigettiExecutor(
        RigettiExecutorConfig(
            quantum_processor_id="2q-qvm",
            as_qvm=True,
            poll_interval_seconds=0.1,
            timeout_seconds=30.0,
        )
    )

    status = await executor.get_status()
    if status.status != "online":
        pytest.skip("Local QVM/quilc not reachable; start Docker containers per CONTRIBUTING.md §4")

    result = await executor.execute(bell_state(), shots=200)

    assert result.backend == "rigetti:2q-qvm"
    assert result.shots == 200
    assert sum(result.counts.values()) == 200
    assert set(result.counts.keys()).issubset({"00", "11"})
