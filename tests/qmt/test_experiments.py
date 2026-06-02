"""Tests for QMT cross-talk experiment runner."""

from __future__ import annotations

from unittest.mock import PropertyMock, patch

import pytest

from marqov.circuits import Circuit
from marqov.executors.base import BaseExecutor, ExecutionResult
from marqov.qmt.characterization.experiments import (
    CrossTalkExperiment,
    ExperimentResult,
    run_cross_talk_experiment,
)


class FakeExecutor(BaseExecutor):
    """Fake executor that returns predetermined counts for testing."""

    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts

    async def execute(self, circuit, shots=1000, **kwargs) -> ExecutionResult:
        return ExecutionResult(
            counts=self._counts,
            backend="fake",
            execution_time_ms=1.0,
            shots=shots,
        )


@pytest.mark.asyncio
async def test_single_tenant_baseline() -> None:
    executor = FakeExecutor({"00": 500, "11": 500})
    experiment = CrossTalkExperiment(
        target_qubits=[0, 1],
        neighbor_qubits=[],
        benchmark="ghz",
        shots=1000,
    )
    result = await run_cross_talk_experiment(experiment, executor)
    assert isinstance(result, ExperimentResult)
    assert result.target_counts is not None
    assert sum(result.target_counts.values()) == 1000


@pytest.mark.asyncio
async def test_single_tenant_ghz_fidelity() -> None:
    executor = FakeExecutor({"00": 500, "11": 500})
    experiment = CrossTalkExperiment(
        target_qubits=[0, 1],
        neighbor_qubits=[],
        benchmark="ghz",
        shots=1000,
    )
    result = await run_cross_talk_experiment(experiment, executor)
    assert result.target_fidelity == 1.0


@pytest.mark.asyncio
async def test_multi_tenant_experiment() -> None:
    # Target qubits [0,1], neighbor qubits [3,4] → total 5 physical qubits (0-4)
    # Composite bitstring has 5 chars; all zeros.
    executor = FakeExecutor({"00000": 1000})
    experiment = CrossTalkExperiment(
        target_qubits=[0, 1],
        neighbor_qubits=[3, 4],
        benchmark="ghz",
        shots=1000,
    )
    with patch.object(Circuit, "num_qubits", new_callable=PropertyMock, return_value=5):
        result = await run_cross_talk_experiment(experiment, executor)
    assert result.target_counts is not None
    assert result.neighbor_counts is not None
    # Target sees "00" (qubits 0,1), neighbor sees "00" (qubits 3,4)
    assert result.target_counts == {"00": 1000}
    assert result.neighbor_counts == {"00": 1000}


@pytest.mark.asyncio
async def test_mirror_benchmark_fidelity() -> None:
    # Mirror circuit returns to |00> — all zeros = perfect fidelity
    executor = FakeExecutor({"00": 900, "01": 100})
    experiment = CrossTalkExperiment(
        target_qubits=[0, 1],
        neighbor_qubits=[],
        benchmark="mirror",
        shots=1000,
        benchmark_seed=42,
    )
    result = await run_cross_talk_experiment(experiment, executor)
    assert result.target_fidelity == 0.9
