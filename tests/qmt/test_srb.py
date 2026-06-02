"""Tests for SRB experiment runner."""

from __future__ import annotations

import tempfile

import pytest

from marqov.executors.base import BaseExecutor, ExecutionResult
from marqov.qmt.characterization.srb import (
    SRBConfig,
    SRBResult,
    run_srb,
)


class FakeExecutor(BaseExecutor):
    """Returns fixed survival rates for testing."""

    async def execute(self, circuit, shots=100, **kwargs) -> ExecutionResult:
        n_qubits = 2
        all_zeros = "0" * n_qubits
        correct = int(shots * 0.8)
        wrong = shots - correct
        counts = {all_zeros: correct}
        if wrong > 0:
            counts["1" + "0" * (n_qubits - 1)] = wrong
        return ExecutionResult(
            counts=counts, backend="fake", execution_time_ms=1.0, shots=shots
        )


class TestSRBConfig:
    def test_default_config(self) -> None:
        config = SRBConfig(target_qubits=[0], neighbor_qubits=[2])
        assert config.sequence_lengths == [1, 2, 4, 8, 16, 32, 64, 128, 256]
        assert config.num_sequences_short == 20
        assert config.num_sequences_long == 10
        assert config.length_threshold == 32

    def test_sequences_for_length(self) -> None:
        config = SRBConfig(target_qubits=[0], neighbor_qubits=[2])
        assert config.num_sequences_for_length(8) == 20
        assert config.num_sequences_for_length(32) == 20
        assert config.num_sequences_for_length(64) == 10
        assert config.num_sequences_for_length(256) == 10


@pytest.mark.asyncio
async def test_run_srb_produces_result() -> None:
    config = SRBConfig(
        target_qubits=[0],
        neighbor_qubits=[2],
        sequence_lengths=[1, 2, 4],
        num_sequences_short=3,
        num_sequences_long=2,
        shots=50,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        executor = FakeExecutor()
        result = await run_srb(config, executor, output_dir=tmpdir)

    assert isinstance(result, SRBResult)
    assert len(result.isolated_survival) == 3
    assert len(result.simultaneous_survival) == 3
    assert all(0.0 <= p <= 1.0 for p in result.isolated_survival.values())


@pytest.mark.asyncio
async def test_srb_isolated_only() -> None:
    config = SRBConfig(
        target_qubits=[0],
        neighbor_qubits=[],
        sequence_lengths=[1, 2],
        num_sequences_short=2,
        num_sequences_long=2,
        shots=50,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        executor = FakeExecutor()
        result = await run_srb(config, executor, output_dir=tmpdir)

    assert len(result.isolated_survival) == 2
    assert len(result.simultaneous_survival) == 0
