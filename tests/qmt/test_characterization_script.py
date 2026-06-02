"""Tests for QMT characterization Run 1 script logic."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from marqov.executors.base import BaseExecutor, ExecutionResult
from marqov.qmt.characterization.experiments import ExperimentResult


class FakeExecutor(BaseExecutor):
    """Executor that returns all-zeros with a small error rate."""

    def __init__(self, error_rate: float = 0.02) -> None:
        self._error_rate = error_rate

    async def execute(self, circuit, shots=1000, **kwargs) -> ExecutionResult:
        n_qubits = 8
        all_zeros = "0" * n_qubits
        all_ones = "1" * n_qubits
        correct = int(shots * (1 - self._error_rate))
        error = shots - correct
        counts = {all_zeros: correct // 2, all_ones: correct // 2}
        if error > 0:
            counts["0" * (n_qubits - 1) + "1"] = error
        return ExecutionResult(
            counts=counts,
            backend="fake",
            execution_time_ms=1.0,
            shots=shots,
        )


class FailingExecutor(BaseExecutor):
    """Executor that fails on the 2nd multi-tenant run (buffer=1)."""

    def __init__(self) -> None:
        self._call_count = 0

    async def execute(self, circuit, shots=1000, **kwargs) -> ExecutionResult:
        self._call_count += 1
        if self._call_count == 4:
            raise RuntimeError("Simulated QPU failure")
        n_qubits = 8
        all_zeros = "0" * n_qubits
        all_ones = "1" * n_qubits
        counts = {all_zeros: shots // 2, all_ones: shots // 2}
        return ExecutionResult(
            counts=counts, backend="fake", execution_time_ms=1.0, shots=shots
        )


def _import_script():
    import sys
    sys.path.insert(0, "examples")
    import qmt_characterization_run1 as mod
    return mod


def test_experiment_matrix_structure() -> None:
    mod = _import_script()
    matrix = mod.build_experiment_matrix()
    assert len(matrix) == 8
    for i in range(0, 8, 2):
        assert matrix[i].neighbor_qubits == []
    for i in range(1, 8, 2):
        assert len(matrix[i].neighbor_qubits) > 0


@pytest.mark.asyncio
async def test_run_characterization_produces_noise_profile() -> None:
    mod = _import_script()
    with tempfile.TemporaryDirectory() as tmpdir:
        executor = FakeExecutor(error_rate=0.02)
        result = await mod.run_characterization(
            executor, shots=100, depth_multiplier=1, output_dir=tmpdir
        )
        assert result["device"] is not None
        assert len(result["paired_results"]) == 4
        assert result["noise_profile"] is not None
        assert result["noise_profile"].num_qubits > 0


@pytest.mark.asyncio
async def test_checkpointing_saves_after_each_pair() -> None:
    mod = _import_script()
    with tempfile.TemporaryDirectory() as tmpdir:
        executor = FakeExecutor(error_rate=0.01)
        await mod.run_characterization(
            executor, shots=50, depth_multiplier=1, output_dir=tmpdir
        )
        files = os.listdir(tmpdir)
        assert any(f.startswith("qmt_run1_") and f.endswith(".json") for f in files)
        filepath = os.path.join(tmpdir, files[0])
        with open(filepath) as f:
            data = json.load(f)
        assert len(data["experiments"]) == 4


@pytest.mark.asyncio
async def test_failure_isolation_continues_past_errors() -> None:
    mod = _import_script()
    with tempfile.TemporaryDirectory() as tmpdir:
        executor = FailingExecutor()
        result = await mod.run_characterization(
            executor, shots=50, depth_multiplier=1, output_dir=tmpdir
        )
        assert len(result["paired_results"]) == 4
        _, failed_mt = result["paired_results"][1]
        assert failed_mt is None
        _, first_mt = result["paired_results"][0]
        assert first_mt is not None
