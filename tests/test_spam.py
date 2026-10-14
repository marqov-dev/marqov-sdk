"""Tests for marqov.benchmarking.spam module."""

from typing import Any

import numpy as np
import pytest

from marqov.benchmarking.spam import (
    ConfusionMatrix,
    SPAMResult,
    spam_benchmark,
    build_correction_matrix,
    apply_spam_correction,
)
from marqov.circuits import Circuit
from marqov.executors.local import LocalExecutor, LocalExecutorConfig
from marqov.executors.base import BaseExecutor, ExecutionResult


class TestConfusionMatrix:
    """Tests for ConfusionMatrix dataclass."""

    def test_creation(self) -> None:
        """Creates with four probability values."""
        cm = ConfusionMatrix(p00=0.98, p01=0.02, p10=0.03, p11=0.97)
        assert cm.p00 == 0.98
        assert cm.p01 == 0.02
        assert cm.p10 == 0.03
        assert cm.p11 == 0.97

    def test_readout_fidelity(self) -> None:
        """Readout fidelity is (p00 + p11) / 2."""
        cm = ConfusionMatrix(p00=0.98, p01=0.02, p10=0.03, p11=0.97)
        assert cm.readout_fidelity == pytest.approx(0.975)

    def test_perfect_fidelity(self) -> None:
        """Perfect readout has fidelity 1.0."""
        cm = ConfusionMatrix(p00=1.0, p01=0.0, p10=0.0, p11=1.0)
        assert cm.readout_fidelity == 1.0

    def test_worst_fidelity(self) -> None:
        """Completely wrong readout has fidelity 0.0."""
        cm = ConfusionMatrix(p00=0.0, p01=1.0, p10=1.0, p11=0.0)
        assert cm.readout_fidelity == 0.0


class TestSPAMResult:
    """Tests for SPAMResult dataclass."""

    def test_readout_fidelity_per_qubit(self) -> None:
        """Returns per-qubit fidelity dict."""
        matrices = {
            0: ConfusionMatrix(p00=0.98, p01=0.02, p10=0.03, p11=0.97),
            1: ConfusionMatrix(p00=0.95, p01=0.05, p10=0.04, p11=0.96),
        }
        result = SPAMResult(matrices=matrices, shots=1000, metadata={})
        fidelities = result.readout_fidelity
        assert fidelities[0] == pytest.approx(0.975)
        assert fidelities[1] == pytest.approx(0.955)

    def test_confusion_matrix_accessor(self) -> None:
        """confusion_matrix(qubit) returns the right matrix."""
        cm0 = ConfusionMatrix(p00=0.98, p01=0.02, p10=0.03, p11=0.97)
        result = SPAMResult(matrices={0: cm0}, shots=1000, metadata={})
        assert result.confusion_matrix(0) is cm0

    def test_confusion_matrix_invalid_qubit(self) -> None:
        """confusion_matrix raises KeyError for unknown qubit."""
        result = SPAMResult(matrices={}, shots=1000, metadata={})
        with pytest.raises(KeyError):
            result.confusion_matrix(5)


class FakeExecutor(BaseExecutor):
    """Fake executor for testing SPAM benchmark.

    Results are returned in call order: prep_0 for q0, prep_1 for q0,
    prep_0 for q1, prep_1 for q1, etc.
    """

    def __init__(self, results: dict[int, dict[str, int]]) -> None:
        """results maps call index (0-based) to counts dict."""
        self._results = results
        self._call_count = 0

    async def execute(self, circuit: Circuit, shots: int = 1000, **kwargs: Any) -> ExecutionResult:
        """Return pre-configured results based on call order."""
        self._call_count += 1
        idx = self._call_count - 1
        counts = self._results[idx]
        return ExecutionResult(
            counts=counts,
            backend="fake",
            execution_time_ms=1.0,
            shots=shots,
        )


class TestSpamBenchmark:
    """Tests for spam_benchmark function."""

    @pytest.mark.asyncio
    async def test_perfect_readout(self) -> None:
        """Perfect executor returns fidelity 1.0."""
        executor = LocalExecutor(LocalExecutorConfig(seed=7))
        result = await spam_benchmark(executor, qubits=[0], shots=1000)

        assert result.readout_fidelity[0] == 1.0
        cm = result.confusion_matrix(0)
        assert cm.p00 == 1.0
        assert cm.p01 == 0.0
        assert cm.p10 == 0.0
        assert cm.p11 == 1.0

    @pytest.mark.asyncio
    async def test_imperfect_readout(self) -> None:
        """Noisy executor returns expected confusion matrix."""

        class ReadoutErrorExecutor(LocalExecutor):
            async def execute(
                self,
                circuit: Circuit,
                shots: int = 1000,
                **kwargs: Any,
            ) -> ExecutionResult:
                result = await super().execute(circuit, shots=shots, **kwargs)
                # Apply a deterministic readout channel to real sampled bits;
                # no call-order fixture invents the circuit's register width.
                noisy_counts: dict[str, int] = {}
                for bit, count in result.counts.items():
                    assert len(bit) == 1
                    flipped = "1" if bit == "0" else "0"
                    errors = int(count * (0.02 if bit == "0" else 0.03))
                    noisy_counts[bit] = count - errors
                    noisy_counts[flipped] = errors
                result.counts = noisy_counts
                return result

        executor = ReadoutErrorExecutor(LocalExecutorConfig(seed=7))
        result = await spam_benchmark(executor, qubits=[0], shots=1000)

        cm = result.confusion_matrix(0)
        assert cm.p00 == pytest.approx(0.98)
        assert cm.p01 == pytest.approx(0.02)
        assert cm.p10 == pytest.approx(0.03)
        assert cm.p11 == pytest.approx(0.97)
        assert cm.readout_fidelity == pytest.approx(0.975)

    @pytest.mark.asyncio
    async def test_multiple_qubits(self) -> None:
        """SPAM on multiple qubits returns per-qubit matrices."""
        executor = LocalExecutor(LocalExecutorConfig(seed=7))
        result = await spam_benchmark(executor, qubits=[0, 1, 2], shots=2000)

        assert result.readout_fidelity == {0: 1.0, 1: 1.0, 2: 1.0}
        assert result.counts_prepared_0 == {
            0: {"0": 2000},
            1: {"00": 2000},
            2: {"000": 2000},
        }
        assert result.counts_prepared_1 == {
            0: {"1": 2000},
            1: {"01": 2000},
            2: {"001": 2000},
        }

    @pytest.mark.asyncio
    async def test_nonconsecutive_qubits_keep_absolute_positions(self) -> None:
        executor = LocalExecutor(LocalExecutorConfig(seed=7))
        result = await spam_benchmark(executor, qubits=[2, 0], shots=32)
        assert result.readout_fidelity == {2: 1.0, 0: 1.0}
        assert result.counts_prepared_1[2] == {"001": 32}
        assert result.metadata["qubits"] == [2, 0]

    @pytest.mark.asyncio
    async def test_real_calibration_correction_round_trip(self) -> None:
        """Noiseless real calibration preserves sampled counts and shots exactly."""
        executor = LocalExecutor(LocalExecutorConfig(seed=7))
        calibration = await spam_benchmark(executor, qubits=[0, 1, 2], shots=2000)
        correction = build_correction_matrix(calibration)
        np.testing.assert_array_equal(correction, np.eye(8))
        circuit = Circuit().x(0).h(1).ry(0.37, 2).cnot(1, 2)
        sampled = await executor.execute(circuit, shots=2000)
        assert all(key[0] == "1" for key in sampled.counts)
        corrected = apply_spam_correction(sampled.counts, correction)
        assert corrected == sampled.counts
        assert sum(corrected.values()) == 2000

    @pytest.mark.asyncio
    async def test_validates_empty_qubits(self) -> None:
        """Raises ValueError for empty qubit list."""
        executor = FakeExecutor({})
        with pytest.raises(ValueError, match="qubits"):
            await spam_benchmark(executor, qubits=[], shots=1000)

    @pytest.mark.asyncio
    async def test_validates_negative_qubits(self) -> None:
        """Raises ValueError for negative qubit indices."""
        executor = FakeExecutor({})
        with pytest.raises(ValueError, match="qubit"):
            await spam_benchmark(executor, qubits=[-1], shots=1000)

    @pytest.mark.asyncio
    async def test_result_metadata(self) -> None:
        """Result includes metadata about the benchmark."""
        executor = LocalExecutor(LocalExecutorConfig(seed=7))
        result = await spam_benchmark(executor, qubits=[0], shots=1000)

        assert result.shots == 1000
        assert "qubits" in result.metadata
        assert result.metadata["qubits"] == [0]

    @pytest.mark.asyncio
    async def test_executor_error_propagates(self) -> None:
        """Executor errors propagate immediately, no partial results."""

        class FailingExecutor(BaseExecutor):
            async def execute(
                self, circuit: Circuit, shots: int = 1000, **kwargs: Any
            ) -> ExecutionResult:
                raise RuntimeError("hardware timeout")

        executor = FailingExecutor()
        with pytest.raises(RuntimeError, match="hardware timeout"):
            await spam_benchmark(executor, qubits=[0], shots=1000)


class TestPackageExports:
    """Tests for marqov.benchmarking package exports."""

    def test_import_from_package(self) -> None:
        """All public types importable from marqov.benchmarking."""
        from marqov.benchmarking import ConfusionMatrix, SPAMResult, spam_benchmark

        assert ConfusionMatrix is not None
        assert SPAMResult is not None
        assert spam_benchmark is not None
