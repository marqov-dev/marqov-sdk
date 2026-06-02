"""End-to-end SRB integration test."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from marqov.executors.base import BaseExecutor, ExecutionResult
from marqov.qmt.characterization.srb import SRBConfig, run_srb
from marqov.qmt.characterization.srb_analysis import (
    build_noise_profile_from_srb,
    extract_cross_talk,
)
from marqov.qmt.models import DeviceModality


class DepolarizingExecutor(BaseExecutor):
    """Simulates depolarizing noise with configurable cross-talk."""

    def __init__(self, base_error: float = 0.01, cross_talk_error: float = 0.005) -> None:
        self._base_error = base_error
        self._cross_talk = cross_talk_error
        self._rng = np.random.default_rng(42)

    async def execute(self, circuit, shots=100, **kwargs) -> ExecutionResult:
        gate_count = 0
        qubits = set()
        try:
            for op in circuit._qf._elements:
                gate_count += 1
                for q in op.qubits:
                    qubits.add(q)
        except Exception:
            gate_count = 10
            qubits = {0}

        is_simultaneous = len(qubits) > 1
        error = self._base_error + (self._cross_talk if is_simultaneous else 0)
        p = max(0, 1 - 2 * error)
        effective_depth = max(gate_count / 2, 1)
        survival = 0.5 * (p ** effective_depth) + 0.5

        n_qubits = max(qubits) + 1 if qubits else 1
        n_correct = int(shots * survival)
        n_wrong = shots - n_correct

        all_zeros = "0" * n_qubits
        counts = {all_zeros: n_correct}
        if n_wrong > 0:
            counts["1" + "0" * (n_qubits - 1)] = n_wrong

        return ExecutionResult(
            counts=counts, backend="depolarizing", execution_time_ms=1.0, shots=shots
        )


@pytest.mark.asyncio
async def test_full_srb_pipeline() -> None:
    """Run SRB -> fit decay -> extract cross-talk -> build NoiseProfile."""
    config = SRBConfig(
        target_qubits=[0],
        neighbor_qubits=[2],
        sequence_lengths=[1, 2, 4, 8, 16],
        num_sequences_short=5,
        num_sequences_long=3,
        shots=200,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        executor = DepolarizingExecutor(base_error=0.01, cross_talk_error=0.005)
        result = await run_srb(config, executor, output_dir=tmpdir)

    ct = extract_cross_talk(result)
    assert ct["isolated_error_per_clifford"] >= 0
    assert ct["cross_talk_delta"] is not None

    profile = build_noise_profile_from_srb(
        device_name="test",
        modality=DeviceModality.TRAPPED_ION,
        num_qubits=4,
        srb_results=[result],
    )
    assert profile.num_qubits == 4
    assert profile.qubit_error_rates[0] >= 0
