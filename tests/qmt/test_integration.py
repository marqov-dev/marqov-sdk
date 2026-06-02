"""End-to-end integration test for QMT packing flow.

Tests the full pipeline: create jobs -> group -> pack -> simulate execution -> split results.
Uses FakeExecutor to avoid QuantumFlow mock issues.
"""

from __future__ import annotations

from unittest.mock import PropertyMock, patch

import pytest

from marqov.circuits import Circuit
from marqov.executors.base import BaseExecutor, ExecutionResult
from marqov.qmt.models import DeviceModality, QMTJob
from marqov.qmt.scheduler.grouper import group_jobs
from marqov.qmt.scheduler.packer import pack_jobs
from marqov.qmt.scheduler.splitter import split_results
from marqov.qmt.scheduler.synthetic import generate_synthetic_profile


class FakeExecutor(BaseExecutor):
    """Returns counts based on the number of qubits inferred from the circuit."""

    async def execute(self, circuit, shots=1000, **kwargs) -> ExecutionResult:
        # Return a simple pattern: all-zeros with some noise
        # We don't know exact qubit count from mock, so use a fixed bitstring length
        counts = {"0" * 11: shots}
        return ExecutionResult(
            counts=counts,
            backend="fake",
            execution_time_ms=1.0,
            shots=shots,
        )


@pytest.mark.asyncio
@patch.object(Circuit, "num_qubits", new_callable=PropertyMock, return_value=2)
async def test_full_packing_pipeline(_mock) -> None:
    """Test the complete flow: jobs -> group -> pack -> split."""
    profile = generate_synthetic_profile(
        device_name="test-device",
        modality=DeviceModality.TRAPPED_ION,
        num_qubits=11,
        seed=42,
    )

    # Create two 2-qubit jobs
    jobs = [
        QMTJob(circuit=Circuit().h(0).cnot(0, 1), submitter="alice"),
        QMTJob(circuit=Circuit().h(0).cnot(0, 1), submitter="bob"),
    ]

    # Group
    groups = group_jobs(jobs, device_qubits=11, min_guard_qubits=1)
    assert len(groups) == 1  # Both should fit (2+1+2 = 5 <= 11)

    # Pack
    plan = pack_jobs(groups[0], profile, min_guard_qubits=1)
    assert len(plan.mappings) == 2

    # Verify no qubit overlap
    a_qubits = plan.mappings[0].physical_qubits
    b_qubits = plan.mappings[1].physical_qubits
    assert a_qubits.isdisjoint(b_qubits)

    # Simulate execution with fake composite counts
    # Build a bitstring matching the 11-qubit device
    composite_counts = {"00000000000": 2000}

    # Split
    per_job_results = split_results(composite_counts, plan, shots=2000)
    assert len(per_job_results) == 2

    for job_result in per_job_results:
        assert sum(job_result.counts.values()) == 2000
        assert len(job_result.counts) > 0


@pytest.mark.asyncio
@patch.object(Circuit, "num_qubits", new_callable=PropertyMock, return_value=2)
async def test_single_job_passthrough(_mock) -> None:
    """A single job should work through the full pipeline."""
    profile = generate_synthetic_profile(
        "test", DeviceModality.TRAPPED_ION, 11, seed=42
    )

    jobs = [QMTJob(circuit=Circuit().h(0).cnot(0, 1), submitter="solo")]
    groups = group_jobs(jobs, device_qubits=11)
    plan = pack_jobs(groups[0], profile)

    # Single job, 2 qubits mapped somewhere on an 11-qubit device
    assert len(plan.mappings) == 1

    composite_counts = {"00000000000": 1000}
    per_job = split_results(composite_counts, plan, shots=1000)

    assert len(per_job) == 1
    assert sum(per_job[0].counts.values()) == 1000
