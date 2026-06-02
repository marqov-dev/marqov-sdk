"""Tests for QMT result splitter."""

from __future__ import annotations

from unittest.mock import PropertyMock, patch

import pytest

from marqov.circuits import Circuit
from marqov.qmt.models import (
    PackingPlan,
    QMTJob,
    QubitMapping,
)
from marqov.qmt.scheduler.splitter import split_results


# Circuit.num_qubits is mocked because conftest stubs quantumflow.


class TestSplitResults:
    @staticmethod
    @patch.object(Circuit, "num_qubits", new_callable=PropertyMock, return_value=2)
    def _make_plan(_mock) -> tuple[PackingPlan, str, str]:
        """Create a test plan: job_a on qubits [0,1], job_b on qubit [3]."""
        job_a = QMTJob(circuit=Circuit().h(0).cnot(0, 1), submitter="a")
        # Need to re-patch for job_b which is 1 qubit
        job_b = QMTJob(circuit=Circuit().h(0), submitter="b")
        plan = PackingPlan(
            jobs=[job_a, job_b],
            mappings=[
                QubitMapping(job_id=job_a.job_id, logical_to_physical={0: 0, 1: 1}),
                QubitMapping(job_id=job_b.job_id, logical_to_physical={0: 3}),
            ],
            guard_qubits={2},
            device_name="test",
            total_qubits=5,
        )
        return plan, job_a.job_id, job_b.job_id

    def test_splits_counts_correctly(self) -> None:
        plan, job_a_id, job_b_id = self._make_plan()
        # Bitstring format: 5 qubits, qubit 0 is leftmost
        # "00000": job_a sees "00", job_b sees "0"
        # "11010": job_a sees "11", job_b sees "1"
        composite_counts = {
            "00000": 400,
            "11010": 600,
        }
        results = split_results(composite_counts, plan, shots=1000)
        assert len(results) == 2

        result_a = next(r for r in results if r.job_id == job_a_id)
        result_b = next(r for r in results if r.job_id == job_b_id)

        assert result_a.counts == {"00": 400, "11": 600}
        assert result_b.counts == {"0": 400, "1": 600}

    def test_preserves_shot_count(self) -> None:
        plan, _, _ = self._make_plan()
        composite_counts = {"00000": 500, "11010": 500}
        results = split_results(composite_counts, plan, shots=1000)
        for result in results:
            assert result.shots == 1000

    def test_handles_mixed_bitstrings(self) -> None:
        plan, job_a_id, job_b_id = self._make_plan()
        composite_counts = {
            "00000": 300,  # a="00", b="0"
            "00010": 200,  # a="00", b="1"
            "11000": 250,  # a="11", b="0"
            "11010": 250,  # a="11", b="1"
        }
        results = split_results(composite_counts, plan, shots=1000)
        result_a = next(r for r in results if r.job_id == job_a_id)
        result_b = next(r for r in results if r.job_id == job_b_id)

        assert result_a.counts == {"00": 500, "11": 500}
        assert result_b.counts == {"0": 550, "1": 450}
