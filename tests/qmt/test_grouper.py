"""Tests for QMT job grouper."""

from __future__ import annotations

from unittest.mock import PropertyMock, patch

from marqov.circuits import Circuit
from marqov.qmt.models import QMTJob
from marqov.qmt.scheduler.grouper import group_jobs


# Circuit.num_qubits returns a mock in tests (conftest stubs quantumflow).
# We patch it for tests that depend on qubit counts.


class TestGroupJobs:
    @patch.object(Circuit, "num_qubits", new_callable=PropertyMock, return_value=2)
    def test_two_small_jobs_grouped(self, _mock) -> None:
        jobs = [
            QMTJob(circuit=Circuit().h(0).cnot(0, 1), submitter="a"),
            QMTJob(circuit=Circuit().h(0).cnot(0, 1), submitter="b"),
        ]
        groups = group_jobs(jobs, device_qubits=11)
        assert len(groups) == 1
        assert len(groups[0]) == 2

    @patch.object(Circuit, "num_qubits", new_callable=PropertyMock, return_value=3)
    def test_jobs_exceeding_capacity_split(self, _mock) -> None:
        jobs = [
            QMTJob(circuit=Circuit().h(0).cnot(0, 1).cnot(1, 2), submitter="a"),
            QMTJob(circuit=Circuit().h(0).cnot(0, 1).cnot(1, 2), submitter="b"),
        ]
        # 3 + 1 guard + 3 = 7 > 5
        groups = group_jobs(jobs, device_qubits=5, min_guard_qubits=1)
        assert len(groups) == 2

    def test_empty_queue(self) -> None:
        groups = group_jobs([], device_qubits=11)
        assert groups == []

    @patch.object(Circuit, "num_qubits", new_callable=PropertyMock, return_value=1)
    def test_single_job_gets_own_group(self, _mock) -> None:
        jobs = [QMTJob(circuit=Circuit().h(0), submitter="a")]
        groups = group_jobs(jobs, device_qubits=11)
        assert len(groups) == 1
        assert len(groups[0]) == 1

    @patch.object(Circuit, "num_qubits", new_callable=PropertyMock, return_value=1)
    def test_many_small_jobs_packed_efficiently(self, _mock) -> None:
        jobs = [QMTJob(circuit=Circuit().h(0), submitter=f"r{i}") for i in range(10)]
        groups = group_jobs(jobs, device_qubits=11, min_guard_qubits=1)
        # Each job: 1 qubit + 1 guard = 2 per job; last doesn't need guard
        # 6 jobs fit: 6*1 + 5*1 = 11
        assert len(groups) == 2
        assert len(groups[0]) == 6
        assert len(groups[1]) == 4

    @patch.object(Circuit, "num_qubits", new_callable=PropertyMock, return_value=1)
    def test_respects_priority_order(self, _mock) -> None:
        jobs = [
            QMTJob(circuit=Circuit().h(0), submitter="low", priority=0),
            QMTJob(circuit=Circuit().h(0), submitter="high", priority=10),
        ]
        groups = group_jobs(jobs, device_qubits=11)
        assert groups[0][0].submitter == "high"
