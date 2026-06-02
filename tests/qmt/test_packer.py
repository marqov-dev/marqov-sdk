"""Tests for QMT qubit packer."""

from __future__ import annotations

from unittest.mock import PropertyMock, patch

import numpy as np
import pytest

from marqov.circuits import Circuit
from marqov.qmt.models import DeviceModality, NoiseProfile, QMTJob
from marqov.qmt.scheduler.packer import pack_jobs
from marqov.qmt.scheduler.synthetic import generate_synthetic_profile


# Circuit.num_qubits is mocked because conftest stubs quantumflow.


class TestPackJobs:
    @patch.object(Circuit, "num_qubits", new_callable=PropertyMock)
    def test_two_jobs_no_overlap(self, mock_nq) -> None:
        mock_nq.return_value = 2
        profile = generate_synthetic_profile(
            "test", DeviceModality.TRAPPED_ION, 11, seed=42
        )
        jobs = [
            QMTJob(circuit=Circuit().h(0).cnot(0, 1), submitter="a"),
            QMTJob(circuit=Circuit().h(0).cnot(0, 1), submitter="b"),
        ]
        mock_nq.return_value = 2
        plan = pack_jobs(jobs, profile, min_guard_qubits=1)
        all_qubits: list[set[int]] = []
        for mapping in plan.mappings:
            all_qubits.append(mapping.physical_qubits)
        for i in range(len(all_qubits)):
            for j in range(i + 1, len(all_qubits)):
                assert all_qubits[i].isdisjoint(all_qubits[j])

    @patch.object(Circuit, "num_qubits", new_callable=PropertyMock, return_value=2)
    def test_guard_qubits_between_jobs(self, _mock) -> None:
        profile = generate_synthetic_profile(
            "test", DeviceModality.TRAPPED_ION, 11, seed=42
        )
        jobs = [
            QMTJob(circuit=Circuit().h(0).cnot(0, 1), submitter="a"),
            QMTJob(circuit=Circuit().h(0).cnot(0, 1), submitter="b"),
        ]
        plan = pack_jobs(jobs, profile, min_guard_qubits=2)
        a_qubits = plan.mappings[0].physical_qubits
        b_qubits = plan.mappings[1].physical_qubits
        min_distance = min(abs(i - j) for i in a_qubits for j in b_qubits)
        assert min_distance >= 3

    @patch.object(Circuit, "num_qubits", new_callable=PropertyMock, return_value=1)
    def test_avoids_hotspot(self, _mock) -> None:
        profile = generate_synthetic_profile(
            "test",
            DeviceModality.TRAPPED_ION,
            num_qubits=11,
            hotspot_qubits=[4, 5, 6],
            hotspot_strength=0.5,
            seed=42,
        )
        jobs = [
            QMTJob(circuit=Circuit().h(0), submitter="a"),
            QMTJob(circuit=Circuit().h(0), submitter="b"),
        ]
        plan = pack_jobs(jobs, profile, min_guard_qubits=1)
        a_qubits = plan.mappings[0].physical_qubits
        b_qubits = plan.mappings[1].physical_qubits
        all_used = a_qubits | b_qubits
        hotspot = {4, 5, 6}
        assert not (a_qubits <= hotspot and b_qubits <= hotspot)

    @patch.object(Circuit, "num_qubits", new_callable=PropertyMock, return_value=2)
    def test_single_job_plan(self, _mock) -> None:
        profile = generate_synthetic_profile(
            "test", DeviceModality.TRAPPED_ION, 11, seed=42
        )
        jobs = [QMTJob(circuit=Circuit().h(0).cnot(0, 1), submitter="a")]
        plan = pack_jobs(jobs, profile)
        assert len(plan.mappings) == 1
        assert len(plan.mappings[0].physical_qubits) == 2

    @patch.object(Circuit, "num_qubits", new_callable=PropertyMock, return_value=1)
    def test_plan_has_correct_device_name(self, _mock) -> None:
        profile = generate_synthetic_profile(
            "ionq-aria-1", DeviceModality.TRAPPED_ION, 11, seed=42
        )
        jobs = [QMTJob(circuit=Circuit().h(0), submitter="a")]
        plan = pack_jobs(jobs, profile)
        assert plan.device_name == "ionq-aria-1"
