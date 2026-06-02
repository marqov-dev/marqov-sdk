"""Tests for QMT shared data contract."""

from __future__ import annotations

from unittest.mock import PropertyMock, patch

import numpy as np
import pytest

from marqov.circuits import Circuit
from marqov.qmt.models import (
    DeviceModality,
    NoiseProfile,
    PackingPlan,
    PackingResult,
    QMTJob,
    QubitMapping,
)


# Circuit.num_qubits returns a mock object in tests because conftest stubs
# quantumflow. We patch it for tests that assert on qubit counts.


class TestQMTJob:
    @patch.object(Circuit, "num_qubits", new_callable=PropertyMock, return_value=2)
    def test_create_job(self, _mock: PropertyMock) -> None:
        circuit = Circuit().h(0).cnot(0, 1)
        job = QMTJob(circuit=circuit, submitter="researcher-1")
        assert job.num_qubits == 2
        assert job.submitter == "researcher-1"
        assert job.priority == 0
        assert job.job_id is not None

    def test_job_id_unique(self) -> None:
        c = Circuit().h(0)
        job_a = QMTJob(circuit=c, submitter="a")
        job_b = QMTJob(circuit=c, submitter="b")
        assert job_a.job_id != job_b.job_id

    @patch.object(Circuit, "num_qubits", new_callable=PropertyMock, return_value=3)
    def test_num_qubits_from_circuit(self, _mock: PropertyMock) -> None:
        circuit = Circuit().h(0).h(1).h(2)
        job = QMTJob(circuit=circuit, submitter="test")
        assert job.num_qubits == 3


class TestNoiseProfile:
    def test_create_empty_profile(self) -> None:
        profile = NoiseProfile(
            device_name="ionq-aria-1",
            modality=DeviceModality.TRAPPED_ION,
            num_qubits=11,
        )
        assert profile.device_name == "ionq-aria-1"
        assert profile.num_qubits == 11
        assert profile.cross_talk_matrix.shape == (11, 11)
        assert profile.qubit_error_rates.shape == (11,)

    def test_cross_talk_between_qubits(self) -> None:
        profile = NoiseProfile(
            device_name="test",
            modality=DeviceModality.TRAPPED_ION,
            num_qubits=4,
            cross_talk_matrix=np.array([
                [0.0, 0.05, 0.01, 0.00],
                [0.05, 0.0, 0.05, 0.01],
                [0.01, 0.05, 0.0, 0.05],
                [0.00, 0.01, 0.05, 0.0],
            ]),
        )
        assert profile.cross_talk_between({0, 1}, {2, 3}) == pytest.approx(0.01 + 0.01 + 0.00 + 0.05)

    def test_is_stale(self) -> None:
        from datetime import datetime, timedelta, timezone

        old_time = datetime.now(timezone.utc) - timedelta(hours=25)
        profile = NoiseProfile(
            device_name="test",
            modality=DeviceModality.TRAPPED_ION,
            num_qubits=4,
            timestamp=old_time,
            max_age_hours=24.0,
        )
        assert profile.is_stale is True

    def test_not_stale(self) -> None:
        profile = NoiseProfile(
            device_name="test",
            modality=DeviceModality.TRAPPED_ION,
            num_qubits=4,
            max_age_hours=24.0,
        )
        assert profile.is_stale is False


class TestPackingPlan:
    def test_create_plan(self) -> None:
        c1 = Circuit().h(0)
        c2 = Circuit().h(0).cnot(0, 1)
        job_a = QMTJob(circuit=c1, submitter="a")
        job_b = QMTJob(circuit=c2, submitter="b")

        plan = PackingPlan(
            jobs=[job_a, job_b],
            mappings=[
                QubitMapping(job_id=job_a.job_id, logical_to_physical={0: 0}),
                QubitMapping(job_id=job_b.job_id, logical_to_physical={0: 3, 1: 4}),
            ],
            guard_qubits={1, 2},
            device_name="ionq-aria-1",
            total_qubits=5,
        )
        assert plan.total_qubits == 5
        assert len(plan.jobs) == 2
        assert plan.guard_qubits == {1, 2}

    def test_physical_qubits_used(self) -> None:
        c = Circuit().h(0)
        job = QMTJob(circuit=c, submitter="a")
        plan = PackingPlan(
            jobs=[job],
            mappings=[QubitMapping(job_id=job.job_id, logical_to_physical={0: 5})],
            guard_qubits=set(),
            device_name="test",
            total_qubits=11,
        )
        assert plan.physical_qubits_used == {5}

    def test_no_overlap_validation(self) -> None:
        c = Circuit().h(0)
        job_a = QMTJob(circuit=c, submitter="a")
        job_b = QMTJob(circuit=c, submitter="b")
        with pytest.raises(ValueError, match="overlap"):
            PackingPlan(
                jobs=[job_a, job_b],
                mappings=[
                    QubitMapping(job_id=job_a.job_id, logical_to_physical={0: 0}),
                    QubitMapping(job_id=job_b.job_id, logical_to_physical={0: 0}),
                ],
                guard_qubits=set(),
                device_name="test",
                total_qubits=2,
            )


class TestPackingResult:
    def test_fidelity_ratio(self) -> None:
        result = PackingResult(
            job_id="job-1",
            counts={"00": 480, "11": 520},
            shots=1000,
            single_tenant_fidelity=0.99,
            multi_tenant_fidelity=0.95,
        )
        assert result.fidelity_ratio == pytest.approx(0.95 / 0.99, rel=1e-3)

    def test_fidelity_ratio_none_when_no_baseline(self) -> None:
        result = PackingResult(
            job_id="job-1",
            counts={"0": 1000},
            shots=1000,
        )
        assert result.fidelity_ratio is None
