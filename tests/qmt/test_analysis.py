"""Tests for noise profile extraction from experiment results."""

from __future__ import annotations

import pytest

from marqov.qmt.characterization.analysis import build_noise_profile
from marqov.qmt.characterization.experiments import (
    CrossTalkExperiment,
    ExperimentResult,
)
from marqov.qmt.models import DeviceModality


def _baseline(target_qubits, fidelity, counts=None):
    if counts is None:
        counts = {"00": 500, "11": 500}
    return ExperimentResult(
        experiment=CrossTalkExperiment(
            target_qubits=target_qubits, neighbor_qubits=[], benchmark="ghz"
        ),
        target_counts=counts,
        target_fidelity=fidelity,
    )


def _multi_tenant(target_qubits, neighbor_qubits, fidelity, counts=None):
    if counts is None:
        counts = {"00": 450, "11": 450, "01": 50, "10": 50}
    return ExperimentResult(
        experiment=CrossTalkExperiment(
            target_qubits=target_qubits, neighbor_qubits=neighbor_qubits, benchmark="ghz",
        ),
        target_counts=counts,
        target_fidelity=fidelity,
    )


class TestBuildNoiseProfile:
    def test_builds_profile_from_paired_results(self) -> None:
        baseline = _baseline([0, 1], fidelity=1.0)
        with_neighbor = _multi_tenant([0, 1], [3, 4], fidelity=0.9)
        profile = build_noise_profile(
            device_name="test-device",
            modality=DeviceModality.TRAPPED_ION,
            num_qubits=6,
            paired_results=[(baseline, with_neighbor)],
        )
        assert profile.device_name == "test-device"
        assert profile.num_qubits == 6
        ct = profile.cross_talk_between({0, 1}, {3, 4})
        assert ct > 0

    def test_no_cross_talk_from_baseline_only_pair(self) -> None:
        baseline = _baseline([0, 1], fidelity=1.0)
        profile = build_noise_profile(
            device_name="test",
            modality=DeviceModality.TRAPPED_ION,
            num_qubits=4,
            paired_results=[(baseline, None)],
        )
        assert profile.cross_talk_between({0, 1}, {2, 3}) == 0.0

    def test_qubit_error_rates_from_baselines(self) -> None:
        baseline = _baseline([0, 1], fidelity=0.9, counts={"00": 450, "11": 450, "01": 50, "10": 50})
        profile = build_noise_profile(
            device_name="test",
            modality=DeviceModality.TRAPPED_ION,
            num_qubits=4,
            paired_results=[(baseline, None)],
        )
        assert profile.qubit_error_rates[0] > 0
        assert profile.qubit_error_rates[1] > 0
        assert profile.qubit_error_rates[2] == 0.0

    def test_drift_normalization_uses_paired_baseline(self) -> None:
        pair1 = (_baseline([0, 1], fidelity=1.0), _multi_tenant([0, 1], [3, 4], fidelity=0.9))
        pair2 = (_baseline([0, 1], fidelity=0.95), _multi_tenant([0, 1], [3, 4], fidelity=0.90))
        profile = build_noise_profile(
            device_name="test",
            modality=DeviceModality.TRAPPED_ION,
            num_qubits=6,
            paired_results=[pair1, pair2],
        )
        ct = profile.cross_talk_between({0, 1}, {3, 4})
        # pair1: 0.1 loss / 4 pairs = 0.025 per pair
        # pair2: 0.05 loss / 4 pairs = 0.0125 per pair
        # Average: 0.01875 per pair, 4 pairs total = 0.075
        assert ct == pytest.approx(0.075, abs=0.001)

    def test_multiple_buffer_distances(self) -> None:
        adjacent = (_baseline([0, 1], fidelity=1.0), _multi_tenant([0, 1], [2, 3], fidelity=0.85))
        distant = (_baseline([0, 1], fidelity=1.0), _multi_tenant([0, 1], [6, 7], fidelity=0.97))
        profile = build_noise_profile(
            device_name="test",
            modality=DeviceModality.TRAPPED_ION,
            num_qubits=10,
            paired_results=[adjacent, distant],
        )
        ct_adjacent = profile.cross_talk_between({0, 1}, {2, 3})
        ct_distant = profile.cross_talk_between({0, 1}, {6, 7})
        assert ct_adjacent > ct_distant
