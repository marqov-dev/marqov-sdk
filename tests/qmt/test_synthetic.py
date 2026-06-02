"""Tests for synthetic noise profile generator."""

from __future__ import annotations

import numpy as np
import pytest

from marqov.qmt.models import DeviceModality
from marqov.qmt.scheduler.synthetic import generate_synthetic_profile


class TestSyntheticProfile:
    def test_trapped_ion_profile_shape(self) -> None:
        profile = generate_synthetic_profile(
            device_name="test-ion",
            modality=DeviceModality.TRAPPED_ION,
            num_qubits=11,
            seed=42,
        )
        assert profile.num_qubits == 11
        assert profile.cross_talk_matrix.shape == (11, 11)
        assert profile.qubit_error_rates.shape == (11,)

    def test_neutral_atom_distance_decay(self) -> None:
        profile = generate_synthetic_profile(
            device_name="test-atom",
            modality=DeviceModality.NEUTRAL_ATOM,
            num_qubits=20,
            seed=42,
        )
        adjacent = profile.cross_talk_matrix[0, 1]
        distant = profile.cross_talk_matrix[0, 10]
        assert adjacent > distant

    def test_hotspot_injection(self) -> None:
        profile = generate_synthetic_profile(
            device_name="test",
            modality=DeviceModality.TRAPPED_ION,
            num_qubits=10,
            hotspot_qubits=[3, 4, 5],
            hotspot_strength=0.2,
            seed=42,
        )
        hotspot_ct = profile.cross_talk_matrix[3, 4]
        normal_ct = profile.cross_talk_matrix[0, 1]
        assert hotspot_ct > normal_ct

    def test_symmetric_matrix(self) -> None:
        profile = generate_synthetic_profile(
            device_name="test",
            modality=DeviceModality.TRAPPED_ION,
            num_qubits=8,
            seed=42,
        )
        np.testing.assert_array_almost_equal(
            profile.cross_talk_matrix,
            profile.cross_talk_matrix.T,
        )

    def test_zero_diagonal(self) -> None:
        profile = generate_synthetic_profile(
            device_name="test",
            modality=DeviceModality.NEUTRAL_ATOM,
            num_qubits=8,
            seed=42,
        )
        np.testing.assert_array_equal(
            np.diag(profile.cross_talk_matrix),
            np.zeros(8),
        )

    def test_seed_reproducibility(self) -> None:
        p1 = generate_synthetic_profile("test", DeviceModality.TRAPPED_ION, 10, seed=99)
        p2 = generate_synthetic_profile("test", DeviceModality.TRAPPED_ION, 10, seed=99)
        np.testing.assert_array_equal(p1.cross_talk_matrix, p2.cross_talk_matrix)

    def test_drift_rate_set(self) -> None:
        profile = generate_synthetic_profile(
            device_name="test",
            modality=DeviceModality.TRAPPED_ION,
            num_qubits=5,
            drift_rate=0.02,
            seed=42,
        )
        assert profile.drift_rate == 0.02
