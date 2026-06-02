"""Synthetic noise profile generator for testing the scheduler without real QPU data.

Generates NoiseProfiles with configurable cross-talk hotspots, drift rates,
and modality-appropriate topology (all-to-all for trapped ion, distance-based
for neutral atom).
"""

from __future__ import annotations

import numpy as np

from marqov.qmt.models import DeviceModality, NoiseProfile


def generate_synthetic_profile(
    device_name: str,
    modality: DeviceModality,
    num_qubits: int,
    *,
    seed: int | None = None,
    hotspot_qubits: list[int] | None = None,
    hotspot_strength: float = 0.1,
    base_error_rate: float = 0.01,
    base_cross_talk: float = 0.02,
    drift_rate: float = 0.0,
) -> NoiseProfile:
    """Generate a synthetic noise profile for scheduler testing.

    Args:
        device_name: Identifier for the synthetic device.
        modality: Hardware modality — determines cross-talk topology.
        num_qubits: Number of qubits on the synthetic device.
        seed: Random seed for reproducibility.
        hotspot_qubits: Qubit indices that form a high-cross-talk cluster.
        hotspot_strength: Cross-talk magnitude within the hotspot cluster.
        base_error_rate: Mean single-qubit error rate.
        base_cross_talk: Baseline cross-talk magnitude.
        drift_rate: Calibration drift rate (stored on the profile).

    Returns:
        A fully populated NoiseProfile.
    """
    rng = np.random.default_rng(seed)
    qubit_error_rates = base_error_rate + rng.uniform(0, base_error_rate * 0.5, num_qubits)

    if modality == DeviceModality.NEUTRAL_ATOM:
        cross_talk = _neutral_atom_cross_talk(num_qubits, base_cross_talk, rng)
    else:
        cross_talk = _trapped_ion_cross_talk(num_qubits, base_cross_talk, rng)

    if hotspot_qubits:
        for i in hotspot_qubits:
            for j in hotspot_qubits:
                if i != j:
                    cross_talk[i, j] = hotspot_strength + rng.uniform(0, 0.02)

    cross_talk = (cross_talk + cross_talk.T) / 2
    np.fill_diagonal(cross_talk, 0.0)

    return NoiseProfile(
        device_name=device_name,
        modality=modality,
        num_qubits=num_qubits,
        cross_talk_matrix=cross_talk,
        qubit_error_rates=qubit_error_rates,
        drift_rate=drift_rate,
    )


def _trapped_ion_cross_talk(
    num_qubits: int, base: float, rng: np.random.Generator
) -> np.ndarray:
    """All-to-all cross-talk with uniform random magnitude."""
    matrix = rng.uniform(base * 0.5, base * 1.5, (num_qubits, num_qubits))
    np.fill_diagonal(matrix, 0.0)
    return matrix


def _neutral_atom_cross_talk(
    num_qubits: int, base: float, rng: np.random.Generator
) -> np.ndarray:
    """Distance-decay cross-talk — nearby qubits couple more strongly."""
    matrix = np.zeros((num_qubits, num_qubits))
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            distance = abs(i - j)
            value = base / (distance**2) + rng.uniform(0, base * 0.1)
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix
