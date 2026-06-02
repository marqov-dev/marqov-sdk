"""Noise profile extraction from cross-talk experiment results.

Takes paired (baseline, multi-tenant) experiment results and builds a
NoiseProfile. Each multi-tenant result is normalized against its specific
paired baseline, enabling drift separation when baselines are interleaved.
"""

from __future__ import annotations

import numpy as np

from marqov.qmt.characterization.experiments import ExperimentResult
from marqov.qmt.models import DeviceModality, NoiseProfile


def build_noise_profile(
    device_name: str,
    modality: DeviceModality,
    num_qubits: int,
    paired_results: list[tuple[ExperimentResult, ExperimentResult | None]],
) -> NoiseProfile:
    qubit_error_rates = np.zeros(num_qubits)
    cross_talk_matrix = np.zeros((num_qubits, num_qubits))
    cross_talk_counts = np.zeros((num_qubits, num_qubits))

    for baseline, multi_tenant in paired_results:
        if baseline.target_fidelity is not None:
            error = 1.0 - baseline.target_fidelity
            for q in baseline.experiment.target_qubits:
                if q < num_qubits:
                    qubit_error_rates[q] = error / len(baseline.experiment.target_qubits)

        if multi_tenant is None or multi_tenant.target_fidelity is None:
            continue
        if baseline.target_fidelity is None:
            continue

        fidelity_loss = max(0.0, baseline.target_fidelity - multi_tenant.target_fidelity)
        target_qs = multi_tenant.experiment.target_qubits
        neighbor_qs = multi_tenant.experiment.neighbor_qubits
        num_pairs = len(target_qs) * len(neighbor_qs)
        if num_pairs == 0:
            continue

        ct_per_pair = fidelity_loss / num_pairs
        for t in target_qs:
            for n in neighbor_qs:
                if t < num_qubits and n < num_qubits:
                    cross_talk_matrix[t, n] += ct_per_pair
                    cross_talk_matrix[n, t] += ct_per_pair
                    cross_talk_counts[t, n] += 1
                    cross_talk_counts[n, t] += 1

    nonzero = cross_talk_counts > 0
    cross_talk_matrix[nonzero] /= cross_talk_counts[nonzero]

    return NoiseProfile(
        device_name=device_name,
        modality=modality,
        num_qubits=num_qubits,
        cross_talk_matrix=cross_talk_matrix,
        qubit_error_rates=qubit_error_rates,
    )
