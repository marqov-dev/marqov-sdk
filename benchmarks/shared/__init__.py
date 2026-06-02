"""Shared benchmark components for consistent testing across orchestration frameworks."""

from .vqe_h2 import (
    H2_HAMILTONIAN,
    H2_GROUND_STATE_ENERGY,
    DEFAULT_SHOTS,
    DEFAULT_MAX_ITERATIONS,
    run_vqe_optimization,
    compute_h2_energy,
    compute_expectation_from_counts,
    build_braket_circuit,
    get_pauli_terms,
)

__all__ = [
    'H2_HAMILTONIAN',
    'H2_GROUND_STATE_ENERGY',
    'DEFAULT_SHOTS',
    'DEFAULT_MAX_ITERATIONS',
    'run_vqe_optimization',
    'compute_h2_energy',
    'compute_expectation_from_counts',
    'build_braket_circuit',
    'get_pauli_terms',
]
