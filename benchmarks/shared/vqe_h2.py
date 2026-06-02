#!/usr/bin/env python3
"""
Shared VQE H₂ implementation for consistent benchmarking across all approaches.

This module provides the core VQE logic that can be called from any orchestration
framework (raw SDK, Temporal, Covalent Direct, Covalent Hybrid).

Key design decisions:
- 2-qubit ansatz (minimal H₂ representation)
- 5 Pauli terms measured separately (Z0, Z1, Z0Z1, X0X1, Y0Y1)
- scipy COBYLA optimizer for reliable convergence
- 1000 shots per measurement
- Max 30 iterations

This ensures:
1. All benchmarks converge to the same answer
2. Fair comparison of orchestration overhead
3. Realistic VQE workflow (actual optimization, not naive stepping)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from typing import Any, Callable, Protocol


# H₂ Hamiltonian coefficients at R=0.75Å
# Source: O'Malley et al. 2016 (arXiv:1512.06860v2), Table I, R=0.75Å
# Bravyi-Kitaev transformation with symmetry reduction to 2 qubits
# Note: identity coefficient adjusted from O'Malley's +0.2252 to account for
# different energy reference (O'Malley uses separated atoms; this uses ionized atoms)
# CRITICAL: Jordan-Wigner requires BOTH XX and YY for electron hopping
H2_HAMILTONIAN = {
    'identity': -0.4804,    # Constant term
    'z0': 0.3435,          # ZI (Z on qubit 0)
    'z1': -0.4347,         # IZ (Z on qubit 1)
    'z0z1': 0.5716,        # ZZ interaction
    'x0x1': 0.0910,        # XX interaction (electron hopping)
    'y0y1': 0.0910,        # YY interaction (electron hopping - DO NOT REMOVE!)
}

# Exact ground state energy for this Hamiltonian (computed via diagonalization)
# This is the theoretical minimum eigenvalue, NOT from O'Malley paper directly
# (their energy reference differs due to the identity term)
H2_GROUND_STATE_ENERGY = -1.851199  # Hartree

# VQE parameters
DEFAULT_SHOTS = 1000
DEFAULT_MAX_ITERATIONS = 30
DEFAULT_OPTIMIZER_RHOBEG = 0.5


class QuantumDevice(Protocol):
    """Protocol for quantum device interface."""
    def run(self, circuit: Any, **kwargs) -> Any: ...


def create_ansatz(theta: float) -> dict:
    """
    Create ansatz circuit specification for H₂ VQE.

    Implements the Unitary Coupled Cluster (UCC) ansatz from O'Malley et al. 2016:
        |ϕ(θ)⟩ = exp(-iθ X₀Y₁) |01⟩

    Circuit matches O'Malley Figure 1 (page 2):
        1. X(q0) - prepare Hartree-Fock state |01⟩
        2. Basis rotations: Ry(π/2, q1), Rx(-π/2, q0)
        3. CNOT(q1 → q0) - entangle
        4. Rz(θ, q0) - apply rotation
        5. CNOT(q1 → q0) - disentangle
        6. Inverse basis rotations: Ry(-π/2, q1), Rx(π/2, q0)

    Args:
        theta: The variational parameter

    Returns:
        Dict describing the circuit (device-agnostic)
    """
    return {
        'type': 'h2_ucc_ansatz',
        'qubits': 2,
        'theta': theta,
        'gates': [
            ('x', 0),                       # Prepare |01⟩ HF reference
            ('ry', 1, np.pi/2),            # Basis rotation Y→Z
            ('rx', 0, -np.pi/2),           # Basis rotation X→Z
            ('cnot', 1, 0),                # Entangle (control=q1, target=q0)
            ('rz', 0, theta),              # Apply rotation
            ('cnot', 1, 0),                # Disentangle
            ('ry', 1, -np.pi/2),           # Inverse basis rotation
            ('rx', 0, np.pi/2),            # Inverse basis rotation
        ]
    }


def create_measurement_circuit(ansatz_spec: dict, pauli_string: str) -> dict:
    """
    Create circuit for measuring a Pauli string expectation value.

    Args:
        ansatz_spec: The ansatz circuit specification
        pauli_string: 2-character string like 'zi', 'zz', 'xx', 'yy'

    Returns:
        Dict describing the measurement circuit
    """
    circuit = {
        'type': 'measurement',
        'base_ansatz': ansatz_spec,
        'pauli_string': pauli_string,
        'basis_rotations': [],
    }

    for i, pauli in enumerate(pauli_string):
        if pauli == 'x':
            circuit['basis_rotations'].append(('h', i))
        elif pauli == 'y':
            circuit['basis_rotations'].append(('rx', i, -np.pi/2))

    return circuit


def compute_expectation_from_counts(counts: dict[str, int], pauli_string: str) -> float:
    """
    Compute expectation value of a Pauli string from measurement counts.

    Args:
        counts: Measurement counts, e.g. {'00': 480, '01': 20, '10': 15, '11': 485}
        pauli_string: 2-character string like 'zi', 'zz', 'xx', 'yy'

    Returns:
        Expectation value in [-1, 1]
    """
    total = sum(counts.values())
    expectation = 0.0

    for bitstring, count in counts.items():
        parity = 1
        # Braket uses big-endian: bitstring[0] is qubit 0
        for i, pauli in enumerate(pauli_string):
            if pauli in ['x', 'y', 'z']:
                # Eigenvalue is +1 for |0⟩, -1 for |1⟩
                if bitstring[i] == '1':
                    parity *= -1
        expectation += parity * count / total

    return expectation


def compute_h2_energy(
    theta: float,
    measure_pauli: Callable[[str, float], dict[str, int]],
    shots: int = DEFAULT_SHOTS,
) -> tuple[float, dict[str, float]]:
    """
    Compute H₂ energy for a given variational parameter.

    Args:
        theta: Variational parameter
        measure_pauli: Function that measures a Pauli string and returns counts.
                      Signature: measure_pauli(pauli_string, theta) -> counts
        shots: Number of shots per measurement

    Returns:
        Tuple of (energy, expectation_values_dict)
    """
    energy = H2_HAMILTONIAN['identity']
    expectation_values = {}

    # Single-qubit Z terms
    # Z0: measure in Z basis, look at qubit 0
    counts_z0 = measure_pauli('zi', theta)
    exp_z0 = compute_expectation_from_counts(counts_z0, 'zi')
    energy += H2_HAMILTONIAN['z0'] * exp_z0
    expectation_values['z0'] = exp_z0

    # Z1: measure in Z basis, look at qubit 1
    counts_z1 = measure_pauli('iz', theta)
    exp_z1 = compute_expectation_from_counts(counts_z1, 'iz')
    energy += H2_HAMILTONIAN['z1'] * exp_z1
    expectation_values['z1'] = exp_z1

    # ZZ term
    counts_zz = measure_pauli('zz', theta)
    exp_zz = compute_expectation_from_counts(counts_zz, 'zz')
    energy += H2_HAMILTONIAN['z0z1'] * exp_zz
    expectation_values['z0z1'] = exp_zz

    # XX term (requires H gates for X basis measurement)
    counts_xx = measure_pauli('xx', theta)
    exp_xx = compute_expectation_from_counts(counts_xx, 'xx')
    energy += H2_HAMILTONIAN['x0x1'] * exp_xx
    expectation_values['x0x1'] = exp_xx

    # YY term (requires Rx(-π/2) for Y basis measurement)
    counts_yy = measure_pauli('yy', theta)
    exp_yy = compute_expectation_from_counts(counts_yy, 'yy')
    energy += H2_HAMILTONIAN['y0y1'] * exp_yy
    expectation_values['y0y1'] = exp_yy

    return energy, expectation_values


def run_vqe_optimization(
    measure_pauli: Callable[[str, float], dict[str, int]],
    shots: int = DEFAULT_SHOTS,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    initial_theta: float = 0.0,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run VQE optimization to find H₂ ground state energy.

    Args:
        measure_pauli: Function that measures a Pauli string and returns counts.
                      Signature: measure_pauli(pauli_string, theta) -> counts
        shots: Number of shots per measurement
        max_iterations: Maximum optimizer iterations
        initial_theta: Initial variational parameter
        verbose: Print progress

    Returns:
        Dict with optimization results
    """
    iteration_data = []
    circuit_count = [0]

    def objective(params):
        theta = params[0]

        def counted_measure(pauli, t):
            circuit_count[0] += 1
            return measure_pauli(pauli, t)

        energy, exp_vals = compute_h2_energy(theta, counted_measure, shots)

        iteration_data.append({
            'iteration': len(iteration_data) + 1,
            'theta': theta,
            'energy': energy,
            'expectation_values': exp_vals,
            'circuits_so_far': circuit_count[0],
        })

        if verbose:
            print(f"    Iter {len(iteration_data):2d}: θ={theta:+.4f}, E={energy:.6f} Ha")

        return energy

    result = minimize(
        objective,
        np.array([initial_theta]),
        method='COBYLA',
        options={'maxiter': max_iterations, 'rhobeg': DEFAULT_OPTIMIZER_RHOBEG}
    )

    final_energy = result.fun
    optimal_theta = result.x[0]
    energy_error = abs(final_energy - H2_GROUND_STATE_ENERGY)

    return {
        'final_energy': final_energy,
        'optimal_theta': optimal_theta,
        'known_ground_state': H2_GROUND_STATE_ENERGY,
        'energy_error': energy_error,
        'energy_error_mhartree': energy_error * 1000,
        'total_iterations': len(iteration_data),
        'total_circuits': circuit_count[0],
        'converged': result.success,
        'optimizer_message': result.message,
        'iteration_data': iteration_data,
        'hamiltonian': H2_HAMILTONIAN,
    }


# Convenience function for building Braket circuits
def build_braket_circuit(theta: float, pauli_string: str):
    """
    Build a Braket Circuit for measuring a Pauli expectation value.

    Args:
        theta: Variational parameter
        pauli_string: 2-character Pauli string ('zi', 'iz', 'zz', 'xx', 'yy')

    Returns:
        braket.circuits.Circuit
    """
    from braket.circuits import Circuit

    circuit = Circuit()

    # Build ansatz
    circuit.x(1)                    # HF reference |01⟩
    circuit.ry(0, theta)            # Excitation
    circuit.ry(1, -theta)
    circuit.cnot(0, 1)

    # Add basis rotations for measurement
    for i, pauli in enumerate(pauli_string):
        if pauli == 'x':
            circuit.h(i)
        elif pauli == 'y':
            circuit.rx(i, -np.pi/2)

    return circuit


def get_pauli_terms() -> list[str]:
    """Return the list of Pauli terms to measure."""
    return ['zi', 'iz', 'zz', 'xx', 'yy']
