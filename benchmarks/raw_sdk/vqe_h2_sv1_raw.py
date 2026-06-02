#!/usr/bin/env python3
"""
VQE H₂ benchmark using raw Braket SDK.

Implements Variational Quantum Eigensolver for H₂ ground state energy.
Uses 4-qubit circuit with UCCSD-inspired ansatz, COBYLA optimizer, 50 iterations.
"""

import os
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

from braket.aws import AwsDevice
from braket.circuits import Circuit

# Add parent to path for harness import
import sys
sys.path.insert(0, str(Path(__file__).parent))

from harness import run_benchmark, save_results, print_summary


# Use marqov-dev profile
os.environ.setdefault("AWS_PROFILE", "marqov-dev")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

# SV1 simulator ARN
SV1_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"

# S3 bucket for results
S3_BUCKET = "amazon-braket-marqov-dev"
S3_PREFIX = "benchmarks"

# H₂ Hamiltonian coefficients (at equilibrium bond length ~0.735 Å)
# Simplified 4-qubit representation in Jordan-Wigner encoding
# H = g0*I + g1*Z0 + g2*Z1 + g3*Z2 + g4*Z3 + g5*Z0Z1 + g6*Z0Z2 + g7*Z1Z3 + g8*Z2Z3 + ...
# Using pre-computed coefficients for H₂ at equilibrium
H2_COEFFICIENTS = {
    'identity': -0.8105,      # Constant term
    'z0': 0.1721,
    'z1': 0.1721,
    'z2': -0.2232,
    'z3': -0.2232,
    'z0z1': 0.1686,
    'z0z2': 0.1205,
    'z0z3': 0.1659,
    'z1z2': 0.1659,
    'z1z3': 0.1205,
    'z2z3': 0.1743,
    'x0x1y2y3': -0.0453,
    'x0y1y2x3': 0.0453,
    'y0x1x2y3': 0.0453,
    'y0y1x2x3': -0.0453,
}

# Known ground state energy for H₂ at this geometry (for validation)
H2_GROUND_STATE_ENERGY = -1.1372  # Hartree


def create_ansatz(params: np.ndarray) -> Circuit:
    """
    Create parameterized ansatz circuit for H₂ VQE.

    Uses a hardware-efficient ansatz with Ry rotations and CNOT entanglement.
    4 qubits, 2 layers = 8 parameters.
    """
    circuit = Circuit()
    n_qubits = 4
    n_layers = 2
    param_idx = 0

    for layer in range(n_layers):
        # Ry rotation on each qubit
        for q in range(n_qubits):
            circuit.ry(q, params[param_idx])
            param_idx += 1

        # Entangling CNOTs (linear connectivity)
        if layer < n_layers - 1:  # Skip entanglement on last layer
            for q in range(n_qubits - 1):
                circuit.cnot(q, q + 1)

    return circuit


def measure_expectation(circuit: Circuit, pauli_string: str, device: AwsDevice, shots: int = 1000) -> float:
    """
    Measure expectation value of a Pauli string.

    For Z measurements, we can measure directly.
    For X/Y, we need basis rotation.
    """
    meas_circuit = circuit.copy()

    # Apply basis rotations for X and Y measurements
    for i, pauli in enumerate(pauli_string):
        if pauli == 'x':
            meas_circuit.h(i)  # X basis: H then measure
        elif pauli == 'y':
            meas_circuit.rx(i, -np.pi/2)  # Y basis: Rx(-π/2) then measure
        # Z basis: no rotation needed

    # Run circuit
    task = device.run(
        meas_circuit,
        s3_destination_folder=(S3_BUCKET, S3_PREFIX),
        shots=shots,
    )
    result = task.result()
    counts = result.measurement_counts

    # Calculate expectation value
    # For Pauli Z: eigenvalue is +1 for |0⟩, -1 for |1⟩
    total = sum(counts.values())
    expectation = 0.0

    for bitstring, count in counts.items():
        # Calculate parity of relevant qubits
        parity = 1
        for i, pauli in enumerate(pauli_string):
            if pauli in ['x', 'y', 'z']:
                # Bit ordering: bitstring[0] is qubit 0
                if bitstring[i] == '1':
                    parity *= -1
        expectation += parity * count / total

    return expectation


def compute_energy(params: np.ndarray, device: AwsDevice, shots: int = 1000) -> float:
    """
    Compute H₂ energy expectation value for given parameters.

    This is the cost function for VQE optimization.
    """
    circuit = create_ansatz(params)
    energy = H2_COEFFICIENTS['identity']

    # Single-qubit Z terms
    for i in range(4):
        key = f'z{i}'
        if key in H2_COEFFICIENTS:
            exp_val = measure_expectation(circuit, f"{'i'*i}z{'i'*(3-i)}", device, shots)
            energy += H2_COEFFICIENTS[key] * exp_val

    # Two-qubit ZZ terms
    zz_terms = [('z0z1', 'zzii'), ('z0z2', 'zizi'), ('z0z3', 'ziiz'),
                ('z1z2', 'izzi'), ('z1z3', 'iziz'), ('z2z3', 'iizz')]
    for key, pauli in zz_terms:
        if key in H2_COEFFICIENTS:
            exp_val = measure_expectation(circuit, pauli, device, shots)
            energy += H2_COEFFICIENTS[key] * exp_val

    # XXYY and similar terms (these capture electron correlation)
    # x0x1y2y3 means X⊗X⊗Y⊗Y
    xxxx_terms = [
        ('x0x1y2y3', 'xxyy'),
        ('x0y1y2x3', 'xyyx'),
        ('y0x1x2y3', 'yxxy'),
        ('y0y1x2x3', 'yyxx'),
    ]
    for key, pauli in xxxx_terms:
        if key in H2_COEFFICIENTS:
            exp_val = measure_expectation(circuit, pauli, device, shots)
            energy += H2_COEFFICIENTS[key] * exp_val

    return energy


def run_vqe_h2() -> tuple[dict, float | None]:
    """
    Run VQE for H₂ ground state on SV1.

    Returns:
        Tuple of (result_data, quantum_task_duration)
    """
    device = AwsDevice(SV1_ARN)

    # Initialize parameters (8 params for 4 qubits, 2 layers)
    n_params = 8
    # Use known good starting point near ground state
    initial_params = np.array([0.1, 0.1, -0.5, 0.5, 0.2, -0.2, 0.3, -0.3])

    # Track optimization progress
    iteration_energies = []
    iteration_count = [0]  # Use list to allow modification in closure

    def objective(params):
        energy = compute_energy(params, device, shots=1000)
        iteration_count[0] += 1
        iteration_energies.append(energy)
        if iteration_count[0] % 5 == 0:
            print(f"    Iteration {iteration_count[0]}: E = {energy:.4f} Ha")
        return energy

    # Run COBYLA optimizer (reduced iterations for benchmarking)
    print(f"  Starting VQE optimization (max 20 iterations)...")
    result = minimize(
        objective,
        initial_params,
        method='COBYLA',
        options={'maxiter': 20, 'rhobeg': 0.3}
    )

    final_energy = result.fun
    optimal_params = result.x.tolist()

    # Calculate error from known ground state
    error = abs(final_energy - H2_GROUND_STATE_ENERGY)

    return {
        'final_energy': final_energy,
        'known_ground_state': H2_GROUND_STATE_ENERGY,
        'energy_error': error,
        'optimal_params': optimal_params,
        'iterations': len(iteration_energies),
        'iteration_energies': iteration_energies,
        'converged': result.success,
        'optimizer_message': result.message,
    }, None  # SV1 doesn't report quantum time


def main():
    print("VQE H₂ Benchmark - Raw Braket SDK")
    print("=" * 50)
    print(f"Device: SV1 Simulator")
    print(f"Qubits: 4")
    print(f"Optimizer: COBYLA")
    print(f"Max iterations: 20")
    print(f"Shots per expectation: 1000")
    print(f"Runs: 1")  # Single run - VQE is very expensive
    print()

    # Verify device access
    print("Verifying device access...")
    device = AwsDevice(SV1_ARN)
    print(f"Device: {device.name}")
    print(f"Status: {device.status}")
    print()

    # Run benchmarks (single run due to cost - VQE is very expensive)
    print("Running VQE benchmarks...")
    results = run_benchmark(
        name="vqe_h2_raw_braket",
        fn=run_vqe_h2,
        runs=1,  # VQE is very expensive - each iteration is ~15 circuit runs
        backend="sv1",
    )

    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "m0-benchmarking" / "raw-sdk" / "braket" / "vqe-h2"
    output_path = save_results(results, output_dir)
    print(f"\nResults saved to: {output_path}")

    # Print summary
    print_summary(results)

    # Show VQE-specific results
    successful = [r for r in results if r.error is None]
    if successful:
        print(f"\nVQE Results:")
        for r in successful:
            data = r.result_data
            print(f"  Run {r.run_number}:")
            print(f"    Final energy: {data['final_energy']:.4f} Ha")
            print(f"    Known ground state: {data['known_ground_state']:.4f} Ha")
            print(f"    Error: {data['energy_error']:.4f} Ha")
            print(f"    Iterations: {data['iterations']}")
            print(f"    Converged: {data['converged']}")


if __name__ == "__main__":
    main()
