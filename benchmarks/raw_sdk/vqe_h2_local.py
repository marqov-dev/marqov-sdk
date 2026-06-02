#!/usr/bin/env python3
"""
VQE H₂ benchmark using Braket LocalSimulator (naive implementation).

This is intentionally inefficient - measures each Pauli term separately.
Purpose: Establish worst-case baseline for orchestration comparison.

Uses LocalSimulator which is free and runs locally.
"""

import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

from braket.devices import LocalSimulator
from braket.circuits import Circuit

# Add parent to path for harness import
import sys
sys.path.insert(0, str(Path(__file__).parent))

from harness import BenchmarkResult, save_results, print_summary
from datetime import datetime


# H₂-like Hamiltonian for VQE benchmarking (2-qubit)
# H = g0*I + g1*Z0 + g2*Z1 + g3*Z0Z1 + g4*X0X1 + g5*Y0Y1
#
# Note: These coefficients are for algorithm benchmarking.
# Exact diagonalization gives ground state = -1.8512 Ha
H2_COEFFICIENTS = {
    'identity': -0.4804,
    'z0': 0.3435,
    'z1': -0.4347,
    'z0z1': 0.5716,
    'x0x1': 0.0910,
    'y0y1': 0.0910,
}

# Ground state from exact diagonalization of above Hamiltonian
H2_GROUND_STATE_ENERGY = -1.8512  # Hartree


def create_ansatz(params: np.ndarray) -> Circuit:
    """
    Create parameterized ansatz circuit for H₂ VQE (2 qubits).

    Uses hardware-efficient ansatz: Ry rotations + CNOT entanglement.
    Single parameter theta controls the ansatz.

    For H₂, the optimal state is approximately:
    cos(θ/2)|01⟩ - sin(θ/2)|10⟩ (singlet state)
    """
    circuit = Circuit()

    # Start in |01⟩ (HF reference for H₂)
    circuit.x(1)

    # Parameterized excitation (single parameter for simplicity)
    # This creates superposition between |01⟩ and |10⟩
    theta = params[0]
    circuit.ry(0, theta)
    circuit.ry(1, -theta)
    circuit.cnot(0, 1)

    return circuit


def measure_expectation(circuit: Circuit, pauli_string: str, device, shots: int = 1000) -> float:
    """Measure expectation value of a Pauli string (naive - one circuit per term)."""
    meas_circuit = circuit.copy()

    for i, pauli in enumerate(pauli_string):
        if pauli == 'x':
            meas_circuit.h(i)
        elif pauli == 'y':
            meas_circuit.rx(i, -np.pi/2)

    result = device.run(meas_circuit, shots=shots).result()
    counts = result.measurement_counts

    total = sum(counts.values())
    expectation = 0.0

    for bitstring, count in counts.items():
        parity = 1
        # Braket bitstring: qubit 0 is LAST character (little-endian)
        # Reverse to make qubit 0 first for easier indexing
        bits = bitstring[::-1]
        for i, pauli in enumerate(pauli_string):
            if pauli in ['x', 'y', 'z']:
                if bits[i] == '1':
                    parity *= -1
        expectation += parity * count / total

    return expectation


def compute_energy(params: np.ndarray, device, shots: int = 1000) -> tuple[float, int]:
    """
    Compute H₂ energy (naive - separate circuit for each Pauli term).

    2-qubit Hamiltonian: H = g0*I + g1*Z0 + g2*Z1 + g3*Z0Z1 + g4*X0X1 + g5*Y0Y1

    Returns: (energy, circuit_count)
    """
    circuit = create_ansatz(params)
    energy = H2_COEFFICIENTS['identity']
    circuit_count = 0

    # Single-qubit Z terms
    for i in range(2):
        key = f'z{i}'
        if key in H2_COEFFICIENTS:
            pauli = 'z' + 'i' if i == 0 else 'i' + 'z'
            exp_val = measure_expectation(circuit, pauli, device, shots)
            energy += H2_COEFFICIENTS[key] * exp_val
            circuit_count += 1

    # Two-qubit ZZ term
    if 'z0z1' in H2_COEFFICIENTS:
        exp_val = measure_expectation(circuit, 'zz', device, shots)
        energy += H2_COEFFICIENTS['z0z1'] * exp_val
        circuit_count += 1

    # XX term
    if 'x0x1' in H2_COEFFICIENTS:
        exp_val = measure_expectation(circuit, 'xx', device, shots)
        energy += H2_COEFFICIENTS['x0x1'] * exp_val
        circuit_count += 1

    # YY term
    if 'y0y1' in H2_COEFFICIENTS:
        exp_val = measure_expectation(circuit, 'yy', device, shots)
        energy += H2_COEFFICIENTS['y0y1'] * exp_val
        circuit_count += 1

    return energy, circuit_count


def run_vqe_local():
    """Run naive VQE on LocalSimulator."""
    device = LocalSimulator()

    # Single parameter for 2-qubit ansatz
    n_params = 1
    initial_params = np.array([0.0])  # Start at HF reference

    iteration_data = []
    total_circuits = [0]

    def objective(params):
        iter_start = time.perf_counter()
        energy, circuits = compute_energy(params, device, shots=1000)
        iter_time = time.perf_counter() - iter_start

        total_circuits[0] += circuits
        iteration_data.append({
            'iteration': len(iteration_data) + 1,
            'energy': energy,
            'circuits': circuits,
            'time': iter_time,
        })

        if len(iteration_data) % 10 == 0:
            print(f"    Iter {len(iteration_data)}: E={energy:.4f} Ha, {circuits} circuits, {iter_time:.2f}s")

        return energy

    print("  Starting VQE optimization (50 iterations, naive implementation)...")
    start_time = time.perf_counter()

    result = minimize(
        objective,
        initial_params,
        method='COBYLA',
        options={'maxiter': 50, 'rhobeg': 0.5}
    )

    total_time = time.perf_counter() - start_time

    return {
        'final_energy': result.fun,
        'known_ground_state': H2_GROUND_STATE_ENERGY,
        'energy_error': abs(result.fun - H2_GROUND_STATE_ENERGY),
        'optimal_params': result.x.tolist(),
        'total_iterations': len(iteration_data),
        'total_circuits': total_circuits[0],
        'total_time': total_time,
        'iteration_data': iteration_data,
        'converged': result.success,
        'optimizer_message': result.message,
        'implementation': 'naive',
        'backend': 'local',
    }


def main():
    print("VQE H₂ Benchmark - LocalSimulator (Naive Implementation)")
    print("=" * 60)
    print("This intentionally inefficient implementation measures each")
    print("Pauli term separately (5 circuits per iteration).")
    print("=" * 60)
    print(f"Device: LocalSimulator")
    print(f"Qubits: 2 (parity-reduced encoding)")
    print(f"Optimizer: COBYLA")
    print(f"Max iterations: 50")
    print(f"Shots per measurement: 1000")
    print(f"Pauli terms: 5 (Z0, Z1, Z0Z1, X0X1, Y0Y1)")
    print()

    print("Running naive VQE...")
    start = time.perf_counter()

    try:
        result_data = run_vqe_local()
        wall_time = time.perf_counter() - start
        error = None
    except Exception as e:
        wall_time = time.perf_counter() - start
        result_data = {}
        error = str(e)
        print(f"Error: {e}")

    # Create benchmark result
    results = [BenchmarkResult(
        name="vqe_h2_local_naive",
        run_number=1,
        wall_time_seconds=wall_time,
        quantum_time_seconds=None,
        result_data=result_data,
        timestamp=datetime.utcnow().isoformat(),
        backend="local",
        error=error,
    )]

    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "m0-benchmarking" / "raw-sdk" / "braket" / "vqe-h2"
    output_path = save_results(results, output_dir)
    print(f"\nResults saved to: {output_path}")

    # Print summary
    if error is None:
        print(f"\n{'=' * 60}")
        print("VQE Results (Naive Implementation)")
        print(f"{'=' * 60}")
        print(f"Final energy:      {result_data['final_energy']:.4f} Ha")
        print(f"Known ground state: {result_data['known_ground_state']:.4f} Ha")
        print(f"Energy error:      {result_data['energy_error']:.4f} Ha")
        print(f"Total iterations:  {result_data['total_iterations']}")
        print(f"Total circuits:    {result_data['total_circuits']}")
        print(f"Total time:        {result_data['total_time']:.1f}s")
        print(f"Avg time/iter:     {result_data['total_time']/result_data['total_iterations']:.2f}s")
        print(f"Converged:         {result_data['converged']}")

        # Show what this would cost on SV1
        sv1_time_estimate = result_data['total_circuits'] * 3.5  # ~3.5s per circuit on SV1
        print(f"\nEstimated SV1 time: {sv1_time_estimate/60:.1f} minutes")
        print(f"(Based on ~3.5s API overhead per circuit)")


if __name__ == "__main__":
    main()
