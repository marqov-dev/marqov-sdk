#!/usr/bin/env python3
"""
Bell State benchmark on DM1 Density Matrix Simulator.

Creates a 2-qubit Bell state |Φ+⟩ = (|00⟩ + |11⟩) / √2
Runs on DM1 simulator with noise model to simulate real hardware.

DM1 supports noise models, allowing us to simulate:
- Depolarizing noise (random Pauli errors)
- Bit-flip and phase-flip errors
- Amplitude damping (energy relaxation)

Issue: Part of simulator benchmarking progression
"""

import os
from pathlib import Path
from datetime import datetime

from braket.aws import AwsDevice
from braket.circuits import Circuit, Noise

# Add parent to path for harness import
import sys
sys.path.insert(0, str(Path(__file__).parent))

from harness import run_benchmark, save_results, print_summary


# Use marqov-dev profile
os.environ.setdefault("AWS_PROFILE", "marqov-dev")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

# DM1 Density Matrix Simulator ARN
DM1_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/dm1"

# S3 bucket for results
S3_BUCKET = "amazon-braket-marqov-dev"
S3_PREFIX = "benchmarks/dm1"

# Noise parameters (approximate real hardware error rates)
# Typical superconducting qubit error rates: 0.1% - 1% per gate
DEPOLARIZING_RATE = 0.005  # 0.5% depolarizing noise per gate
READOUT_ERROR_RATE = 0.01  # 1% readout error


def create_bell_circuit_with_noise() -> Circuit:
    """
    Create a Bell state circuit with noise model.

    Applies depolarizing noise after each gate to simulate
    real hardware behavior.
    """
    circuit = Circuit()

    # Hadamard on qubit 0
    circuit.h(0)
    circuit.apply_gate_noise(Noise.Depolarizing(probability=DEPOLARIZING_RATE), target_qubits=[0])

    # CNOT with control=0, target=1
    circuit.cnot(0, 1)
    # Apply depolarizing to both qubits after 2-qubit gate
    circuit.apply_gate_noise(Noise.Depolarizing(probability=DEPOLARIZING_RATE), target_qubits=[0, 1])

    # Readout errors (bit flips before measurement)
    circuit.apply_readout_noise(Noise.BitFlip(probability=READOUT_ERROR_RATE), target_qubits=[0, 1])

    return circuit


def create_bell_circuit_clean() -> Circuit:
    """Create a Bell state circuit without noise (for comparison)."""
    circuit = Circuit()
    circuit.h(0)
    circuit.cnot(0, 1)
    return circuit


def run_bell_state_dm1_noisy() -> tuple[dict, float | None]:
    """
    Run Bell state circuit on DM1 with noise.

    Returns:
        Tuple of (result_data, quantum_task_duration)
    """
    device = AwsDevice(DM1_ARN)
    circuit = create_bell_circuit_with_noise()

    task = device.run(
        circuit,
        s3_destination_folder=(S3_BUCKET, S3_PREFIX),
        shots=1000,
    )

    result = task.result()
    counts = result.measurement_counts

    metadata = task.metadata()
    quantum_time = None
    if "executionDuration" in metadata:
        quantum_time = metadata["executionDuration"] / 1000.0

    # Calculate fidelity
    total = sum(counts.values())
    bell_states = counts.get("00", 0) + counts.get("11", 0)
    fidelity = bell_states / total

    return {
        "counts": dict(counts),
        "shots": 1000,
        "task_arn": task.id,
        "fidelity": fidelity,
        "noise_model": {
            "depolarizing_rate": DEPOLARIZING_RATE,
            "readout_error_rate": READOUT_ERROR_RATE,
        },
    }, quantum_time


def run_bell_state_dm1_clean() -> tuple[dict, float | None]:
    """
    Run Bell state circuit on DM1 without noise (baseline).

    Returns:
        Tuple of (result_data, quantum_task_duration)
    """
    device = AwsDevice(DM1_ARN)
    circuit = create_bell_circuit_clean()

    task = device.run(
        circuit,
        s3_destination_folder=(S3_BUCKET, S3_PREFIX),
        shots=1000,
    )

    result = task.result()
    counts = result.measurement_counts

    metadata = task.metadata()
    quantum_time = None
    if "executionDuration" in metadata:
        quantum_time = metadata["executionDuration"] / 1000.0

    total = sum(counts.values())
    bell_states = counts.get("00", 0) + counts.get("11", 0)
    fidelity = bell_states / total

    return {
        "counts": dict(counts),
        "shots": 1000,
        "task_arn": task.id,
        "fidelity": fidelity,
        "noise_model": None,
    }, quantum_time


def main():
    print("Bell State Benchmark - DM1 Density Matrix Simulator")
    print("=" * 60)
    print(f"Device: DM1 (Density Matrix Simulator)")
    print(f"Shots: 1000")
    print(f"Runs: 5 clean + 5 noisy")
    print()
    print("Noise Model (for noisy runs):")
    print(f"  Depolarizing rate: {DEPOLARIZING_RATE*100:.1f}% per gate")
    print(f"  Readout error rate: {READOUT_ERROR_RATE*100:.1f}%")
    print()

    # Verify device access
    print("Verifying device access...")
    device = AwsDevice(DM1_ARN)
    print(f"Device: {device.name}")
    print(f"Status: {device.status}")
    print()

    # Run clean benchmarks first
    print("Running clean (no noise) benchmarks...")
    clean_results = run_benchmark(
        name="bell_state_dm1_clean",
        fn=run_bell_state_dm1_clean,
        runs=5,
        backend="dm1-clean",
    )

    # Run noisy benchmarks
    print("\nRunning noisy benchmarks...")
    noisy_results = run_benchmark(
        name="bell_state_dm1_noisy",
        fn=run_bell_state_dm1_noisy,
        runs=5,
        backend="dm1-noisy",
    )

    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "m0-benchmarking" / "dm1" / "bell-state"
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_path = save_results(clean_results, output_dir, suffix="_clean")
    noisy_path = save_results(noisy_results, output_dir, suffix="_noisy")

    print(f"\nClean results saved to: {clean_path}")
    print(f"Noisy results saved to: {noisy_path}")

    # Print summaries
    print("\n" + "=" * 60)
    print("CLEAN (No Noise) Results:")
    print("=" * 60)
    print_summary(clean_results)

    clean_successful = [r for r in clean_results if r.error is None]
    if clean_successful:
        fidelities = [r.result_data.get("fidelity", 0) for r in clean_successful]
        avg_fidelity = sum(fidelities) / len(fidelities)
        print(f"\nAverage fidelity: {avg_fidelity:.1%}")

    print("\n" + "=" * 60)
    print("NOISY Results:")
    print("=" * 60)
    print_summary(noisy_results)

    noisy_successful = [r for r in noisy_results if r.error is None]
    if noisy_successful:
        fidelities = [r.result_data.get("fidelity", 0) for r in noisy_successful]
        avg_fidelity = sum(fidelities) / len(fidelities)
        print(f"\nAverage fidelity: {avg_fidelity:.1%}")

        # Show measurement distribution from last noisy run
        last_counts = noisy_successful[-1].result_data.get("counts", {})
        if last_counts:
            print(f"\nMeasurement distribution (last noisy run):")
            total = sum(last_counts.values())
            for state, count in sorted(last_counts.items()):
                pct = count / total * 100
                print(f"  |{state}⟩: {count:4d} ({pct:.1f}%)")

    # Compare to real hardware
    print("\n" + "=" * 60)
    print("Comparison:")
    print("=" * 60)
    print(f"DM1 clean fidelity:     100% (expected)")
    if noisy_successful:
        noisy_fidelity = sum(r.result_data.get("fidelity", 0) for r in noisy_successful) / len(noisy_successful)
        print(f"DM1 noisy fidelity:     {noisy_fidelity:.1%}")
    print(f"Rigetti Ankaa-3:        95.0% (from real hardware)")
    print(f"IonQ Forte-1:           ~99% (expected, pending)")


if __name__ == "__main__":
    main()
