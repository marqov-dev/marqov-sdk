#!/usr/bin/env python3
"""
Bell State benchmark on TN1 Tensor Network Simulator.

Creates a 2-qubit Bell state |Φ+⟩ = (|00⟩ + |11⟩) / √2
Runs on TN1 simulator which uses tensor network methods.

TN1 is optimized for:
- Circuits with low entanglement
- Larger qubit counts (up to 50 qubits)
- Circuits where full state vector would be too large

For a simple 2-qubit Bell state, TN1 is overkill but we include
it for completeness in our simulator benchmarking.

Cost: ~$0.275 per minute (more expensive than SV1/DM1)
"""

import os
from pathlib import Path

from braket.aws import AwsDevice
from braket.circuits import Circuit

# Add parent to path for harness import
import sys
sys.path.insert(0, str(Path(__file__).parent))

from harness import run_benchmark, save_results, print_summary


# Use marqov-dev profile
os.environ.setdefault("AWS_PROFILE", "marqov-dev")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

# TN1 Tensor Network Simulator ARN
TN1_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/tn1"

# S3 bucket for results
S3_BUCKET = "amazon-braket-marqov-dev"
S3_PREFIX = "benchmarks/tn1"


def create_bell_circuit() -> Circuit:
    """Create a Bell state circuit: H on qubit 0, then CNOT(0, 1)."""
    circuit = Circuit()
    circuit.h(0)
    circuit.cnot(0, 1)
    return circuit


def run_bell_state_tn1() -> tuple[dict, float | None]:
    """
    Run Bell state circuit on TN1.

    Returns:
        Tuple of (result_data, quantum_task_duration)
    """
    device = AwsDevice(TN1_ARN)
    circuit = create_bell_circuit()

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
    }, quantum_time


def main():
    print("Bell State Benchmark - TN1 Tensor Network Simulator")
    print("=" * 60)
    print(f"Device: TN1 (Tensor Network Simulator)")
    print(f"Shots: 1000")
    print(f"Runs: 5")
    print(f"Cost: ~$0.275/min (higher than SV1/DM1)")
    print()

    # Verify device access
    print("Verifying device access...")
    device = AwsDevice(TN1_ARN)
    print(f"Device: {device.name}")
    print(f"Status: {device.status}")
    print()

    # Run benchmarks
    print("Running benchmarks...")
    results = run_benchmark(
        name="bell_state_tn1",
        fn=run_bell_state_tn1,
        runs=5,
        backend="tn1",
    )

    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "m0-benchmarking" / "tn1" / "bell-state"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_results(results, output_dir)
    print(f"\nResults saved to: {output_path}")

    # Print summary
    print_summary(results)

    # Show fidelity
    successful = [r for r in results if r.error is None]
    if successful:
        fidelities = [r.result_data.get("fidelity", 0) for r in successful]
        avg_fidelity = sum(fidelities) / len(fidelities)
        print(f"\nAverage fidelity: {avg_fidelity:.1%}")

        # Show measurement distribution from last run
        last_counts = successful[-1].result_data.get("counts", {})
        if last_counts:
            print(f"\nMeasurement distribution (last run):")
            total = sum(last_counts.values())
            for state, count in sorted(last_counts.items()):
                pct = count / total * 100
                print(f"  |{state}⟩: {count:4d} ({pct:.1f}%)")

    # Comparison
    print("\n" + "=" * 60)
    print("Simulator Comparison:")
    print("=" * 60)
    print(f"SV1 (State Vector):     3.6s mean, 100% fidelity, $0.075/min")
    print(f"DM1 (Density Matrix):   3.5s mean, 100% fidelity, $0.075/min")
    if successful:
        wall_times = [r.wall_time_seconds for r in successful]
        mean_time = sum(wall_times) / len(wall_times)
        print(f"TN1 (Tensor Network):   {mean_time:.1f}s mean, {avg_fidelity:.0%} fidelity, $0.275/min")


if __name__ == "__main__":
    main()
