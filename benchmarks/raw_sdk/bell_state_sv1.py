#!/usr/bin/env python3
"""
Bell State benchmark using raw Braket SDK.

Creates a 2-qubit Bell state |Φ+⟩ = (|00⟩ + |11⟩) / √2
Runs on SV1 simulator with 1000 shots.
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

# SV1 simulator ARN
SV1_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"

# S3 bucket for results (Braket requires this)
S3_BUCKET = "amazon-braket-marqov-dev"
S3_PREFIX = "benchmarks"


def create_bell_circuit() -> Circuit:
    """Create a Bell state circuit: H on qubit 0, then CNOT(0, 1)."""
    circuit = Circuit()
    circuit.h(0)      # Hadamard on qubit 0
    circuit.cnot(0, 1)  # CNOT with control=0, target=1
    return circuit


def run_bell_state() -> tuple[dict, float | None]:
    """
    Run Bell state circuit on SV1.

    Returns:
        Tuple of (result_data, quantum_task_duration)
    """
    device = AwsDevice(SV1_ARN)
    circuit = create_bell_circuit()

    # Run with 1000 shots
    task = device.run(
        circuit,
        s3_destination_folder=(S3_BUCKET, S3_PREFIX),
        shots=1000,
    )

    # Wait for result
    result = task.result()

    # Extract measurements
    counts = result.measurement_counts

    # Get quantum task metadata
    metadata = task.metadata()
    quantum_time = None

    # Try to get execution duration from metadata
    if "executionDuration" in metadata:
        # Duration is in milliseconds
        quantum_time = metadata["executionDuration"] / 1000.0

    return {
        "counts": dict(counts),
        "shots": 1000,
        "task_arn": task.id,
    }, quantum_time


def main():
    print("Bell State Benchmark - Raw Braket SDK")
    print("=" * 50)
    print(f"Device: SV1 Simulator")
    print(f"Shots: 1000")
    print(f"Runs: 10")
    print()

    # First, verify we can access the device
    print("Verifying device access...")
    device = AwsDevice(SV1_ARN)
    print(f"Device: {device.name}")
    print(f"Status: {device.status}")
    print()

    # Check S3 bucket exists (Braket needs this)
    print(f"S3 bucket: s3://{S3_BUCKET}/{S3_PREFIX}")
    print()

    # Run benchmarks
    print("Running benchmarks...")
    results = run_benchmark(
        name="bell_state_raw_braket",
        fn=run_bell_state,
        runs=10,
        backend="sv1",
    )

    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_path = save_results(results, output_dir)
    print(f"\nResults saved to: {output_path}")

    # Print summary
    print_summary(results)

    # Show measurement distribution from last successful run
    successful = [r for r in results if r.error is None]
    if successful:
        last_counts = successful[-1].result_data.get("counts", {})
        print(f"\nMeasurement distribution (last run):")
        for state, count in sorted(last_counts.items()):
            pct = count / 10  # 1000 shots, so divide by 10 for percentage
            print(f"  |{state}⟩: {count:4d} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
