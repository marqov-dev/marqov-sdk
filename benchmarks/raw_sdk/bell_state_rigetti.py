#!/usr/bin/env python3
"""
Bell State benchmark on Rigetti Ankaa-3 QPU.

Creates a 2-qubit Bell state |Φ+⟩ = (|00⟩ + |11⟩) / √2
Runs on real Rigetti superconducting quantum hardware.

COST WARNING: ~$1.20 per circuit (1000 shots)
Total estimated cost for 5 runs: ~$6.00

Issue: #171
"""

import os
import time
from pathlib import Path
from datetime import datetime

import boto3
from braket.aws import AwsDevice, AwsSession
from braket.circuits import Circuit

# Add parent to path for harness import
import sys
sys.path.insert(0, str(Path(__file__).parent))

from harness import BenchmarkResult, save_results, print_summary


# Use marqov-dev profile - Rigetti is in us-west-1
os.environ.setdefault("AWS_PROFILE", "marqov-dev")

# Rigetti Ankaa-3 ARN (us-west-1)
RIGETTI_ARN = "arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3"

# S3 bucket for results - must be in same region as device
S3_BUCKET = "amazon-braket-marqov-dev-west"
S3_PREFIX = "benchmarks/rigetti"


def get_aws_session():
    """Create AWS session for us-west-1 (Rigetti region)."""
    boto_session = boto3.Session(profile_name="marqov-dev", region_name="us-west-1")
    return AwsSession(boto_session=boto_session)


def create_bell_circuit() -> Circuit:
    """Create a Bell state circuit: H on qubit 0, then CNOT(0, 1)."""
    circuit = Circuit()
    circuit.h(0)
    circuit.cnot(0, 1)
    return circuit


def run_bell_state_rigetti(shots: int = 1000) -> tuple[dict, float | None]:
    """
    Run Bell state circuit on Rigetti Ankaa-3.

    Returns:
        Tuple of (result_data, quantum_task_duration)
    """
    aws_session = get_aws_session()
    device = AwsDevice(RIGETTI_ARN, aws_session=aws_session)
    circuit = create_bell_circuit()

    # Submit task
    submit_time = time.perf_counter()
    task = device.run(
        circuit,
        s3_destination_folder=(S3_BUCKET, S3_PREFIX),
        shots=shots,
    )

    # Wait for result
    result = task.result()
    total_time = time.perf_counter() - submit_time

    # Extract measurements
    counts = result.measurement_counts

    # Get task metadata
    metadata = task.metadata()
    quantum_time = None
    queue_time = None

    # Try to extract timing info
    if "executionDuration" in metadata:
        quantum_time = metadata["executionDuration"] / 1000.0

    # Calculate queue time (total - execution)
    if quantum_time:
        queue_time = total_time - quantum_time

    return {
        "counts": dict(counts),
        "shots": shots,
        "task_arn": task.id,
        "device": RIGETTI_ARN,
        "queue_time": queue_time,
        "quantum_time": quantum_time,
        "total_time": total_time,
    }, quantum_time


def main():
    print("Bell State Benchmark - Rigetti Ankaa-3 QPU")
    print("=" * 55)
    print(f"Device: Rigetti Ankaa-3 (superconducting, 82 qubits)")
    print(f"Region: us-west-1")
    print(f"Shots: 1000")
    print(f"Runs: 5")
    print()
    print("COST WARNING:")
    print(f"  Per-task fee: $0.30")
    print(f"  Per-shot fee: $0.00090 × 1000 = $0.90")
    print(f"  Total per run: ~$1.20")
    print(f"  Estimated total: ~$6.00")
    print()

    # Verify device access
    print("Verifying device access...")
    aws_session = get_aws_session()

    try:
        device = AwsDevice(RIGETTI_ARN, aws_session=aws_session)
        print(f"Device: {device.name}")
        print(f"Status: {device.status}")
        print(f"Qubits: {device.properties.paradigm.qubitCount}")

        queue_info = device.queue_depth()
        print(f"Queue depth: Normal={queue_info.quantum_tasks.get('Normal', 'N/A')}, Priority={queue_info.quantum_tasks.get('Priority', 'N/A')}")
        print()

        if device.status != "ONLINE":
            print(f"ERROR: Device is {device.status}, not ONLINE")
            print("Cannot run benchmark on offline device.")
            return

    except Exception as e:
        print(f"ERROR: Could not access device: {e}")
        return

    # Confirm before running (costs real money)
    print("=" * 55)
    response = input("Proceed with benchmark? This will cost ~$6. [y/N]: ")
    if response.lower() != 'y':
        print("Benchmark cancelled.")
        return
    print()

    # Run benchmarks
    print("Running benchmarks on Rigetti Ankaa-3...")
    print("(Each run submits to real QPU and waits for execution)")
    print()

    results = []
    for i in range(5):
        print(f"  Run {i + 1}/5...", end=" ", flush=True)

        start = time.perf_counter()
        try:
            result_data, quantum_time = run_bell_state_rigetti(shots=1000)
            wall_time = time.perf_counter() - start
            error = None

            queue_time = result_data.get('queue_time', 0) or 0
            print(f"done ({wall_time:.1f}s total, ~{queue_time:.1f}s queue)")

        except Exception as e:
            wall_time = time.perf_counter() - start
            result_data = {"error_details": str(e)}
            quantum_time = None
            error = str(e)
            print(f"error: {e}")

        results.append(BenchmarkResult(
            name="bell_state_rigetti",
            run_number=i + 1,
            wall_time_seconds=wall_time,
            quantum_time_seconds=quantum_time,
            result_data=result_data,
            timestamp=datetime.now().isoformat(),
            backend="rigetti-ankaa3",
            error=error,
        ))

    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "m0-benchmarking" / "rigetti" / "bell-state"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_results(results, output_dir)
    print(f"\nResults saved to: {output_path}")

    # Print summary
    print_summary(results)

    # Show measurement distribution from last successful run
    successful = [r for r in results if r.error is None]
    if successful:
        last_counts = successful[-1].result_data.get("counts", {})
        if last_counts:
            print(f"\nMeasurement distribution (last run):")
            total_shots = sum(last_counts.values())
            for state, count in sorted(last_counts.items()):
                pct = count / total_shots * 100
                print(f"  |{state}⟩: {count:4d} ({pct:.1f}%)")

            # Calculate fidelity estimate
            bell_states = last_counts.get("00", 0) + last_counts.get("11", 0)
            error_states = last_counts.get("01", 0) + last_counts.get("10", 0)
            fidelity = bell_states / total_shots
            print(f"\nBell state fidelity estimate: {fidelity:.1%}")
            print(f"  (Ideal: 100% in |00⟩ and |11⟩, 0% in |01⟩ and |10⟩)")

    # Compare to SV1 simulator
    print("\n" + "=" * 55)
    print("Comparison to SV1 Simulator:")
    print("=" * 55)
    sv1_mean = 3.624  # From previous benchmarks

    if successful:
        wall_times = [r.wall_time_seconds for r in successful]
        rigetti_mean = sum(wall_times) / len(wall_times)

        print(f"SV1 mean time:     {sv1_mean:.1f}s")
        print(f"Rigetti mean time: {rigetti_mean:.1f}s")
        print(f"Difference:        {rigetti_mean - sv1_mean:+.1f}s")

        # Queue time analysis
        queue_times = [r.result_data.get('queue_time', 0) or 0 for r in successful]
        if any(queue_times):
            avg_queue = sum(queue_times) / len(queue_times)
            print(f"\nAverage queue time: {avg_queue:.1f}s")


if __name__ == "__main__":
    main()
