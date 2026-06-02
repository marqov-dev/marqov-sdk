#!/usr/bin/env python3
"""
Bell State benchmark on IonQ Forte-1 QPU.

Creates a 2-qubit Bell state |Φ+⟩ = (|00⟩ + |11⟩) / √2
Runs on real IonQ trapped-ion quantum hardware.

COST WARNING: IonQ Forte is expensive!
- Per-task: $0.30
- Per-shot: $0.08
- 100 shots = $0.30 + $8.00 = $8.30 per circuit

Issue: #173
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


# Use marqov-dev profile - IonQ is in us-east-1
os.environ.setdefault("AWS_PROFILE", "marqov-dev")

# IonQ Forte-1 ARN (us-east-1)
IONQ_ARN = "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1"

# S3 bucket for results
S3_BUCKET = "amazon-braket-marqov-dev"
S3_PREFIX = "benchmarks/ionq"

# Use fewer shots to control cost
SHOTS = 100  # $8.30 per run instead of $80.30


def get_aws_session():
    """Create AWS session for us-east-1 (IonQ region)."""
    boto_session = boto3.Session(profile_name="marqov-dev", region_name="us-east-1")
    return AwsSession(boto_session=boto_session)


def create_bell_circuit() -> Circuit:
    """Create a Bell state circuit: H on qubit 0, then CNOT(0, 1)."""
    circuit = Circuit()
    circuit.h(0)
    circuit.cnot(0, 1)
    return circuit


def run_bell_state_ionq(shots: int = SHOTS) -> tuple[dict, float | None]:
    """
    Run Bell state circuit on IonQ Forte-1.

    Returns:
        Tuple of (result_data, quantum_task_duration)
    """
    aws_session = get_aws_session()
    device = AwsDevice(IONQ_ARN, aws_session=aws_session)
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
        "device": IONQ_ARN,
        "queue_time": queue_time,
        "quantum_time": quantum_time,
        "total_time": total_time,
    }, quantum_time


def main():
    print("Bell State Benchmark - IonQ Forte-1 QPU")
    print("=" * 55)
    print(f"Device: IonQ Forte-1 (trapped-ion, 36 qubits)")
    print(f"Region: us-east-1")
    print(f"Shots: {SHOTS}")
    print(f"Runs: 3")
    print()
    print("COST WARNING (IonQ Forte is expensive!):")
    print(f"  Per-task fee: $0.30")
    print(f"  Per-shot fee: $0.08 × {SHOTS} = ${0.08 * SHOTS:.2f}")
    print(f"  Total per run: ${0.30 + 0.08 * SHOTS:.2f}")
    print(f"  Estimated total (3 runs): ${3 * (0.30 + 0.08 * SHOTS):.2f}")
    print()

    # Verify device access
    print("Verifying device access...")
    aws_session = get_aws_session()

    try:
        device = AwsDevice(IONQ_ARN, aws_session=aws_session)
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
    total_cost = 3 * (0.30 + 0.08 * SHOTS)
    response = input(f"Proceed with benchmark? This will cost ~${total_cost:.2f}. [y/N]: ")
    if response.lower() != 'y':
        print("Benchmark cancelled.")
        return
    print()

    # Run benchmarks
    print("Running benchmarks on IonQ Forte-1...")
    print("(Each run submits to real QPU and waits for execution)")
    print("(IonQ trapped-ion systems may have longer execution times)")
    print()

    results = []
    num_runs = 3  # Fewer runs due to high cost

    for i in range(num_runs):
        print(f"  Run {i + 1}/{num_runs}...", end=" ", flush=True)

        start = time.perf_counter()
        try:
            result_data, quantum_time = run_bell_state_ionq(shots=SHOTS)
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
            name="bell_state_ionq",
            run_number=i + 1,
            wall_time_seconds=wall_time,
            quantum_time_seconds=quantum_time,
            result_data=result_data,
            timestamp=datetime.now().isoformat(),
            backend="ionq-forte1",
            error=error,
        ))

    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "m0-benchmarking" / "ionq" / "bell-state"
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
            fidelity = bell_states / total_shots
            print(f"\nBell state fidelity estimate: {fidelity:.1%}")
            print(f"  (Ideal: 100% in |00⟩ and |11⟩, 0% in |01⟩ and |10⟩)")

    # Compare to other backends
    print("\n" + "=" * 55)
    print("Comparison to Other Backends:")
    print("=" * 55)
    sv1_mean = 3.624
    rigetti_mean = 3.487

    if successful:
        wall_times = [r.wall_time_seconds for r in successful]
        ionq_mean = sum(wall_times) / len(wall_times)

        print(f"SV1 simulator mean:    {sv1_mean:.1f}s")
        print(f"Rigetti Ankaa-3 mean:  {rigetti_mean:.1f}s (95.4% fidelity)")
        print(f"IonQ Forte-1 mean:     {ionq_mean:.1f}s")

        # Fidelity comparison
        if successful:
            last_counts = successful[-1].result_data.get("counts", {})
            if last_counts:
                total = sum(last_counts.values())
                bell_states = last_counts.get("00", 0) + last_counts.get("11", 0)
                ionq_fidelity = bell_states / total
                print(f"\nFidelity comparison:")
                print(f"  SV1 simulator:  100% (perfect)")
                print(f"  Rigetti Ankaa-3: 95.4%")
                print(f"  IonQ Forte-1:   {ionq_fidelity:.1%}")


if __name__ == "__main__":
    main()
