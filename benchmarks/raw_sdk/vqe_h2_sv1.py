#!/usr/bin/env python3
"""
VQE H₂ benchmark on SV1 (State Vector Simulator) using raw Braket SDK.

Uses the shared VQE implementation for consistency across all approaches.
Measures all 5 Pauli terms with scipy COBYLA optimizer.

Cost estimate: ~$0.50 per run
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from harness import BenchmarkResult, save_results, print_summary
from shared.vqe_h2 import (
    run_vqe_optimization,
    build_braket_circuit,
    H2_GROUND_STATE_ENERGY,
    DEFAULT_SHOTS,
    DEFAULT_MAX_ITERATIONS,
)

os.environ.setdefault("AWS_PROFILE", "marqov-dev")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

DEVICE_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
S3_BUCKET = "amazon-braket-marqov-dev"
S3_PREFIX = "benchmarks/raw-sdk/vqe-sv1"


def create_measure_pauli_fn(device):
    """Create a function that measures a Pauli string on the device."""
    def measure_pauli(pauli_string: str, theta: float) -> dict[str, int]:
        circuit = build_braket_circuit(theta, pauli_string)
        task = device.run(
            circuit,
            s3_destination_folder=(S3_BUCKET, S3_PREFIX),
            shots=DEFAULT_SHOTS,
        )
        result = task.result()
        return dict(result.measurement_counts)
    return measure_pauli


def run_vqe() -> tuple[dict, float | None]:
    """Run VQE for H₂."""
    from braket.aws import AwsDevice

    device = AwsDevice(DEVICE_ARN)
    measure_pauli = create_measure_pauli_fn(device)

    print("  Running VQE optimization...")
    result = run_vqe_optimization(
        measure_pauli=measure_pauli,
        shots=DEFAULT_SHOTS,
        max_iterations=DEFAULT_MAX_ITERATIONS,
        initial_theta=0.0,
        verbose=True,
    )

    return result, None


def main():
    print("VQE H₂ Benchmark - Raw SDK + SV1 (State Vector Simulator)")
    print("=" * 60)
    print(f"Device: SV1 (State Vector Simulator)")
    print(f"Qubits: 2")
    print(f"Optimizer: COBYLA")
    print(f"Max iterations: {DEFAULT_MAX_ITERATIONS}")
    print(f"Shots per measurement: {DEFAULT_SHOTS}")
    print(f"Pauli terms: 5 (Z0, Z1, Z0Z1, X0X1, Y0Y1)")
    print(f"Circuits per iteration: 5")
    print(f"Known ground state: {H2_GROUND_STATE_ENERGY:.4f} Ha")
    print()

    from braket.aws import AwsDevice
    device = AwsDevice(DEVICE_ARN)
    print(f"Device: {device.name}")
    print(f"Status: {device.status}")
    print()

    results = []
    for i in range(2):
        print(f"Run {i + 1}/2")
        start = time.perf_counter()

        try:
            result_data, quantum_time = run_vqe()
            wall_time = time.perf_counter() - start
            error = None
            print(f"  Completed in {wall_time:.1f}s")
            print(f"  Final energy: {result_data['final_energy']:.6f} Ha")
            print(f"  Error: {result_data['energy_error']*1000:.2f} mHa")
        except Exception as e:
            wall_time = time.perf_counter() - start
            result_data = {}
            quantum_time = None
            error = str(e)
            print(f"  Error: {e}")

        results.append(BenchmarkResult(
            name="vqe_h2_raw_sdk_sv1",
            run_number=i + 1,
            wall_time_seconds=wall_time,
            quantum_time_seconds=quantum_time,
            result_data=result_data,
            timestamp=datetime.now(timezone.utc).isoformat(),
            backend="sv1",
            error=error,
        ))
        print()

    output_dir = Path(__file__).parent.parent / "results" / "raw_sdk"
    save_results(results, output_dir)
    print_summary(results)


if __name__ == "__main__":
    main()
