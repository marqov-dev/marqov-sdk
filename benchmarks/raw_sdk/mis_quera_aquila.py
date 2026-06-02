#!/usr/bin/env python3
"""
Maximum Independent Set (MIS) benchmark on QuEra Aquila QPU.

QuEra Aquila uses Analog Hamiltonian Simulation (AHS), not gate-based circuits.
This is fundamentally different from our Bell State and VQE benchmarks.

The MIS problem: Find the largest set of vertices in a graph such that no two
vertices in the set are adjacent. This maps naturally to Rydberg atom physics:
- Atoms represent graph vertices
- Rydberg blockade (atoms can't both be excited if too close) encodes edges
- Ground state of the system encodes the MIS solution

This benchmark creates a simple 1D chain and finds the anti-ferromagnetic
ground state (alternating excited/ground pattern = MIS for 1D chain).

COST:
- Per-shot: $0.01
- 100 shots = $1.00
- Estimated total (5 runs): ~$5.00
"""

import os
import time
from pathlib import Path
from datetime import datetime
from collections import Counter

import boto3
from braket.aws import AwsDevice, AwsSession
from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
from braket.ahs.atom_arrangement import AtomArrangement
from braket.ahs.driving_field import DrivingField

# Add parent to path for harness import
import sys
sys.path.insert(0, str(Path(__file__).parent))

from harness import BenchmarkResult, save_results, print_summary


# Use marqov-dev profile
os.environ.setdefault("AWS_PROFILE", "marqov-dev")

# QuEra Aquila ARN
QUERA_AQUILA_ARN = "arn:aws:braket:us-east-1::device/qpu/quera/Aquila"

# S3 bucket for results
S3_BUCKET = "amazon-braket-marqov-dev"
S3_PREFIX = "benchmarks/quera/mis"

# MIS parameters
NUM_ATOMS = 9  # 1D chain of 9 atoms
ATOM_SEPARATION = 6.1e-6  # meters (ensures Rydberg blockade between neighbors)
SHOTS = 100


def get_aws_session():
    """Create AWS session for us-east-1 (QuEra region)."""
    boto_session = boto3.Session(profile_name="marqov-dev", region_name="us-east-1")
    return AwsSession(boto_session=boto_session)


def create_1d_chain_register(num_atoms: int, separation: float) -> AtomArrangement:
    """Create a 1D chain of atoms for MIS problem.

    For a 1D chain, the MIS is the anti-ferromagnetic pattern:
    excited-ground-excited-ground-... (or vice versa)
    """
    register = AtomArrangement()
    for k in range(num_atoms):
        register.add([k * separation, 0])
    return register


def create_adiabatic_drive() -> DrivingField:
    """Create adiabatic evolution drive for ground state preparation.

    The key is to slowly ramp the detuning from negative to positive
    while maintaining a driving amplitude. This adiabatically prepares
    the ground state of the many-body Hamiltonian.
    """
    # Time points (in seconds)
    # Total evolution: 3 microseconds
    time_points = [0, 2.5e-7, 2.75e-6, 3e-6]

    # Rabi frequency Omega (rad/s) - the driving amplitude
    # Ramps up, stays constant, ramps down
    amplitude_values = [0, 1.57e7, 1.57e7, 0]

    # Detuning Delta (rad/s) - controls the energy landscape
    # Starts negative (ground state favored), ends positive (Rydberg state favored)
    # The slow ramp is key to adiabatic preparation
    detuning_values = [-5.5e7, -5.5e7, 5.5e7, 5.5e7]

    # Phase (constant at 0)
    phase_values = [0, 0, 0, 0]

    return DrivingField.from_lists(
        time_points,
        amplitude_values,
        detuning_values,
        phase_values
    )


def run_mis_aquila(shots: int = SHOTS) -> tuple[dict, float | None]:
    """
    Run MIS problem on QuEra Aquila.

    Returns:
        Tuple of (result_data, quantum_task_duration)
    """
    aws_session = get_aws_session()
    device = AwsDevice(QUERA_AQUILA_ARN, aws_session=aws_session)

    # Create the atom register (1D chain)
    register = create_1d_chain_register(NUM_ATOMS, ATOM_SEPARATION)

    # Create the adiabatic drive
    drive = create_adiabatic_drive()

    # Create the AHS program
    program = AnalogHamiltonianSimulation(register=register, hamiltonian=drive)

    # Submit task
    submit_time = time.perf_counter()
    task = device.run(
        program,
        s3_destination_folder=(S3_BUCKET, S3_PREFIX),
        shots=shots,
    )

    # Wait for result
    result = task.result()
    total_time = time.perf_counter() - submit_time

    # Analyze measurements
    # Each measurement is a list of states: 'g' (ground), 'r' (Rydberg), 'e' (empty/lost)
    measurements = result.measurements

    # Convert measurements to bitstrings (1=Rydberg/excited, 0=ground)
    bitstrings = []
    for shot in measurements:
        # pre_sequence shows which atoms were successfully trapped
        # post_sequence shows final state after evolution
        pre = shot.pre_sequence
        post = shot.post_sequence

        # Build bitstring: 1 if Rydberg (in MIS), 0 if ground
        bits = ""
        for i in range(len(post)):
            if pre[i] == 0:  # Atom was lost before measurement
                bits += "?"
            elif post[i] == 1:  # Rydberg state
                bits += "1"
            else:  # Ground state
                bits += "0"
        bitstrings.append(bits)

    # Count occurrences of each pattern
    pattern_counts = Counter(bitstrings)

    # For a 1D chain, the ideal MIS patterns are:
    # "101010101" (5 atoms in MIS) or "010101010" (4 atoms in MIS)
    ideal_patterns = ["101010101", "010101010"]

    # Calculate success rate (how often we get ideal anti-ferromagnetic order)
    ideal_count = sum(pattern_counts.get(p, 0) for p in ideal_patterns)
    valid_shots = sum(1 for b in bitstrings if "?" not in b)
    success_rate = ideal_count / valid_shots if valid_shots > 0 else 0

    # Get task metadata
    metadata = task.metadata()
    quantum_time = None

    if "executionDuration" in metadata:
        quantum_time = metadata["executionDuration"] / 1000.0

    return {
        "pattern_counts": dict(pattern_counts.most_common(10)),
        "total_shots": shots,
        "valid_shots": valid_shots,
        "ideal_patterns": ideal_patterns,
        "ideal_count": ideal_count,
        "success_rate": success_rate,
        "num_atoms": NUM_ATOMS,
        "atom_separation_um": ATOM_SEPARATION * 1e6,
        "task_arn": task.id,
        "device": QUERA_AQUILA_ARN,
        "total_time": total_time,
        "quantum_time": quantum_time,
    }, quantum_time


def main():
    print("MIS (Maximum Independent Set) Benchmark - QuEra Aquila QPU")
    print("=" * 65)
    print(f"Device: QuEra Aquila (neutral-atom, 256 qubits)")
    print(f"Region: us-east-1")
    print(f"Problem: 1D chain of {NUM_ATOMS} atoms")
    print(f"Atom separation: {ATOM_SEPARATION * 1e6:.1f} μm")
    print(f"Shots: {SHOTS}")
    print(f"Runs: 5")
    print()
    print("This is ANALOG Hamiltonian Simulation, not gate-based!")
    print("We're finding the Maximum Independent Set via adiabatic evolution.")
    print()
    print("COST:")
    print(f"  Per-shot fee: $0.01 × {SHOTS} = ${0.01 * SHOTS:.2f}")
    print(f"  Estimated total (5 runs): ${5 * 0.01 * SHOTS:.2f}")
    print()

    # Verify device access
    print("Verifying device access...")
    aws_session = get_aws_session()

    try:
        device = AwsDevice(QUERA_AQUILA_ARN, aws_session=aws_session)
        print(f"Device: {device.name}")
        print(f"Status: {device.status}")

        # Get paradigm info
        paradigm = device.properties.paradigm
        print(f"Paradigm: {paradigm.paradigmType if hasattr(paradigm, 'paradigmType') else 'AHS'}")

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

    # Confirm before running
    print("=" * 65)
    total_cost = 5 * 0.01 * SHOTS
    response = input(f"Proceed with benchmark? This will cost ~${total_cost:.2f}. [y/N]: ")
    if response.lower() != 'y':
        print("Benchmark cancelled.")
        return
    print()

    # Run benchmarks
    print("Running MIS benchmarks on QuEra Aquila...")
    print("(Each run submits to real QPU and waits for execution)")
    print()

    results = []
    num_runs = 5

    for i in range(num_runs):
        print(f"  Run {i + 1}/{num_runs}...", end=" ", flush=True)

        start = time.perf_counter()
        try:
            result_data, quantum_time = run_mis_aquila(shots=SHOTS)
            wall_time = time.perf_counter() - start
            error = None

            success_rate = result_data.get('success_rate', 0)
            print(f"done ({wall_time:.1f}s, {success_rate:.1%} ideal MIS)")

        except Exception as e:
            wall_time = time.perf_counter() - start
            result_data = {"error_details": str(e)}
            quantum_time = None
            error = str(e)
            print(f"error: {e}")

        results.append(BenchmarkResult(
            name="mis_quera_aquila",
            run_number=i + 1,
            wall_time_seconds=wall_time,
            quantum_time_seconds=quantum_time,
            result_data=result_data,
            timestamp=datetime.now().isoformat(),
            backend="quera-aquila",
            error=error,
        ))

    # Save results
    output_dir = Path(__file__).parent.parent / "results" / "m0-benchmarking" / "quera" / "mis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_results(results, output_dir)
    print(f"\nResults saved to: {output_path}")

    # Print summary
    print_summary(results)

    # Show pattern distribution from last successful run
    successful = [r for r in results if r.error is None]
    if successful:
        last_result = successful[-1].result_data
        print(f"\nMost common measurement patterns (last run):")
        for pattern, count in last_result.get("pattern_counts", {}).items():
            pct = count / last_result.get("total_shots", 1) * 100
            # Mark ideal patterns
            marker = " ← IDEAL MIS" if pattern in last_result.get("ideal_patterns", []) else ""
            print(f"  {pattern}: {count:4d} ({pct:.1f}%){marker}")

        print(f"\nMIS Success Rate: {last_result.get('success_rate', 0):.1%}")
        print(f"  (Fraction achieving ideal anti-ferromagnetic order)")

    # Comparison note
    print("\n" + "=" * 65)
    print("Note on QuEra Aquila:")
    print("=" * 65)
    print("This is ANALOG quantum simulation, not gate-based computing.")
    print("- Cannot run Bell States or VQE (different paradigm)")
    print("- Excels at optimization (MIS, MaxCut) and simulation")
    print("- 256 qubits available, we only used 9 for this demo")
    print("- Much cheaper than gate-based QPUs ($0.01/shot vs $0.03-$0.08)")


if __name__ == "__main__":
    main()
