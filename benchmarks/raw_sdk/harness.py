"""Benchmark harness for collecting timing metrics."""

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    name: str
    run_number: int
    wall_time_seconds: float
    quantum_time_seconds: float | None
    result_data: dict[str, Any]
    timestamp: str
    backend: str
    error: str | None = None


def run_benchmark(
    name: str,
    fn: Callable[[], tuple[dict[str, Any], float | None]],
    runs: int = 10,
    backend: str = "sv1",
) -> list[BenchmarkResult]:
    """
    Run a benchmark function multiple times and collect metrics.

    Args:
        name: Name of the benchmark
        fn: Function that returns (result_data, quantum_time_seconds)
        runs: Number of times to run
        backend: Backend identifier

    Returns:
        List of BenchmarkResult objects
    """
    results = []

    for i in range(runs):
        print(f"  Run {i + 1}/{runs}...", end=" ", flush=True)

        start = time.perf_counter()
        try:
            result_data, quantum_time = fn()
            wall_time = time.perf_counter() - start
            error = None
            print(f"done ({wall_time:.2f}s)")
        except Exception as e:
            wall_time = time.perf_counter() - start
            result_data = {}
            quantum_time = None
            error = str(e)
            print(f"error: {e}")

        results.append(BenchmarkResult(
            name=name,
            run_number=i + 1,
            wall_time_seconds=wall_time,
            quantum_time_seconds=quantum_time,
            result_data=result_data,
            timestamp=datetime.utcnow().isoformat(),
            backend=backend,
            error=error,
        ))

    return results


def save_results(results: list[BenchmarkResult], output_dir: Path, suffix: str = "") -> Path:
    """Save benchmark results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if results:
        name = results[0].name
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}{suffix}.json"
    else:
        filename = "empty_results.json"

    output_path = output_dir / filename

    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    return output_path


def print_summary(results: list[BenchmarkResult]) -> None:
    """Print summary statistics for benchmark results."""
    if not results:
        print("No results to summarize.")
        return

    successful = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]

    print(f"\n{'=' * 50}")
    print(f"Benchmark: {results[0].name}")
    print(f"Backend: {results[0].backend}")
    print(f"{'=' * 50}")
    print(f"Total runs: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        wall_times = [r.wall_time_seconds for r in successful]
        print(f"\nWall time (seconds):")
        print(f"  Min: {min(wall_times):.3f}")
        print(f"  Max: {max(wall_times):.3f}")
        print(f"  Mean: {sum(wall_times) / len(wall_times):.3f}")

        quantum_times = [r.quantum_time_seconds for r in successful if r.quantum_time_seconds]
        if quantum_times:
            print(f"\nQuantum time (seconds):")
            print(f"  Min: {min(quantum_times):.3f}")
            print(f"  Max: {max(quantum_times):.3f}")
            print(f"  Mean: {sum(quantum_times) / len(quantum_times):.3f}")
