"""Run a small, executor-agnostic circuit benchmark suite."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marqov.circuits import Circuit
from marqov.executors.base import BaseExecutor
from marqov.executors.local import LocalExecutor, LocalExecutorConfig


@dataclass(frozen=True)
class BenchmarkRow:
    """Result from one executor and circuit combination."""

    backend: str
    circuit: str
    shots: int
    exec_time_ms: float
    top_3_outcomes: dict[str, int]


def build_circuits(seed: int = 0) -> dict[str, Circuit]:
    """Build the standard Bell, GHZ, and depth-five random circuits."""
    rng = random.Random(seed)
    random_circuit = Circuit()
    single_qubit_gates = (random_circuit.h, random_circuit.x, random_circuit.y)
    for _ in range(5):
        for qubit in range(3):
            rng.choice(single_qubit_gates)(qubit)
        control = rng.randrange(3)
        target = (control + rng.choice((1, 2))) % 3
        random_circuit.cnot(control, target)

    return {
        "bell": Circuit().h(0).cnot(0, 1),
        "ghz": Circuit().h(0).cnot(0, 1).cnot(1, 2),
        "random_d5": random_circuit,
    }


def top_outcomes(counts: Mapping[str, int], limit: int = 3) -> dict[str, int]:
    """Return the most frequent outcomes with deterministic tie-breaking."""
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit])


async def run_suite(
    executors: Mapping[str, BaseExecutor],
    shots: int = 1000,
    circuits: Mapping[str, Circuit] | None = None,
    stderr: TextIO = sys.stderr,
) -> list[BenchmarkRow]:
    """Run every circuit on each executor, skipping backends that fail."""
    benchmark_circuits = circuits or build_circuits()
    rows: list[BenchmarkRow] = []

    for backend, executor in executors.items():
        backend_rows: list[BenchmarkRow] = []
        try:
            for circuit_name, circuit in benchmark_circuits.items():
                result = await executor.execute(circuit, shots=shots)
                backend_rows.append(
                    BenchmarkRow(
                        backend=backend,
                        circuit=circuit_name,
                        shots=result.shots,
                        exec_time_ms=result.execution_time_ms,
                        top_3_outcomes=top_outcomes(result.counts),
                    )
                )
        except Exception as exc:
            print(f"Skipping backend {backend}: {exc}", file=stderr)
            continue
        rows.extend(backend_rows)

    return rows


def format_table(rows: Sequence[BenchmarkRow]) -> str:
    """Format benchmark rows using the documented column layout."""
    headers = ("backend", "circuit", "shots", "exec_time_ms", "top_3_outcomes")
    values = [
        (
            row.backend,
            row.circuit,
            str(row.shots),
            f"{row.exec_time_ms:.1f}",
            json.dumps(row.top_3_outcomes, sort_keys=True),
        )
        for row in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in values))
        for index in range(len(headers))
    ]

    def format_row(row: Sequence[str]) -> str:
        return " | ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip()

    separator = " | ".join("-" * width for width in widths)
    return "\n".join([format_row(headers), separator, *(format_row(row) for row in values)])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executor",
        action="append",
        choices=("local",),
        default=None,
        help="Executor to benchmark; repeat to select multiple executors.",
    )
    parser.add_argument("--shots", type=int, default=1000, help="Shots per circuit.")
    parser.add_argument(
        "--seed", type=int, default=0, help="Random circuit and local sampler seed."
    )
    return parser.parse_args(argv)


async def async_main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line benchmark suite."""
    args = parse_args(argv)
    if args.shots <= 0:
        raise ValueError("--shots must be positive")

    executor_names = args.executor or ["local"]
    executors = {
        name: LocalExecutor(LocalExecutorConfig(seed=args.seed)) for name in executor_names
    }
    rows = await run_suite(executors, shots=args.shots, circuits=build_circuits(args.seed))
    print(format_table(rows))
    return 0 if rows else 1


def main() -> int:
    """Run the benchmark suite CLI."""
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
