#!/usr/bin/env python3
"""Unified benchmark suite for Marqov executors."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from dataclasses import dataclass
from typing import TextIO

from marqov.circuits import Circuit, bell_state, ghz_state
from marqov.executors.base import BaseExecutor
from marqov.executors.local import LocalExecutor, LocalExecutorConfig


DEFAULT_SHOTS = 1000
RANDOM_CIRCUIT_SEED = 42
RANDOM_CIRCUIT_QUBITS = 3
RANDOM_CIRCUIT_DEPTH = 5
TABLE_COLUMNS = ["backend", "circuit", "shots", "exec_time_ms", "top_3_outcomes"]


@dataclass(frozen=True)
class BenchmarkRow:
    """One benchmark table row."""

    backend: str
    circuit: str
    shots: int
    exec_time_ms: float
    top_3_outcomes: dict[str, int]


def make_random_d5(seed: int = RANDOM_CIRCUIT_SEED) -> Circuit:
    """Build a deterministic 3-qubit depth-5 random circuit."""
    rng = random.Random(seed)
    circuit = Circuit()
    single_qubit_gates = ("h", "x", "y", "z", "s", "t")

    for _ in range(RANDOM_CIRCUIT_DEPTH):
        for qubit in range(RANDOM_CIRCUIT_QUBITS):
            getattr(circuit, rng.choice(single_qubit_gates))(qubit)

        control = rng.randrange(RANDOM_CIRCUIT_QUBITS)
        target = rng.randrange(RANDOM_CIRCUIT_QUBITS - 1)
        if target >= control:
            target += 1
        circuit.cnot(control, target)

    return circuit


def default_circuits() -> dict[str, Circuit]:
    """Return the default benchmark circuits."""
    return {
        "bell": bell_state(),
        "ghz": ghz_state(3),
        "random_d5": make_random_d5(),
    }


def top_outcomes(counts: dict[str, int], limit: int = 3) -> dict[str, int]:
    """Return the top measurement outcomes by count."""
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return dict(ordered[:limit])


async def run_suite(
    executors: dict[str, BaseExecutor],
    shots: int = DEFAULT_SHOTS,
    circuits: dict[str, Circuit] | None = None,
    stderr: TextIO = sys.stderr,
) -> list[BenchmarkRow]:
    """Run every configured executor against every benchmark circuit.

    Executor failures are logged and skipped so one backend cannot abort the
    full comparison.
    """
    benchmark_circuits = circuits or default_circuits()
    rows: list[BenchmarkRow] = []

    for backend_name, executor in executors.items():
        for circuit_name, circuit in benchmark_circuits.items():
            try:
                result = await executor.execute(circuit, shots=shots)
            except Exception as exc:
                print(
                    f"Skipping {backend_name}/{circuit_name}: {exc}",
                    file=stderr,
                )
                continue

            rows.append(
                BenchmarkRow(
                    backend=backend_name,
                    circuit=circuit_name,
                    shots=result.shots or shots,
                    exec_time_ms=result.execution_time_ms,
                    top_3_outcomes=top_outcomes(result.counts),
                )
            )

    return rows


def format_outcomes(outcomes: dict[str, int]) -> str:
    """Format outcomes as compact JSON for table output."""
    return json.dumps(outcomes)


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format benchmark results as a markdown table."""
    table_rows = [
        [
            row.backend,
            row.circuit,
            str(row.shots),
            f"{row.exec_time_ms:.1f}",
            format_outcomes(row.top_3_outcomes),
        ]
        for row in rows
    ]
    widths = [
        max(len(TABLE_COLUMNS[index]), *(len(row[index]) for row in table_rows))
        if table_rows
        else len(TABLE_COLUMNS[index])
        for index in range(len(TABLE_COLUMNS))
    ]

    def render(values: list[str]) -> str:
        return "| " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        ) + " |"

    header = render(TABLE_COLUMNS)
    separator = "|-" + "-|-".join("-" * width for width in widths) + "-|"
    return "\n".join([header, separator, *(render(row) for row in table_rows)])


def create_executors(executor_names: list[str], seed: int | None = RANDOM_CIRCUIT_SEED) -> dict[str, BaseExecutor]:
    """Create executors by CLI name."""
    executors: dict[str, BaseExecutor] = {}
    for name in executor_names:
        normalized = name.lower()
        if normalized == "local":
            executors["local"] = LocalExecutor(LocalExecutorConfig(seed=seed))
            continue
        raise ValueError(f"Unsupported executor: {name}. Supported executors: local")
    return executors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run Marqov's multi-backend benchmark suite.")
    parser.add_argument(
        "--executor",
        action="append",
        default=None,
        help="Executor to benchmark. Repeat for multiple executors. Defaults to local.",
    )
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS, help="Shots per circuit.")
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_CIRCUIT_SEED,
        help="Seed for local sampling and deterministic random circuit generation.",
    )
    return parser.parse_args(argv)


async def async_main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    args = parse_args(argv)
    executor_names = args.executor or ["local"]
    executors = create_executors(executor_names, seed=args.seed)
    rows = await run_suite(executors, shots=args.shots, circuits=default_circuits())
    print(format_rows(rows))
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
