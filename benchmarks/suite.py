"""Unified executor benchmark harness."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from dataclasses import dataclass
from typing import Callable, Sequence

from marqov.circuits import Circuit, bell_state, ghz_state
from marqov.executors.base import BaseExecutor, ExecutionResult
from marqov.executors.local import LocalExecutor


@dataclass(frozen=True)
class BenchmarkCase:
    """A named circuit benchmark."""

    name: str
    build: Callable[[], Circuit]


@dataclass(frozen=True)
class BenchmarkRow:
    """One rendered benchmark result row."""

    backend: str
    circuit: str
    shots: int
    exec_time_ms: float
    top_3_outcomes: dict[str, int]


def random_depth_5(seed: int = 42) -> Circuit:
    """Build a deterministic 3-qubit depth-5 random circuit."""
    rng = random.Random(seed)
    circuit = Circuit()
    one_qubit_gates = ["h", "x", "y", "z", "s", "t"]

    for _ in range(5):
        gate = rng.choice(one_qubit_gates)
        qubit = rng.randrange(3)
        getattr(circuit, gate)(qubit)

        control, target = rng.sample(range(3), 2)
        circuit.cnot(control, target)

    return circuit


def benchmark_cases() -> list[BenchmarkCase]:
    """Return the default benchmark circuit suite."""
    return [
        BenchmarkCase("bell", bell_state),
        BenchmarkCase("ghz", lambda: ghz_state(3)),
        BenchmarkCase("random_d5", random_depth_5),
    ]


def create_executor(name: str) -> tuple[str, BaseExecutor]:
    """Create an executor by CLI name."""
    if name == "local":
        return "local", LocalExecutor()
    raise ValueError(f"Unsupported executor: {name}. Supported executors: local")


def top_outcomes(counts: dict[str, int], limit: int = 3) -> dict[str, int]:
    """Return the top measurement outcomes sorted by count descending."""
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit])


async def run_suite(
    executors: Sequence[tuple[str, BaseExecutor]],
    cases: Sequence[BenchmarkCase],
    shots: int,
) -> list[BenchmarkRow]:
    """Run all benchmark cases against all executors."""
    rows: list[BenchmarkRow] = []

    for backend_name, executor in executors:
        for case in cases:
            try:
                result = await executor.execute(case.build(), shots=shots)
            except Exception as exc:
                print(
                    f"skipping {backend_name}/{case.name}: {exc}",
                    file=sys.stderr,
                )
                continue

            rows.append(row_from_result(backend_name, case.name, result))

    return rows


def row_from_result(backend_name: str, circuit_name: str, result: ExecutionResult) -> BenchmarkRow:
    """Convert an execution result to a benchmark row."""
    return BenchmarkRow(
        backend=backend_name,
        circuit=circuit_name,
        shots=result.shots,
        exec_time_ms=result.execution_time_ms,
        top_3_outcomes=top_outcomes(result.counts),
    )


def format_rows(rows: Sequence[BenchmarkRow]) -> str:
    """Render benchmark rows as the CONTRIBUTING.md §5 table format."""
    lines = [
        "backend | circuit | shots | exec_time_ms | top_3_outcomes",
    ]

    for row in rows:
        lines.append(
            " | ".join(
                [
                    row.backend,
                    row.circuit,
                    str(row.shots),
                    f"{row.exec_time_ms:.1f}",
                    json.dumps(row.top_3_outcomes, sort_keys=True),
                ]
            )
        )

    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run Marqov executor benchmarks.")
    parser.add_argument(
        "--executor",
        action="append",
        default=None,
        help="Executor to benchmark. May be passed multiple times. Defaults to local.",
    )
    parser.add_argument("--shots", type=int, default=1000, help="Number of shots per circuit.")
    return parser.parse_args(argv)


async def async_main(argv: Sequence[str] | None = None) -> int:
    """Async CLI entrypoint."""
    args = parse_args(argv)
    executor_names = args.executor or ["local"]
    executors = [create_executor(name) for name in executor_names]
    rows = await run_suite(executors, benchmark_cases(), shots=args.shots)
    print(format_rows(rows))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
