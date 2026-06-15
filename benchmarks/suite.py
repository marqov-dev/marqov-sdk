"""Unified multi-backend benchmark harness.

Runs a fixed set of reference circuits (Bell, 3-qubit GHZ, and a deterministic
depth-5 random circuit) against any configured executor and prints a comparison
table in the CONTRIBUTING.md §5 column format.

Works out of the box with ``LocalExecutor`` — no credentials required::

    python benchmarks/suite.py --executor local --shots 1000

``run_suite()`` is executor-agnostic: pass any mapping of name -> ``BaseExecutor``
to benchmark cloud backends programmatically.

    from benchmarks.suite import run_suite, format_table
    from marqov.executors import ExecutorFactory

    executors = {
        "local": LocalExecutor(),
        "sv1": ExecutorFactory.create_executor("sv1", braket_config),
    }
    rows = await run_suite(executors, shots=1000)
    print(format_table(rows))
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TextIO

from marqov.circuits import Circuit, bell_state, ghz_state
from marqov.executors.base import BaseExecutor
from marqov.executors.local import LocalExecutor, LocalExecutorConfig

# Column order is fixed by CONTRIBUTING.md §5 and must not change.
TABLE_COLUMNS: tuple[str, ...] = (
    "backend",
    "circuit",
    "shots",
    "exec_time_ms",
    "top_3_outcomes",
)

DEFAULT_SHOTS = 1000
DEFAULT_SEED = 1234

# Single-qubit gates for the random benchmark circuit. Restricted to the
# canonical gate set (CONTRIBUTING.md §1) so every backend can run it.
_RANDOM_SINGLE_QUBIT_GATES: tuple[str, ...] = ("h", "x", "y", "z", "s", "t")
_RANDOM_QUBITS = 3
_RANDOM_DEPTH = 5


@dataclass(frozen=True)
class BenchmarkRow:
    """One rendered row of the benchmark comparison table."""

    backend: str
    circuit: str
    shots: int
    exec_time_ms: float
    top_3_outcomes: dict[str, int]


def random_depth5_circuit(seed: int = DEFAULT_SEED) -> Circuit:
    """Build a deterministic pseudo-random 3-qubit, depth-5 circuit.

    Each of the five layers applies one random single-qubit gate to a random
    qubit followed by a CNOT between two distinct random qubits. Seeding makes
    the circuit identical across runs, which keeps the suite usable for
    regression testing.

    Args:
        seed: Seed for the pseudo-random generator.

    Returns:
        A reproducible Circuit using only canonical gates.
    """
    rng = random.Random(seed)
    circuit = Circuit()
    for _ in range(_RANDOM_DEPTH):
        gate = rng.choice(_RANDOM_SINGLE_QUBIT_GATES)
        getattr(circuit, gate)(rng.randrange(_RANDOM_QUBITS))
        control = rng.randrange(_RANDOM_QUBITS)
        # Offset by 1..n-1 (mod n) so control and target are always distinct.
        target = (control + rng.randrange(1, _RANDOM_QUBITS)) % _RANDOM_QUBITS
        circuit.cnot(control, target)
    return circuit


def default_circuits(seed: int = DEFAULT_SEED) -> dict[str, Circuit]:
    """Return the standard benchmark circuits keyed by name.

    Args:
        seed: Seed for the random depth-5 circuit.

    Returns:
        Ordered mapping of circuit name -> Circuit (bell, ghz, random_d5).
    """
    return {
        "bell": bell_state(),
        "ghz": ghz_state(3),
        "random_d5": random_depth5_circuit(seed),
    }


def top_outcomes(counts: Mapping[str, int], limit: int = 3) -> dict[str, int]:
    """Return the most frequent measurement outcomes.

    Outcomes are ordered by count descending, with bitstring ascending as a
    deterministic tie-breaker. The returned dict's insertion order carries this
    ranking and must be preserved when serialising — do not sort JSON keys.

    Args:
        counts: Measurement outcome counts.
        limit: Maximum number of outcomes to return.

    Returns:
        Ordered dict of the top ``limit`` outcomes.
    """
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return dict(ranked[:limit])


async def run_suite(
    executors: Mapping[str, BaseExecutor],
    shots: int = DEFAULT_SHOTS,
    circuits: Mapping[str, Circuit] | None = None,
    stderr: TextIO | None = None,
) -> list[BenchmarkRow]:
    """Benchmark every circuit against every executor.

    Each backend is treated atomically: if any circuit raises, the whole backend
    is skipped, the failure is logged to stderr (naming the backend and the
    circuit that failed), and the suite continues with the next backend — it
    never aborts.

    Args:
        executors: Ordered mapping of backend name -> executor.
        shots: Number of measurement shots per circuit.
        circuits: Circuits to run; defaults to the standard suite.
        stderr: Stream for skip messages; defaults to ``sys.stderr``.

    Returns:
        One BenchmarkRow per (backend, circuit) that completed successfully.
    """
    error_stream = stderr if stderr is not None else sys.stderr
    benchmark_circuits = circuits if circuits is not None else default_circuits()
    rows: list[BenchmarkRow] = []

    for backend, executor in executors.items():
        backend_rows: list[BenchmarkRow] = []
        in_flight: str | None = None
        try:
            for circuit_name, circuit in benchmark_circuits.items():
                in_flight = circuit_name
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
            # Atomic skip: discard any partial rows for this backend and move on.
            print(f"skipping {backend} (failed on {in_flight}): {exc}", file=error_stream)
            continue
        rows.extend(backend_rows)

    return rows


def format_table(rows: Sequence[BenchmarkRow]) -> str:
    """Render rows as a GitHub-flavoured markdown table (CONTRIBUTING.md §5).

    Columns are padded to a common width and wrapped in pipes with a separator
    row, matching the documented format exactly. Empty ``rows`` still yields a
    well-formed header and separator rather than crashing.

    Args:
        rows: Benchmark rows to render.

    Returns:
        The markdown table as a string (no trailing newline).
    """
    cells = [
        (
            row.backend,
            row.circuit,
            str(row.shots),
            f"{row.exec_time_ms:.1f}",
            json.dumps(row.top_3_outcomes),
        )
        for row in rows
    ]

    widths: list[int] = []
    for index, column in enumerate(TABLE_COLUMNS):
        column_cells = [len(cell[index]) for cell in cells]
        widths.append(max(len(column), *column_cells) if column_cells else len(column))

    def render(values: Sequence[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(values)) + " |"

    separator = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
    return "\n".join([render(TABLE_COLUMNS), separator, *(render(cell) for cell in cells)])


def _build_executor(name: str, seed: int) -> BaseExecutor:
    """Create a single executor for a CLI ``--executor`` name."""
    # Only ``local`` is wired here. Richer backends need provider config (ARNs,
    # tokens, S3 buckets) that a zero-arg CLI can't supply, so they go through
    # run_suite() with a caller-built BaseExecutor rather than ExecutorFactory.
    if name == "local":
        return LocalExecutor(LocalExecutorConfig(seed=seed))
    raise ValueError(
        f"Unsupported executor '{name}'. The CLI supports: local. "
        f"To benchmark other backends, call run_suite() with any BaseExecutor."
    )


def build_executors(names: Sequence[str], seed: int) -> dict[str, BaseExecutor]:
    """Create executors for the given CLI names, preserving order and de-duping.

    Args:
        names: Executor names from the CLI.
        seed: Seed forwarded to executors that support reproducibility.

    Returns:
        Ordered mapping of name -> executor.
    """
    executors: dict[str, BaseExecutor] = {}
    for name in names:
        if name not in executors:
            executors[name] = _build_executor(name, seed)
    return executors


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Marqov's multi-backend benchmark suite.")
    parser.add_argument(
        "--executor",
        action="append",
        default=None,
        help="Executor to benchmark (repeatable). Defaults to 'local'.",
    )
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS, help="Shots per circuit.")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for the random circuit and local sampler (reproducible runs).",
    )
    return parser.parse_args(argv)


async def async_main(argv: Sequence[str] | None = None) -> int:
    """Async CLI entry point. Returns a process exit code."""
    args = parse_args(argv)
    if args.shots <= 0:
        raise ValueError("--shots must be a positive integer")

    names = args.executor or ["local"]
    executors = build_executors(names, args.seed)
    rows = await run_suite(executors, shots=args.shots, circuits=default_circuits(args.seed))
    print(format_table(rows))
    # Non-zero only when every backend failed, so CI can flag a fully broken run.
    return 0 if rows else 1


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code.

    Invalid arguments (unknown executor, non-positive shots) are reported to
    stderr and produce exit code 2 rather than an uncaught traceback.
    """
    try:
        return asyncio.run(async_main(argv))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
