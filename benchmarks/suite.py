"""Unified benchmark suite for Marqov executors."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marqov.circuits import Circuit, bell_state, ghz_state
from marqov.executors.base import BaseExecutor
from marqov.executors.local import LocalExecutor, LocalExecutorConfig


@dataclass(frozen=True)
class BenchmarkCase:
    """Circuit builder used by the benchmark suite."""

    name: str
    circuit: Circuit


@dataclass(frozen=True)
class BenchmarkRow:
    """One rendered benchmark result row."""

    backend: str
    circuit: str
    shots: int
    exec_time_ms: float
    top_3_outcomes: dict[str, int]


def random_depth5_circuit(seed: int = 0) -> Circuit:
    """Build a deterministic three-qubit depth-5 circuit."""
    rng = random.Random(seed)
    circuit = Circuit()

    for _ in range(5):
        gate = rng.choice(("h", "x", "y", "z", "s", "t", "rx", "ry", "rz", "cnot", "cz", "swap"))
        if gate in {"h", "x", "y", "z", "s", "t"}:
            getattr(circuit, gate)(rng.randrange(3))
        elif gate in {"rx", "ry", "rz"}:
            getattr(circuit, gate)(rng.choice((0.25, 0.5, 1.0)), rng.randrange(3))
        else:
            control = rng.randrange(3)
            target = (control + rng.randrange(1, 3)) % 3
            getattr(circuit, gate)(control, target)

    return circuit


def benchmark_cases() -> list[BenchmarkCase]:
    """Return the standard circuit set required by CONTRIBUTING.md."""
    return [
        BenchmarkCase("bell", bell_state()),
        BenchmarkCase("ghz", ghz_state(3)),
        BenchmarkCase("random_d5", random_depth5_circuit()),
    ]


def top_outcomes(counts: Mapping[str, int], limit: int = 3) -> dict[str, int]:
    """Return the highest-count measurement outcomes with stable tie ordering."""
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit])


async def run_suite(
    executors: Mapping[str, BaseExecutor],
    shots: int = 1000,
    stderr: TextIO | None = None,
) -> list[BenchmarkRow]:
    """Run the standard benchmark circuits against each configured executor.

    If an executor fails on any circuit, that backend is skipped and the suite
    continues with the remaining backends.
    """
    error_stream = stderr or sys.stderr
    rows: list[BenchmarkRow] = []

    for backend, executor in executors.items():
        try:
            for case in benchmark_cases():
                result = await executor.execute(case.circuit, shots=shots)
                rows.append(
                    BenchmarkRow(
                        backend=backend,
                        circuit=case.name,
                        shots=result.shots,
                        exec_time_ms=result.execution_time_ms,
                        top_3_outcomes=top_outcomes(result.counts),
                    )
                )
        except Exception as exc:
            print(f"{backend}: {exc}", file=error_stream)
            continue

    return rows


def format_table(rows: list[BenchmarkRow]) -> str:
    """Render rows in the documented benchmark table column order."""
    lines = ["backend | circuit | shots | exec_time_ms | top_3_outcomes"]
    for row in rows:
        outcomes = json.dumps(row.top_3_outcomes)
        lines.append(
            f"{row.backend} | {row.circuit} | {row.shots} | {row.exec_time_ms:.2f} | {outcomes}"
        )
    return "\n".join(lines)


def _split_executor_names(value: str) -> list[str]:
    return [name.strip() for name in value.split(",") if name.strip()]


def load_executor_config(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load optional executor factory configuration from JSON."""
    if path is None:
        return {}
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("executor config must be a JSON object keyed by executor name")
    return data


def create_executors(
    names: list[str],
    configs: Mapping[str, dict[str, Any]] | None = None,
    seed: int | None = 0,
) -> dict[str, BaseExecutor]:
    """Create executors requested by CLI name."""
    resolved: dict[str, BaseExecutor] = {}
    configs = configs or {}

    for name in names:
        if name == "local":
            resolved[name] = LocalExecutor(LocalExecutorConfig(seed=seed))
            continue

        if name not in configs:
            raise ValueError(
                f"missing config for executor '{name}'. "
                f"Use --config with a JSON object keyed by executor name."
            )

        from marqov.executors.factory import ExecutorFactory

        resolved[name] = ExecutorFactory.create_executor(name, configs[name])

    return resolved


async def amain(argv: list[str] | None = None) -> int:
    """Async CLI entry point."""
    parser = argparse.ArgumentParser(description="Run Marqov benchmark circuits.")
    parser.add_argument(
        "--executor",
        default="local",
        help="Executor name or comma-separated names. 'local' works without credentials.",
    )
    parser.add_argument("--shots", type=int, default=1000, help="Shots per circuit.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON config for non-local executors.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for local/random circuits.")
    args = parser.parse_args(argv)

    executors = create_executors(
        _split_executor_names(args.executor),
        load_executor_config(args.config),
        seed=args.seed,
    )
    rows = await run_suite(executors, shots=args.shots)
    print(format_table(rows))
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    return asyncio.run(amain(argv))


if __name__ == "__main__":
    raise SystemExit(main())
