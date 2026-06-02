"""Tests for the unified benchmark suite."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from marqov.circuits import Circuit
from marqov.executors.base import BaseExecutor, ExecutionResult


@dataclass
class FixedExecutor(BaseExecutor):
    """Executor that returns deterministic counts for benchmark tests."""

    backend: str = "fixed"

    async def execute(
        self,
        circuit: Circuit,
        shots: int = 1000,
        **kwargs: object,
    ) -> ExecutionResult:
        return ExecutionResult(
            counts={"11": 7, "00": 5, "10": 3, "01": 1},
            backend=self.backend,
            execution_time_ms=12.345,
            shots=shots,
        )


class FailingExecutor(BaseExecutor):
    """Executor that always fails, used to verify skip-and-log behavior."""

    async def execute(
        self,
        circuit: Circuit,
        shots: int = 1000,
        **kwargs: object,
    ) -> ExecutionResult:
        raise RuntimeError("backend unavailable")


def test_top_outcomes_returns_three_highest_counts() -> None:
    """Top outcome selection is stable and limited to the three highest counts."""
    from benchmarks.suite import top_outcomes

    assert top_outcomes({"00": 5, "01": 5, "10": 1, "11": 9}) == {
        "11": 9,
        "00": 5,
        "01": 5,
    }


def test_format_table_matches_contributing_columns() -> None:
    """The rendered table uses the documented benchmark columns."""
    from benchmarks.suite import BenchmarkRow, format_table

    table = format_table(
        [
            BenchmarkRow(
                backend="local",
                circuit="bell",
                shots=100,
                exec_time_ms=12.3456,
                top_3_outcomes={"00": 52, "11": 48},
            )
        ]
    )

    lines = table.splitlines()
    assert lines[0] == "backend | circuit | shots | exec_time_ms | top_3_outcomes"
    assert lines[1] == 'local | bell | 100 | 12.35 | {"00": 52, "11": 48}'


@pytest.mark.parametrize("shots", [10, 25])
def test_run_suite_executes_three_standard_circuits(shots: int) -> None:
    """The suite runs Bell, GHZ, and random depth-5 circuits for each executor."""
    from benchmarks.suite import run_suite

    rows = asyncio.run(run_suite({"fixed": FixedExecutor()}, shots=shots))

    assert [row.circuit for row in rows] == ["bell", "ghz", "random_d5"]
    assert {row.backend for row in rows} == {"fixed"}
    assert {row.shots for row in rows} == {shots}
    assert all(row.top_3_outcomes == {"11": 7, "00": 5, "10": 3} for row in rows)


def test_run_suite_skips_backend_on_executor_error(capsys: pytest.CaptureFixture[str]) -> None:
    """Executor failures are logged to stderr without aborting other backends."""
    from benchmarks.suite import run_suite

    rows = asyncio.run(
        run_suite(
            {
                "broken": FailingExecutor(),
                "fixed": FixedExecutor(),
            },
            shots=10,
        )
    )

    assert [row.backend for row in rows] == ["fixed", "fixed", "fixed"]
    assert "broken: backend unavailable" in capsys.readouterr().err
