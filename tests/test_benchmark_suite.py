"""Tests for the unified benchmark suite."""

from __future__ import annotations

import pytest

from benchmarks.suite import (
    BenchmarkCase,
    BenchmarkRow,
    format_rows,
    row_from_result,
    run_suite,
    top_outcomes,
)
from marqov.circuits import Circuit
from marqov.executors.base import BaseExecutor, ExecutionResult


class FakeExecutor(BaseExecutor):
    """Executor that returns a fixed result."""

    async def execute(self, circuit: Circuit, shots: int = 1000, **kwargs) -> ExecutionResult:
        return ExecutionResult(
            counts={"11": 2, "00": 5, "01": 3, "10": 1},
            backend="fake",
            execution_time_ms=12.34,
            shots=shots,
        )


class FailingExecutor(BaseExecutor):
    """Executor that always fails."""

    async def execute(self, circuit: Circuit, shots: int = 1000, **kwargs) -> ExecutionResult:
        raise RuntimeError("backend offline")


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ({"00": 5, "11": 2, "01": 3, "10": 1}, {"00": 5, "01": 3, "11": 2}),
        ({"1": 7, "0": 7, "10": 1}, {"0": 7, "1": 7, "10": 1}),
        ({}, {}),
    ],
)
def test_top_outcomes(counts: dict[str, int], expected: dict[str, int]) -> None:
    assert top_outcomes(counts) == expected


def test_row_from_result() -> None:
    result = ExecutionResult(
        counts={"00": 4, "11": 6},
        backend="ignored",
        execution_time_ms=8.25,
        shots=10,
    )

    row = row_from_result("local", "bell", result)

    assert row == BenchmarkRow(
        backend="local",
        circuit="bell",
        shots=10,
        exec_time_ms=8.25,
        top_3_outcomes={"11": 6, "00": 4},
    )


def test_format_rows_matches_contributing_columns() -> None:
    rendered = format_rows(
        [
            BenchmarkRow(
                backend="local",
                circuit="bell",
                shots=1000,
                exec_time_ms=12.34,
                top_3_outcomes={"00": 503, "11": 497},
            )
        ]
    )

    assert rendered.splitlines() == [
        "backend | circuit | shots | exec_time_ms | top_3_outcomes",
        'local | bell | 1000 | 12.3 | {"00": 503, "11": 497}',
    ]


@pytest.mark.asyncio
async def test_run_suite_returns_rows_for_each_case() -> None:
    rows = await run_suite(
        [("fake", FakeExecutor())],
        [
            BenchmarkCase("bell", Circuit),
            BenchmarkCase("ghz", Circuit),
        ],
        shots=25,
    )

    assert [row.circuit for row in rows] == ["bell", "ghz"]
    assert all(row.backend == "fake" for row in rows)
    assert all(row.shots == 25 for row in rows)


@pytest.mark.asyncio
async def test_run_suite_skips_and_logs_executor_errors(capsys: pytest.CaptureFixture) -> None:
    rows = await run_suite(
        [
            ("bad", FailingExecutor()),
            ("good", FakeExecutor()),
        ],
        [BenchmarkCase("bell", Circuit)],
        shots=10,
    )

    assert [row.backend for row in rows] == ["good"]
    assert "skipping bad/bell: backend offline" in capsys.readouterr().err
