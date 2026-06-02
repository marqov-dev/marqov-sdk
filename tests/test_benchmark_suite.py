"""Tests for the unified benchmark suite."""

from __future__ import annotations

from io import StringIO
from typing import Any

import pytest

from benchmarks.suite import (
    BenchmarkRow,
    format_rows,
    run_suite,
    top_outcomes,
)
from marqov.circuits import Circuit
from marqov.executors.base import BaseExecutor, ExecutionResult


class FakeExecutor(BaseExecutor):
    """Deterministic executor for benchmark suite tests."""

    def __init__(self, counts: dict[str, int] | None = None, error: Exception | None = None) -> None:
        self.counts = counts or {"00": 7, "11": 3}
        self.error = error

    async def execute(
        self,
        circuit: Circuit,
        shots: int = 1000,
        **kwargs: Any,
    ) -> ExecutionResult:
        if self.error is not None:
            raise self.error
        return ExecutionResult(
            counts=self.counts,
            backend="fake",
            execution_time_ms=12.345,
            shots=shots,
        )


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ({"00": 5, "11": 9, "01": 1, "10": 2}, {"11": 9, "00": 5, "10": 2}),
        ({"10": 4, "00": 4, "11": 3}, {"00": 4, "10": 4, "11": 3}),
    ],
)
def test_top_outcomes_returns_three_highest_counts(
    counts: dict[str, int],
    expected: dict[str, int],
) -> None:
    """Top outcomes are sorted by count, then bitstring."""
    assert top_outcomes(counts) == expected


def test_format_rows_outputs_expected_table_columns() -> None:
    """Rows render with the documented benchmark table columns."""
    output = format_rows(
        [
            BenchmarkRow(
                backend="local",
                circuit="bell",
                shots=1000,
                exec_time_ms=12.345,
                top_3_outcomes={"00": 503, "11": 497},
            )
        ]
    )

    assert output.splitlines() == [
        "| backend | circuit | shots | exec_time_ms | top_3_outcomes         |",
        "|---------|---------|-------|--------------|------------------------|",
        '| local   | bell    | 1000  | 12.3         | {"00": 503, "11": 497} |',
    ]


@pytest.mark.asyncio
async def test_run_suite_collects_rows_for_each_circuit() -> None:
    """Suite creates one row per executor/circuit pair."""
    rows = await run_suite(
        {"fake": FakeExecutor({"00": 8, "11": 2})},
        shots=10,
        circuits={"bell": Circuit(), "ghz": Circuit()},
    )

    assert len(rows) == 2
    assert rows[0].backend == "fake"
    assert rows[0].circuit == "bell"
    assert rows[0].shots == 10
    assert rows[0].top_3_outcomes == {"00": 8, "11": 2}
    assert rows[1].circuit == "ghz"


@pytest.mark.asyncio
async def test_run_suite_skips_and_logs_executor_errors() -> None:
    """Executor errors are logged to stderr without aborting the suite."""
    stderr = StringIO()

    rows = await run_suite(
        {
            "broken": FakeExecutor(error=RuntimeError("backend unavailable")),
            "fake": FakeExecutor(),
        },
        shots=10,
        circuits={"bell": Circuit()},
        stderr=stderr,
    )

    assert len(rows) == 1
    assert rows[0].backend == "fake"
    assert "Skipping broken/bell: backend unavailable" in stderr.getvalue()
