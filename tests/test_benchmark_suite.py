"""Tests for the unified executor benchmark suite."""

from io import StringIO

import pytest

from benchmarks.suite import BenchmarkRow, build_circuits, format_table, run_suite, top_outcomes
from marqov.circuits import Circuit
from marqov.executors.base import BaseExecutor, ExecutionResult
from marqov.executors.local import LocalExecutor, LocalExecutorConfig


class FailingExecutor(BaseExecutor):
    """Executor that always fails, used to verify skip-and-log behavior."""

    async def execute(
        self, circuit: Circuit, shots: int = 1000, **kwargs: object
    ) -> ExecutionResult:
        raise RuntimeError("backend unavailable")


@pytest.mark.parametrize(
    ("name", "num_qubits"),
    [("bell", 2), ("ghz", 3), ("random_d5", 3)],
)
def test_build_circuits(name: str, num_qubits: int) -> None:
    """The suite contains all documented circuits with the expected width."""
    circuit = build_circuits()[name]
    assert len(circuit.simulate().qubits) == num_qubits


def test_top_outcomes_limits_and_sorts_counts() -> None:
    """Only the three most common outcomes are returned."""
    assert top_outcomes({"11": 2, "00": 4, "10": 3, "01": 2}) == {
        "00": 4,
        "10": 3,
        "01": 2,
    }


@pytest.mark.asyncio
async def test_run_suite_with_local_executor() -> None:
    """The complete suite runs without credentials on the local executor."""
    rows = await run_suite(
        {"local": LocalExecutor(LocalExecutorConfig(seed=0))},
        shots=100,
    )

    assert [row.circuit for row in rows] == ["bell", "ghz", "random_d5"]
    assert all(row.backend == "local" for row in rows)
    assert all(row.shots == 100 for row in rows)
    assert all(sum(row.top_3_outcomes.values()) <= 100 for row in rows)


@pytest.mark.asyncio
async def test_run_suite_skips_and_logs_failing_backend() -> None:
    """A failed backend is logged and does not abort remaining backends."""
    stderr = StringIO()
    rows = await run_suite(
        {
            "broken": FailingExecutor(),
            "local": LocalExecutor(LocalExecutorConfig(seed=0)),
        },
        shots=10,
        circuits={"bell": build_circuits()["bell"]},
        stderr=stderr,
    )

    assert [row.backend for row in rows] == ["local"]
    assert "Skipping backend broken: backend unavailable" in stderr.getvalue()


def test_format_table_uses_documented_columns() -> None:
    """Table output exactly uses the documented five-column layout."""
    table = format_table([BenchmarkRow("local", "bell", 1000, 12.34, {"00": 503, "11": 497})])
    assert table.splitlines()[0].split(" | ") == [
        "backend",
        "circuit",
        "shots",
        "exec_time_ms",
        "top_3_outcomes",
    ]
    assert "local" in table
    assert "12.3" in table
    assert '{"00": 503, "11": 497}' in table
