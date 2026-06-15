"""Tests for the unified multi-backend benchmark suite (``benchmarks/suite.py``)."""

from __future__ import annotations

import io
from typing import Any

import pytest

from benchmarks.suite import (
    BenchmarkRow,
    build_executors,
    default_circuits,
    format_table,
    main,
    random_depth5_circuit,
    run_suite,
    top_outcomes,
)
from marqov.circuits import Circuit
from marqov.executors.base import BaseExecutor, ExecutionResult
from marqov.executors.local import LocalExecutor, LocalExecutorConfig


class _FixedExecutor(BaseExecutor):
    """Executor that returns fixed counts — for deterministic table assertions."""

    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts

    async def execute(self, circuit: Circuit, shots: int = 1000, **kwargs: Any) -> ExecutionResult:
        return ExecutionResult(
            counts=dict(self._counts),
            backend="fixed",
            execution_time_ms=12.34,
            shots=shots,
        )


class _FailOnNthExecutor(BaseExecutor):
    """Executor that raises on its N-th ``execute`` call and succeeds otherwise.

    Used to verify that a backend which fails partway through is skipped
    *atomically* (no partial rows survive).
    """

    def __init__(self, fail_on_call: int) -> None:
        self._fail_on_call = fail_on_call
        self._calls = 0

    async def execute(self, circuit: Circuit, shots: int = 1000, **kwargs: Any) -> ExecutionResult:
        self._calls += 1
        if self._calls == self._fail_on_call:
            raise RuntimeError("backend unavailable")
        return ExecutionResult(
            counts={"00": shots}, backend="flaky", execution_time_ms=1.0, shots=shots
        )


class _ClampedShotsExecutor(BaseExecutor):
    """Executor that runs fewer shots than requested — like a shot-capped backend.

    The table must report ``result.shots`` (what actually ran), not the requested
    value, so a clamped run is never silently misreported.
    """

    def __init__(self, actual_shots: int) -> None:
        self._actual_shots = actual_shots

    async def execute(self, circuit: Circuit, shots: int = 1000, **kwargs: Any) -> ExecutionResult:
        return ExecutionResult(
            counts={"00": self._actual_shots},
            backend="clamped",
            execution_time_ms=1.0,
            shots=self._actual_shots,
        )


# --------------------------------------------------------------------------- #
# top_outcomes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        # Ordered by count descending, then bitstring ascending; limited to 3.
        ({"00": 5, "11": 2, "01": 3, "10": 1}, {"00": 5, "01": 3, "11": 2}),
        # Ties broken by bitstring ascending.
        ({"10": 4, "00": 4, "11": 3}, {"00": 4, "10": 4, "11": 3}),
        # The count-descending order must win even when the top key sorts last —
        # this is the case a naive json.dumps(sort_keys=True) would corrupt.
        ({"11": 600, "00": 400}, {"11": 600, "00": 400}),
        # Fewer than the limit, and empty.
        ({"0": 7}, {"0": 7}),
        ({}, {}),
    ],
)
def test_top_outcomes(counts: dict[str, int], expected: dict[str, int]) -> None:
    """top_outcomes ranks by count (desc) then bitstring (asc), capped at 3."""
    result = top_outcomes(counts)
    assert result == expected
    # dict equality ignores order, so assert the ranking order explicitly too.
    assert list(result.items()) == list(expected.items())


# --------------------------------------------------------------------------- #
# format_table — exact CONTRIBUTING.md §5 column format
# --------------------------------------------------------------------------- #


def test_format_table_matches_contributing_section5() -> None:
    """A single row renders as the exact markdown table documented in §5."""
    table = format_table(
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

    assert table.splitlines() == [
        "| backend | circuit | shots | exec_time_ms | top_3_outcomes         |",
        "|---------|---------|-------|--------------|------------------------|",
        '| local   | bell    | 1000  | 12.3         | {"00": 503, "11": 497} |',
    ]


def test_format_table_preserves_count_descending_outcome_order() -> None:
    """The rendered JSON keeps count-descending order (no sort_keys corruption)."""
    table = format_table(
        [BenchmarkRow("local", "bell", 1000, 1.0, top_outcomes({"11": 600, "00": 400}))]
    )
    assert '{"11": 600, "00": 400}' in table


def test_format_table_empty_rows_is_wellformed() -> None:
    """Rendering with no rows yields just the header + separator, never crashes."""
    lines = format_table([]).splitlines()
    assert lines == [
        "| backend | circuit | shots | exec_time_ms | top_3_outcomes |",
        "|---------|---------|-------|--------------|----------------|",
    ]


# --------------------------------------------------------------------------- #
# run_suite — end-to-end with the real LocalExecutor (no credentials)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_run_suite_local_end_to_end() -> None:
    """The full suite runs against a real LocalExecutor with no credentials.

    The assertions pin physics invariants of the sampling + aggregation pipeline
    (total conservation, the support of Bell/GHZ, a roughly balanced split) so a
    numerically-wrong regression — e.g. counts silently halved — is caught, not
    just the structural shape.
    """
    shots = 200
    rows = await run_suite(
        {"local": LocalExecutor(LocalExecutorConfig(seed=0))},
        shots=shots,
        circuits=default_circuits(seed=0),
    )

    assert [row.circuit for row in rows] == ["bell", "ghz", "random_d5"]
    assert all(row.backend == "local" for row in rows)
    assert all(row.shots == shots for row in rows)
    assert all(row.exec_time_ms >= 0.0 for row in rows)

    by_circuit = {row.circuit: row.top_3_outcomes for row in rows}

    # Bell and GHZ each have exactly two outcomes, so top_3 captures *all* counts:
    # the sum must equal shots exactly (catches a halving / dropped-shot bug) and
    # the split must stay roughly balanced (catches a collapsed distribution).
    for name, support in (("bell", {"00", "11"}), ("ghz", {"000", "111"})):
        outcomes = by_circuit[name]
        assert set(outcomes) <= support
        assert sum(outcomes.values()) == shots
        assert all(0.3 * shots < count < 0.7 * shots for count in outcomes.values())

    # random_d5 may have more than three outcomes, so top_3 can sum below shots —
    # but never above it, and at least one outcome must have been observed.
    random_d5 = by_circuit["random_d5"]
    assert len(random_d5) <= 3
    assert 0 < sum(random_d5.values()) <= shots


@pytest.mark.asyncio
async def test_run_suite_outcomes_are_pinned_for_seed() -> None:
    """Exact-value regression guard for the suite's reproducibility promise.

    With both the circuit and the sampler seeded to 0 the whole pipeline is
    deterministic, so the exact counts are pinned. This flags any drift in
    circuit construction, simulation, sampling, or aggregation. The values are
    tied to the locked numpy (<2.4) and quantumflow (v1.4.0) versions; re-baseline
    them deliberately if those pins change.
    """
    rows = await run_suite(
        {"local": LocalExecutor(LocalExecutorConfig(seed=0))},
        shots=200,
        circuits=default_circuits(seed=0),
    )

    assert {row.circuit: row.top_3_outcomes for row in rows} == {
        "bell": {"11": 112, "00": 88},
        "ghz": {"111": 105, "000": 95},
        "random_d5": {"001": 109, "000": 91},
    }


@pytest.mark.asyncio
async def test_run_suite_is_reproducible_with_seed() -> None:
    """Identical seeds yield identical outcome rows — usable for regression tests."""
    circuits = default_circuits(seed=0)
    first = await run_suite(
        {"local": LocalExecutor(LocalExecutorConfig(seed=0))}, shots=500, circuits=circuits
    )
    second = await run_suite(
        {"local": LocalExecutor(LocalExecutorConfig(seed=0))}, shots=500, circuits=circuits
    )
    assert [r.top_3_outcomes for r in first] == [r.top_3_outcomes for r in second]


@pytest.mark.asyncio
async def test_run_suite_records_executor_shots_not_request() -> None:
    """The shots column reflects what the backend actually ran, not what was asked.

    Real hardware clamps shot counts, so the row must carry ``result.shots`` —
    recording the requested value instead would silently misreport a capped run.
    """
    rows = await run_suite(
        {"clamped": _ClampedShotsExecutor(actual_shots=64)},
        shots=1000,
        circuits={"bell": Circuit()},
    )
    assert [row.shots for row in rows] == [64]


# --------------------------------------------------------------------------- #
# run_suite — skip-and-log error handling (atomic per backend)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_run_suite_skips_and_logs_failing_backend() -> None:
    """A backend that always fails is logged to stderr and produces no rows."""
    stderr = io.StringIO()
    rows = await run_suite(
        {
            "broken": _FailOnNthExecutor(fail_on_call=1),
            "local": LocalExecutor(LocalExecutorConfig(seed=0)),
        },
        shots=50,
        stderr=stderr,
    )

    assert {row.backend for row in rows} == {"local"}
    assert "skipping broken (failed on bell)" in stderr.getvalue()
    assert "backend unavailable" in stderr.getvalue()


@pytest.mark.asyncio
async def test_run_suite_skips_whole_backend_atomically() -> None:
    """A backend that fails on its 2nd circuit contributes *zero* rows (atomic)."""
    stderr = io.StringIO()
    rows = await run_suite(
        {
            "flaky": _FailOnNthExecutor(fail_on_call=2),  # passes bell, fails ghz
            "local": LocalExecutor(LocalExecutorConfig(seed=0)),
        },
        shots=50,
        stderr=stderr,
    )

    # No partial 'flaky' row leaks through despite bell having succeeded.
    assert all(row.backend != "flaky" for row in rows)
    assert [row.circuit for row in rows] == ["bell", "ghz", "random_d5"]
    assert "skipping flaky (failed on ghz)" in stderr.getvalue()


# --------------------------------------------------------------------------- #
# random_depth5_circuit
# --------------------------------------------------------------------------- #


def test_random_depth5_circuit_is_deterministic() -> None:
    """The same seed always produces the same circuit; different seeds differ."""
    assert random_depth5_circuit(7).to_dict() == random_depth5_circuit(7).to_dict()
    assert random_depth5_circuit(7).to_dict() != random_depth5_circuit(8).to_dict()


def test_random_depth5_circuit_uses_canonical_gates_on_three_qubits() -> None:
    """The random circuit stays within 3 qubits and the canonical gate set."""
    # QuantumFlow op names the generator may emit (single-qubit gates + CNOT).
    allowed = {"H", "X", "Y", "Z", "S", "T", "CNot"}
    circuit = random_depth5_circuit(0)
    for gate in circuit.to_dict()["gates"]:
        assert gate["gate"] in allowed
        assert all(0 <= qubit < 3 for qubit in gate["qubits"])


# --------------------------------------------------------------------------- #
# build_executors + CLI
# --------------------------------------------------------------------------- #


def test_build_executors_dedupes_local() -> None:
    """Repeated names collapse to a single ordered entry."""
    executors = build_executors(["local", "local"], seed=0)
    assert list(executors) == ["local"]
    assert isinstance(executors["local"], LocalExecutor)


def test_build_executors_rejects_unknown_name() -> None:
    """An unknown executor name raises a helpful error."""
    with pytest.raises(ValueError, match="Unsupported executor 'rigetti'"):
        build_executors(["rigetti"], seed=0)


def test_cli_main_runs_local_end_to_end(capsys: pytest.CaptureFixture[str]) -> None:
    """`main --executor local` prints the table and exits 0."""
    exit_code = main(["--executor", "local", "--shots", "50", "--seed", "0"])
    assert exit_code == 0

    out_lines = capsys.readouterr().out.splitlines()
    assert out_lines[0].startswith("| backend | circuit")
    data_rows = [line for line in out_lines if line.startswith("| local")]
    assert len(data_rows) == 3


def test_cli_main_rejects_nonpositive_shots(capsys: pytest.CaptureFixture[str]) -> None:
    """`--shots 0` exits 2 with a clean stderr message, not a traceback."""
    exit_code = main(["--executor", "local", "--shots", "0"])
    assert exit_code == 2
    assert "--shots must be a positive integer" in capsys.readouterr().err


def test_cli_main_rejects_unknown_executor(capsys: pytest.CaptureFixture[str]) -> None:
    """An unknown `--executor` exits 2 with a helpful stderr message."""
    exit_code = main(["--executor", "bogus"])
    assert exit_code == 2
    assert "Unsupported executor 'bogus'" in capsys.readouterr().err


def test_cli_main_exits_1_when_all_backends_fail(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """When every backend fails, the CLI exits non-zero so CI can flag a broken run.

    The shipped CLI only builds the always-succeeding ``local`` executor, so the
    all-fail path (``return 0 if rows else 1``) is driven here by substituting a
    failing backend.
    """
    monkeypatch.setattr(
        "benchmarks.suite.build_executors",
        lambda names, seed: {"broken": _FailOnNthExecutor(fail_on_call=1)},
    )
    exit_code = main(["--executor", "broken", "--shots", "50"])
    assert exit_code == 1
    assert "skipping broken" in capsys.readouterr().err
