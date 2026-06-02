"""Cross-talk experiment runner.

Runs benchmark circuits in single-tenant and multi-tenant configurations
to measure cross-talk effects. Uses FakeExecutor-compatible interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from marqov.circuits import Circuit
from marqov.executors.base import BaseExecutor
from marqov.qmt.characterization.benchmarks import (
    ghz_on_qubits,
    mirror_circuit,
    random_circuit,
)
from marqov.qmt.models import PackingPlan, QMTJob, QubitMapping
from marqov.qmt.scheduler.splitter import split_results


@dataclass
class CrossTalkExperiment:
    """Defines a single cross-talk measurement experiment."""

    target_qubits: list[int]
    neighbor_qubits: list[int]
    benchmark: str  # "ghz", "mirror", "random"
    shots: int = 1000
    benchmark_seed: int | None = None
    benchmark_depth: int = 2


@dataclass
class ExperimentResult:
    """Result of a cross-talk experiment."""

    experiment: CrossTalkExperiment
    target_counts: dict[str, int]
    neighbor_counts: dict[str, int] | None = None
    target_fidelity: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _make_benchmark(
    qubits: list[int],
    benchmark: str,
    seed: int | None,
    depth: int,
) -> Circuit:
    """Create a benchmark circuit on the given qubits."""
    if benchmark == "ghz":
        return ghz_on_qubits(qubits, depth_multiplier=depth)
    elif benchmark == "mirror":
        return mirror_circuit(qubits, depth=depth, seed=seed)
    elif benchmark == "random":
        return random_circuit(qubits, depth=depth, seed=seed)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def _compute_ghz_fidelity(counts: dict[str, int], num_qubits: int) -> float:
    """GHZ fidelity: fraction of shots in |0...0> or |1...1>."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    all_zeros = "0" * num_qubits
    all_ones = "1" * num_qubits
    correct = counts.get(all_zeros, 0) + counts.get(all_ones, 0)
    return correct / total


def _compute_mirror_fidelity(counts: dict[str, int], num_qubits: int) -> float:
    """Mirror fidelity: fraction of shots returning to |0...0>."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    all_zeros = "0" * num_qubits
    return counts.get(all_zeros, 0) / total


def _compute_fidelity(
    counts: dict[str, int], num_qubits: int, benchmark: str
) -> float:
    """Dispatch to benchmark-specific fidelity computation."""
    if benchmark == "ghz":
        return _compute_ghz_fidelity(counts, num_qubits)
    elif benchmark == "mirror":
        return _compute_mirror_fidelity(counts, num_qubits)
    else:
        # Random circuits have no fixed ideal output
        return 0.0


def _merge_circuits(a: Circuit, b: Circuit) -> Circuit:
    """Merge two circuits on disjoint qubits into one composite circuit.

    Replays the operations from both circuits into a new composite circuit.
    The circuits must operate on disjoint physical qubits.
    """
    composite = Circuit()
    for op in a._qf._elements:
        composite._qf += op
    for op in b._qf._elements:
        composite._qf += op
    return composite


async def run_cross_talk_experiment(
    experiment: CrossTalkExperiment,
    executor: BaseExecutor,
) -> ExperimentResult:
    """Run a cross-talk experiment and return the result.

    In single-tenant mode (no neighbor qubits), runs the benchmark circuit
    alone. In multi-tenant mode, merges target and neighbor benchmark
    circuits into a composite and splits the results back per-job.

    Args:
        experiment: The experiment configuration.
        executor: The executor to run circuits on.

    Returns:
        ExperimentResult with per-job counts and fidelity.
    """
    target_circuit = _make_benchmark(
        experiment.target_qubits,
        experiment.benchmark,
        experiment.benchmark_seed,
        experiment.benchmark_depth,
    )

    # Single-tenant: run target circuit alone
    if not experiment.neighbor_qubits:
        result = await executor.execute(target_circuit, shots=experiment.shots)
        fidelity = _compute_fidelity(
            result.counts, len(experiment.target_qubits), experiment.benchmark
        )
        return ExperimentResult(
            experiment=experiment,
            target_counts=result.counts,
            target_fidelity=fidelity,
        )

    # Multi-tenant: merge target + neighbor circuits and split results
    neighbor_circuit = _make_benchmark(
        experiment.neighbor_qubits,
        experiment.benchmark,
        (experiment.benchmark_seed + 1)
        if experiment.benchmark_seed is not None
        else None,
        experiment.benchmark_depth,
    )

    composite = _merge_circuits(target_circuit, neighbor_circuit)
    result = await executor.execute(composite, shots=experiment.shots)

    # Build a PackingPlan so the splitter can attribute bits to each job
    target_job = QMTJob(circuit=target_circuit, submitter="target")
    neighbor_job = QMTJob(circuit=neighbor_circuit, submitter="neighbor")

    all_qubits = experiment.target_qubits + experiment.neighbor_qubits
    total_qubits = max(all_qubits) + 1

    plan = PackingPlan(
        jobs=[target_job, neighbor_job],
        mappings=[
            QubitMapping(
                job_id=target_job.job_id,
                logical_to_physical={
                    i: q for i, q in enumerate(experiment.target_qubits)
                },
            ),
            QubitMapping(
                job_id=neighbor_job.job_id,
                logical_to_physical={
                    i: q for i, q in enumerate(experiment.neighbor_qubits)
                },
            ),
        ],
        guard_qubits=set(),
        device_name="characterization",
        total_qubits=total_qubits,
    )

    split = split_results(result.counts, plan, shots=experiment.shots)
    target_result = next(r for r in split if r.job_id == target_job.job_id)
    neighbor_result = next(r for r in split if r.job_id == neighbor_job.job_id)

    fidelity = _compute_fidelity(
        target_result.counts, len(experiment.target_qubits), experiment.benchmark
    )

    return ExperimentResult(
        experiment=experiment,
        target_counts=target_result.counts,
        neighbor_counts=neighbor_result.counts,
        target_fidelity=fidelity,
    )
