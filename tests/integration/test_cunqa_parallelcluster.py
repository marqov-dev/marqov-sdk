"""End-to-end smoke test: a real no-comm QPE run distributed across CUNQA
vQPUs on a live AWS ParallelCluster.

Skipped unless MARQOV_CUNQA_INTEGRATION=1 is set. This test spends real
Slurm allocation time and should not run in CI by default.

The circuit here is the same tests/_qpe_reference.build_no_comm_qpe_circuit
used by test_cunqa_qpe_reference_circuit.py, which already proves the
exponent convention and the Circuit.from_qiskit round-trip are both correct
using only a local qiskit-aer simulation, zero cluster dependency. If this
test ever fails, check squeue/sacct and CUNQAExecutor's behavior first, not
the circuit -- its correctness is independently established and re-checked
on every plain `pytest` run. Note this test does NOT call add_measure_all
itself -- CUNQAExecutor.execute() does that internally; passing an
already-measured circuit here would be wrong (Circuit.from_qiskit would
just strip it again on the way through anyway)."""
import asyncio
import logging
import os

import pytest

from marqov.circuits import Circuit
from marqov.executors import CUNQAExecutor, CUNQAExecutorConfig
from tests._qpe_reference import build_no_comm_qpe_circuit, expected_phase_bin

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.skipif(
    os.environ.get("MARQOV_CUNQA_INTEGRATION") != "1",
    reason="requires a live CUNQA cluster; set MARQOV_CUNQA_INTEGRATION=1 to run",
)


def test_no_comm_qpe_recovers_correct_phase_at_n4() -> None:
    n_counting_qubits = 4
    true_phase = 0.3125  # = 5/16, exactly representable at n=4
    expected_bin = expected_phase_bin(n_counting_qubits, true_phase)

    qiskit_circuit = build_no_comm_qpe_circuit(n_counting_qubits, true_phase)
    circuit = Circuit.from_qiskit(qiskit_circuit)

    config = CUNQAExecutorConfig(n_qpus=4, walltime="00:10:00", mem_per_qpu_gb=4)
    executor = CUNQAExecutor(config)  # real cunqa client, no injected double

    result = asyncio.run(executor.execute(circuit, shots=1000))

    modal_outcome = max(result.counts, key=result.counts.get)
    assert modal_outcome == expected_bin, (
        f"expected modal bin {expected_bin} ({true_phase=}), got {modal_outcome} "
        f"from counts={result.counts}"
    )

    # First live check of the seeding gap flagged in Task 1: if CUNQA seeds
    # every vQPU identically by default, all 4 entries here will be equal.
    logger.warning("per-vQPU counts (check for identical entries): %s", result.metadata["per_qpu_counts"])
