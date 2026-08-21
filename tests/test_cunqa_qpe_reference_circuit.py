"""Committed, re-runnable verification for the no-comm QPE circuit used by
tests/integration/test_cunqa_parallelcluster.py.

Pure-Python, no-cluster checks. Both circuit tests measure ALL qubits (via
add_measure_all, the same helper CUNQAExecutor itself uses).
"""

from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from marqov.circuits import Circuit
from marqov.executors.cunqa import add_measure_all
from tests._qpe_reference import build_no_comm_qpe_circuit, expected_phase_bin


def test_qpe_exponent_convention_recovers_correct_phase_bin() -> None:
    """The bit-ordering check: 2**q on counting qubit q must pair correctly
    with Qiskit's QFT(inverse=True) convention."""
    n, phase = 4, 0.3125  # = 5/16, exactly representable at n=4
    expected_bin = expected_phase_bin(n, phase)

    qc = add_measure_all(build_no_comm_qpe_circuit(n, phase))
    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=2000, seed_simulator=42).result()
    counts = result.get_counts()

    modal = max(counts, key=counts.get)
    assert modal == expected_bin
    assert counts[expected_bin] / 2000 > 0.95


def test_from_qiskit_round_trip_still_recovers_correct_phase_bin() -> None:
    """Semantic round-trip check: the actual property Task 4 depends on."""
    n, phase = 4, 0.3125
    expected_bin = expected_phase_bin(n, phase)

    round_tripped = Circuit.from_qiskit(build_no_comm_qpe_circuit(n, phase)).to_qiskit()
    measured = add_measure_all(round_tripped)

    sim = AerSimulator()
    tqc = transpile(measured, sim)
    counts = sim.run(tqc, shots=2000, seed_simulator=42).result().get_counts()

    assert counts[expected_bin] / 2000 > 0.95


def test_add_measure_all_rejects_any_preexisting_classical_bits() -> None:
    """add_measure_all must reject ANY input with clbits already present —
    not just a partial register. Circuit.to_qiskit() never produces one
    (it's unitary-only), so there's nothing legitimate to tolerate."""
    import pytest

    qc = QuantumCircuit(3)
    qc.add_register(ClassicalRegister(1, "partial"))

    with pytest.raises(ValueError, match="measurement-free"):
        add_measure_all(qc)


def test_add_measure_all_rejects_full_width_unmeasured_circuit() -> None:
    """A circuit with a full-width classical register but zero measure
    instructions must be rejected, not silently passed through unmeasured —
    the bare-unitary bug this function exists to prevent."""
    import pytest

    qc = QuantumCircuit(3)
    qc.add_register(ClassicalRegister(3, "c"))
    qc.h(0)  # some gate, deliberately no .measure() call

    with pytest.raises(ValueError, match="measurement-free"):
        add_measure_all(qc)
