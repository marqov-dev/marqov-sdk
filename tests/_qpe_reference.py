"""Shared no-comm QPE circuit builder for the CUNQA PoC's Phase 0 tests.

Used by both tests/test_cunqa_qpe_reference_circuit.py (pure-Python,
no-cluster verification) and tests/integration/test_cunqa_parallelcluster.py
(the real cluster-gated run). A single source avoids the two tests drifting
apart, which duplicating this function across both files would risk.
"""

import math

from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT


def build_no_comm_qpe_circuit(n_counting_qubits: int, phase: float) -> QuantumCircuit:
    """Standard textbook QPE: n counting qubits + 1 eigenstate qubit.

    Exponent convention (2**q on counting qubit q) verified against Qiskit's
    QFT(inverse=True): gives the exact expected bin in 2000/2000 shots at
    phase=5/16, n=4 (see test_cunqa_qpe_reference_circuit.py). Matches the
    structure of arXiv:2511.05209's QPE benchmark. Uses n=4 here for the
    Phase 0 smoke test, NOT the paper's n=16 — that's a Phase 3 scaling
    concern.
    """
    qc = QuantumCircuit(n_counting_qubits + 1)
    qc.x(n_counting_qubits)  # eigenstate qubit starts in |1>

    for q in range(n_counting_qubits):
        qc.h(q)

    for q in range(n_counting_qubits):
        angle = 2 * math.pi * phase * (2**q)
        qc.cp(angle, q, n_counting_qubits)

    qc.append(QFT(n_counting_qubits, inverse=True), range(n_counting_qubits))
    return qc


def expected_phase_bin(n_counting_qubits: int, phase: float) -> str:
    """The expected measurement bitstring for build_no_comm_qpe_circuit's
    output once every qubit is measured (marqov.executors.cunqa.add_measure_all).

    Leftmost bit is the eigenstate qubit (index n_counting_qubits) — always
    |1>, followed by the n-bit phase estimate. Confirmed empirically
    (2000/2000 exact match via qiskit-aer).
    """
    return "1" + format(round(phase * (2**n_counting_qubits)), f"0{n_counting_qubits}b")
