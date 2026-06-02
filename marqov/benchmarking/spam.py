"""SPAM (State Preparation And Measurement) benchmarking.

Characterizes per-qubit measurement readout errors by preparing known
states and measuring how accurately they are read out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from marqov.circuits import Circuit
from marqov.executors.base import BaseExecutor


@dataclass(frozen=True)
class ConfusionMatrix:
    """2x2 readout confusion matrix for a single qubit.

    p00: P(measure 0 | prepared 0) — correct |0⟩ readout
    p01: P(measure 1 | prepared 0) — false positive
    p10: P(measure 0 | prepared 1) — false negative
    p11: P(measure 1 | prepared 1) — correct |1⟩ readout
    """

    p00: float
    p01: float
    p10: float
    p11: float

    @property
    def readout_fidelity(self) -> float:
        """Average readout fidelity: (p00 + p11) / 2."""
        return (self.p00 + self.p11) / 2


@dataclass
class SPAMResult:
    """Result of SPAM benchmarking across multiple qubits."""

    matrices: dict[int, ConfusionMatrix]
    shots: int
    counts_prepared_0: dict[int, dict[str, int]] = field(default_factory=dict)
    counts_prepared_1: dict[int, dict[str, int]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def readout_fidelity(self) -> dict[int, float]:
        """Per-qubit readout fidelity."""
        return {q: m.readout_fidelity for q, m in self.matrices.items()}

    def confusion_matrix(self, qubit: int) -> ConfusionMatrix:
        """Get confusion matrix for a specific qubit."""
        return self.matrices[qubit]


async def spam_benchmark(
    executor: BaseExecutor,
    *,
    qubits: list[int],
    shots: int = 1000,
) -> SPAMResult:
    """Run per-qubit SPAM benchmarking.

    For each qubit, prepares |0⟩ and |1⟩ states and measures readout
    accuracy. Returns confusion matrices characterizing measurement errors.

    Args:
        executor: Quantum executor to run circuits on.
        qubits: List of qubit indices to benchmark.
        shots: Number of measurement shots per preparation state.

    Returns:
        SPAMResult with per-qubit confusion matrices.

    Raises:
        ValueError: If qubits list is empty or contains negative indices.
    """
    if not qubits:
        raise ValueError("qubits list must not be empty")
    for q in qubits:
        if q < 0:
            raise ValueError(f"qubit index must be non-negative, got {q}")

    matrices: dict[int, ConfusionMatrix] = {}
    counts_prep_0: dict[int, dict[str, int]] = {}
    counts_prep_1: dict[int, dict[str, int]] = {}

    for qubit in qubits:
        # Prepare |0⟩ on target qubit (X·X = identity, registers the qubit)
        circuit_0 = Circuit()
        circuit_0.x(qubit).x(qubit)
        result_0 = await executor.execute(circuit_0, shots=shots)
        counts_prep_0[qubit] = result_0.counts

        # Prepare |1⟩ on target qubit (X gate then measure)
        circuit_1 = Circuit()
        circuit_1.x(qubit)
        result_1 = await executor.execute(circuit_1, shots=shots)
        counts_prep_1[qubit] = result_1.counts

        # Extract confusion matrix from counts
        # Count how many times the target qubit was measured as 0 or 1
        total_0 = sum(result_0.counts.values())
        total_1 = sum(result_1.counts.values())

        meas_0_given_prep_0 = 0
        meas_1_given_prep_0 = 0
        for bitstring, count in result_0.counts.items():
            bit = bitstring[qubit] if qubit < len(bitstring) else "0"
            if bit == "0":
                meas_0_given_prep_0 += count
            else:
                meas_1_given_prep_0 += count

        meas_0_given_prep_1 = 0
        meas_1_given_prep_1 = 0
        for bitstring, count in result_1.counts.items():
            bit = bitstring[qubit] if qubit < len(bitstring) else "0"
            if bit == "0":
                meas_0_given_prep_1 += count
            else:
                meas_1_given_prep_1 += count

        p00 = meas_0_given_prep_0 / total_0 if total_0 > 0 else 0
        p01 = meas_1_given_prep_0 / total_0 if total_0 > 0 else 0
        p10 = meas_0_given_prep_1 / total_1 if total_1 > 0 else 0
        p11 = meas_1_given_prep_1 / total_1 if total_1 > 0 else 0

        matrices[qubit] = ConfusionMatrix(p00=p00, p01=p01, p10=p10, p11=p11)

    return SPAMResult(
        matrices=matrices,
        shots=shots,
        counts_prepared_0=counts_prep_0,
        counts_prepared_1=counts_prep_1,
        metadata={"qubits": qubits},
    )


def build_correction_matrix(spam: SPAMResult) -> np.ndarray:
    """Build a correction matrix from SPAM benchmark results.

    Constructs the joint confusion matrix as the Kronecker product of
    per-qubit confusion matrices (sorted by qubit index), then inverts it.

    Args:
        spam: SPAM benchmark result containing per-qubit confusion matrices.

    Returns:
        Inverse of the joint confusion matrix (2^n x 2^n).

    Raises:
        ValueError: If SPAMResult has no confusion matrices.
        np.linalg.LinAlgError: If the joint matrix is singular.
    """
    if not spam.matrices:
        raise ValueError("SPAMResult has no confusion matrices")

    sorted_qubits = sorted(spam.matrices.keys())

    # Build joint confusion matrix via Kronecker product
    joint: np.ndarray | None = None
    for q in sorted_qubits:
        cm = spam.matrices[q]
        # Column-stochastic: columns are conditional probability distributions
        # Column 0: P(meas|prep=0), Column 1: P(meas|prep=1)
        q_matrix = np.array([[cm.p00, cm.p10], [cm.p01, cm.p11]])
        if joint is None:
            joint = q_matrix
        else:
            joint = np.kron(joint, q_matrix)

    assert joint is not None
    return np.linalg.inv(joint)


def apply_spam_correction(
    counts: dict[str, int],
    correction: np.ndarray,
) -> dict[str, int]:
    """Apply SPAM correction to measurement counts.

    Multiplies the correction matrix by the counts vector, clamps negative
    values to zero, and re-normalizes to preserve total shot count.

    Args:
        counts: Measurement counts keyed by bitstring.
        correction: Correction matrix from build_correction_matrix().

    Returns:
        Corrected counts with non-negative integer values.

    Raises:
        ValueError: If bitstring length doesn't match correction matrix size.
    """
    if not counts:
        return {}

    # Determine qubit count from bitstring length
    sample_key = next(iter(counts))
    n_qubits = len(sample_key)
    dim = 2**n_qubits

    if correction.shape != (dim, dim):
        raise ValueError(
            f"Correction matrix dimension {correction.shape[0]} "
            f"does not match {n_qubits}-qubit state space (2^{n_qubits}={dim})"
        )

    total_shots = sum(counts.values())

    # Build counts vector
    vec = np.zeros(dim)
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        vec[idx] = count

    # Apply correction
    corrected_vec = correction @ vec

    # Clamp negatives to zero
    corrected_vec = np.maximum(corrected_vec, 0)

    # Re-normalize to preserve total shot count
    current_sum = corrected_vec.sum()
    if current_sum > 0:
        corrected_vec = corrected_vec * (total_shots / current_sum)

    # Convert back to dict, rounding to integers
    result: dict[str, int] = {}
    for i in range(dim):
        count = int(round(corrected_vec[i]))
        if count > 0:
            bitstring = format(i, f"0{n_qubits}b")
            result[bitstring] = count

    return result
