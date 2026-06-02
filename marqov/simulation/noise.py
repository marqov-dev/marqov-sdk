"""Noise modeling for quantum simulation.

Provides noise channel types and a NoiseModel container for configuring
realistic quantum hardware error simulation. Only works with the 'aer'
backend in qristal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _validate_probability(value: float, name: str) -> None:
    """Validate that a value is a valid probability [0, 1]."""
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")


@dataclass(frozen=True)
class Depolarizing:
    """Depolarizing noise channel — random errors with given probability.

    Applies random Pauli errors (X, Y, Z) with total probability p.
    """

    probability: float

    def __post_init__(self) -> None:
        _validate_probability(self.probability, "probability")

    def __repr__(self) -> str:
        return f"Depolarizing(probability={self.probability})"


@dataclass(frozen=True)
class AmplitudeDamping:
    """Amplitude damping channel — models T1 energy loss.

    A |1⟩ state decays to |0⟩ with rate gamma.
    """

    gamma: float

    def __post_init__(self) -> None:
        _validate_probability(self.gamma, "gamma")

    def __repr__(self) -> str:
        return f"AmplitudeDamping(gamma={self.gamma})"


@dataclass(frozen=True)
class PhaseDamping:
    """Phase damping channel — models T2 dephasing.

    Phase information is lost at rate gamma without energy change.
    """

    gamma: float

    def __post_init__(self) -> None:
        _validate_probability(self.gamma, "gamma")

    def __repr__(self) -> str:
        return f"PhaseDamping(gamma={self.gamma})"


@dataclass(frozen=True)
class ReadoutError:
    """Measurement readout error.

    p0_given1: probability of reading 0 when state is 1
    p1_given0: probability of reading 1 when state is 0
    """

    p0_given1: float
    p1_given0: float

    def __post_init__(self) -> None:
        _validate_probability(self.p0_given1, "p0_given1")
        _validate_probability(self.p1_given0, "p1_given0")

    def __repr__(self) -> str:
        return f"ReadoutError(p0_given1={self.p0_given1}, p1_given0={self.p1_given0})"


# Union type for all noise channels
NoiseChannel = Depolarizing | AmplitudeDamping | PhaseDamping | ReadoutError


class NoiseModel:
    """Container for noise channel assignments.

    Maps noise channels to qubit indices. Each add() call applies the
    channel independently to each listed qubit (single-qubit noise only;
    two-qubit correlated noise is not supported in this version).
    """

    def __init__(self) -> None:
        self._entries: list[tuple[NoiseChannel, list[int]]] = []

    def add(self, channel: NoiseChannel, *, qubits: list[int]) -> None:
        """Add a noise channel applied to the given qubits.

        Args:
            channel: The noise channel to apply.
            qubits: List of qubit indices. Must be non-empty, non-negative.

        Raises:
            ValueError: If qubits list is empty or contains negative indices.
        """
        if not qubits:
            raise ValueError("qubits must be a non-empty list")
        if any(q < 0 for q in qubits):
            raise ValueError("qubit indices must be non-negative")
        self._entries.append((channel, list(qubits)))

    def entries(self) -> list[tuple[NoiseChannel, list[int]]]:
        """Return all (channel, qubits) pairs."""
        return list(self._entries)

    @classmethod
    def depolarizing_uniform(cls, *, p: float, num_qubits: int) -> NoiseModel:
        """Create a model with uniform depolarizing noise on all qubits.

        Args:
            p: Depolarizing probability.
            num_qubits: Number of qubits in the circuit.
        """
        model = cls()
        model.add(Depolarizing(p), qubits=list(range(num_qubits)))
        return model

    @classmethod
    def realistic_device(
        cls,
        *,
        t1: float,
        t2: float,
        gate_time: float,
        num_qubits: int,
    ) -> NoiseModel:
        """Create a model from physical device parameters.

        Computes damping rates from T1/T2 relaxation times and gate duration:
          gamma_ad = 1 - exp(-gate_time / T1)
          gamma_pd = 1 - exp(-gate_time / T2)

        Args:
            t1: T1 relaxation time (seconds).
            t2: T2 dephasing time (seconds).
            gate_time: Single gate duration (seconds).
            num_qubits: Number of qubits in the circuit.
        """
        gamma_ad = 1 - math.exp(-gate_time / t1)
        gamma_pd = 1 - math.exp(-gate_time / t2)
        qubits = list(range(num_qubits))

        model = cls()
        model.add(AmplitudeDamping(gamma_ad), qubits=qubits)
        model.add(PhaseDamping(gamma_pd), qubits=qubits)
        return model
