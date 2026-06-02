"""Benchmark circuit generators for cross-talk characterization.

All generators accept a `qubits` parameter specifying which physical
qubit indices to use, rather than always starting from qubit 0.
"""

from __future__ import annotations

import random as stdlib_random

from marqov.circuits import Circuit


def ghz_on_qubits(qubits: list[int], depth_multiplier: int = 1) -> Circuit:
    """GHZ state preparation on specified physical qubits.

    Creates a circuit that prepares the GHZ state across the given
    qubit indices: (|0...0> + |1...1>) / sqrt(2).

    When ``depth_multiplier`` > 1, CNOT-CNOT identity pairs are appended
    (circuit folding) to increase circuit depth without changing the
    logical output state.

    Args:
        qubits: Physical qubit indices to entangle.
        depth_multiplier: Number of times the CNOT chain depth appears.
            1 means no folding (default).

    Returns:
        Circuit preparing the GHZ state on the specified qubits.
    """
    circuit = Circuit()
    circuit.h(qubits[0])
    for i in range(len(qubits) - 1):
        circuit.cnot(qubits[i], qubits[i + 1])

    # Circuit folding: append CNOT-CNOT identity pairs
    for _ in range(depth_multiplier - 1):
        for i in range(len(qubits) - 1):
            circuit.cnot(qubits[i], qubits[i + 1])
            circuit.cnot(qubits[i], qubits[i + 1])

    return circuit


def mirror_circuit(
    qubits: list[int], depth: int, seed: int | None = None
) -> Circuit:
    """Self-inverting mirror circuit.

    Applies random single-qubit gates and CNOTs for `depth` layers,
    then applies the inverse in reverse order. On a noiseless device
    the output is always |0...0>.

    Args:
        qubits: Physical qubit indices to use.
        depth: Number of forward layers before mirroring.
        seed: Random seed for reproducibility.

    Returns:
        Mirror circuit that should return to the all-zeros state.
    """
    rng = stdlib_random.Random(seed)
    circuit = Circuit()
    forward_ops: list[tuple[str, list[int], list[float]]] = []

    single_gates = ["h", "s", "t"]
    for _ in range(depth):
        for q in qubits:
            gate = rng.choice(single_gates)
            if gate == "h":
                circuit.h(q)
                forward_ops.append(("h", [q], []))
            elif gate == "s":
                circuit.s(q)
                forward_ops.append(("s_inv", [q], []))
            elif gate == "t":
                circuit.t(q)
                forward_ops.append(("t_inv", [q], []))

        if len(qubits) > 1:
            i = rng.randrange(len(qubits) - 1)
            circuit.cnot(qubits[i], qubits[i + 1])
            forward_ops.append(("cnot", [qubits[i], qubits[i + 1]], []))

    # Apply inverse in reverse order
    for gate, gate_qubits, _params in reversed(forward_ops):
        if gate == "h":
            circuit.h(gate_qubits[0])
        elif gate == "s_inv":
            # S^dagger = Z * S
            circuit.z(gate_qubits[0])
            circuit.s(gate_qubits[0])
        elif gate == "t_inv":
            # T^dagger via Rz(-pi/4)
            circuit.rz(-3.14159265 / 4, gate_qubits[0])
        elif gate == "cnot":
            circuit.cnot(gate_qubits[0], gate_qubits[1])

    return circuit


def random_circuit(
    qubits: list[int], depth: int, seed: int | None = None
) -> Circuit:
    """Random circuit on specified qubits for randomized benchmarking.

    Each layer applies a random rotation (Rx, Ry, or Rz with uniform
    random angle) to every qubit, followed by a CNOT between a random
    adjacent pair.

    Args:
        qubits: Physical qubit indices to use.
        depth: Number of layers.
        seed: Random seed for reproducibility.

    Returns:
        Random circuit on the specified qubits.
    """
    rng = stdlib_random.Random(seed)
    circuit = Circuit()

    for _ in range(depth):
        for q in qubits:
            axis = rng.choice(["rx", "ry", "rz"])
            angle = rng.uniform(0, 6.283185307)
            getattr(circuit, axis)(angle, q)

        if len(qubits) > 1:
            i = rng.randrange(len(qubits) - 1)
            circuit.cnot(qubits[i], qubits[i + 1])

    return circuit
