"""Refuse modifier data before mapping or SWAP recognition can discard it."""

from unittest.mock import patch

import numpy as np
import pytest

from marqov import Circuit

quil = pytest.importorskip("pyquil")
from pyquil.gates import CNOT, CZ, H, RX, RY, RZ, S, SWAP, T, X, Y, Z  # noqa: E402
from pyquil.simulation.tools import program_unitary  # noqa: E402


@pytest.mark.parametrize("gate", [
    RX(0.37, 0).dagger(),
    S(0).dagger(),
    RX(0.37, 0).controlled(1),
    RX(0.37, 1).controlled(0),
    H(0).controlled(1),
    RZ(0.37, 0).forked(1, [0.91]),
    RZ(0.37, 1).forked(0, [-0.91]),
    RX(0.37, 0).controlled(1).controlled(2),
    RY(0.37, 0).controlled(1).dagger(),
    RY(0.37, 0).dagger().controlled(1),
    RZ(0.37, 0).forked(1, [0.91]).dagger(),
    RX(0.37, 0).dagger().dagger(),
])
def test_modifiers_rejected_before_mapping(gate):
    source = quil.Program(H(0), gate, X(0))
    with patch.object(
        Circuit, "_try_consume_swap_from_cnots", wraps=Circuit._try_consume_swap_from_cnots
    ) as swap:
        with pytest.raises(NotImplementedError, match=f"gate '{gate.name}' at index 1"):
            Circuit.from_pyquil(source)
        swap.assert_not_called()


@pytest.mark.parametrize("position", [0, 1, 2])
def test_modified_cnot_cannot_bypass_guard_through_swap_shortcut(position):
    gates = [CNOT(0, 1), CNOT(1, 0), CNOT(0, 1)]
    gates[position] = gates[position].dagger()
    with pytest.raises(NotImplementedError, match=f"DAGGER.*gate 'CNOT' at index {position}"):
        Circuit.from_pyquil(quil.Program(*gates))


@pytest.mark.parametrize("angle", [0.37, -1.1, 4.0])
@pytest.mark.parametrize("control, target", [(0, 1), (1, 0)])
def test_unmodified_canonical_import_matches_source(angle, control, target):
    source = quil.Program(
        H(control), X(target), Y(control), Z(target), S(control), T(target),
        RX(angle, target), RY(-angle, control), RZ(angle, target),
        CNOT(control, target), CZ(target, control), SWAP(target, control),
    )
    restored = Circuit.from_pyquil(source)
    # PyQuil orders basis states |q1 q0>; QuantumFlow orders |q0 q1>.
    # Explicitly reverse both row and column bit indices before comparing.
    permutation = [0, 2, 1, 3]
    expected = program_unitary(source, n_qubits=2)[np.ix_(permutation, permutation)]
    assert tuple(restored._qf.qubits) == (0, 1)
    assert np.allclose(restored._qf.asgate().asoperator(), expected, atol=1e-12)


def test_unmodified_three_cnot_swap_matches_source():
    source = quil.Program(H(0), RY(0.37, 1), CNOT(0, 1), CNOT(1, 0), CNOT(0, 1))
    restored = Circuit.from_pyquil(source)
    permutation = [0, 2, 1, 3]
    expected = program_unitary(source, n_qubits=2)[np.ix_(permutation, permutation)]
    assert restored.to_dict()["gates"][-1] == {"gate": "Swap", "qubits": [0, 1], "params": []}
    assert np.allclose(restored._qf.asgate().asoperator(), expected, atol=1e-12)
