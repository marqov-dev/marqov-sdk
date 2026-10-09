"""Qualify public local simulation against the installed QuantumFlow dependency."""

import numpy as np
import pytest

from marqov import Circuit


def test_grover_native_sampling_preserves_state():
    circuit = Circuit().h(0).h(1).cz(0, 1).h(0).h(1).x(0).x(1).cz(0, 1).x(0).x(1).h(0).h(1)
    state = circuit.simulate()
    original = state.tensor.copy()
    probabilities = state.probabilities().copy()
    np.testing.assert_allclose(probabilities, [[0, 0], [0, 1]], atol=1e-14, rtol=0)

    np.testing.assert_array_equal(state.sample(1000), [[0, 0], [0, 1000]])

    np.testing.assert_array_equal(state.tensor, original)
    np.testing.assert_array_equal(state.probabilities(), probabilities)


@pytest.mark.parametrize("wire,expected", [(0, [[0, 0], [1000, 0]]), (1, [[0, 1000], [0, 0]])])
def test_native_sampling_retains_asymmetric_wire_order(wire, expected):
    state = Circuit().z(0).z(1).x(wire).simulate()
    assert state.qubits == (0, 1)
    np.testing.assert_array_equal(state.sample(1000), expected)
