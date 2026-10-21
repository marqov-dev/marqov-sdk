"""The H2 benchmark circuit implements the documented UCC ansatz and reaches the ground state."""

import numpy as np
import pytest

from benchmarks.shared.vqe_h2 import (
    H2_GROUND_STATE_ENERGY,
    H2_HAMILTONIAN,
    build_braket_circuit,
    compute_h2_energy,
    create_ansatz,
)

pytest.importorskip("braket.circuits")

_I = np.eye(2)
_X = np.array([[0, 1], [1, 0]])
_Y = np.array([[0, -1j], [1j, 0]])
_Z = np.diag([1, -1])


def _hamiltonian() -> np.ndarray:
    # Braket is big-endian: qubit 0 is the leftmost tensor factor
    h = H2_HAMILTONIAN
    return (h['identity'] * np.kron(_I, _I) + h['z0'] * np.kron(_Z, _I)
            + h['z1'] * np.kron(_I, _Z) + h['z0z1'] * np.kron(_Z, _Z)
            + h['x0x1'] * np.kron(_X, _X) + h['y0y1'] * np.kron(_Y, _Y))


def _statevector(circuit) -> np.ndarray:
    return circuit.to_unitary()[:, 0]


def _reference_ansatz_unitary(theta: float) -> np.ndarray:
    from braket.circuits import Circuit

    circuit = Circuit()
    for name, *args in create_ansatz(theta)['gates']:
        getattr(circuit, name)(*args)
    return circuit.to_unitary()


def _exact_measure_pauli(pauli_string: str, theta: float) -> dict[str, float]:
    probs = np.abs(_statevector(build_braket_circuit(theta, pauli_string))) ** 2
    return {format(i, '02b'): p for i, p in enumerate(probs)}


def test_stated_ground_energy_matches_diagonalization():
    assert np.linalg.eigvalsh(_hamiltonian())[0] == pytest.approx(H2_GROUND_STATE_ENERGY, abs=1e-6)


@pytest.mark.parametrize("theta", [0.0, 0.2293, -0.7, 1.3, np.pi, -2.5])
def test_benchmark_circuit_matches_create_ansatz_unitary(theta):
    # 'zz' adds no basis rotations, so this is the bare ansatz
    actual = build_braket_circuit(theta, 'zz').to_unitary()
    expected = _reference_ansatz_unitary(theta)
    overlap = np.trace(expected.conj().T @ actual) / 4
    assert abs(overlap) == pytest.approx(1.0, abs=1e-12)


def test_benchmark_ansatz_reaches_exact_ground_energy():
    hamiltonian = _hamiltonian()
    exact = np.linalg.eigvalsh(hamiltonian)[0]
    energies = []
    for theta in np.linspace(-np.pi, np.pi, 4001):
        state = _statevector(build_braket_circuit(theta, 'zz'))
        energies.append(np.real(state.conj() @ hamiltonian @ state))
    assert min(energies) == pytest.approx(exact, abs=1e-6)


def test_measurement_pipeline_reaches_ground_energy_at_optimal_theta():
    energy, _ = compute_h2_energy(0.2293, _exact_measure_pauli)
    assert abs(energy - H2_GROUND_STATE_ENERGY) * 1000 < 1.0
