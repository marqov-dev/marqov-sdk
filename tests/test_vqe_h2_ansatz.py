"""The H2 benchmark circuit implements the documented UCC ansatz and reaches the ground state."""

import asyncio
import importlib.util
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from scipy.linalg import expm  # type: ignore[import-untyped]

from benchmarks.shared.vqe_h2 import (
    DEFAULT_SHOTS,
    H2_GROUND_STATE_ENERGY,
    H2_HAMILTONIAN,
    build_braket_ansatz,
    build_braket_circuit,
    compute_h2_energy,
    create_ansatz,
    create_measurement_circuit,
    run_vqe_optimization,
)
from marqov.circuits import Circuit
from marqov.executors.local import LocalExecutor, LocalExecutorConfig

pytest.importorskip("braket.circuits")

_I = np.eye(2)
_X = np.array([[0, 1], [1, 0]])
_Y = np.array([[0, -1j], [1j, 0]])
_Z = np.diag([1, -1])
_P0 = np.diag([1, 0])
_P1 = np.diag([0, 1])


def _on_qubit(gate: np.ndarray, qubit: int) -> np.ndarray:
    # Big-endian, as Braket: qubit 0 is the leftmost tensor factor
    return np.kron(gate, _I) if qubit == 0 else np.kron(_I, gate)


def _rotation(pauli: np.ndarray, angle: float) -> np.ndarray:
    return np.asarray(np.cos(angle / 2) * _I - 1j * np.sin(angle / 2) * pauli)


def _gate_list_unitary(gates: list[tuple[Any, ...]]) -> np.ndarray:
    """Multiply out a create_ansatz gate list with plain numpy matrices."""
    paulis = {'rx': _X, 'ry': _Y, 'rz': _Z}
    unitary = np.eye(4, dtype=complex)
    for name, *args in gates:
        if name in paulis:
            qubit, angle = args
            step = _on_qubit(_rotation(paulis[name], angle), qubit)
        elif name == 'x':
            step = _on_qubit(_X, args[0])
        elif name == 'cnot':
            control, target = args
            step = _on_qubit(_P0, control) + _on_qubit(_P1, control) @ _on_qubit(_X, target)
        else:
            raise AssertionError(f"unexpected gate {name}")
        unitary = step @ unitary
    return unitary


def _hamiltonian() -> np.ndarray:
    h = H2_HAMILTONIAN
    return np.asarray(
        h['identity'] * np.kron(_I, _I) + h['z0'] * np.kron(_Z, _I)
        + h['z1'] * np.kron(_I, _Z) + h['z0z1'] * np.kron(_Z, _Z)
        + h['x0x1'] * np.kron(_X, _X) + h['y0y1'] * np.kron(_Y, _Y)
    )


def _statevector(circuit: Any) -> np.ndarray:
    return np.asarray(circuit.to_unitary()[:, 0])


def _phase_free_overlap(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(np.trace(a.conj().T @ b)) / a.shape[0])


def _exact_measure_pauli(pauli_string: str, theta: float) -> dict[str, int]:
    # Exact probabilities stand in for counts; compute_expectation_from_counts
    # only uses count / total, so fractional "counts" give the exact expectation
    probs = np.abs(_statevector(build_braket_circuit(theta, pauli_string))) ** 2
    return {format(i, '02b'): p for i, p in enumerate(probs)}


def test_stated_ground_energy_matches_diagonalization() -> None:
    assert np.linalg.eigvalsh(_hamiltonian())[0] == pytest.approx(H2_GROUND_STATE_ENERGY, abs=1e-6)


@pytest.mark.parametrize("theta", [0.0, 0.2293, -0.7, 1.3, np.pi, -2.5])
def test_benchmark_circuit_matches_create_ansatz_unitary(theta: float) -> None:
    # 'zz' adds no basis rotations, so this is the bare ansatz
    actual = np.asarray(build_braket_circuit(theta, 'zz').to_unitary())
    expected = _gate_list_unitary(create_ansatz(theta)['gates'])
    assert _phase_free_overlap(expected, actual) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("theta", [0.0, 0.2293, -0.7, 1.3])
def test_benchmark_ansatz_is_ucc_exponential(theta: float) -> None:
    # In create_ansatz's angle convention, Rz(theta) = exp(-i theta Z / 2), the
    # gate list prepares exp(+i theta/2 X0 Y1) applied to the q0=1, q1=0 reference
    reference = np.zeros(4)
    reference[0b10] = 1
    expected = expm(0.5j * theta * np.kron(_X, _Y)) @ reference
    actual = _statevector(build_braket_ansatz(theta))
    assert abs(np.vdot(expected, actual)) == pytest.approx(1.0, abs=1e-12)


def test_benchmark_ansatz_reaches_exact_ground_energy() -> None:
    hamiltonian = _hamiltonian()
    exact = np.linalg.eigvalsh(hamiltonian)[0]
    energies = []
    for theta in np.linspace(-np.pi, np.pi, 4001):
        state = _statevector(build_braket_circuit(theta, 'zz'))
        energies.append(np.real(state.conj() @ hamiltonian @ state))
    assert min(energies) == pytest.approx(exact, abs=1e-6)


def test_measurement_pipeline_reaches_ground_energy_at_optimal_theta() -> None:
    energy, _ = compute_h2_energy(0.2293, _exact_measure_pauli)
    assert abs(energy - H2_GROUND_STATE_ENERGY) * 1000 < 1.0


def _marqov_circuit(theta: float, pauli_string: str) -> Circuit:
    spec = create_measurement_circuit(create_ansatz(theta), pauli_string)
    circuit = Circuit()
    for name, *args in spec['base_ansatz']['gates'] + spec['basis_rotations']:
        if name in ('rx', 'ry', 'rz'):
            qubit, angle = args
            getattr(circuit, name)(angle, qubit)
        else:
            getattr(circuit, name)(*args)
    return circuit


def _local_measure_pauli(pauli_string: str, theta: float, seed: int | None = None) -> dict[str, int]:
    executor = LocalExecutor(LocalExecutorConfig(seed=seed))
    circuit = _marqov_circuit(theta, pauli_string)
    return asyncio.run(executor.execute(circuit, shots=DEFAULT_SHOTS)).counts


def test_seeded_vqe_on_local_executor_reaches_ground_state() -> None:
    """Seeded end-to-end VQE on the local executor lands within 20 mHa.

    Measured at DEFAULT_SHOTS over seeds 0..29: 0.50 to 16.44 mHa
    (mean 8.71). Seed 1234 gives 11.65 mHa.
    """
    result = run_vqe_optimization(_local_measure_pauli, verbose=False, seed=1234)
    assert result['energy_error_mhartree'] < 20


def test_same_seed_reproduces_iteration_data() -> None:
    first = run_vqe_optimization(_local_measure_pauli, verbose=False, seed=1234)
    second = run_vqe_optimization(_local_measure_pauli, verbose=False, seed=1234)
    other = run_vqe_optimization(_local_measure_pauli, verbose=False, seed=4321)
    assert first['seed'] == second['seed'] == 1234
    assert json.dumps(first['iteration_data']) == json.dumps(second['iteration_data'])
    assert json.dumps(first['iteration_data']) != json.dumps(other['iteration_data'])


def test_unseeded_run_keeps_two_argument_callback() -> None:
    calls = []

    def measure_pauli(pauli_string: str, theta: float) -> dict[str, int]:
        calls.append(pauli_string)
        return _local_measure_pauli(pauli_string, theta)

    result = run_vqe_optimization(measure_pauli, max_iterations=3, verbose=False)
    assert result['seed'] is None
    assert calls


_LOCAL_BENCHMARK_IMPORTS = ("harness", "shared", "shared.vqe_h2")


@pytest.fixture
def local_benchmark(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "raw_sdk" / "vqe_h2_local.py"
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in _LOCAL_BENCHMARK_IMPORTS:
        monkeypatch.delitem(sys.modules, name, raising=False)
    spec = importlib.util.spec_from_file_location("vqe_h2_local_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    for name in _LOCAL_BENCHMARK_IMPORTS:
        sys.modules.pop(name, None)


@pytest.mark.parametrize("theta", [0.0, 0.2293, -0.7])
def test_local_benchmark_uses_shared_ansatz(local_benchmark: ModuleType, theta: float) -> None:
    actual = np.asarray(local_benchmark.create_ansatz(np.array([theta])).to_unitary())
    expected = _gate_list_unitary(create_ansatz(theta)['gates'])
    assert _phase_free_overlap(expected, actual) == pytest.approx(1.0, abs=1e-12)


def test_local_benchmark_reads_braket_bitstrings_big_endian(local_benchmark: ModuleType) -> None:
    from braket.circuits import Circuit as BraketCircuit  # type: ignore[import-untyped]
    from braket.devices import LocalSimulator  # type: ignore[import-untyped]

    # X on qubit 0 only: <Z0> = -1 and <Z1> = +1 with no sampling noise
    circuit = BraketCircuit().x(0).i(1)
    device = LocalSimulator()
    assert local_benchmark.measure_expectation(circuit, 'zi', device, shots=10) == -1.0
    assert local_benchmark.measure_expectation(circuit, 'iz', device, shots=10) == 1.0
