"""Circuit width is the largest qubit index plus one (marqov-sdk#154).

Untouched indices below the largest one are identity wires: every export and
``LocalExecutor`` keep qubit ``i`` at register index ``i`` and at character
``i`` of a counts key (qubit 0 leftmost), whatever else the circuit touches.
"""

import asyncio
import hashlib
import json

import numpy as np
import pytest

from marqov.circuits import Circuit
from marqov.executors.local import LocalExecutor, LocalExecutorConfig


# Y and Rz are left out: QuantumFlow's Y.run and Rz.run differ from their own
# operators by a global phase (see test_circuits.py), and these checks compare
# complex amplitudes exactly.


def _sparse() -> Circuit:
    """Asymmetric, complex-amplitude circuit on wires {0, 1, 3}; wire 2 is idle."""
    return Circuit().h(3).s(3).cnot(3, 1).rx(0.4, 0).t(1).ry(0.9, 3)


def _sparse_low() -> Circuit:
    """Wires {2, 4}; wires 0, 1 and 3 are idle."""
    return Circuit().h(2).cnot(2, 4).t(4).s(2).ry(0.6, 4).rx(0.2, 2)


# (builder, expected width): the width is pinned independently of num_qubits.
SPARSE = [(_sparse, 4), (_sparse_low, 5)]


def _counts(circuit: Circuit, shots: int = 64) -> dict[str, int]:
    executor = LocalExecutor(LocalExecutorConfig(seed=42))
    return asyncio.run(executor.execute(circuit, shots=shots)).counts


def _reference(circuit: Circuit, width: int | None = None) -> np.ndarray:
    """Statevector from Circuit.simulate(), qubit 0 most significant."""
    state = circuit.simulate().tensor.flatten()
    if width is not None:
        assert state.shape == (2**width,)
    return state


# Width


def test_width_is_max_index_plus_one():
    assert Circuit().x(2).num_qubits == 3
    assert Circuit().h(3).cnot(3, 1).x(0).num_qubits == 4
    assert _sparse_low().num_qubits == 5
    assert Circuit().h(0).cnot(0, 1).num_qubits == 2
    assert Circuit().num_qubits == 0
    assert type(Circuit().x(2).num_qubits) is int


def test_simulate_keeps_idle_wires():
    state = Circuit().x(2).simulate()
    assert tuple(state.qubits) == (0, 1, 2)
    expected = np.zeros(8, dtype=complex)
    expected[0b001] = 1  # qubit 0 is the most significant tensor axis
    np.testing.assert_array_equal(state.tensor.flatten(), expected)


def test_repr_reports_width():
    assert repr(Circuit().x(2)) == "Circuit(qubits=3, gates=1)"


# Counts keys on LocalExecutor


def test_local_executor_places_qubit_2_at_index_2():
    """Documented convention: qubit 0 is the leftmost character of a key."""
    assert _counts(Circuit().x(2), shots=8) == {"001": 8}
    doc = Circuit.num_qubits.__doc__
    assert "leftmost" in doc and "001" in doc
    assert "leftmost" in Circuit.__doc__


def test_x2_and_x0_produce_different_counts_keys():
    assert set(_counts(Circuit().x(2))) != set(_counts(Circuit().x(0)))
    assert _counts(Circuit().x(0), shots=8) == {"1": 8}


def test_bit_order_with_entangled_sparse_wires():
    # Qubit 0 excited; qubits 1 and 3 in a Bell pair; qubit 2 idle.
    counts = _counts(Circuit().h(3).cnot(3, 1).x(0), shots=256)
    assert set(counts) == {"1000", "1101"}


def test_qubit_index_keeps_its_counts_position_regardless_of_other_wires():
    for others in ([], [0], [1], [0, 1, 3]):
        circuit = Circuit().x(2)
        for q in others:
            circuit.z(q)  # touches the wire without changing |0>
        (key,) = _counts(circuit, shots=4)
        assert key[2] == "1"
        assert key.count("1") == 1


# Exports: full complex statevector equality, qubit i -> register index i


@pytest.mark.parametrize(("build", "width"), SPARSE)
def test_qiskit_export_statevector_matches_simulate(build, width):
    from qiskit.quantum_info import Statevector

    circuit = build()
    qc = circuit.to_qiskit()
    assert qc.num_qubits == width
    # The first gate keeps its source label as its register index.
    source = circuit.to_dict()["gates"][0]["qubits"]
    assert [qc.find_bit(q).index for q in qc.data[0].qubits] == source
    # Qiskit indexes amplitudes little-endian; reverse so qubit 0 is most significant.
    exported = Statevector(qc).reverse_qargs().data
    np.testing.assert_allclose(exported, _reference(circuit, width), atol=1e-12, rtol=0)


@pytest.mark.parametrize(("build", "width"), SPARSE)
def test_braket_export_statevector_matches_simulate(build, width):
    pytest.importorskip("braket")
    from braket.devices import LocalSimulator

    circuit = build()
    bk = circuit.to_braket()
    assert [int(q) for q in bk.qubits] == list(range(width))
    result = LocalSimulator().run(bk.copy().state_vector(), shots=0).result()
    np.testing.assert_allclose(result.values[0], _reference(circuit, width), atol=1e-12, rtol=0)


@pytest.mark.parametrize(("build", "width"), SPARSE)
def test_cirq_export_statevector_matches_simulate(build, width):
    import cirq

    circuit = build()
    exported = circuit.to_cirq()
    qubits = cirq.LineQubit.range(width)
    assert sorted(exported.all_qubits()) == qubits
    state = cirq.Simulator(dtype=np.complex128).simulate(exported, qubit_order=qubits)
    actual, reference = state.final_state_vector, _reference(circuit, width)
    # Cirq's power gates carry a global phase relative to QuantumFlow's
    # rotations; remove that single phase, then compare amplitudes exactly.
    pivot = int(np.argmax(np.abs(reference)))
    phase = actual[pivot] / reference[pivot]
    assert abs(abs(phase) - 1) < 1e-7
    np.testing.assert_allclose(actual, phase * reference, atol=1e-7, rtol=0)


@pytest.mark.parametrize(("build", "width"), SPARSE)
def test_pennylane_export_statevector_matches_simulate(build, width):
    qml = pytest.importorskip("pennylane")

    circuit = build()
    script = circuit.to_pennylane()
    wires = list(range(width))
    assert sorted(int(w) for w in script.wires) == wires
    (state,) = qml.execute(
        [qml.tape.QuantumScript(script.operations, [qml.state()])],
        qml.device("default.qubit", wires=wires),
    )
    np.testing.assert_allclose(state, _reference(circuit, width), atol=1e-12, rtol=0)


def test_pyquil_export_declares_every_wire():
    pytest.importorskip("pyquil")
    program = Circuit().h(3).cnot(3, 1).x(0).to_pyquil()
    assert sorted(program.get_qubit_indices()) == [0, 1, 2, 3]
    assert "I 2" in program.out()


def test_pytket_export_declares_every_wire():
    pytest.importorskip("pytket")
    assert Circuit().x(2).to_pytket().n_qubits == 3


def test_openqasm_exports_declare_the_width():
    circuit = Circuit().x(2)
    assert "qreg q[3];" in circuit.to_openqasm(2)
    assert "x q[2];" in circuit.to_openqasm(2)
    assert "qubit[3] q;" in circuit.to_openqasm(3)
    assert "x q[2];" in circuit.to_openqasm(3)


def test_braket_local_simulator_counts_match_local_executor_keys():
    pytest.importorskip("braket")
    from braket.devices import LocalSimulator

    result = LocalSimulator().run(Circuit().x(2).to_braket(), shots=8).result()
    assert dict(result.measurement_counts) == {"001": 8}


@pytest.mark.parametrize("build", [build for build, _ in SPARSE])
def test_sparse_exports_are_byte_identical_when_repeated(build):
    pytest.importorskip("braket")
    first, second = build(), build()
    assert first.to_openqasm(2) == second.to_openqasm(2)
    assert first.to_openqasm(3) == second.to_openqasm(3)
    assert first.to_braket().to_ir().json() == second.to_braket().to_ir().json()
    assert repr(first.to_cirq()) == repr(second.to_cirq())


# Round trips through each framework's own importer


@pytest.mark.parametrize("build", [build for build, _ in SPARSE])
def test_qiskit_and_openqasm_round_trips_keep_width_and_gates(build):
    circuit = build()
    for restored in (
        Circuit.from_qiskit(circuit.to_qiskit()),
        Circuit.from_openqasm(circuit.to_openqasm(2)),
        Circuit.from_openqasm(circuit.to_openqasm(3)),
    ):
        assert restored.num_qubits == circuit.num_qubits
        np.testing.assert_allclose(_reference(restored), _reference(circuit), atol=1e-12)


@pytest.mark.parametrize("build", [build for build, _ in SPARSE])
def test_cirq_round_trip_keeps_width_and_drops_the_identity_padding(build):
    circuit = build()
    restored = Circuit.from_cirq(circuit.to_cirq())
    assert restored.num_qubits == circuit.num_qubits
    assert "I" not in {gate["gate"] for gate in restored.to_dict()["gates"]}


@pytest.mark.parametrize("build", [build for build, _ in SPARSE])
def test_pennylane_round_trip_keeps_width_and_drops_the_identity_padding(build):
    pytest.importorskip("pennylane")
    circuit = build()
    restored = Circuit.from_pennylane(circuit.to_pennylane())
    assert restored.to_dict() == circuit.to_dict()
    assert restored.num_qubits == circuit.num_qubits


def _chain() -> Circuit:
    """Sparse circuit whose gates each share a wire with the previous one.

    Qiskit's DAG and Cirq's moments may reorder gates on disjoint wires (dense
    circuits too), so a chain is the circuit whose `to_dict()` order every
    framework keeps. Wire 2 is idle.
    """
    return Circuit().h(3).cnot(3, 1).t(1).cnot(1, 0).s(0).x(0).cz(0, 3).swap(3, 1)


def _chain_low() -> Circuit:
    """Wires {2, 4}; wires 0, 1 and 3 are idle."""
    return Circuit().h(2).cnot(2, 4).t(4).s(4).cz(4, 2).x(2).swap(2, 4)


def _round_trips():
    """Every exporter/importer pair, as (name, round trip)."""
    return [
        ("qiskit", lambda c: Circuit.from_qiskit(c.to_qiskit())),
        ("openqasm2", lambda c: Circuit.from_openqasm(c.to_openqasm(2))),
        ("openqasm3", lambda c: Circuit.from_openqasm(c.to_openqasm(3))),
        ("braket", lambda c: Circuit.from_braket(c.to_braket())),
        ("cirq", lambda c: Circuit.from_cirq(c.to_cirq())),
        ("pyquil", lambda c: Circuit.from_pyquil(c.to_pyquil())),
        ("pennylane", lambda c: Circuit.from_pennylane(c.to_pennylane())),
    ]


@pytest.mark.parametrize("build", [_chain, _chain_low])
@pytest.mark.parametrize(("name", "round_trip"), _round_trips(), ids=lambda v: v if isinstance(v, str) else "")
def test_sparse_round_trip_through_every_framework_is_exact(build, name, round_trip):
    pytest.importorskip("braket")
    pytest.importorskip("pyquil")
    pytest.importorskip("pennylane")
    circuit = build()
    restored = round_trip(circuit)
    assert restored.to_dict() == circuit.to_dict(), name
    assert restored.num_qubits == circuit.num_qubits, name


def _x2_after_identity_on_1(framework: str):
    """A native program with an identity on wire 1 and X on wire 2."""
    if framework == "qiskit":
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(3)
        qc.id(1)
        qc.x(2)
        return Circuit.from_qiskit(qc)
    if framework == "openqasm2":
        return Circuit.from_openqasm(
            'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\nid q[1];\nx q[2];'
        )
    if framework == "braket":
        from braket.circuits import Circuit as BraketCircuit

        return Circuit.from_braket(BraketCircuit().i(1).x(2))
    if framework == "cirq":
        import cirq

        q = cirq.LineQubit.range(3)
        return Circuit.from_cirq(cirq.Circuit([cirq.I(q[1]), cirq.X(q[2])]))
    if framework == "pyquil":
        from pyquil import Program
        from pyquil.gates import I, X

        return Circuit.from_pyquil(Program(I(1), X(2)))
    import pennylane as qml

    return Circuit.from_pennylane(qml.tape.QuantumScript([qml.Identity(1), qml.PauliX(2)]))


# OpenQASM 3 is absent: Qiskit's QASM 3 reader parses `id` as a U gate, not an
# identity, so from_openqasm decomposes it. to_openqasm(3) never emits `id`.
@pytest.mark.parametrize("framework", ["qiskit", "openqasm2", "braket", "cirq", "pyquil", "pennylane"])
def test_importers_drop_identity_gates(framework):
    pytest.importorskip("braket")
    pytest.importorskip("pyquil")
    pytest.importorskip("pennylane")
    imported = _x2_after_identity_on_1(framework)
    assert imported.to_dict() == {"gates": [{"gate": "X", "qubits": [2], "params": []}]}
    assert imported.num_qubits == 3


def test_from_dict_keeps_identity_gates():
    """Only the framework importers drop identities; the gate-record format keeps them."""
    data = {"gates": [{"gate": "I", "qubits": [1], "params": []}]}
    assert Circuit.from_dict(data).to_dict() == data


# Dense circuits: exports and counts pinned from the implementation before marqov-sdk#154


def _dense() -> Circuit:
    return (
        Circuit().h(0).cnot(0, 1).rx(0.3, 2).s(1).t(2).cz(2, 0).swap(1, 2)
        .ry(0.7, 0).rz(1.1, 1).y(2).z(0).x(1)
    )


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


DENSE_QASM2 = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\nh q[0];\ncx q[0],q[1];\n'
    "rx(0.3) q[2];\ns q[1];\nt q[2];\ncz q[2],q[0];\nswap q[1],q[2];\nry(0.7) q[0];\n"
    "rz(1.1) q[1];\ny q[2];\nz q[0];\nx q[1];"
)
DENSE_QASM3 = (
    'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[3] q;\nh q[0];\ncx q[0], q[1];\n'
    "rx(0.3) q[2];\ns q[1];\nt q[2];\ncz q[2], q[0];\nswap q[1], q[2];\nry(0.7) q[0];\n"
    "rz(1.1) q[1];\ny q[2];\nz q[0];\nx q[1];\n"
)
DENSE_PYQUIL = (
    "H 0\nCNOT 0 1\nRX(0.3) 2\nS 1\nT 2\nCZ 2 0\nSWAP 1 2\nRY(0.7) 0\nRZ(1.1) 1\nY 2\nZ 0\nX 1\n"
)


def test_dense_circuit_exports_and_counts_are_unchanged():
    pytest.importorskip("braket")
    pytest.importorskip("pyquil")
    pytest.importorskip("pennylane")
    pytest.importorskip("pytket")
    circuit = _dense()
    assert circuit.num_qubits == 3
    assert circuit.to_openqasm(2) == DENSE_QASM2
    assert circuit.to_openqasm(3) == DENSE_QASM3
    assert circuit.to_pyquil().out() == DENSE_PYQUIL
    assert len(circuit.to_qiskit().qubits) == 3
    assert list(circuit.simulate().qubits) == [0, 1, 2]
    assert _sha(circuit.to_braket().to_ir().json()) == (
        "159026a6a5e25d9ebdfdef79912deb06d5173c585772d27368aa9401ae5f244a"
    )
    assert _sha(repr(circuit.to_cirq())) == (
        "e2b3d7bddf8c7c9afd3a6190a6351fc0ebd36c683b32dc196fa05da3b197dac3"
    )
    assert _sha(repr(circuit.to_pennylane().operations)) == (
        "f3aa267c24960dfcd4dc488f727abfc906ede30d6bace5e354e58324448f7f6a"
    )
    assert _sha(repr(circuit.to_pytket().get_commands())) == (
        "b51d895076d6aa4d43c53db6c8ec70ec4d339b3c523922d9a353d58d18520fe9"
    )
    assert _sha(json.dumps(circuit.to_dict(), sort_keys=True)) == (
        "cb1ea9c0fe805b33e799e4de622cc6c48932725b04f2cd99243d119dfb319a04"
    )
    assert _counts(circuit, shots=256) == {
        "001": 1, "010": 13, "011": 116, "100": 3, "110": 112, "111": 11,
    }
