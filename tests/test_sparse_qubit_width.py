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
    touched = sorted({q for gate in circuit.to_dict()["gates"] for q in gate["qubits"]})
    # Labels are kept, not compacted; no identity instruction is emitted, because
    # not every Braket QPU lists `i` among its supported operations.
    assert [int(q) for q in bk.qubits] == touched
    assert "I" not in {instruction.operator.name for instruction in bk.instructions}
    result = LocalSimulator().run(bk.copy().state_vector(), shots=0).result()
    # Braket orders amplitudes over its sorted qubits; an untouched wire is |0>,
    # so place each Braket qubit i on axis i and the idle axes at index 0.
    exported = np.zeros((2,) * width, dtype=complex)
    index = tuple(slice(None) if q in touched else 0 for q in range(width))
    exported[index] = result.values[0].reshape((2,) * len(touched))
    np.testing.assert_allclose(exported.flatten(), _reference(circuit, width), atol=1e-12, rtol=0)


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


def test_pyquil_export_keeps_labels_and_sends_no_identity():
    """Quil labels are physical qubits on QCS, so idle wires are not named."""
    pytest.importorskip("pyquil")
    program = Circuit().h(3).cnot(3, 1).x(0).to_pyquil()
    assert sorted(program.get_qubit_indices()) == [0, 1, 3]
    assert "I " not in program.out()


def test_pytket_export_declares_every_wire():
    pytest.importorskip("pytket")
    assert Circuit().x(2).to_pytket().n_qubits == 3


def test_openqasm_exports_declare_the_width():
    circuit = Circuit().x(2)
    assert "qreg q[3];" in circuit.to_openqasm(2)
    assert "x q[2];" in circuit.to_openqasm(2)
    assert "qubit[3] q;" in circuit.to_openqasm(3)
    assert "x q[2];" in circuit.to_openqasm(3)


def _braket_entry_points(monkeypatch):
    """BraketExecutor and cloud MarqovDevice, both backed by a real LocalSimulator."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock

    from braket.devices import LocalSimulator

    from marqov.device import MarqovDevice
    from marqov.executors.braket import BraketExecutor, BraketExecutorConfig

    no_cloud = Mock(side_effect=AssertionError("Unexpected AWS initialization"))
    monkeypatch.setattr("braket.aws.AwsDevice", no_cloud)
    monkeypatch.setattr("marqov.executors.braket.AwsDevice", no_cloud)
    submitted = []

    def run(circuit, *args, shots, **kwargs):
        submitted.append(circuit)
        result = LocalSimulator().run(circuit, shots=shots).result()
        return SimpleNamespace(id="offline-task", result=lambda: result, metadata=lambda **k: {})

    provider = SimpleNamespace(name="offline", run=run)
    arn = "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1"
    executor = BraketExecutor(
        BraketExecutorConfig(device_arn=arn, s3_bucket="unused", s3_prefix="test")
    )
    executor._get_device = AsyncMock(return_value=provider)
    device = MarqovDevice("ionq", {"device_arn": arn, "s3_destination_folder": ("unused", "test")})
    device._get_provider_device = Mock(return_value=provider)
    return executor, device, submitted


@pytest.mark.parametrize(
    ("circuit", "expected"),
    [
        (Circuit().x(2), {"001"}),
        (Circuit().h(3).cnot(3, 1).x(0), {"1000", "1101"}),
        (Circuit().x(1).x(4), {"01001"}),
        (Circuit().x(0).cnot(0, 1), {"11"}),  # dense: provider keys pass through
    ],
    ids=["x2", "bell-1-3", "x1-x4", "dense"],
)
def test_braket_counts_use_the_widened_convention(monkeypatch, circuit, expected):
    """Untouched wires are not sent to Braket; their '0' is inserted in each key."""
    pytest.importorskip("braket")
    from marqov.device import MarqovDevice

    executor, device, submitted = _braket_entry_points(monkeypatch)
    outcome = asyncio.run(executor.execute(circuit, shots=64))
    cloud = device.run(circuit, shots=64)
    local = MarqovDevice("local", {}).run(circuit, shots=64)
    reference = _counts(circuit, shots=64)
    for counts in (outcome.counts, cloud, local):
        assert set(counts) == expected == set(reference)
        assert sum(counts.values()) == 64
    assert all("I" not in {i.operator.name for i in c.instructions} for c in submitted)
    # Provenance still describes the provider's own counts in raw_result.
    provenance = outcome.metadata["measurement_provenance"]
    touched = sorted({q for gate in circuit.to_dict()["gates"] for q in gate["qubits"]})
    assert provenance["measured_qubits"] == touched
    idle = [q for q in range(circuit.num_qubits) if q not in touched]
    assert outcome.metadata.get("identity_wires", []) == idle
    assert ("identity_wires" in outcome.metadata) == bool(idle)


def test_braket_counts_refuse_measured_wires_that_are_not_the_circuit_wires(monkeypatch):
    """Without a wire-for-wire match there is no safe place for the idle '0's."""
    pytest.importorskip("braket")
    from types import SimpleNamespace

    from marqov._braket_native import widen_logical_counts

    result = SimpleNamespace(measured_qubits=[5, 6])
    with pytest.raises(ValueError, match="measured qubits"):
        widen_logical_counts(Circuit().x(2).h(0), {"10": 4}, result)
    assert widen_logical_counts(Circuit().x(2).h(0), {"01": 4}, SimpleNamespace()) == {"001": 4}


def test_braket_counts_refuse_keys_whose_length_is_not_the_measured_width():
    """A key-length mismatch gets its own message, not a measured-qubit one."""
    pytest.importorskip("braket")
    from types import SimpleNamespace

    from marqov._braket_native import widen_logical_counts

    result = SimpleNamespace(measured_qubits=[0, 2])
    with pytest.raises(ValueError, match=r"key '1' has 1 characters, not the 2 measured") as error:
        widen_logical_counts(Circuit().x(2).h(0), {"1": 4}, result)
    assert "do not match" not in str(error.value)


@pytest.mark.parametrize("build", [build for build, _ in SPARSE])
def test_sparse_exports_are_byte_identical_when_repeated(build):
    pytest.importorskip("braket")
    first, second = build(), build()
    assert first.to_openqasm(2) == second.to_openqasm(2)
    assert first.to_openqasm(3) == second.to_openqasm(3)
    assert first.to_braket().to_ir().json() == second.to_braket().to_ir().json()
    assert repr(first.to_cirq()) == repr(second.to_cirq())


# Round trips through each framework's own importer


@pytest.mark.parametrize(("build", "width"), SPARSE)
def test_qiskit_and_openqasm_round_trips_keep_width_and_gates(build, width):
    circuit = build()
    for restored in (
        Circuit.from_qiskit(circuit.to_qiskit()),
        Circuit.from_openqasm(circuit.to_openqasm(2)),
        Circuit.from_openqasm(circuit.to_openqasm(3)),
    ):
        assert restored.num_qubits == width
        np.testing.assert_allclose(_reference(restored, width), _reference(circuit, width), atol=1e-12)


@pytest.mark.parametrize(("build", "width"), SPARSE)
def test_cirq_round_trip_keeps_width_and_drops_the_identity_padding(build, width):
    circuit = build()
    restored = Circuit.from_cirq(circuit.to_cirq())
    assert restored.num_qubits == width
    assert "I" not in {gate["gate"] for gate in restored.to_dict()["gates"]}


@pytest.mark.parametrize(("build", "width"), SPARSE)
def test_pennylane_round_trip_keeps_width_and_drops_the_identity_padding(build, width):
    pytest.importorskip("pennylane")
    circuit = build()
    restored = Circuit.from_pennylane(circuit.to_pennylane())
    assert restored.to_dict() == circuit.to_dict()
    assert restored.num_qubits == width


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


def _import(framework: str, ops: list[tuple[str, int]]) -> Circuit:
    """Import a native program of single-qubit ``h``/``x``/``i`` ops on ``max+1`` wires."""
    width = max(q for _, q in ops) + 1
    if framework in ("qiskit", "openqasm2"):
        from qiskit import QuantumCircuit, qasm2

        qc = QuantumCircuit(width)
        for name, q in ops:
            getattr(qc, "id" if name == "i" else name)(q)
        if framework == "qiskit":
            return Circuit.from_qiskit(qc)
        return Circuit.from_openqasm(qasm2.dumps(qc))
    if framework == "braket":
        from braket.circuits import Circuit as BraketCircuit

        bk = BraketCircuit()
        for name, q in ops:
            getattr(bk, name)(q)
        return Circuit.from_braket(bk)
    if framework == "cirq":
        import cirq

        gates = {"h": cirq.H, "x": cirq.X, "i": cirq.I}
        return Circuit.from_cirq(cirq.Circuit([gates[n](cirq.LineQubit(q)) for n, q in ops]))
    if framework == "pyquil":
        from pyquil import Program
        from pyquil.gates import H, I, X

        gates = {"h": H, "x": X, "i": I}
        return Circuit.from_pyquil(Program(*[gates[n](q) for n, q in ops]))
    import pennylane as qml

    gates = {"h": qml.Hadamard, "x": qml.PauliX, "i": qml.Identity}
    return Circuit.from_pennylane(qml.tape.QuantumScript([gates[n](q) for n, q in ops]))


IMPORT_FRAMEWORKS = ["qiskit", "openqasm2", "braket", "cirq", "pyquil", "pennylane"]


def _gate(name: str, qubit: int) -> dict:
    return {"gate": name, "qubits": [qubit], "params": []}


# OpenQASM 3 is absent: Qiskit's QASM 3 reader parses `id` as a U gate, not an
# identity, so from_openqasm decomposes it. to_openqasm(3) never emits `id`.
@pytest.mark.parametrize("framework", IMPORT_FRAMEWORKS)
def test_importers_drop_identity_gates_below_the_largest_other_index(framework):
    pytest.importorskip("braket")
    pytest.importorskip("pyquil")
    pytest.importorskip("pennylane")
    imported = _import(framework, [("i", 1), ("x", 2)])
    assert imported.to_dict() == {"gates": [_gate("X", 2)]}
    assert imported.num_qubits == 3


@pytest.mark.parametrize("framework", IMPORT_FRAMEWORKS)
def test_importers_keep_a_trailing_identity_so_the_width_is_kept(framework):
    """An identity above every other gate is the only record of that wire."""
    pytest.importorskip("braket")
    pytest.importorskip("pyquil")
    pytest.importorskip("pennylane")
    imported = _import(framework, [("h", 0), ("i", 1)])
    assert imported.to_dict() == {"gates": [_gate("H", 0), _gate("I", 1)]}
    assert imported.num_qubits == 2
    mixed = _import(framework, [("h", 0), ("i", 1), ("x", 2), ("i", 3)])
    assert mixed.to_dict() == {"gates": [_gate("H", 0), _gate("X", 2), _gate("I", 3)]}
    assert mixed.num_qubits == 4


def test_cirq_identity_after_a_terminal_measurement_is_not_mid_circuit():
    import cirq

    q = cirq.LineQubit.range(2)
    imported = Circuit.from_cirq(cirq.Circuit([cirq.X(q[1]), cirq.measure(q[1]), cirq.I(q[1])]))
    assert imported.to_dict() == {"gates": [_gate("X", 1)]}


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


def _azure_cirq_submission(circuit: Circuit, target_name: str):
    """The cirq program the Azure executor submits, captured before any network call."""
    pytest.importorskip("azure.quantum.cirq")
    from unittest.mock import Mock

    from azure.quantum.cirq import AzureQuantumService

    from marqov.executors.azure import AzureQuantumExecutor, AzureQuantumExecutorConfig

    submitted = []

    def submit(**kwargs):
        submitted.append(kwargs["program"])
        raise RuntimeError("offline: stop after capturing the program")

    target = Mock()
    target.submit.side_effect = submit
    service = AzureQuantumService(workspace=Mock(), default_target=target_name)
    service.get_target = Mock(return_value=target)
    executor = AzureQuantumExecutor(
        AzureQuantumExecutorConfig(
            subscription_id="subscription", resource_group="group", workspace_name="workspace",
            location="eastus", target=target_name, framework="cirq",
        )
    )
    executor._backend = service
    with pytest.raises(RuntimeError, match="offline"):
        asyncio.run(executor.execute(circuit, shots=5))
    (program,) = submitted
    return program


# Gates IonQ's Cirq serializer accepts (it has no CZ or SWAP, sparse or dense).
@pytest.mark.parametrize(
    "build", [lambda: Circuit().h(2), lambda: Circuit().h(3).cnot(3, 1).x(0), _sparse_low]
)
def test_azure_cirq_submission_serializes_for_ionq_and_quantinuum(build):
    """IonQ's serializer refuses cirq.I, so idle wires are measured, not padded."""
    import cirq

    circuit = build()
    program = _azure_cirq_submission(circuit, "ionq.simulator")
    assert not any(isinstance(op.gate, cirq.IdentityGate) for op in program.all_operations())
    (measure,) = [op for op in program.all_operations() if cirq.is_measurement(op)]
    assert list(measure.qubits) == cirq.LineQubit.range(circuit.num_qubits)

    from azure.quantum.cirq.targets import IonQTarget, QuantinuumTarget

    serialized = IonQTarget._translate_cirq_circuit(program)
    assert serialized.body["qubits"] == circuit.num_qubits
    qasm = QuantinuumTarget._translate_cirq_circuit(program)
    assert f"qreg q[{circuit.num_qubits}];" in qasm
    assert qasm.count("measure q[") == circuit.num_qubits


def test_azure_cirq_submission_of_a_dense_circuit_is_unchanged():
    import cirq

    circuit = _dense()
    expected = circuit.to_cirq()
    expected.append(cirq.measure(*sorted(expected.all_qubits()), key="result"))
    assert _azure_cirq_submission(circuit, "ionq.simulator") == expected


# A wire touched only by identity gates is an identity wire too: no provider is
# sent an identity, and its '0' is restored in the counts keys.


def _top_identity() -> Circuit:
    """H on wire 0; wire 1 holds only an identity (as from_qiskit keeps it)."""
    return Circuit.from_dict({"gates": [_gate("H", 0), _gate("I", 1)]})


def _identity_only() -> Circuit:
    return Circuit.from_dict({"gates": [_gate("I", 0), _gate("I", 2)]})


def test_identity_only_wires_are_identity_wires():
    assert _top_identity().num_qubits == 2
    assert _top_identity()._idle_qubits() == [1]
    assert _identity_only()._idle_qubits() == [0, 1, 2]
    assert set(_counts(_top_identity())) == {"00", "10"}


@pytest.mark.parametrize("build", [_top_identity, _identity_only])
def test_azure_ionq_cirq_submission_sends_no_identity_on_an_identity_only_wire(build):
    import cirq

    circuit = build()
    program = _azure_cirq_submission(circuit, "ionq.simulator")
    assert not any(isinstance(op.gate, cirq.IdentityGate) for op in program.all_operations())
    (measure,) = [op for op in program.all_operations() if cirq.is_measurement(op)]
    assert list(measure.qubits) == cirq.LineQubit.range(circuit.num_qubits)
    from azure.quantum.cirq.targets import IonQTarget

    assert IonQTarget._translate_cirq_circuit(program).body["qubits"] == circuit.num_qubits


def test_braket_sends_no_identity_for_an_identity_only_top_wire(monkeypatch):
    pytest.importorskip("braket")
    from marqov.device import MarqovDevice

    circuit = _top_identity()
    assert "I" not in {i.operator.name for i in circuit.to_braket().instructions}
    executor, device, submitted = _braket_entry_points(monkeypatch)
    outcome = asyncio.run(executor.execute(circuit, shots=64))
    cloud = device.run(circuit, shots=64)
    local = MarqovDevice("local", {}).run(circuit, shots=64)
    for counts in (outcome.counts, cloud, local):
        assert set(counts) == {"00", "10"}
    assert all("I" not in {i.operator.name for i in c.instructions} for c in submitted)
    assert outcome.metadata["identity_wires"] == [1]


def test_rigetti_widens_an_identity_only_top_wire_instead_of_measuring_it():
    pytest.importorskip("pyquil")
    from types import SimpleNamespace

    from pyquil.pyqvm import PyQVM

    from marqov.executors.rigetti import RigettiExecutor, RigettiExecutorConfig

    circuit = _top_identity()
    assert "I " not in circuit.to_pyquil().out()
    qvm = PyQVM(n_qubits=2, seed=17)
    qc = SimpleNamespace(qam=qvm, compile=lambda p: p, run=qvm.run)
    executor = RigettiExecutor(RigettiExecutorConfig(quantum_processor_id="2q-qvm"), qc=qc)
    out = asyncio.run(executor.execute(circuit, shots=64))
    assert out.metadata["measured_qubits"] == [0]
    assert out.metadata["identity_wires"] == [1]
    assert set(out.counts) == set(_counts(circuit)) == {"00", "10"}
