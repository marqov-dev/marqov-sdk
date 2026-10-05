"""Lossless dictionary reconstruction and unchanged serialization output."""

import json

import numpy as np
import pytest
import quantumflow as qf

from marqov.circuits import Circuit

# Derive Braket coverage from its pinned converter map, not the reader map.
# XY is mapped but not importable in marqov-quantumflow 1.0.0 (marqov-sdk#195).
from quantumflow.xbraket import BRAKET_TO_QF

BRAKET_GATES = sorted(name for name in BRAKET_TO_QF if name != "XY")


def imported_braket_gate(name, angle):
    from braket.circuits import Circuit as BK, Instruction, gates
    from braket.circuits.angled_gate import AngledGate

    constructor = getattr(gates, name)
    operator = constructor(angle) if issubclass(constructor, AngledGate) else constructor()
    targets = list(reversed(range(constructor.fixed_qubit_count())))
    native = BK().add_instruction(Instruction(operator, targets))
    return Circuit.from_braket(native)


@pytest.mark.parametrize("name", BRAKET_GATES)
@pytest.mark.parametrize("angle", [0.37, -1.1, 4.0])
def test_braket_gate_roundtrip(name, angle):
    pytest.importorskip("braket")
    original = imported_braket_gate(name, angle)
    data = original.to_dict()
    restored = Circuit.from_dict(json.loads(json.dumps(data)))
    assert restored.to_dict() == data
    assert np.allclose(
        original._qf.asgate().asoperator(), restored._qf.asgate().asoperator(), atol=1e-12
    )


def test_importer_gate_inventory():
    assert len(BRAKET_GATES) == 30
    assert set(Circuit._DICT_GATE_MAP) == {BRAKET_TO_QF[name] for name in BRAKET_GATES}
    # Once XY becomes importable, deliberately update the inventory and tests.
    pytest.importorskip("braket")
    from braket.circuits import Circuit as BK

    with pytest.raises(UnboundLocalError):
        Circuit.from_braket(BK().xy(0, 1, 0.37))


def test_asymmetric_roundtrip_preserves_complex_amplitudes():
    original = Circuit().h(0).ry(0.37, 1).t(0).cnot(1, 0)
    original._qf += qf.V(0)
    original._qf += qf.ISwap(1, 0)
    data = original.to_dict()
    restored = Circuit.from_dict(data)
    assert restored.to_dict() == data
    assert np.allclose(original.simulate().tensor, restored.simulate().tensor, atol=1e-12)


def test_all_braket_gates_in_one_circuit():
    pytest.importorskip("braket")
    original = Circuit()
    for name in BRAKET_GATES:
        original._qf += imported_braket_gate(name, 0.37)._qf
    restored = Circuit.from_dict(original.to_dict())
    assert restored.to_dict() == original.to_dict()
    assert np.allclose(original.simulate().tensor, restored.simulate().tensor, atol=1e-12)


@pytest.mark.parametrize("name", ["UnknownGate", "UnitaryGate"])
def test_unsupported_gate_between_known_gates_raises(name):
    data = {"gates": [
        {"gate": "H", "qubits": [0], "params": []},
        {"gate": name, "qubits": [0], "params": []},
        {"gate": "X", "qubits": [0], "params": []},
    ]}
    with pytest.raises(ValueError, match=f"unsupported gate '{name}' at index 1"):
        Circuit.from_dict(data)


@pytest.mark.parametrize("gate, qubits, params", [
    ("H", [0, 1], []), ("H", [0], [0.3]), ("H", [], []),
    ("Rx", [0], []), ("Rx", [0], [0.3, 0.4]), ("Rx", [], [0.3, 0.4]),
    ("CNot", [0], []), ("CNot", [0, 1, 2], []),
])
def test_gate_arity_rejected(gate, qubits, params):
    with pytest.raises(ValueError, match=f"gate '{gate}' at index 0 expects"):
        Circuit.from_dict({"gates": [{"gate": gate, "qubits": qubits, "params": params}]})


def test_empty_dictionary_compatibility():
    assert Circuit.from_dict({"gates": []}).to_dict() == {"gates": []}
    assert Circuit.from_dict({}).to_dict() == {"gates": []}


def test_sparse_wire_labels_preserved():
    original = Circuit().h(5).ry(0.37, 2).cnot(5, 2)
    restored = Circuit.from_dict(original.to_dict())
    assert restored.to_dict() == original.to_dict()
    assert restored._qf.qubits == original._qf.qubits
    assert np.allclose(original._qf.asgate().asoperator(), restored._qf.asgate().asoperator())


def test_symbolic_parameter_preserved_in_memory():
    import sympy

    theta = sympy.Symbol("theta")
    data = {"gates": [{"gate": "Ry", "qubits": [0], "params": [theta]}]}
    assert Circuit.from_dict(data).to_dict() == data


@pytest.mark.parametrize("name", ["XX", "YY", "ZZ"])
def test_quantumflow_parameter_convention_preserved(name):
    pytest.importorskip("braket")
    original = imported_braket_gate(name, 0.37)
    data = original.to_dict()
    assert data == {"gates": [{"gate": name, "qubits": [1, 0], "params": [0.37 / np.pi]}]}
    assert Circuit.from_dict(data).to_dict() == data


def test_to_dict_literal_fixtures():
    # Literal baseline output pins the writer without requiring optional SDKs.
    original = Circuit().h(0).x(1).rx(0.37, 1).cnot(1, 0)
    assert original.to_dict() == {"gates": [
        {"gate": "H", "qubits": [0], "params": []},
        {"gate": "X", "qubits": [1], "params": []},
        {"gate": "Rx", "qubits": [1], "params": [0.37]},
        {"gate": "CNot", "qubits": [1, 0], "params": []},
    ]}
    richer = Circuit()
    richer._qf += qf.V(1)
    richer._qf += qf.ISwap(1, 0)
    assert richer.to_dict() == {"gates": [
        {"gate": "V", "qubits": [1], "params": []},
        {"gate": "ISwap", "qubits": [1, 0], "params": []},
    ]}
