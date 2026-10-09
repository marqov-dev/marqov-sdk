"""Lossless dictionary reconstruction and unchanged serialization output."""

import json

import numpy as np
import pytest
import quantumflow as qf

from marqov.circuits import Circuit

# Derive Braket coverage from its pinned converter map, not the reader map.
from quantumflow.xbraket import BRAKET_TO_QF

BRAKET_GATES = sorted(BRAKET_TO_QF)


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
    assert len(BRAKET_GATES) == 31
    assert set(Circuit._DICT_GATE_MAP) == {BRAKET_TO_QF[name] for name in BRAKET_GATES}


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


@pytest.mark.parametrize("angle", [0.37, -1.1, 4.0])
@pytest.mark.parametrize("targets", [(0, 1), (1, 0)])
def test_xy_import_and_dictionary_match_braket(angle, targets):
    pytest.importorskip("braket")
    from braket.circuits import Circuit as BK

    source = BK().h(0).rx(1, 0.83).xy(*targets, angle).rz(0, -0.29)
    original = Circuit.from_braket(source)
    data = json.loads(json.dumps(original.to_dict()))
    assert data["gates"][2] == {
        "gate": "XY", "qubits": list(targets), "params": [-angle / (2 * np.pi)],
    }
    restored = Circuit.from_dict(data)
    assert restored.to_dict() == data
    # Braket and QuantumFlow order active axes |q0 q1>, most significant first.
    np.testing.assert_allclose(
        original._qf.asgate().asoperator(), source.to_unitary(), atol=1e-12, rtol=1e-12
    )
    np.testing.assert_allclose(
        restored._qf.asgate().asoperator(), source.to_unitary(), atol=1e-12, rtol=1e-12
    )


@pytest.mark.parametrize("modifiers", [{"control": 2}, {"power": 0.5}])
def test_xy_still_refuses_instruction_modifiers(modifiers):
    pytest.importorskip("braket")
    from braket.circuits import Circuit as BK

    with pytest.raises(NotImplementedError, match="gate 'XY' at index 0"):
        Circuit.from_braket(BK().xy(1, 0, 0.37, **modifiers))


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int32, np.int64])
def test_numpy_numeric_parameters_json_roundtrip(dtype):
    angle = dtype(1) if issubclass(dtype, np.integer) else dtype(0.37)
    original = Circuit().h(0).rx(angle, 1).cnot(1, 0)
    data = original.to_dict()
    parameter = data["gates"][1]["params"][0]
    assert type(parameter) is (int if issubclass(dtype, np.integer) else float)
    assert parameter == angle
    restored = Circuit.from_dict(json.loads(json.dumps(data)))
    assert restored.to_dict() == data
    np.testing.assert_allclose(
        original.simulate().tensor, restored.simulate().tensor, atol=1e-12, rtol=1e-12
    )
    assert original._qf[1].params[0] is angle


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint64])
def test_numpy_qubit_labels_json_roundtrip(dtype):
    original = Circuit().h(dtype(1)).cnot(dtype(1), dtype(0))
    data = original.to_dict()
    assert data == {"gates": [
        {"gate": "H", "qubits": [1], "params": []},
        {"gate": "CNot", "qubits": [1, 0], "params": []},
    ]}
    assert all(type(q) is int for gate in data["gates"] for q in gate["qubits"])
    restored = Circuit.from_dict(json.loads(json.dumps(data)))
    np.testing.assert_array_equal(original.simulate().tensor, restored.simulate().tensor)
    assert all(isinstance(q, dtype) for q in original._qf.qubits)


def test_numpy_integer_parameter_keeps_exact_value():
    angle = np.uint64(2**63 + 1)
    data = Circuit().rx(angle, 0).to_dict()
    assert type(data["gates"][0]["params"][0]) is int
    assert json.loads(json.dumps(data))["gates"][0]["params"] == [2**63 + 1]


def test_extended_precision_parameter_is_not_rounded_for_json():
    angle = np.longdouble("0.1234567890123456789")
    data = Circuit().rx(angle, 0).to_dict()
    assert data["gates"][0]["params"][0] is angle
    with pytest.raises(TypeError):
        json.dumps(data)
