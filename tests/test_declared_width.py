"""Declared width on Circuit: kept as a field, with no identity marker.

A Qiskit or OpenQASM import keeps the source's declared qubits, and
``Circuit(num_qubits=n)`` declares a width directly. Gate-built circuits keep
the inferred width.
"""

from __future__ import annotations

import asyncio
import copy
import pickle

import cloudpickle
import pytest

from marqov.circuits import Circuit
from marqov.executors.local import LocalExecutor

qiskit = pytest.importorskip("qiskit")


def _declared_six() -> Circuit:
    """Six declared qubits with gates on q0 and q3 only."""
    return Circuit(num_qubits=6).x(0).x(3)


def _gate_names(circuit: Circuit) -> list[str]:
    return [gate["gate"] for gate in circuit.to_dict()["gates"]]


class TestDeclaredWidthOnTheCircuit:
    def test_declared_width_is_reported(self):
        assert Circuit(num_qubits=6).num_qubits == 6

    def test_declared_width_survives_gate_appends(self):
        assert _declared_six().num_qubits == 6

    def test_no_identity_marker_is_added(self):
        assert _gate_names(_declared_six()) == ["X", "X"]

    def test_gate_built_circuit_keeps_inferred_width(self):
        assert Circuit().x(2).num_qubits == 3
        assert Circuit().num_qubits == 0

    def test_gate_above_declared_width_widens_the_circuit(self):
        assert Circuit(num_qubits=2).x(4).num_qubits == 5

    def test_idle_qubits_include_the_declared_tail(self):
        assert _declared_six()._idle_qubits() == [1, 2, 4, 5]

    @pytest.mark.parametrize("bad", [-1, True, 2.0, "6"])
    def test_invalid_declared_width_is_rejected(self, bad):
        with pytest.raises(ValueError):
            Circuit(num_qubits=bad)

    def test_repr_shows_declared_width(self):
        assert repr(_declared_six()) == "Circuit(qubits=6, gates=2)"


class TestDeclaredWidthSurvivesCopies:
    def test_deepcopy(self):
        assert copy.deepcopy(_declared_six()).num_qubits == 6

    def test_pickle(self):
        assert pickle.loads(pickle.dumps(_declared_six())).num_qubits == 6

    def test_cloudpickle(self):
        assert cloudpickle.loads(cloudpickle.dumps(_declared_six())).num_qubits == 6

    def test_dict_round_trip(self):
        restored = Circuit.from_dict(_declared_six().to_dict())
        assert restored.num_qubits == 6
        assert _gate_names(restored) == ["X", "X"]

    def test_dict_of_gate_built_circuit_is_unchanged(self):
        assert Circuit().x(2).to_dict() == {
            "gates": [{"gate": "X", "qubits": [2], "params": []}]
        }

    def test_dict_without_width_still_loads(self):
        data = {"gates": [{"gate": "X", "qubits": [2], "params": []}]}
        assert Circuit.from_dict(data).num_qubits == 3

    @pytest.mark.parametrize("bad", [-1, True, 2.0, "6"])
    def test_dict_with_invalid_width_is_rejected(self, bad):
        with pytest.raises(ValueError):
            Circuit.from_dict({"gates": [], "num_qubits": bad})

    def test_dict_omits_a_width_the_gates_already_reach(self):
        assert "num_qubits" not in Circuit(num_qubits=2).x(4).to_dict()

    def test_object_pickled_before_the_field_existed(self):
        old = Circuit().x(1)
        del old.__dict__["_declared_width"]
        assert pickle.loads(pickle.dumps(old)).num_qubits == 2


class TestImportersKeepTheDeclaredWidth:
    def test_from_qiskit(self):
        qc = qiskit.QuantumCircuit(6)
        qc.x(0)
        qc.x(3)
        circuit = Circuit.from_qiskit(qc)
        assert circuit.num_qubits == 6
        assert _gate_names(circuit) == ["X", "X"]

    def test_from_qiskit_gate_free(self):
        assert Circuit.from_qiskit(qiskit.QuantumCircuit(6)).num_qubits == 6

    def test_from_qiskit_with_measured_top_qubit(self):
        qc = qiskit.QuantumCircuit(6, 1)
        qc.x(0)
        qc.measure(5, 0)
        assert Circuit.from_qiskit(qc).num_qubits == 6

    def test_from_openqasm_2(self):
        src = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[6]; x q[0]; x q[3];'
        assert Circuit.from_openqasm(src).num_qubits == 6

    def test_from_openqasm_3(self):
        src = 'OPENQASM 3.0; include "stdgates.inc"; qubit[6] q; x q[0]; x q[3];'
        assert Circuit.from_openqasm(src).num_qubits == 6


class TestExportersReadTheDeclaredWidth:
    def test_to_qiskit(self):
        assert _declared_six().to_qiskit().num_qubits == 6

    def test_to_openqasm_2(self):
        assert "qreg q[6];" in _declared_six().to_openqasm(2)

    def test_to_openqasm_3(self):
        text = _declared_six().to_openqasm(3)
        assert "qubit[6] q;" in text
        assert "q[5]" not in text

    def test_to_cirq(self):
        pytest.importorskip("cirq")
        assert len(_declared_six().to_cirq().all_qubits()) == 6

    def test_to_pennylane(self):
        pytest.importorskip("pennylane")
        assert len(_declared_six().to_pennylane().wires) == 6

    def test_to_pytket(self):
        pytest.importorskip("pytket")
        assert _declared_six().to_pytket().n_qubits == 6

    @pytest.mark.parametrize("via", ["qiskit", "qasm2", "qasm3", "cirq", "pennylane"])
    def test_round_trip_keeps_width(self, via):
        circuit = _declared_six()
        if via == "qiskit":
            back = Circuit.from_qiskit(circuit.to_qiskit())
        elif via == "qasm2":
            back = Circuit.from_openqasm(circuit.to_openqasm(2))
        elif via == "qasm3":
            back = Circuit.from_openqasm(circuit.to_openqasm(3))
        elif via == "cirq":
            pytest.importorskip("cirq")
            back = Circuit.from_cirq(circuit.to_cirq())
        else:
            pytest.importorskip("pennylane")
            back = Circuit.from_pennylane(circuit.to_pennylane())
        assert back.num_qubits == 6
        assert back._idle_qubits() == [1, 2, 4, 5]


class TestExecutionUsesTheDeclaredWidth:
    def test_local_counts(self):
        result = asyncio.run(LocalExecutor().execute(_declared_six(), shots=16))
        assert result.counts == {"100100": 16}

    def test_gate_free_declared_circuit_runs_locally(self):
        result = asyncio.run(LocalExecutor().execute(Circuit(num_qubits=6), shots=8))
        assert result.counts == {"000000": 8}

    def test_simulate_state_has_declared_width(self):
        assert _declared_six().simulate().qubit_nb == 6


class TestMeasuredQubitsCountTowardTheWidth:
    """A qubit the source program measures is part of the program's width."""

    def test_from_braket(self):
        braket = pytest.importorskip("braket.circuits")
        circuit = Circuit.from_braket(braket.Circuit().h(0).measure([0, 1]))
        assert circuit.num_qubits == 2
        assert _gate_names(circuit) == ["H"]

    def test_from_cirq(self):
        cirq = pytest.importorskip("cirq")
        q0, q1 = cirq.LineQubit.range(2)
        circuit = Circuit.from_cirq(cirq.Circuit([cirq.H(q0), cirq.measure(q0, q1)]))
        assert circuit.num_qubits == 2
        assert _gate_names(circuit) == ["H"]

    def test_from_pyquil(self):
        pyquil = pytest.importorskip("pyquil")
        from pyquil.gates import MEASURE, H

        program = pyquil.Program()
        ro = program.declare("ro", "BIT", 2)
        program += H(0)
        program += MEASURE(0, ro[0])
        program += MEASURE(1, ro[1])
        circuit = Circuit.from_pyquil(program)
        assert circuit.num_qubits == 2
        assert _gate_names(circuit) == ["H"]

    def test_measured_qubit_below_the_gates_changes_nothing(self):
        braket = pytest.importorskip("braket.circuits")
        circuit = Circuit.from_braket(braket.Circuit().h(1).measure([0, 1]))
        assert circuit.to_dict() == {"gates": [{"gate": "H", "qubits": [1], "params": []}]}


class TestIonQV04RefusesIdleWires:
    def test_declared_idle_wires_are_refused_with_a_clear_message(self):
        from marqov.executors.ionq import IonQExecutor, IonQExecutorConfig

        executor = IonQExecutor(
            IonQExecutorConfig(target="simulator", api_key="offline-placeholder", api_version="0.4")
        )
        with pytest.raises(ValueError, match=r"idle wires \[1, 2, 3, 4, 5\]"):
            asyncio.run(executor._execute_v04(Circuit(num_qubits=6).x(0), 8))
