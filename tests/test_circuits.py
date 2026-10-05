"""Tests for marqov.circuits module."""

import pytest

from marqov.circuits import Circuit, bell_state, ghz_state


class TestCircuit:
    """Tests for Circuit class."""

    def test_empty_circuit(self) -> None:
        """Empty circuit has zero qubits."""
        circuit = Circuit()
        assert circuit.num_qubits == 0

    def test_single_qubit_gates(self) -> None:
        """Single-qubit gates work correctly."""
        circuit = Circuit().h(0).x(1).y(2).z(3)
        assert circuit.num_qubits == 4

    def test_rotation_gates(self) -> None:
        """Rotation gates accept angle parameters."""
        import math

        circuit = Circuit()
        circuit.rx(math.pi / 2, 0)
        circuit.ry(math.pi / 4, 1)
        circuit.rz(math.pi, 2)
        assert circuit.num_qubits == 3

    def test_two_qubit_gates(self) -> None:
        """Two-qubit gates work correctly."""
        circuit = Circuit().cnot(0, 1).cz(1, 2).swap(0, 2)
        assert circuit.num_qubits == 3

    def test_cx_alias(self) -> None:
        """CX is an alias for CNOT."""
        circuit = Circuit().cx(0, 1)
        assert circuit.num_qubits == 2

    def test_method_chaining(self) -> None:
        """All gate methods return self for chaining."""
        circuit = Circuit().h(0).cnot(0, 1).x(1)
        assert isinstance(circuit, Circuit)

    def test_repr(self) -> None:
        """String representation shows qubit and gate count."""
        circuit = Circuit().h(0).cnot(0, 1)
        repr_str = repr(circuit)
        assert "Circuit" in repr_str
        assert "qubits" in repr_str


class TestSimulation:
    """Tests for circuit simulation."""

    def test_simulate_returns_state(self) -> None:
        """Simulation returns a QuantumFlow State object."""
        circuit = Circuit().h(0)
        state = circuit.simulate()
        # State should have a tensor attribute
        assert hasattr(state, "tensor")

    def test_bell_state_simulation(self) -> None:
        """Bell state produces entangled state vector."""
        import numpy as np

        circuit = bell_state()
        state = circuit.simulate()

        # Get amplitudes
        amplitudes = state.tensor.flatten()

        # Bell state: |00⟩ and |11⟩ should have equal probability
        prob_00 = abs(amplitudes[0]) ** 2
        prob_11 = abs(amplitudes[3]) ** 2

        assert np.isclose(prob_00, 0.5, atol=0.01)
        assert np.isclose(prob_11, 0.5, atol=0.01)

class TestToPytket:
    """Tests for Circuit.to_pytket()."""
    
    def test_to_pytket_bell_state(self) -> None:
        """Bell state converts to pytket Circuit object."""

        from pytket import Circuit as PytketCircuit
        from pytket.circuit import OpType

        expected = PytketCircuit(2)
        expected.H(0)
        expected.add_gate(OpType.CX, [0, 1])

        result = bell_state().to_pytket()
        assert isinstance(result, PytketCircuit)
        assert result.n_qubits == expected.n_qubits

    @pytest.mark.parametrize("targets", [(0, 1), (1, 0)])
    def test_braket_extended_gates_preserve_complex_circuit(self, targets) -> None:
        import numpy as np
        from braket.circuits import Circuit as BK

        source = BK().h(0).rx(1, 0.37).v(1).cnot(*targets).iswap(*targets).ry(0, -0.29)
        imported = Circuit.from_braket(source)
        exported = imported.to_pytket()
        np.testing.assert_allclose(
            exported.get_unitary(), source.to_unitary(), atol=1e-12, rtol=1e-12,
        )
        np.testing.assert_allclose(
            exported.get_statevector(), imported.simulate().tensor.flatten(),
            atol=1e-12, rtol=1e-12,
        )

    def test_converter_failure_is_chained_not_implemented(self, monkeypatch) -> None:
        import pytket.extensions.qiskit as extension

        failure = ValueError("Unsupported gate 'custom_gate'")

        def fail(_):
            raise failure

        monkeypatch.setattr(extension, "qiskit_to_tk", fail)
        with pytest.raises(NotImplementedError, match="custom_gate") as info:
            bell_state().to_pytket()
        assert info.value.__cause__ is failure
        assert str(failure) in str(info.value)


class TestConvenienceConstructors:
    """Tests for convenience circuit constructors."""

    def test_bell_state(self) -> None:
        """bell_state creates a 2-qubit circuit."""
        circuit = bell_state()
        assert circuit.num_qubits == 2

    def test_ghz_state(self) -> None:
        """ghz_state creates an n-qubit GHZ circuit."""
        for n in [3, 4, 5]:
            circuit = ghz_state(n)
            assert circuit.num_qubits == n


class TestBackendConversion:
    """Tests for backend conversion methods."""

    def test_to_braket(self) -> None:
        """Conversion to Braket circuit works."""
        circuit = Circuit().h(0).cnot(0, 1)
        braket_circuit = circuit.to_braket()
        # Should be a Braket Circuit object
        assert braket_circuit is not None

    def test_from_braket_roundtrip(self) -> None:
        """Import from Braket preserves circuit structure."""
        original = Circuit().h(0).cnot(0, 1)
        braket_circuit = original.to_braket()
        imported = Circuit.from_braket(braket_circuit)
        assert imported.num_qubits == original.num_qubits


class TestFromQiskit:
    """Tests for Circuit.from_qiskit()."""

    def test_bell_state_roundtrip(self) -> None:
        """Bell state survives Qiskit roundtrip (state vector preserved)."""
        import numpy as np

        original = bell_state()
        qiskit_circuit = original.to_qiskit()
        imported = Circuit.from_qiskit(qiskit_circuit)

        orig_amps = original.simulate().tensor.flatten()
        imported_amps = imported.simulate().tensor.flatten()
        assert np.allclose(np.abs(orig_amps), np.abs(imported_amps))

    def test_single_qubit_gates(self) -> None:
        """All single-qubit gates convert correctly."""
        original = Circuit().h(0).x(1).y(2).z(3).s(4).t(5)
        qiskit_circuit = original.to_qiskit()
        imported = Circuit.from_qiskit(qiskit_circuit)
        assert imported.num_qubits == original.num_qubits

    def test_rotation_gates_preserve_angles(self) -> None:
        """Parameterized rotation gates preserve angles through roundtrip."""
        import math
        import numpy as np

        original = Circuit().rx(math.pi / 3, 0).ry(math.pi / 5, 1).rz(math.pi / 7, 2)
        qiskit_circuit = original.to_qiskit()
        imported = Circuit.from_qiskit(qiskit_circuit)

        orig_amps = original.simulate().tensor.flatten()
        imported_amps = imported.simulate().tensor.flatten()
        assert np.allclose(np.abs(orig_amps), np.abs(imported_amps))

    def test_two_qubit_gates(self) -> None:
        """Two-qubit gates (CNOT, CZ, SWAP) convert correctly."""
        import numpy as np

        original = Circuit().h(0).cnot(0, 1).cz(1, 2).swap(0, 2)
        qiskit_circuit = original.to_qiskit()
        imported = Circuit.from_qiskit(qiskit_circuit)

        orig_amps = original.simulate().tensor.flatten()
        imported_amps = imported.simulate().tensor.flatten()
        assert np.allclose(np.abs(orig_amps), np.abs(imported_amps))

    def test_toffoli_decomposes(self) -> None:
        """Toffoli (CCX) gate is decomposed and produces correct output."""
        import numpy as np
        from qiskit import QuantumCircuit

        # Build a Qiskit circuit with Toffoli (not in our basis set)
        qc = QuantumCircuit(3)
        qc.x(0)
        qc.x(1)
        qc.ccx(0, 1, 2)  # Toffoli: flips qubit 2 when 0 and 1 are |1⟩

        imported = Circuit.from_qiskit(qc)

        # |110⟩ -> Toffoli -> |111⟩
        amps = imported.simulate().tensor.flatten()
        # Qubit ordering: |q2 q1 q0⟩ — |111⟩ = index 7
        assert np.abs(amps[7]) ** 2 > 0.99

    def test_barriers_ignored(self) -> None:
        """Barrier instructions are silently skipped."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.barrier()
        qc.cx(0, 1)

        imported = Circuit.from_qiskit(qc)
        assert imported.num_qubits == 2

    def test_type_error_on_wrong_input(self) -> None:
        """TypeError raised for non-QuantumCircuit input."""
        with pytest.raises(TypeError, match="Expected a Qiskit QuantumCircuit"):
            Circuit.from_qiskit("not a circuit")

    def test_native_qiskit_circuit(self) -> None:
        """A circuit built entirely in Qiskit can be imported."""
        import numpy as np
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        imported = Circuit.from_qiskit(qc)
        amps = imported.simulate().tensor.flatten()

        # Bell state: equal probability on |00⟩ and |11⟩
        assert np.isclose(np.abs(amps[0]) ** 2, 0.5, atol=0.01)
        assert np.isclose(np.abs(amps[3]) ** 2, 0.5, atol=0.01)

    def test_terminal_measurement_skipped(self) -> None:
        """A measurement with nothing after it is silently skipped."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure(0, 0)
        qc.measure(1, 1)

        imported = Circuit.from_qiskit(qc)
        assert imported.num_qubits == 2

    def test_mid_circuit_measurement_raises(self) -> None:
        """A measurement followed by another op on the same qubit raises."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        qc.x(0)

        with pytest.raises(ValueError, match="mid-circuit measurement"):
            Circuit.from_qiskit(qc)

    def test_classically_conditioned_measurement_raises(self) -> None:
        """A measurement whose classical bit conditions a later op (even on a
        different qubit, as in teleportation) is not terminal and raises."""
        from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

        qr = QuantumRegister(2)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)
        qc.h(0)
        qc.measure(0, 0)
        with qc.if_test((cr[0], 1)):
            qc.x(1)

        with pytest.raises(ValueError, match="mid-circuit measurement"):
            Circuit.from_qiskit(qc)

    def test_reset_raises(self) -> None:
        """A reset instruction is never implicit and always raises."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.h(0)
        qc.reset(0)

        with pytest.raises(ValueError, match="reset"):
            Circuit.from_qiskit(qc)

    def test_delay_raises(self) -> None:
        """A delay instruction is meaningful for timing and always raises."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1)
        qc.h(0)
        qc.delay(100, 0)

        with pytest.raises(ValueError, match="delay"):
            Circuit.from_qiskit(qc)

    def test_barrier_after_terminal_measurement_still_skipped(self) -> None:
        """A trailing barrier doesn't make a terminal measurement look mid-circuit."""
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        qc.barrier(0)

        imported = Circuit.from_qiskit(qc)
        assert imported.num_qubits == 1


class TestFromCirq:
    """Tests for Circuit.from_cirq()."""

    def test_bell_state_roundtrip(self) -> None:
        """Bell state survives Cirq roundtrip (state vector preserved)."""
        import numpy as np

        original = bell_state()
        cirq_circuit = original.to_cirq()
        imported = Circuit.from_cirq(cirq_circuit)

        orig_amps = original.simulate().tensor.flatten()
        imported_amps = imported.simulate().tensor.flatten()
        assert np.allclose(np.abs(orig_amps), np.abs(imported_amps))

    def test_single_qubit_gates(self) -> None:
        """All single-qubit gates convert correctly."""
        original = Circuit().h(0).x(1).y(2).z(3).s(4).t(5)
        cirq_circuit = original.to_cirq()
        imported = Circuit.from_cirq(cirq_circuit)
        assert imported.num_qubits == original.num_qubits

    def test_rotation_gates_preserve_angles(self) -> None:
        """Parameterized rotation gates preserve angles through roundtrip."""
        import math
        import numpy as np

        original = Circuit().rx(math.pi / 3, 0).ry(math.pi / 5, 1).rz(math.pi / 7, 2)
        cirq_circuit = original.to_cirq()
        imported = Circuit.from_cirq(cirq_circuit)

        orig_amps = original.simulate().tensor.flatten()
        imported_amps = imported.simulate().tensor.flatten()
        assert np.allclose(np.abs(orig_amps), np.abs(imported_amps))

    def test_two_qubit_gates(self) -> None:
        """Two-qubit gates (CNOT, CZ, SWAP) convert correctly."""
        import numpy as np

        original = Circuit().h(0).cnot(0, 1).cz(1, 2).swap(0, 2)
        cirq_circuit = original.to_cirq()
        imported = Circuit.from_cirq(cirq_circuit)

        orig_amps = original.simulate().tensor.flatten()
        imported_amps = imported.simulate().tensor.flatten()
        assert np.allclose(np.abs(orig_amps), np.abs(imported_amps))

    def test_toffoli_decomposes(self) -> None:
        """Toffoli (CCX) gate is decomposed and produces correct output."""
        import cirq
        import numpy as np

        q0, q1, q2 = cirq.LineQubit.range(3)
        cc = cirq.Circuit([
            cirq.X(q0),
            cirq.X(q1),
            cirq.TOFFOLI(q0, q1, q2),
        ])

        imported = Circuit.from_cirq(cc)

        # |110⟩ -> Toffoli -> |111⟩
        amps = imported.simulate().tensor.flatten()
        # Qubit ordering: |q2 q1 q0⟩ — |111⟩ = index 7
        assert np.abs(amps[7]) ** 2 > 0.99

    def test_measurements_skipped(self) -> None:
        """Measurement gates are silently skipped."""
        import cirq

        q0, q1 = cirq.LineQubit.range(2)
        cc = cirq.Circuit([
            cirq.H(q0),
            cirq.CNOT(q0, q1),
            cirq.measure(q0, q1, key="result"),
        ])

        imported = Circuit.from_cirq(cc)
        assert imported.num_qubits == 2

    def test_type_error_on_wrong_input(self) -> None:
        """TypeError raised for non-Circuit input."""
        with pytest.raises(TypeError, match="Expected a Cirq Circuit"):
            Circuit.from_cirq("not a circuit")

    def test_gridqubit_rejected(self) -> None:
        """GridQubit circuits raise TypeError with clear message."""
        import cirq

        q = cirq.GridQubit(0, 0)
        cc = cirq.Circuit([cirq.H(q)])

        with pytest.raises(TypeError, match="LineQubit"):
            Circuit.from_cirq(cc)

    def test_native_cirq_circuit(self) -> None:
        """A circuit built entirely in Cirq can be imported."""
        import cirq
        import numpy as np

        q0, q1 = cirq.LineQubit.range(2)
        cc = cirq.Circuit([
            cirq.H(q0),
            cirq.CNOT(q0, q1),
        ])

        imported = Circuit.from_cirq(cc)
        amps = imported.simulate().tensor.flatten()

        # Bell state: equal probability on |00⟩ and |11⟩
        assert np.isclose(np.abs(amps[0]) ** 2, 0.5, atol=0.01)
        assert np.isclose(np.abs(amps[3]) ** 2, 0.5, atol=0.01)

    def test_mid_circuit_measurement_raises(self) -> None:
        """A measurement followed by another op on the same qubit raises."""
        import cirq

        q0 = cirq.LineQubit(0)
        cc = cirq.Circuit([
            cirq.H(q0),
            cirq.measure(q0, key="m"),
            cirq.X(q0),
        ])

        with pytest.raises(ValueError, match="mid-circuit measurement"):
            Circuit.from_cirq(cc)

    def test_classically_controlled_measurement_raises(self) -> None:
        """A measurement whose key conditions a later op (even on a
        different qubit, as in teleportation) is not terminal and raises."""
        import cirq

        q0, q1 = cirq.LineQubit.range(2)
        cc = cirq.Circuit([
            cirq.H(q0),
            cirq.measure(q0, key="m"),
            cirq.X(q1).with_classical_controls("m"),
        ])

        with pytest.raises(ValueError, match="mid-circuit measurement"):
            Circuit.from_cirq(cc)

    def test_reset_raises(self) -> None:
        """A reset instruction is never implicit and always raises, naming
        the qubit (not just accidentally falling into the generic
        unsupported-gate path)."""
        import cirq

        q0 = cirq.LineQubit(0)
        cc = cirq.Circuit([cirq.H(q0), cirq.reset(q0)])

        with pytest.raises(ValueError, match=r"does not support 'reset'.*qubits \[0\]"):
            Circuit.from_cirq(cc)

    def test_delay_raises(self) -> None:
        """A wait/delay instruction is meaningful for timing and always raises."""
        import cirq

        q0 = cirq.LineQubit(0)
        cc = cirq.Circuit([cirq.H(q0), cirq.wait(q0, nanos=10)])

        with pytest.raises(ValueError, match="delay"):
            Circuit.from_cirq(cc)

    def test_terminal_measurement_inside_subcircuit_still_skipped(self) -> None:
        """A measurement nested in a CircuitOperation subcircuit, with nothing
        after it, is terminal and must not be flagged as mid-circuit."""
        import cirq

        q0 = cirq.LineQubit(0)
        sub = cirq.FrozenCircuit([cirq.H(q0), cirq.measure(q0, key="m")])
        cc = cirq.Circuit([cirq.CircuitOperation(sub)])

        imported = Circuit.from_cirq(cc)
        assert imported.num_qubits == 1

    def test_mid_circuit_measurement_in_repeated_subcircuit_raises(self) -> None:
        """A measurement repeated via CircuitOperation(repetitions>1) is only
        terminal on its last repetition — the earlier ones are genuinely
        mid-circuit and must raise, even though Cirq's decomposition reuses
        the same operation object across every repetition."""
        import cirq

        q0 = cirq.LineQubit(0)
        sub = cirq.FrozenCircuit([cirq.H(q0), cirq.measure(q0, key="m")])
        cc = cirq.Circuit([cirq.CircuitOperation(sub, repetitions=3)])

        with pytest.raises(ValueError, match="mid-circuit measurement"):
            Circuit.from_cirq(cc)


class TestFromPennylane:
    """Tests for Circuit.from_pennylane()."""

    def _make_tape(self, ops):
        """Helper to create a PennyLane QuantumTape from operations."""
        import pennylane as qml
        return qml.tape.QuantumTape(ops)

    def test_bell_state_roundtrip(self) -> None:
        """Bell state survives PennyLane roundtrip (state vector preserved)."""
        import numpy as np
        import pennylane as qml

        tape = self._make_tape([qml.Hadamard(wires=0), qml.CNOT(wires=[0, 1])])
        imported = Circuit.from_pennylane(tape)

        # Compare with our bell_state()
        orig_amps = bell_state().simulate().tensor.flatten()
        imported_amps = imported.simulate().tensor.flatten()
        assert np.allclose(np.abs(orig_amps), np.abs(imported_amps))

    def test_single_qubit_gates(self) -> None:
        """All single-qubit gates convert correctly."""
        import pennylane as qml

        tape = self._make_tape([
            qml.Hadamard(wires=0),
            qml.PauliX(wires=1),
            qml.PauliY(wires=2),
            qml.PauliZ(wires=3),
            qml.S(wires=4),
            qml.T(wires=5),
        ])
        imported = Circuit.from_pennylane(tape)
        assert imported.num_qubits == 6

    def test_rotation_gates_preserve_angles(self) -> None:
        """Parameterized rotation gates preserve angles through roundtrip."""
        import math
        import numpy as np
        import pennylane as qml

        tape = self._make_tape([
            qml.RX(math.pi / 3, wires=0),
            qml.RY(math.pi / 5, wires=1),
            qml.RZ(math.pi / 7, wires=2),
        ])
        imported = Circuit.from_pennylane(tape)

        # Build equivalent Marqov circuit
        original = Circuit().rx(math.pi / 3, 0).ry(math.pi / 5, 1).rz(math.pi / 7, 2)
        orig_amps = original.simulate().tensor.flatten()
        imported_amps = imported.simulate().tensor.flatten()
        assert np.allclose(np.abs(orig_amps), np.abs(imported_amps))

    def test_two_qubit_gates(self) -> None:
        """Two-qubit gates (CNOT, CZ, SWAP) convert correctly."""
        import numpy as np
        import pennylane as qml

        tape = self._make_tape([
            qml.Hadamard(wires=0),
            qml.CNOT(wires=[0, 1]),
            qml.CZ(wires=[1, 2]),
            qml.SWAP(wires=[0, 2]),
        ])
        imported = Circuit.from_pennylane(tape)

        original = Circuit().h(0).cnot(0, 1).cz(1, 2).swap(0, 2)
        orig_amps = original.simulate().tensor.flatten()
        imported_amps = imported.simulate().tensor.flatten()
        assert np.allclose(np.abs(orig_amps), np.abs(imported_amps))

    def test_toffoli_decomposes(self) -> None:
        """Toffoli gate is decomposed and produces correct output."""
        import numpy as np
        import pennylane as qml

        tape = self._make_tape([
            qml.PauliX(wires=0),
            qml.PauliX(wires=1),
            qml.Toffoli(wires=[0, 1, 2]),
        ])
        imported = Circuit.from_pennylane(tape)

        # |110⟩ -> Toffoli -> |111⟩
        amps = imported.simulate().tensor.flatten()
        assert np.abs(amps[7]) ** 2 > 0.99

    def test_type_error_on_wrong_input(self) -> None:
        """TypeError raised for non-tape input."""
        with pytest.raises(TypeError, match="Expected a PennyLane"):
            Circuit.from_pennylane("not a tape")

    def test_string_wire_rejected(self) -> None:
        """String wires raise TypeError with clear message."""
        import pennylane as qml

        tape = self._make_tape([qml.Hadamard(wires="a")])

        with pytest.raises(TypeError, match="integer wires"):
            Circuit.from_pennylane(tape)

    @pytest.mark.parametrize("dtype", ["int32", "int64", "uint64"])
    def test_numpy_integer_wires_preserve_complex_state(self, dtype) -> None:
        import numpy as np
        import pennylane as qml

        wires = np.arange(2, dtype=dtype)
        ops = [
            qml.Hadamard(wires=wires[0]),
            qml.RX(0.37, wires=wires[1]),
            qml.CNOT(wires=[wires[1], wires[0]]),
            qml.RY(-0.29, wires=wires[0]),
        ]
        tape = self._make_tape(ops)
        imported = Circuit.from_pennylane(tape)
        reference = qml.execute(
            [qml.tape.QuantumScript(ops, [qml.state()])],
            qml.device("default.qubit", wires=[0, 1]),
        )[0]
        np.testing.assert_allclose(
            imported.simulate().tensor.flatten(), reference, atol=1e-12, rtol=1e-12
        )
        assert all(type(q) is int for gate in imported._qf for q in gate.qubits)

    @pytest.mark.parametrize("wire", [True, False, 0.0])
    def test_non_integer_wire_rejected(self, wire) -> None:
        import pennylane as qml

        with pytest.raises(TypeError, match="integer wires"):
            Circuit.from_pennylane(self._make_tape([qml.Hadamard(wires=wire)]))

    def test_nested_decomposition_error_propagates(self) -> None:
        import pennylane as qml

        class Inner(qml.operation.Operation):
            num_wires = 1

            def decomposition(self):
                raise RuntimeError("boom inside inner decomposition")

        class Outer(qml.operation.Operation):
            num_wires = 1

            def decomposition(self):
                return [Inner(wires=self.wires)]

        with pytest.raises(RuntimeError, match="boom inside inner decomposition"):
            Circuit.from_pennylane(self._make_tape([Outer(wires=0)]))

    def test_nested_unmappable_gate_reports_leaf(self, monkeypatch) -> None:
        import sys
        from types import SimpleNamespace
        import pennylane as qml

        messages = []
        monkeypatch.setitem(sys.modules, "sentry_sdk", SimpleNamespace(
            capture_message=lambda message, **kwargs: messages.append(message)
        ))

        class Inner(qml.operation.Operation):
            num_wires = 1

        class Outer(qml.operation.Operation):
            num_wires = 1

            def decomposition(self):
                return [Inner(wires=self.wires)]

        with pytest.raises(ValueError, match="Unsupported PennyLane gate 'Inner'"):
            Circuit.from_pennylane(self._make_tape([Outer(wires=0)]))
        assert messages == ["Circuit.from_pennylane(): unmappable gate 'Inner'"]

    def test_native_pennylane_tape(self) -> None:
        """A tape built entirely in PennyLane can be imported."""
        import numpy as np
        import pennylane as qml

        tape = self._make_tape([
            qml.Hadamard(wires=0),
            qml.CNOT(wires=[0, 1]),
        ])

        imported = Circuit.from_pennylane(tape)
        amps = imported.simulate().tensor.flatten()

        # Bell state: equal probability on |00⟩ and |11⟩
        assert np.isclose(np.abs(amps[0]) ** 2, 0.5, atol=0.01)
        assert np.isclose(np.abs(amps[3]) ** 2, 0.5, atol=0.01)


class TestFromPyquil:
    """Tests for Circuit.from_pyquil()."""

    def test_bell_state_roundtrip(self) -> None:
        """Bell state survives PyQuil roundtrip (state vector preserved)."""
        import numpy as np

        original = bell_state()
        pyquil_program = original.to_pyquil()
        imported = Circuit.from_pyquil(pyquil_program)

        orig_amps = original.simulate().tensor.flatten()
        imported_amps = imported.simulate().tensor.flatten()
        assert imported.num_qubits == original.num_qubits
        assert np.allclose(np.abs(orig_amps), np.abs(imported_amps))

    def test_single_qubit_gates(self) -> None:
        """All single-qubit gates convert correctly."""
        from pyquil import Program
        from pyquil.gates import H, S, T, X, Y, Z

        original = Circuit().h(0).x(1).y(2).z(3).s(4).t(5)
        pyquil_program = Program(
            H(0),
            X(1),
            Y(2),
            Z(3),
            S(4),
            T(5),
        )
        imported = Circuit.from_pyquil(pyquil_program)
        assert imported.num_qubits == original.num_qubits

    def test_rotation_gates_preserve_angles(self) -> None:
        """Parameterized rotation gates preserve angles through roundtrip."""
        import math
        import numpy as np

        original = Circuit().rx(math.pi / 3, 0).ry(math.pi / 5, 1).rz(math.pi / 7, 2)
        imported = Circuit.from_pyquil(original.to_pyquil())

        orig_amps = original.simulate().tensor.flatten()
        imported_amps = imported.simulate().tensor.flatten()
        assert np.allclose(np.abs(orig_amps), np.abs(imported_amps))

    def test_two_qubit_gates(self) -> None:
        """Two-qubit gates (CNOT, CZ, SWAP) convert correctly."""
        import numpy as np

        original = Circuit().h(0).cnot(0, 1).cz(1, 2).swap(0, 2)
        imported = Circuit.from_pyquil(original.to_pyquil())

        orig_amps = original.simulate().tensor.flatten()
        imported_amps = imported.simulate().tensor.flatten()
        assert np.allclose(np.abs(orig_amps), np.abs(imported_amps))

    def test_native_swap_instruction(self) -> None:
        """Native PyQuil SWAP is supported directly."""
        import numpy as np
        from pyquil import Program
        from pyquil.gates import SWAP, X

        original = Circuit().x(0).swap(0, 1)
        imported = Circuit.from_pyquil(Program(X(0), SWAP(0, 1)))

        orig_amps = original.simulate().tensor.flatten()
        imported_amps = imported.simulate().tensor.flatten()
        assert np.allclose(np.abs(orig_amps), np.abs(imported_amps))

    def test_swap_cnot_decomposition(self) -> None:
        """SWAP decomposed as three CNOTs is recognized and imported."""
        import numpy as np
        from pyquil import Program
        from pyquil.gates import CNOT, X

        original = Circuit().x(0).swap(0, 1)
        imported = Circuit.from_pyquil(
            Program(
                X(0),
                CNOT(0, 1),
                CNOT(1, 0),
                CNOT(0, 1),
            )
        )

        orig_amps = original.simulate().tensor.flatten()
        imported_amps = imported.simulate().tensor.flatten()
        assert np.allclose(np.abs(orig_amps), np.abs(imported_amps))

    def test_measurements_skipped(self) -> None:
        """Measurement instructions are silently skipped."""
        from pyquil import Program
        from pyquil.gates import CNOT, H, MEASURE

        program = Program()
        program.declare("ro", "BIT", 2)
        program += Program(H(0), CNOT(0, 1))
        program += MEASURE(0, ("ro", 0))
        program += MEASURE(1, ("ro", 1))

        imported = Circuit.from_pyquil(program)
        assert imported.num_qubits == 2

    def test_type_error_on_wrong_input(self) -> None:
        """TypeError raised for non-Program input."""
        with pytest.raises(TypeError, match="Expected a PyQuil Program"):
            Circuit.from_pyquil("not a program")

    def test_not_implemented_for_unsupported_gate(self) -> None:
        """Quil-native gates outside the canonical set raise NotImplementedError."""
        from pyquil import Program
        from pyquil.gates import CCNOT

        with pytest.raises(NotImplementedError, match="CCNOT"):
            Circuit.from_pyquil(Program(CCNOT(0, 1, 2)))

    def test_symbolic_rotation_parameter_raises_not_implemented(self) -> None:
        """Symbolic PyQuil rotation parameters fail with a clear error."""
        from pyquil import Program
        from pyquil.gates import RX
        from pyquil.quilatom import Parameter

        with pytest.raises(NotImplementedError, match="numeric real-valued"):
            Circuit.from_pyquil(Program(RX(Parameter("theta"), 0)))

    def test_native_pyquil_bell_pair(self) -> None:
        """A hand-written PyQuil Bell pair produces the correct state vector."""
        import numpy as np
        from pyquil import Program
        from pyquil.gates import CNOT, H

        imported = Circuit.from_pyquil(Program(H(0), CNOT(0, 1)))
        amps = imported.simulate().tensor.flatten()

        assert np.isclose(np.abs(amps[0]) ** 2, 0.5, atol=0.01)
        assert np.isclose(np.abs(amps[3]) ** 2, 0.5, atol=0.01)

    def test_mid_circuit_measurement_raises(self) -> None:
        """A measurement followed by another gate on the same qubit raises."""
        from pyquil import Program
        from pyquil.gates import H, MEASURE, X

        program = Program()
        ro = program.declare("ro", "BIT", 1)
        program += H(0)
        program += MEASURE(0, ro[0])
        program += X(0)

        with pytest.raises(NotImplementedError, match="mid-circuit measurement"):
            Circuit.from_pyquil(program)


class TestOpenQASM:
    """Tests for OpenQASM import and export."""

    def test_to_openqasm2_basic(self) -> None:
        """to_openqasm() produces valid QASM 2.0 string."""
        circuit = Circuit().h(0).cnot(0, 1)
        qasm = circuit.to_openqasm(version=2)
        assert qasm.startswith("OPENQASM 2.0")
        assert "h " in qasm or "h(" in qasm

    def test_to_openqasm3_basic(self) -> None:
        """to_openqasm(version=3) produces valid QASM 3.0 string."""
        circuit = Circuit().h(0).cnot(0, 1)
        qasm = circuit.to_openqasm(version=3)
        assert "OPENQASM 3" in qasm

    def test_to_openqasm_invalid_version(self) -> None:
        """to_openqasm rejects invalid version."""
        circuit = Circuit().h(0)
        with pytest.raises(ValueError, match="Unsupported QASM version"):
            circuit.to_openqasm(version=4)

    def test_from_openqasm2(self) -> None:
        """from_openqasm parses QASM 2.0 string."""
        circuit = Circuit().h(0).cnot(0, 1)
        qasm = circuit.to_openqasm(version=2)
        restored = Circuit.from_openqasm(qasm)
        assert restored.num_qubits == 2

    def test_from_openqasm3(self) -> None:
        """from_openqasm parses QASM 3.0 string."""
        circuit = Circuit().h(0).cnot(0, 1)
        qasm = circuit.to_openqasm(version=3)
        restored = Circuit.from_openqasm(qasm)
        assert restored.num_qubits == 2

    def test_roundtrip_bell_state(self) -> None:
        """Bell state survives QASM roundtrip."""
        import numpy as np

        original = bell_state()
        qasm = original.to_openqasm(version=2)
        restored = Circuit.from_openqasm(qasm)

        orig_state = original.simulate().tensor.flatten()
        rest_state = restored.simulate().tensor.flatten()
        assert np.allclose(np.abs(orig_state), np.abs(rest_state))

    def test_roundtrip_with_rotations(self) -> None:
        """Parameterized gates survive QASM roundtrip."""
        import math
        import numpy as np

        original = Circuit().rx(math.pi / 4, 0).ry(math.pi / 3, 1).cnot(0, 1)
        qasm = original.to_openqasm(version=2)
        restored = Circuit.from_openqasm(qasm)

        orig_state = original.simulate().tensor.flatten()
        rest_state = restored.simulate().tensor.flatten()
        assert np.allclose(np.abs(orig_state), np.abs(rest_state), atol=1e-6)

    def test_auto_detects_qasm2(self) -> None:
        """from_openqasm auto-detects QASM 2.0."""
        qasm2_str = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nh q[0];\n'
        circuit = Circuit.from_openqasm(qasm2_str)
        assert circuit.num_qubits >= 1

    def test_auto_detects_qasm3(self) -> None:
        """from_openqasm auto-detects QASM 3.0."""
        qasm3_str = (
            "OPENQASM 3.0;\n"
            'include "stdgates.inc";\n'
            "qubit[2] q;\n"
            "h q[0];\n"
            "cx q[0], q[1];\n"
        )
        circuit = Circuit.from_openqasm(qasm3_str)
        assert circuit.num_qubits == 2

    def test_malformed_qasm_raises(self) -> None:
        """Malformed QASM string raises an error."""
        with pytest.raises(ValueError):
            Circuit.from_openqasm("this is not valid QASM")


    @pytest.mark.parametrize("prefix", [
        "// leading comment\n\n/* block\ncomment */\n",
        "  /* OPENQASM 2.0; */ // ignore this header\n",
        "/* first */ /* second */\n\n",
    ])
    def test_commented_qasm3_preserves_complex_state(self, prefix) -> None:
        import numpy as np

        original = Circuit().h(0).rx(0.37, 1).cnot(1, 0).ry(-0.29, 0)
        restored = Circuit.from_openqasm(prefix + original.to_openqasm(version=3))
        np.testing.assert_allclose(
            restored.simulate().tensor.flatten(),
            original.simulate().tensor.flatten(),
            atol=1e-12, rtol=1e-12,
        )

    def test_many_leading_comments_preserve_version_detection(self) -> None:
        source = 'OPENQASM 3.0; include "stdgates.inc"; qubit q; h q;'
        prefix = "/*" + "*//*" * 1000 + "*/\n"
        restored = Circuit.from_openqasm(prefix + source)
        assert restored.num_qubits == 1

    @pytest.mark.parametrize("version", [2, 3])
    def test_parser_error_is_chained_value_error(self, version) -> None:
        from qiskit.qasm2 import QASM2ParseError
        from qiskit.qasm3 import QASM3ImporterError

        with pytest.raises(ValueError, match=f"OpenQASM {version}") as info:
            Circuit.from_openqasm(f"OPENQASM {version}.0;\ninvalid syntax;\n")
        expected = QASM2ParseError if version == 2 else QASM3ImporterError
        assert isinstance(info.value.__cause__, expected)
        assert str(info.value.__cause__) in str(info.value)

    def test_headerless_qasm_defaults_to_version2(self) -> None:
        restored = Circuit.from_openqasm('include "qelib1.inc"; qreg q[1]; h q[0];')
        assert restored.num_qubits == 1

    def test_missing_qasm3_dependency_still_raises_import_error(self, monkeypatch) -> None:
        from qiskit import qasm3

        def missing_dependency(_):
            raise ImportError("missing optional QASM 3 importer")

        monkeypatch.setattr(qasm3, "loads", missing_dependency)
        with pytest.raises(ImportError, match="missing optional QASM 3 importer"):
            Circuit.from_openqasm("OPENQASM 3.0;")

    def test_non_comment_preamble_keeps_qasm2_default(self, monkeypatch) -> None:
        from qiskit import qasm2

        calls = []

        def parser(source):
            calls.append(source)
            raise qasm2.QASM2ParseError("preamble is not supported")

        monkeypatch.setattr(qasm2, "loads", parser)
        source = "pragma test;\nOPENQASM 3.0;"
        with pytest.raises(ValueError, match="OpenQASM 2"):
            Circuit.from_openqasm(source)
        assert calls == [source]

    def test_qasm3_conversion_error_is_not_wrapped(self) -> None:
        qasm = 'OPENQASM 3.0; qubit q; reset q;'
        with pytest.raises(ValueError, match="does not support 'reset'") as info:
            Circuit.from_openqasm(qasm)
        assert info.value.__cause__ is None

class TestSerialization:
    """Tests for circuit serialization."""

    def test_to_dict_simple(self) -> None:
        """to_dict serializes basic gates."""
        circuit = Circuit().h(0).x(1)
        data = circuit.to_dict()
        assert "gates" in data
        assert len(data["gates"]) == 2
        assert data["gates"][0]["gate"] == "H"
        assert data["gates"][1]["gate"] == "X"

    def test_to_dict_with_params(self) -> None:
        """to_dict includes rotation parameters."""
        import math
        circuit = Circuit().rx(math.pi / 2, 0)
        data = circuit.to_dict()
        assert len(data["gates"]) == 1
        assert data["gates"][0]["gate"] == "Rx"
        assert len(data["gates"][0]["params"]) > 0

    def test_from_dict_roundtrip(self) -> None:
        """from_dict reconstructs circuit correctly."""
        original = Circuit().h(0).cnot(0, 1).x(1)
        data = original.to_dict()
        restored = Circuit.from_dict(data)
        import numpy as np

        assert restored.to_dict() == data
        assert np.allclose(original.simulate().tensor, restored.simulate().tensor)

    def test_bell_state_roundtrip(self) -> None:
        """Bell state survives serialization roundtrip."""
        original = bell_state()
        data = original.to_dict()
        restored = Circuit.from_dict(data)

        # Both should produce same simulation results
        import numpy as np
        orig_state = original.simulate().tensor.flatten()
        rest_state = restored.simulate().tensor.flatten()
        assert np.allclose(orig_state, rest_state)


class TestToPennylane:
    """Tests for Circuit.to_pennylane(), the single Circuit -> PennyLane boundary.

    The gate-by-gate expectations follow marqov-dev/marqov-sdk#38 (Siddha
    Mavuram, "to_pennylane Closes #27"); the exact-state and refusal checks
    are added here.
    """

    @staticmethod
    def _every_gate_circuit(angle: float = 0.5) -> Circuit:
        return (
            Circuit()
            .cnot(0, 1).h(0).x(0).y(0).z(0).s(0).t(0)
            .rx(angle, 0).ry(angle, 1).rz(angle, 0)
            .cz(0, 1).swap(0, 1)
        )

    def test_every_canonical_gate_maps_to_its_pennylane_operation(self) -> None:
        """Each canonical gate maps one-to-one, in order, with its angle."""
        import pennylane as qml

        script = self._every_gate_circuit(0.5).to_pennylane()
        expected = [
            qml.CNOT(wires=[0, 1]), qml.Hadamard(0), qml.PauliX(0), qml.PauliY(0),
            qml.PauliZ(0), qml.S(0), qml.T(0), qml.RX(0.5, wires=0),
            qml.RY(0.5, wires=1), qml.RZ(0.5, wires=0), qml.CZ(wires=[0, 1]),
            qml.SWAP(wires=[0, 1]),
        ]
        assert len(script.operations) == len(expected)
        for got, want in zip(script.operations, expected):
            qml.assert_equal(got, want)

    def test_exported_unitary_matches_quantumflow_exactly(self) -> None:
        """The whole-circuit unitary, including global phase, is unchanged.

        Compared against QuantumFlow's operator, not ``simulate()``: in
        marqov-quantumflow 1.0.0, ``Y.run`` and ``Rz.run`` differ from their own
        ``asoperator()`` by a global phase, so the simulated state is only
        equal up to phase (see the probability check below).
        """
        import numpy as np
        import pennylane as qml

        circuit = self._every_gate_circuit(0.37).ry(1.1, 2).cnot(2, 0)
        script = circuit.to_pennylane()
        wires = sorted(script.wires.tolist())
        dim = 2 ** len(wires)
        expected = circuit._qf.asgate().asoperator().reshape(dim, dim)
        assert np.allclose(qml.matrix(script, wire_order=wires), expected)

    def test_exported_probabilities_match_quantumflow(self) -> None:
        """Computational-basis probabilities match the QuantumFlow simulation."""
        import numpy as np
        import pennylane as qml

        circuit = self._every_gate_circuit(0.37).ry(1.1, 2).cnot(2, 0)
        script = circuit.to_pennylane()
        wires = sorted(script.wires.tolist())
        tape = qml.tape.QuantumScript(script.operations, [qml.probs(wires=wires)])
        (probs,) = qml.execute([tape], qml.device("default.qubit", wires=wires))
        expected = np.abs(circuit.simulate().tensor.flatten()) ** 2
        assert np.allclose(probs, expected)

    def test_carries_no_measurements_and_no_shots(self) -> None:
        """Measurement and shots belong to the executor, not to the gate export."""
        script = bell_state().to_pennylane()
        assert script.measurements == []
        assert script.shots.total_shots is None

    def test_roundtrip_preserves_state(self) -> None:
        """to_pennylane -> from_pennylane reproduces the exact state."""
        import numpy as np

        original = self._every_gate_circuit(0.9)
        restored = Circuit.from_pennylane(original.to_pennylane())
        assert np.allclose(
            original.simulate().tensor.flatten(), restored.simulate().tensor.flatten()
        )

    def test_unbound_parameter_is_refused(self) -> None:
        """A symbolic (unbound) angle raises instead of being cast or dropped."""
        import sympy

        circuit = Circuit().rx(sympy.Symbol("theta"), 0)
        with pytest.raises(ValueError, match="unbound"):
            circuit.to_pennylane()

    def test_gate_outside_canonical_set_is_refused(self) -> None:
        """A QuantumFlow gate with no mapping raises NotImplementedError."""
        import quantumflow as qf

        circuit = Circuit().h(0)
        circuit._qf += qf.CCNot(0, 1, 2)
        with pytest.raises(NotImplementedError, match="CCNot"):
            circuit.to_pennylane()
