"""Tests for marqov.circuits module."""

import pytest

from marqov.circuits import Circuit, bell_state, ghz_state


def _basis_index(bits: str) -> int:
    """Flattened statevector index for a computational basis state.

    When wires ``0..N-1`` are all allocated, QuantumFlow orders the statevector
    with qubit 0 as the most significant bit, so the index is the bitstring read
    left-to-right as ``q0 q1 ... qN``. Example: ``"100"`` -> index 4.
    """
    return int(bits, 2)


def _state_equiv(circuit_a, circuit_b, atol: float = 1e-6) -> bool:
    """True if two circuits produce the same state up to a global phase.

    Compares full complex amplitudes via fidelity ``|<a|b>|^2``, so unlike a
    magnitude-only check it catches qubit permutations and arg-order bugs even
    when amplitudes share the same magnitude.
    """
    import numpy as np

    a = circuit_a.simulate().tensor.flatten()
    b = circuit_b.simulate().tensor.flatten()
    fidelity = np.abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real)
    return bool(np.isclose(fidelity, 1.0, atol=atol))


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

    def test_three_qubit_gates(self) -> None:
        """Three-qubit gates (CCX, CSWAP) build a 3-qubit circuit."""
        circuit = Circuit().ccx(0, 1, 2).cswap(0, 1, 2)
        assert circuit.num_qubits == 3

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


class TestThreeQubitGates:
    """Truth-table correctness for Toffoli (CCX) and Fredkin (CSWAP)."""

    def test_toffoli_flips_when_both_controls_set(self) -> None:
        """CCX flips the target only when both controls are |1>."""
        import numpy as np

        # |110> -> Toffoli -> |111>
        amps = Circuit().x(0).x(1).ccx(0, 1, 2).simulate().tensor.flatten()
        assert np.isclose(np.abs(amps[_basis_index("111")]) ** 2, 1.0)

    def test_toffoli_leaves_target_when_control_unset(self) -> None:
        """CCX leaves the target untouched when a control is |0>."""
        import numpy as np

        # q0=1, q1=0 -> no flip -> |100>
        amps = Circuit().x(0).ccx(0, 1, 2).simulate().tensor.flatten()
        assert np.isclose(np.abs(amps[_basis_index("100")]) ** 2, 1.0)

    def test_fredkin_swaps_when_control_set(self) -> None:
        """CSWAP swaps the two targets when the control is |1>."""
        import numpy as np

        # control q0=1, targets q1=1,q2=0 -> swapped -> q1=0,q2=1 -> |101>
        amps = Circuit().x(0).x(1).cswap(0, 1, 2).simulate().tensor.flatten()
        assert np.isclose(np.abs(amps[_basis_index("101")]) ** 2, 1.0)

    def test_fredkin_no_swap_when_control_unset(self) -> None:
        """CSWAP leaves the targets unchanged when the control is |0>."""
        import numpy as np

        # control q0=0 -> no swap -> q1=1 stays -> |010>
        amps = Circuit().x(1).cswap(0, 1, 2).simulate().tensor.flatten()
        assert np.isclose(np.abs(amps[_basis_index("010")]) ** 2, 1.0)

    def test_toffoli_alias_matches_ccx(self) -> None:
        """toffoli() is an alias for ccx()."""
        assert _state_equiv(
            Circuit().x(0).x(1).toffoli(0, 1, 2),
            Circuit().x(0).x(1).ccx(0, 1, 2),
        )

    def test_fredkin_alias_matches_cswap(self) -> None:
        """fredkin() is an alias for cswap()."""
        assert _state_equiv(
            Circuit().x(0).x(1).fredkin(0, 1, 2),
            Circuit().x(0).x(1).cswap(0, 1, 2),
        )

    def test_ccx_out_of_order_qubits(self) -> None:
        """CCX respects argument order with non-ascending qubit indices."""
        import numpy as np

        # controls q2 and q0, target q1: q0=q2=1 -> flip q1
        # start |101> (q0=1,q1=0,q2=1) -> |111>
        amps = Circuit().x(2).x(0).ccx(2, 0, 1).simulate().tensor.flatten()
        assert np.isclose(np.abs(amps[_basis_index("111")]) ** 2, 1.0)

    def test_cswap_out_of_order_qubits(self) -> None:
        """CSWAP respects argument order with non-ascending qubit indices."""
        import numpy as np

        # control q1, targets q0 and q2: control set, q0=1,q2=0 -> swap to q0=0,q2=1
        # start |110> (q0=1,q1=1,q2=0) -> |011>
        amps = Circuit().x(0).x(1).cswap(1, 0, 2).simulate().tensor.flatten()
        assert np.isclose(np.abs(amps[_basis_index("011")]) ** 2, 1.0)


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

    def test_three_qubit_gates_roundtrip(self) -> None:
        """CCX and CSWAP survive a Braket roundtrip (state preserved up to phase).

        Each prep is asymmetric and exercises its gate non-trivially, so the
        fidelity check would catch a control/target permutation.
        """
        for original in (
            Circuit().ry(0.7, 0).x(1).ccx(0, 1, 2),
            Circuit().ry(0.7, 0).x(1).cswap(0, 1, 2),
        ):
            imported = Circuit.from_braket(original.to_braket())
            assert _state_equiv(original, imported)


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

    def test_toffoli_imported_natively(self) -> None:
        """A Qiskit Toffoli imports as a native CCX (it is in the basis set now)."""
        import numpy as np
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(3)
        qc.x(0)
        qc.x(1)
        qc.ccx(0, 1, 2)  # Toffoli: flips qubit 2 when 0 and 1 are |1⟩

        imported = Circuit.from_qiskit(qc)

        # |110⟩ -> Toffoli -> |111⟩
        amps = imported.simulate().tensor.flatten()
        # |111⟩ is symmetric, so index 7 regardless of bit ordering.
        assert np.abs(amps[7]) ** 2 > 0.99

    def test_unmapped_three_qubit_gate_decomposes(self) -> None:
        """A 3-qubit gate outside the basis (CCZ) still decomposes and imports.

        Guards the transpiler-decompose path now that CCX/CSWAP import natively:
        CCZ has no direct mapping, so it must decompose through the basis
        (CCZ == H(t) . CCX . H(t)).
        """
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(3)
        qc.ry(0.7, 0)
        qc.x(1)
        qc.ccz(0, 1, 2)

        imported = Circuit.from_qiskit(qc)
        equivalent = Circuit().ry(0.7, 0).x(1).h(2).ccx(0, 1, 2).h(2)
        assert _state_equiv(imported, equivalent)

    def test_three_qubit_gates_roundtrip(self) -> None:
        """CCX and CSWAP survive a Qiskit roundtrip (state preserved up to phase)."""
        for original in (
            Circuit().ry(0.7, 0).x(1).ccx(0, 1, 2),
            Circuit().ry(0.7, 0).x(1).cswap(0, 1, 2),
        ):
            imported = Circuit.from_qiskit(original.to_qiskit())
            assert _state_equiv(original, imported)

    def test_native_cswap_imported_directly(self) -> None:
        """A Qiskit cswap is mapped to a native Fredkin, not decomposed away."""
        import numpy as np
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(3)
        qc.x(0)  # control
        qc.x(1)  # target0
        qc.cswap(0, 1, 2)  # swap q1,q2 -> q1=0, q2=1

        imported = Circuit.from_qiskit(qc)
        amps = imported.simulate().tensor.flatten()
        assert np.isclose(np.abs(amps[_basis_index("101")]) ** 2, 1.0)

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

    def test_toffoli_imported_natively(self) -> None:
        """A Cirq TOFFOLI imports as a native CCX (it is in the basis set now)."""
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
        # |111⟩ is symmetric, so index 7 regardless of bit ordering.
        assert np.abs(amps[7]) ** 2 > 0.99

    def test_three_qubit_gates_roundtrip(self) -> None:
        """CCX and CSWAP survive a Cirq roundtrip (state preserved up to phase)."""
        for original in (
            Circuit().ry(0.7, 0).x(1).ccx(0, 1, 2),
            Circuit().ry(0.7, 0).x(1).cswap(0, 1, 2),
        ):
            imported = Circuit.from_cirq(original.to_cirq())
            assert _state_equiv(original, imported)

    def test_native_fredkin_imported_directly(self) -> None:
        """A native Cirq FREDKIN maps to a Fredkin, not a decomposition."""
        import cirq
        import numpy as np

        q0, q1, q2 = cirq.LineQubit.range(3)
        cc = cirq.Circuit([
            cirq.X(q0),  # control
            cirq.X(q1),  # target0
            cirq.FREDKIN(q0, q1, q2),  # swap q1,q2 -> q1=0, q2=1
        ])

        imported = Circuit.from_cirq(cc)
        amps = imported.simulate().tensor.flatten()
        assert np.isclose(np.abs(amps[_basis_index("101")]) ** 2, 1.0)

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

    def test_toffoli_imported_natively(self) -> None:
        """A PennyLane Toffoli imports as a native CCX (it is in the basis set now)."""
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

    def test_three_qubit_gates_match_equivalent_circuit(self) -> None:
        """Toffoli/CSWAP from a tape match an equivalent Marqov circuit.

        PennyLane has no to_pennylane() export yet (issue #27), so we compare a
        natively-built tape against the equivalent Marqov circuit rather than
        doing a to->from roundtrip. Each prep is asymmetric so a control/target
        permutation would change the state.
        """
        import pennylane as qml

        cases = [
            (
                [qml.RY(0.7, wires=0), qml.PauliX(wires=1), qml.Toffoli(wires=[0, 1, 2])],
                Circuit().ry(0.7, 0).x(1).ccx(0, 1, 2),
            ),
            (
                [qml.RY(0.7, wires=0), qml.PauliX(wires=1), qml.CSWAP(wires=[0, 1, 2])],
                Circuit().ry(0.7, 0).x(1).cswap(0, 1, 2),
            ),
        ]
        for ops, original in cases:
            imported = Circuit.from_pennylane(self._make_tape(ops))
            assert _state_equiv(original, imported)

    def test_native_cswap_imported_directly(self) -> None:
        """A native PennyLane CSWAP maps to a Fredkin, not a decomposition."""
        import numpy as np
        import pennylane as qml

        tape = self._make_tape([
            qml.PauliX(wires=0),  # control
            qml.PauliX(wires=1),  # target0
            qml.CSWAP(wires=[0, 1, 2]),  # swap q1,q2 -> q1=0, q2=1
        ])
        imported = Circuit.from_pennylane(tape)
        amps = imported.simulate().tensor.flatten()
        assert np.isclose(np.abs(amps[_basis_index("101")]) ** 2, 1.0)

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

    def test_ccx_roundtrip_v2_and_v3(self) -> None:
        """Toffoli survives OpenQASM 2 and 3 roundtrips (state preserved up to phase).

        The prep is asymmetric and fires the gate (both controls set), so a
        control/target permutation on import fails the fidelity check.
        """
        original = Circuit().ry(0.7, 0).x(1).ccx(0, 1, 2)
        for version in (2, 3):
            restored = Circuit.from_openqasm(original.to_openqasm(version=version))
            assert _state_equiv(original, restored), f"v{version}"

    def test_cswap_roundtrip_v2_and_v3(self) -> None:
        """Fredkin survives OpenQASM 2 and 3 roundtrips (state preserved up to phase).

        QASM2 needs the legacy qelib1 gate set on import — qiskit's default
        QASM2 builtins omit cswap even though it dumps one (see from_openqasm).
        The asymmetric prep makes a qubit-order bug on this custom-parsed path
        actually fail rather than slip through a magnitude-only check.
        """
        original = Circuit().ry(0.7, 0).x(1).cswap(0, 1, 2)
        for version in (2, 3):
            restored = Circuit.from_openqasm(original.to_openqasm(version=version))
            assert _state_equiv(original, restored), f"v{version}"

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
        with pytest.raises(Exception):
            Circuit.from_openqasm("this is not valid QASM")


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
        assert restored.num_qubits == original.num_qubits

    def test_bell_state_roundtrip(self) -> None:
        """Bell state survives serialization roundtrip."""
        original = bell_state()
        data = original.to_dict()
        restored = Circuit.from_dict(data)

        # Both should produce same simulation results
        import numpy as np
        orig_state = original.simulate().tensor.flatten()
        rest_state = restored.simulate().tensor.flatten()
        assert np.allclose(np.abs(orig_state), np.abs(rest_state))

    def test_to_dict_three_qubit_gate_names(self) -> None:
        """to_dict serializes CCX and CSWAP under their QuantumFlow names."""
        data = Circuit().ccx(0, 1, 2).cswap(0, 1, 2).to_dict()
        names = [g["gate"] for g in data["gates"]]
        assert names == ["CCNot", "CSwap"]

    def test_three_qubit_gates_roundtrip(self) -> None:
        """CCX and CSWAP survive a to_dict/from_dict roundtrip (up to phase)."""
        for original in (
            Circuit().ry(0.7, 0).x(1).ccx(0, 1, 2),
            Circuit().ry(0.7, 0).x(1).cswap(0, 1, 2),
        ):
            restored = Circuit.from_dict(original.to_dict())
            assert _state_equiv(original, restored)
