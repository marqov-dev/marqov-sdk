"""Integration tests for MarqovDevice type conversion and execution."""

import pytest

from marqov.circuits import Circuit
from marqov.device import MarqovDevice


@pytest.fixture
def local_device():
    """Create a MarqovDevice targeting the local simulator."""
    return MarqovDevice("local", {"backend": "local"})


class TestNormalizeCircuit:
    """Verify _normalize_circuit converts all supported types to marqov.Circuit."""

    def test_marqov_circuit_passthrough(self, local_device):
        circuit = Circuit().h(0).cnot(0, 1)
        result = local_device._normalize_circuit(circuit)
        assert isinstance(result, Circuit)
        assert result is circuit  # same object, not a copy

    def test_qasm_string(self, local_device):
        qasm = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n"
            "measure q -> c;\n"
        )
        result = local_device._normalize_circuit(qasm)
        assert isinstance(result, Circuit)
        assert result.num_qubits == 2

    def test_braket_circuit(self, local_device):
        from braket.circuits import Circuit as BraketCircuit

        bc = BraketCircuit().h(0).cnot(0, 1)
        result = local_device._normalize_circuit(bc)
        assert isinstance(result, Circuit)
        assert result.num_qubits == 2

    def test_qiskit_circuit(self, local_device):
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        result = local_device._normalize_circuit(qc)
        assert isinstance(result, Circuit)
        assert result.num_qubits == 2

    def test_cirq_circuit(self, local_device):
        import cirq

        q0, q1 = cirq.LineQubit.range(2)
        cc = cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1)])
        result = local_device._normalize_circuit(cc)
        assert isinstance(result, Circuit)
        assert result.num_qubits == 2

    def test_pennylane_tape(self, local_device):
        import pennylane as qml

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])

        result = local_device._normalize_circuit(tape)
        assert isinstance(result, Circuit)
        assert result.num_qubits == 2

    def test_unsupported_type_raises(self, local_device):
        with pytest.raises(TypeError, match="Unsupported circuit type"):
            local_device._normalize_circuit(42)

    def test_unsupported_type_message(self, local_device):
        with pytest.raises(TypeError, match="int"):
            local_device._normalize_circuit(42)


class TestToBackendFormat:
    """Verify _to_backend_format produces correct native types."""

    def test_local_produces_braket(self, local_device):
        from braket.circuits import Circuit as BraketCircuit

        mc = Circuit().h(0).cnot(0, 1)
        result = local_device._to_backend_format(mc)
        assert isinstance(result, BraketCircuit)

    def test_azure_produces_qiskit_with_measurements(self):
        from qiskit import QuantumCircuit

        azure_device = MarqovDevice(
            "quantinuum-syntax-checker",
            {
                "backend": "quantinuum-syntax-checker",
                "azure_subscription_id": "fake-sub-id",
                "azure_resource_group": "fake-rg",
                "azure_workspace_name": "fake-ws",
            },
        )
        mc = Circuit().h(0).cnot(0, 1)
        result = azure_device._to_backend_format(mc)
        assert isinstance(result, QuantumCircuit)
        # Must have classical registers (measurements added)
        assert len(result.cregs) > 0


class TestRunIntegration:
    """End-to-end execution on LocalSimulator with different input types."""

    def _assert_bell_state(self, counts, shots):
        """Assert Bell state properties on measurement counts."""
        assert isinstance(counts, dict)
        assert sum(counts.values()) == shots
        # Bell state: only "00" and "11" outcomes
        for key in counts:
            assert key in ("00", "11"), f"Unexpected outcome: {key}"

    def test_run_marqov_circuit(self, local_device):
        circuit = Circuit().h(0).cnot(0, 1)
        counts = local_device.run(circuit, shots=100)
        self._assert_bell_state(counts, 100)

    def test_run_braket_circuit(self, local_device):
        from braket.circuits import Circuit as BraketCircuit

        bc = BraketCircuit().h(0).cnot(0, 1)
        counts = local_device.run(bc, shots=100)
        self._assert_bell_state(counts, 100)

    def test_run_qiskit_circuit(self, local_device):
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        counts = local_device.run(qc, shots=100)
        self._assert_bell_state(counts, 100)

    def test_run_cirq_circuit(self, local_device):
        import cirq

        q0, q1 = cirq.LineQubit.range(2)
        cc = cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1)])
        counts = local_device.run(cc, shots=100)
        self._assert_bell_state(counts, 100)

    def test_run_pennylane_tape(self, local_device):
        import pennylane as qml

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])

        counts = local_device.run(tape, shots=100)
        self._assert_bell_state(counts, 100)

    def test_run_qasm_string(self, local_device):
        qasm = (
            'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
            "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0],q[1];\n"
            "measure q -> c;\n"
        )
        counts = local_device.run(qasm, shots=100)
        self._assert_bell_state(counts, 100)
