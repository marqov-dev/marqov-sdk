"""Integration tests for MarqovDevice type conversion and execution."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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


class TestBraketVerbatim:
    """verbatim=True must mirror BraketExecutor: validate native gates and wrap
    the circuit in add_verbatim_box before submitting. Without it, Rigetti's
    compiler folds RB sequences to identity and survival comes back flat.
    """

    def _rigetti_device(self) -> MarqovDevice:
        return MarqovDevice(
            "rigetti-cepheus-1",
            {
                "backend": "rigetti-cepheus-1",
                "device_arn": (
                    "arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q"
                ),
                "s3_bucket": "my-bucket",
                "s3_prefix": "my-prefix",
            },
        )

    def _mock_aws_device(self) -> MagicMock:
        task = MagicMock()
        task.result.return_value.measurement_counts = {"0": 100}
        dev = MagicMock()
        dev.run.return_value = task
        return dev

    def _submitted_op_types(self, mock_aws: MagicMock) -> list[str]:
        submitted = mock_aws.run.call_args[0][0]
        return [type(instr.operator).__name__ for instr in submitted.instructions]

    def test_verbatim_wraps_native_circuit_in_verbatim_box(self) -> None:
        device = self._rigetti_device()
        mock_aws = self._mock_aws_device()
        circuit = Circuit().rx(0.5, 0).rz(0.3, 0)  # native gates only
        with patch.object(MarqovDevice, "_get_provider_device", return_value=mock_aws):
            device.run(circuit, shots=100, verbatim=True)
        assert "StartVerbatimBox" in self._submitted_op_types(mock_aws)

    def test_verbatim_rejects_non_native_gates(self) -> None:
        device = self._rigetti_device()
        mock_aws = self._mock_aws_device()
        circuit = Circuit().h(0)  # H is not a Rigetti native gate
        with patch.object(MarqovDevice, "_get_provider_device", return_value=mock_aws):
            with pytest.raises(ValueError, match="native"):
                device.run(circuit, shots=100, verbatim=True)

    def test_no_verbatim_box_by_default(self) -> None:
        device = self._rigetti_device()
        mock_aws = self._mock_aws_device()
        circuit = Circuit().rx(0.5, 0).rz(0.3, 0)
        with patch.object(MarqovDevice, "_get_provider_device", return_value=mock_aws):
            device.run(circuit, shots=100)
        assert "StartVerbatimBox" not in self._submitted_op_types(mock_aws)


class _LoopBoundTask:
    """Mimics Braket's AwsQuantumTask.result(): drives a coroutine via
    asyncio.get_event_loop().run_until_complete — which raises inside a running
    loop unless the caller offloads to a worker thread with its own loop.
    """

    def result(self):
        async def _poll():
            return SimpleNamespace(measurement_counts={"0": 100})

        return asyncio.get_event_loop().run_until_complete(_poll())


class TestBraketEventLoopSafety:
    """MarqovDevice.run must be callable from within a running event loop on a
    real-QPU backend, where Braket's result() uses run_until_complete internally
    (marqov-sdk#67). Simulators don't hit this — only real Braket QPU tasks.
    """

    def _rigetti_device(self) -> MarqovDevice:
        return MarqovDevice(
            "rigetti-cepheus-1",
            {
                "backend": "rigetti-cepheus-1",
                "device_arn": (
                    "arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q"
                ),
                "s3_bucket": "my-bucket",
                "s3_prefix": "my-prefix",
            },
        )

    @pytest.mark.asyncio
    async def test_run_is_safe_from_within_running_loop(self) -> None:
        device = self._rigetti_device()
        mock_aws = MagicMock()
        mock_aws.run.return_value = _LoopBoundTask()
        # Called synchronously from inside this async test — i.e. a running loop.
        with patch.object(MarqovDevice, "_get_provider_device", return_value=mock_aws):
            counts = device.run(Circuit().rx(0.5, 0), shots=100)
        assert counts == {"0": 100}
