"""Integration tests for MarqovDevice type conversion and execution."""

import asyncio
import json
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

    def test_local_ignores_stray_ibm_credentials(self):
        """backend='local' must win over leftover IBM/Azure params.

        _get_provider_device() and run() both check backend in
        ("local", "marqov-sim") before is_ibm/is_azure; this method must
        agree, or a Qiskit circuit ends up fed to the Braket
        LocalSimulator the other two methods build.
        """
        from braket.circuits import Circuit as BraketCircuit

        device = MarqovDevice(
            "local",
            {"backend": "local", "ibm_token": "stale-token"},
        )
        mc = Circuit().h(0).cnot(0, 1)
        result = device._to_backend_format(mc)
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

    def test_run_local_with_stray_ibm_credentials(self):
        """End-to-end regression test for the local+stray-IBM-token bug.

        A user switching backend to "local" for a smoke test, with
        ibm_token still set from an earlier run, must still execute on the
        local simulator rather than crash on a format/device mismatch.
        """
        device = MarqovDevice(
            "local",
            {"backend": "local", "ibm_token": "stale-token"},
        )
        circuit = Circuit().h(0).cnot(0, 1)
        counts = device.run(circuit, shots=100)
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

    def test_verbatim_rejects_explicit_measure(self) -> None:
        """Parity with BraketExecutor: an explicit Measure must raise OUR error.

        Braket refuses to box a measured subcircuit, so allow-listing Measure would
        let the caller past this check and into Braket's opaque error. Measurement is
        applied implicitly via `shots`.
        """
        device = self._rigetti_device()
        mock_aws = self._mock_aws_device()
        # marqov.Circuit has no measure(); inject at the Braket-conversion boundary,
        # which is where an explicit Measure could realistically arrive.
        instr = MagicMock()
        instr.operator.name = "Measure"
        braket_circuit = MagicMock()
        braket_circuit.instructions = [instr]
        circuit = Circuit().rx(0.5, 0)
        with patch.object(MarqovDevice, "_get_provider_device", return_value=mock_aws):
            with patch.object(MarqovDevice, "_to_backend_format", return_value=braket_circuit):
                with pytest.raises(ValueError, match="native") as exc:
                    device.run(circuit, shots=100, verbatim=True)
        assert "Measure" in str(exc.value)

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


class _StubSamplerJob:
    """Stands in for a qiskit_ibm_runtime job handle."""

    def __init__(self, result) -> None:
        self._result = result

    def result(self):
        return self._result


def _stub_sampler_v2(result, recorder: dict):
    """Build a SamplerV2 replacement that hands back a prepared result.

    The stub sits exactly where ``qiskit_ibm_runtime.SamplerV2`` sits, so
    everything between MarqovDevice.run and the provider SDK (transpilation
    and count extraction) is the real code under test.
    """

    class _StubSamplerV2:
        def __init__(self, mode=None) -> None:
            recorder["mode"] = mode

        def run(self, pubs, shots=None):
            recorder["pubs"] = pubs
            recorder["shots"] = shots
            return _StubSamplerJob(result)

    return _StubSamplerV2


class TestRunIBMBranch:
    """MarqovDevice.run down the IBM branch, stubbed at the provider SDK.

    The counts must match IBMExecutor's convention: qubit 0 leftmost. X on
    qubit 0 of a 2-qubit circuit is the probe, because Bell and GHZ states are
    palindromes and so cannot detect a reversed bitstring.
    """

    @staticmethod
    def _ibm_device() -> MarqovDevice:
        return MarqovDevice(
            "ibm-fake-backend",
            {"backend": "ibm-fake-backend", "ibm_token": "fake-token"},
        )

    @staticmethod
    def _fake_backend():
        from qiskit.providers.fake_provider import GenericBackendV2

        return GenericBackendV2(num_qubits=5, seed=42)

    @staticmethod
    def _x0_sampler_result(shots: int = 100):
        """A genuine PrimitiveResult carrying DataBin(meas=BitArray(...))."""
        from qiskit import QuantumCircuit
        from qiskit.primitives import StatevectorSampler

        qc = QuantumCircuit(2)
        qc.x(0)
        qc.measure_all()
        return StatevectorSampler().run([qc], shots=shots).result()

    def _run(self, result, shots: int = 100):
        recorder: dict = {}
        device = self._ibm_device()
        with patch.object(
            MarqovDevice, "_get_provider_device", return_value=self._fake_backend()
        ):
            with patch(
                "qiskit_ibm_runtime.SamplerV2",
                _stub_sampler_v2(result, recorder),
            ):
                counts = device.run(Circuit().x(0).h(1), shots=shots)
        return counts, recorder

    def test_run_returns_counts_with_qubit0_leftmost(self) -> None:
        counts, recorder = self._run(self._x0_sampler_result())

        assert counts == {"10": 100}
        assert recorder["shots"] == 100

    def test_run_agrees_with_ibm_executor_extraction(self) -> None:
        """Both paths share one implementation, so they cannot diverge."""
        from marqov.executors.ibm import IBMExecutor

        result = self._x0_sampler_result()
        counts, _ = self._run(result)

        assert counts == IBMExecutor._extract_counts(result)

    def test_unresolvable_register_raises_with_field_names(self) -> None:
        """A DataBin with no BitArray must raise, not return {} silently."""
        from qiskit.primitives.containers import DataBin, PrimitiveResult
        from qiskit.primitives.containers.pub_result import PubResult

        result = PrimitiveResult([PubResult(DataBin(alpha=1.0))])

        with pytest.raises(ValueError, match="no measurement data") as exc:
            self._run(result)
        assert "alpha" in str(exc.value)

    def test_multiple_classical_registers_raise(self) -> None:
        from qiskit import ClassicalRegister, QuantumCircuit
        from qiskit.primitives import StatevectorSampler

        reg_a, reg_b = ClassicalRegister(1, "a"), ClassicalRegister(1, "b")
        qc = QuantumCircuit(2)
        qc.add_register(reg_a)
        qc.add_register(reg_b)
        qc.x(0)
        qc.measure(0, reg_a[0])
        qc.measure(1, reg_b[0])
        result = StatevectorSampler().run([qc], shots=100).result()

        with pytest.raises(NotImplementedError, match="multiple classical registers"):
            self._run(result)


class _StubAzureJob:
    """Stands in for an azure.quantum Job: serves the raw results blob.

    ``get_results`` is the method the branch must NOT use: it hands back
    normalized probabilities under display keys, not counts.
    """

    def __init__(self, payload: dict, output_data_format: str) -> None:
        self.details = SimpleNamespace(
            output_data_format=output_data_format,
            output_data_uri="https://example.invalid/results.json",
        )
        self._blob = json.dumps(payload).encode("utf8")

    def wait_until_completed(self, **kwargs) -> None:
        return None

    def download_data(self, uri: str) -> bytes:
        assert uri == self.details.output_data_uri
        return self._blob

    def get_results(self, **kwargs):
        raise AssertionError(
            "run() must not use get_results(): it returns probabilities"
        )


class _StubAzureTarget:
    """Stands in for the object workspace.get_targets(...) returns."""

    def __init__(self, job: _StubAzureJob) -> None:
        self._job = job
        self.submitted: dict = {}

    def submit(self, circuit, shots: int = 1000, **kwargs):
        self.submitted = {"circuit": circuit, "shots": shots}
        return self._job


def _v2_payload(histogram: list[dict], shots: int) -> dict:
    """A microsoft.quantum-results.v2 blob as azure-quantum reads it."""
    return {
        "DataFormat": "microsoft.quantum-results.v2",
        "Results": [
            {
                "Histogram": histogram,
                "Shots": [entry["Outcome"] for entry in histogram for _ in range(entry["Count"])],
            }
        ],
    }


class TestRunAzureBranch:
    """MarqovDevice.run down the Azure branch, stubbed at the provider SDK.

    The contract is counts keyed by bitstrings in the SDK's convention
    (qubit 0 leftmost), not azure-quantum's normalized probabilities keyed by
    display strings such as '[0]'.
    """

    @staticmethod
    def _azure_device() -> MarqovDevice:
        return MarqovDevice(
            "quantinuum-sim-h1-1e",
            {
                "backend": "quantinuum-sim-h1-1e",
                "azure_subscription_id": "fake-sub-id",
                "azure_resource_group": "fake-rg",
                "azure_workspace_name": "fake-ws",
            },
        )

    def _run(self, job: _StubAzureJob, shots: int = 100):
        target = _StubAzureTarget(job)
        device = self._azure_device()
        with patch.object(MarqovDevice, "_get_provider_device", return_value=target):
            counts = device.run(Circuit().x(0).h(1), shots=shots)
        return counts, target

    def test_v2_histogram_counts_are_returned_as_integer_counts(self) -> None:
        """X on qubit 0: display '[0, 1]' must come back as '10'."""
        job = _StubAzureJob(
            _v2_payload(
                [
                    {"Outcome": [0, 1], "Display": "[0, 1]", "Count": 60},
                    {"Outcome": [0, 0], "Display": "[0, 0]", "Count": 40},
                ],
                shots=100,
            ),
            "microsoft.quantum-results.v2",
        )

        counts, target = self._run(job, shots=100)

        assert counts == {"10": 60, "00": 40}
        assert all(isinstance(value, int) for value in counts.values())
        assert {len(key) for key in counts} == {2}
        assert set("".join(counts)) <= {"0", "1"}
        assert sum(counts.values()) == 100
        assert target.submitted["shots"] == 100

    def test_v2_bit_order_agrees_with_azure_executor(self) -> None:
        """Same provider outcome, same endianness as AzureQuantumExecutor.

        AzureQuantumExecutor reads the qiskit provider's counts, which join the
        provider's outcome list into a qiskit bitstring, then reverses them
        into the SDK's convention. The device path must land on the same keys.
        """
        histogram = [{"Outcome": [0, 1], "Display": "[0, 1]", "Count": 100}]
        job = _StubAzureJob(
            _v2_payload(histogram, shots=100), "microsoft.quantum-results.v2"
        )

        counts, _ = self._run(job, shots=100)

        # What AzureQuantumExecutor does to the qiskit provider's counts, which
        # for this outcome are keyed '01' (the joined outcome list).
        qiskit_counts = {"01": 100}
        executor_counts = {
            key.replace(" ", "")[::-1]: value for key, value in qiskit_counts.items()
        }
        assert counts == executor_counts

    def test_v1_probability_histogram_is_converted_to_counts(self) -> None:
        """A payload shaped like {'[0]': 0.5, ...} must never be returned as is."""
        job = _StubAzureJob(
            {"Histogram": ["[0, 1]", 0.5, "[1, 1]", 0.5]},
            "microsoft.quantum-results.v1",
        )

        counts, _ = self._run(job, shots=100)

        assert counts == {"10": 50, "11": 50}
        assert all(isinstance(value, int) for value in counts.values())
        assert sum(counts.values()) == 100

    def test_unsupported_output_format_raises_naming_the_format(self) -> None:
        job = _StubAzureJob({"Histogram": []}, "microsoft.quantum-results.v99")

        with pytest.raises(ValueError, match="microsoft.quantum-results.v99"):
            self._run(job)
