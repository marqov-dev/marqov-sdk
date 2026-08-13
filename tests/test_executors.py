"""Tests for marqov.executors module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from marqov.circuits import Circuit, bell_state
from marqov.executors import (
    AzureQuantumExecutor,
    BraketExecutor,
    ExecutionResult,
    ExecutorFactory,
    IBMExecutor,
    IonQExecutor,
    LocalExecutor,
    QuantinuumExecutor,
)
from marqov.executors.azure import AzureQuantumExecutorConfig
from marqov.executors.braket import BraketExecutorConfig, _extract_region_from_arn
from marqov.executors.ibm import IBMExecutorConfig
from marqov.executors.local import LocalExecutorConfig
from marqov.executors.quantinuum import QuantinuumExecutorConfig


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_probabilities(self) -> None:
        """Probabilities are calculated correctly from counts."""
        result = ExecutionResult(
            counts={"00": 500, "11": 500},
            backend="test",
            execution_time_ms=10.0,
            shots=1000,
        )
        probs = result.probabilities
        assert probs["00"] == 0.5
        assert probs["11"] == 0.5

    def test_probabilities_empty(self) -> None:
        """Empty counts return empty probabilities."""
        result = ExecutionResult(
            counts={},
            backend="test",
            execution_time_ms=0.0,
        )
        assert result.probabilities == {}

    def test_metadata_default(self) -> None:
        """Metadata defaults to empty dict."""
        result = ExecutionResult(
            counts={"0": 1},
            backend="test",
            execution_time_ms=0.0,
        )
        assert result.metadata == {}


class TestLocalExecutor:
    """Tests for LocalExecutor."""

    @pytest.mark.asyncio
    async def test_execute_returns_result(self) -> None:
        """Execute returns an ExecutionResult."""
        executor = LocalExecutor()
        circuit = Circuit().h(0)
        result = await executor.execute(circuit, shots=100)

        assert isinstance(result, ExecutionResult)
        assert result.backend == "local"
        assert result.shots == 100

    @pytest.mark.asyncio
    async def test_execute_bell_state(self) -> None:
        """Bell state produces expected measurement distribution."""
        executor = LocalExecutor(LocalExecutorConfig(seed=42))
        circuit = bell_state()
        result = await executor.execute(circuit, shots=1000)

        # Should only have 00 and 11 outcomes
        assert set(result.counts.keys()).issubset({"00", "11"})

        # Total counts should equal shots
        assert sum(result.counts.values()) == 1000

        # Each outcome should be roughly 50% (with some variance)
        for count in result.counts.values():
            assert 400 < count < 600

    @pytest.mark.asyncio
    async def test_execute_reproducible_with_seed(self) -> None:
        """Same seed produces same results."""
        circuit = bell_state()

        executor1 = LocalExecutor(LocalExecutorConfig(seed=123))
        result1 = await executor1.execute(circuit, shots=100)

        executor2 = LocalExecutor(LocalExecutorConfig(seed=123))
        result2 = await executor2.execute(circuit, shots=100)

        assert result1.counts == result2.counts

    @pytest.mark.asyncio
    async def test_execute_metadata(self) -> None:
        """Execution includes simulator metadata."""
        executor = LocalExecutor()
        circuit = Circuit().x(0)
        result = await executor.execute(circuit, shots=10)

        assert "simulator" in result.metadata
        assert result.metadata["simulator"] == "quantumflow"

    @pytest.mark.asyncio
    async def test_execute_timing(self) -> None:
        """Execution time is recorded."""
        executor = LocalExecutor()
        circuit = Circuit().h(0).cnot(0, 1)
        result = await executor.execute(circuit, shots=100)

        assert result.execution_time_ms > 0

    def test_executor_name(self) -> None:
        """Executor name property works."""
        executor = LocalExecutor()
        assert executor.name == "LocalExecutor"

    @pytest.mark.asyncio
    async def test_cancel_returns_false(self) -> None:
        """Cancel returns False (not implemented for local)."""
        executor = LocalExecutor()
        assert await executor.cancel("any-id") is False


class TestBraketExecutorConfig:
    """Tests for BraketExecutorConfig."""

    def test_config_required_fields(self) -> None:
        """Config requires device_arn and s3_bucket."""
        config = BraketExecutorConfig(
            device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
            s3_bucket="my-bucket",
        )
        assert config.device_arn == "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
        assert config.s3_bucket == "my-bucket"

    def test_config_defaults(self) -> None:
        """Config has sensible defaults."""
        config = BraketExecutorConfig(
            device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
            s3_bucket="my-bucket",
        )
        assert config.s3_prefix == "marqov"
        assert config.aws_profile is None
        assert config.aws_region is None
        assert config.poll_interval_seconds == 1.0
        assert config.timeout_seconds is None


class TestRegionExtraction:
    """Tests for ARN region extraction."""

    def test_simulator_arn_returns_us_east_1(self) -> None:
        """Simulator ARNs with empty region default to us-east-1."""
        arn = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
        assert _extract_region_from_arn(arn) == "us-east-1"

    def test_iqm_arn_returns_eu_north_1(self) -> None:
        """IQM ARN returns eu-north-1."""
        arn = "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet"
        assert _extract_region_from_arn(arn) == "eu-north-1"

    def test_rigetti_arn_returns_us_west_1(self) -> None:
        """Rigetti ARN returns us-west-1."""
        arn = "arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3"
        assert _extract_region_from_arn(arn) == "us-west-1"

    def test_ionq_arn_returns_us_east_1(self) -> None:
        """IonQ ARN returns us-east-1."""
        arn = "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1"
        assert _extract_region_from_arn(arn) == "us-east-1"


class TestBraketExecutor:
    """Tests for BraketExecutor with mocked AWS services."""

    @pytest.fixture
    def mock_braket(self) -> MagicMock:
        """Create mock Braket device and task."""
        mock_result = MagicMock()
        mock_result.measurement_counts = {"00": 500, "11": 500}

        mock_task = MagicMock()
        mock_task.id = "arn:aws:braket:us-east-1:123456789:quantum-task/abc123"
        mock_task.result.return_value = mock_result
        mock_task.metadata.return_value = {"executionDuration": 150}

        mock_device = MagicMock()
        mock_device.name = "SV1"
        mock_device.status = "ONLINE"
        mock_device.run.return_value = mock_task

        return {"device": mock_device, "task": mock_task, "result": mock_result}

    @pytest.mark.asyncio
    async def test_execute_returns_result(self, mock_braket: dict) -> None:
        """Execute returns ExecutionResult with correct structure."""
        config = BraketExecutorConfig(
            device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
            s3_bucket="my-bucket",
        )

        with patch("marqov.executors.braket.AwsDevice", return_value=mock_braket["device"]):
            with patch("marqov.executors.braket.boto3.Session"):
                with patch("marqov.executors.braket.AwsSession"):
                    executor = BraketExecutor(config)
                    circuit = bell_state()
                    result = await executor.execute(circuit, shots=1000)

        assert isinstance(result, ExecutionResult)
        assert result.counts == {"00": 500, "11": 500}
        assert result.shots == 1000
        assert "task_arn" in result.metadata
        assert result.metadata["task_arn"] == mock_braket["task"].id

    @pytest.mark.asyncio
    async def test_execute_uses_correct_s3_destination(self, mock_braket: dict) -> None:
        """Execute passes correct S3 destination to device.run()."""
        config = BraketExecutorConfig(
            device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
            s3_bucket="test-bucket",
            s3_prefix="test-prefix",
        )

        with patch("marqov.executors.braket.AwsDevice", return_value=mock_braket["device"]):
            with patch("marqov.executors.braket.boto3.Session"):
                with patch("marqov.executors.braket.AwsSession"):
                    executor = BraketExecutor(config)
                    circuit = Circuit().h(0)
                    await executor.execute(circuit, shots=100)

        # Verify device.run was called with correct S3 destination
        mock_braket["device"].run.assert_called_once()
        call_args = mock_braket["device"].run.call_args
        assert call_args.kwargs["s3_destination_folder"] == ("test-bucket", "test-prefix")

    @pytest.mark.asyncio
    async def test_executor_name(self) -> None:
        """Executor name property works."""
        config = BraketExecutorConfig(
            device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
            s3_bucket="my-bucket",
        )
        executor = BraketExecutor(config)
        assert executor.name == "BraketExecutor"

    @pytest.mark.asyncio
    async def test_is_device_available(self, mock_braket: dict) -> None:
        """is_device_available returns True when device is ONLINE."""
        config = BraketExecutorConfig(
            device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
            s3_bucket="my-bucket",
        )

        with patch("marqov.executors.braket.AwsDevice", return_value=mock_braket["device"]):
            with patch("marqov.executors.braket.boto3.Session"):
                with patch("marqov.executors.braket.AwsSession"):
                    executor = BraketExecutor(config)
                    assert await executor.is_device_available() is True

    @pytest.mark.asyncio
    async def test_cancel_returns_true_on_success(self, mock_braket: dict) -> None:
        """Cancel returns True when cancellation succeeds."""
        config = BraketExecutorConfig(
            device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
            s3_bucket="my-bucket",
        )

        mock_session = MagicMock()
        mock_session.braket_client.cancel_quantum_task.return_value = {}

        with patch("marqov.executors.braket.boto3.Session"):
            with patch("marqov.executors.braket.AwsSession", return_value=mock_session):
                executor = BraketExecutor(config)
                result = await executor.cancel("arn:aws:braket:us-east-1:123:quantum-task/xyz")

        assert result is True

    @pytest.mark.asyncio
    async def test_cancel_returns_false_on_error(self) -> None:
        """Cancel returns False when cancellation fails."""
        config = BraketExecutorConfig(
            device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
            s3_bucket="my-bucket",
        )

        mock_session = MagicMock()
        mock_session.braket_client.cancel_quantum_task.side_effect = Exception("Task not found")

        with patch("marqov.executors.braket.boto3.Session"):
            with patch("marqov.executors.braket.AwsSession", return_value=mock_session):
                executor = BraketExecutor(config)
                result = await executor.cancel("invalid-arn")

        assert result is False

    @pytest.mark.asyncio
    async def test_verbatim_mode_calls_add_verbatim_box(self, mock_braket: dict) -> None:
        """verbatim=True calls add_verbatim_box() before submitting to the device.

        This is the load-bearing regression test for the SRB verbatim bug:
        without add_verbatim_box(), Rigetti's compiler silently optimises Clifford
        sequences to identity and survival ≈ 1.0 at all sequence lengths.
        """
        config = BraketExecutorConfig(
            device_arn="arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q",
            s3_bucket="my-bucket",
        )

        # Track whether add_verbatim_box was called and what was passed to device.run.
        verbatim_wrapper = MagicMock()
        verbatim_wrapper.add_verbatim_box.return_value = MagicMock(name="wrapped_circuit")

        with patch("marqov.executors.braket.AwsDevice", return_value=mock_braket["device"]):
            with patch("marqov.executors.braket.boto3.Session"):
                with patch("marqov.executors.braket.AwsSession"):
                    with patch("marqov.executors.braket.BraketCircuit", return_value=verbatim_wrapper):
                        executor = BraketExecutor(config)
                        circuit = Circuit().rx(0.5, 0)
                        await executor.execute(circuit, shots=100, verbatim=True)

        verbatim_wrapper.add_verbatim_box.assert_called_once()
        submitted = mock_braket["device"].run.call_args.args[0]
        assert submitted is verbatim_wrapper.add_verbatim_box.return_value, (
            "device.run() should receive the verbatim-wrapped circuit, not the raw circuit"
        )

    @pytest.mark.asyncio
    async def test_verbatim_mode_rejects_non_native_gates(self, mock_braket: dict) -> None:
        """verbatim=True raises ValueError if circuit contains non-native gates.

        Catches the case where verbatim=True is passed but the circuit was built
        with clifford_to_circuit() (H/S gates) instead of clifford_to_circuit_native().
        Without this guard, the circuit reaches Rigetti and silently misbehaves.
        """
        config = BraketExecutorConfig(
            device_arn="arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q",
            s3_bucket="my-bucket",
        )

        # Construct a mock braket circuit whose instructions contain a non-native gate.
        mock_instr = MagicMock()
        mock_instr.operator.name = "H"
        mock_braket_circuit = MagicMock()
        mock_braket_circuit.instructions = [mock_instr]

        with patch("marqov.executors.braket.AwsDevice", return_value=mock_braket["device"]):
            with patch("marqov.executors.braket.boto3.Session"):
                with patch("marqov.executors.braket.AwsSession"):
                    executor = BraketExecutor(config)
                    circuit = Circuit().h(0)
                    with patch.object(circuit, "to_braket", return_value=mock_braket_circuit):
                        with pytest.raises(ValueError, match="non-native gates"):
                            await executor.execute(circuit, shots=100, verbatim=True)

    @pytest.mark.asyncio
    async def test_verbatim_mode_rejects_explicit_measure(self, mock_braket: dict) -> None:
        """verbatim=True raises OUR ValueError for an explicit Measure instruction.

        Braket refuses to box a subcircuit containing a measurement — `add_verbatim_box`
        raises `ValueError: cannot measure a subcircuit inside a verbatim box.`
        (braket/circuits/circuit.py). So Measure must NOT be in our allowed set: if it
        is, the caller clears our validator and then hits Braket's opaque error instead
        of our actionable one. Braket applies measurement implicitly via `shots`, so an
        explicit Measure is never needed for SRB anyway.
        """
        config = BraketExecutorConfig(
            device_arn="arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q",
            s3_bucket="my-bucket",
        )

        # Native 1Q/2Q gates plus an explicit Measure — previously allow-listed.
        names = ["Rx", "Rz", "CZ", "Measure"]
        mock_instrs = []
        for name in names:
            instr = MagicMock()
            instr.operator.name = name
            mock_instrs.append(instr)
        mock_braket_circuit = MagicMock()
        mock_braket_circuit.instructions = mock_instrs

        with patch("marqov.executors.braket.AwsDevice", return_value=mock_braket["device"]):
            with patch("marqov.executors.braket.boto3.Session"):
                with patch("marqov.executors.braket.AwsSession"):
                    executor = BraketExecutor(config)
                    circuit = Circuit().rx(0.5, 0)
                    with patch.object(circuit, "to_braket", return_value=mock_braket_circuit):
                        with pytest.raises(ValueError, match="non-native gates") as exc:
                            await executor.execute(circuit, shots=100, verbatim=True)
        assert "Measure" in str(exc.value), (
            "the error should name Measure as the offending instruction"
        )


class TestQuantinuumExecutor:
    """Tests for QuantinuumExecutor."""

    def test_config_validation(self) -> None:
        """Config validates required fields."""
        config = QuantinuumExecutorConfig(
            device_name="H2-1",
            simulator="state-vector",
            group="test-group",
            label="test-label",
        )
        assert config.device_name == "H2-1"
        assert config.simulator == "state-vector"
        assert config.group == "test-group"
        assert config.label == "test-label"
        assert config.provider is None
        assert config.api_handler is None
        assert config.compilation_config is None
        assert config.options == {}
        assert config.poll_interval_seconds == 2.0
        assert config.timeout_seconds == 300.0
        assert config.optimisation_level == 2



    def test_executor_creation(self) -> None:
        """Executor can be created with valid config."""
        config = QuantinuumExecutorConfig(
            device_name="H2-1",
            simulator="state-vector",
            group="test-group",
            label="test-label",
        )
        executor = QuantinuumExecutor(config)
        assert executor.config == config
        assert executor.name == "QuantinuumExecutor"


    def test_pytket_counts_tuple_keys(self) -> None:
        raw = {(0, 0): 512, (1, 1): 488}
        assert QuantinuumExecutor._pytket_counts_to_bitstring(raw) == {"00": 512, "11": 488}
   
    def test_pytket_counts_string_keys_passthrough(self) -> None:
        raw = {"00": 300, "11": 700}
        assert QuantinuumExecutor._pytket_counts_to_bitstring(raw) == {"00": 300, "11": 700}

    @pytest.mark.asyncio
    async def test_execute_converts_tuple_counts(self) -> None:
        """execute() converts pytket tuple counts to bitstring ExecutionResult.counts."""
        mock_result = MagicMock()
        mock_result.get_counts.return_value = {(0, 0): 512, (1, 1): 488}

        config = QuantinuumExecutorConfig(
            device_name="H2-1",
            simulator="state-vector",
            group="test-group",
            label="test-label",
        )
        executor = QuantinuumExecutor(config)

        with patch.object(executor, "_get_backend", new_callable=AsyncMock, return_value=MagicMock()):
            with patch.object(executor, "_run_sync", return_value=(mock_result, "job-123")):
                with patch.object(Circuit, "to_pytket", return_value=MagicMock()):
                    result = await executor.execute(bell_state(), shots=1000)

        assert isinstance(result, ExecutionResult)
        assert result.counts == {"00": 512, "11": 488}
        assert result.shots == 1000
        assert result.metadata["job_id"] == "job-123"

class TestAzureQuantumExecutor:
    """Tests for AzureQuantumExecutor."""

    def test_config_validation(self) -> None:
        """Config validates framework parameter."""
        # Valid frameworks
        config1 = AzureQuantumExecutorConfig(
            subscription_id="test",
            resource_group="test-rg",
            workspace_name="test-ws",
            location="eastus",
            target="ionq.simulator",
            framework="qiskit",
        )
        assert config1.framework == "qiskit"

        config2 = AzureQuantumExecutorConfig(
            subscription_id="test",
            resource_group="test-rg",
            workspace_name="test-ws",
            location="eastus",
            target="ionq.simulator",
            framework="cirq",
        )
        assert config2.framework == "cirq"

        # Invalid framework should raise
        with pytest.raises(ValueError, match="Unsupported framework"):
            AzureQuantumExecutorConfig(
                subscription_id="test",
                resource_group="test-rg",
                workspace_name="test-ws",
                location="eastus",
                target="ionq.simulator",
                framework="invalid",
            )

    def test_executor_creation(self) -> None:
        """Executor can be created with valid config."""
        config = AzureQuantumExecutorConfig(
            subscription_id="test-subscription",
            resource_group="quantum-rg",
            workspace_name="my-workspace",
            location="eastus",
            target="ionq.simulator",
            framework="qiskit",
        )
        executor = AzureQuantumExecutor(config)
        assert executor.config == config
        assert executor.name == "AzureQuantumExecutor"

    # Note: Integration tests requiring real Azure credentials
    # should be marked with @pytest.mark.integration and run separately
    # Example:
    # @pytest.mark.integration
    # @pytest.mark.asyncio
    # async def test_execute_ionq_simulator(self) -> None:
    #     """Execute on Azure IonQ simulator (requires Azure credentials)."""
    #     config = AzureQuantumExecutorConfig(
    #         subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
    #         resource_group=os.getenv("AZURE_RESOURCE_GROUP"),
    #         workspace_name=os.getenv("AZURE_WORKSPACE_NAME"),
    #         location=os.getenv("AZURE_LOCATION", "eastus"),
    #         target="ionq.simulator",
    #         framework="qiskit",
    #     )
    #     executor = AzureQuantumExecutor(config)
    #     circuit = bell_state()
    #     result = await executor.execute(circuit, shots=100)
    #     assert "00" in result.counts or "11" in result.counts
    #     assert result.shots == 100


class TestSmartCircuitErrors:
    """Tests for framework-aware error messages on executors."""

    def _make_mock(self, module: str, qualname: str):
        """Create a mock object whose type reports the given module and qualname."""
        cls = type(qualname, (), {"__module__": module, "__qualname__": qualname})
        return cls()

    @pytest.mark.asyncio
    async def test_qiskit_circuit_gives_conversion_hint(self) -> None:
        """Passing a Qiskit-like object gives a specific conversion hint."""
        executor = LocalExecutor()
        fake_qc = self._make_mock("qiskit.circuit", "QuantumCircuit")

        with pytest.raises(TypeError, match="Circuit.from_qiskit") as exc_info:
            await executor.execute(fake_qc)
        assert "marqov[qiskit]" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_cirq_circuit_gives_conversion_hint(self) -> None:
        """Passing a Cirq-like object gives a specific conversion hint."""
        executor = LocalExecutor()
        fake_cc = self._make_mock("cirq.circuits", "Circuit")

        with pytest.raises(TypeError, match="Circuit.from_cirq") as exc_info:
            await executor.execute(fake_cc)
        assert "marqov[cirq]" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_pennylane_tape_gives_conversion_hint(self) -> None:
        """Passing a PennyLane-like object gives a specific conversion hint."""
        executor = LocalExecutor()
        fake_tape = self._make_mock("pennylane.tape", "QuantumTape")

        with pytest.raises(TypeError, match="Circuit.from_pennylane") as exc_info:
            await executor.execute(fake_tape)
        assert "marqov[pennylane]" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_unknown_type_gives_generic_error(self) -> None:
        """Passing an unrecognized type lists all supported frameworks."""
        executor = LocalExecutor()

        with pytest.raises(TypeError, match="Expected a Marqov Circuit") as exc_info:
            await executor.execute("not a circuit")
        error_msg = str(exc_info.value)
        assert "from_qiskit" in error_msg
        assert "from_cirq" in error_msg
        assert "from_pennylane" in error_msg

    @pytest.mark.asyncio
    async def test_marqov_circuit_passes_through(self) -> None:
        """Marqov Circuit is accepted without error."""
        executor = LocalExecutor()
        circuit = Circuit().h(0)
        result = await executor.execute(circuit, shots=10)
        assert isinstance(result, ExecutionResult)


class TestDeviceStatus:
    """Tests for DeviceStatus dataclass."""

    def test_always_online(self) -> None:
        from marqov.executors.base import DeviceStatus
        status = DeviceStatus.always_online()
        assert status.status == "online"
        assert status.queue_depth == 0
        assert status.queue_time_seconds == 0

    def test_fields(self) -> None:
        from marqov.executors.base import DeviceStatus
        status = DeviceStatus(status="offline", queue_depth=42, queue_time_seconds=1260)
        assert status.status == "offline"
        assert status.queue_depth == 42
        assert status.queue_time_seconds == 1260

    def test_none_queue_fields(self) -> None:
        from marqov.executors.base import DeviceStatus
        status = DeviceStatus(status="maintenance", queue_depth=None, queue_time_seconds=None)
        assert status.queue_depth is None
        assert status.queue_time_seconds is None


class TestBaseExecutorGetStatus:
    """Tests for BaseExecutor.get_status() default."""

    @pytest.mark.asyncio
    async def test_default_returns_always_online(self) -> None:
        executor = LocalExecutor()
        status = await executor.get_status()
        assert status.status == "online"
        assert status.queue_depth == 0
        assert status.queue_time_seconds == 0


class TestBraketExecutorGetStatus:
    """Tests for BraketExecutor.get_status()."""

    @pytest.mark.asyncio
    async def test_online_status(self) -> None:
        config = BraketExecutorConfig(device_arn="arn:aws:braket:::device/qpu/ionq/Aria-1", s3_bucket="unused")
        executor = BraketExecutor(config)
        mock_device = MagicMock()
        mock_device.status = "ONLINE"
        mock_queue = MagicMock()
        mock_queue.quantum_tasks = {"Normal": 5, "Priority": 2}
        mock_device.queue_depth.return_value = mock_queue
        executor._device = mock_device
        status = await executor.get_status()
        assert status.status == "online"
        assert status.queue_depth == 7
        assert status.queue_time_seconds == 210

    @pytest.mark.asyncio
    async def test_offline_status(self) -> None:
        config = BraketExecutorConfig(device_arn="arn:aws:braket:::device/qpu/ionq/Aria-1", s3_bucket="unused")
        executor = BraketExecutor(config)
        mock_device = MagicMock()
        mock_device.status = "OFFLINE"
        executor._device = mock_device
        status = await executor.get_status()
        assert status.status == "offline"

    @pytest.mark.asyncio
    async def test_retired_maps_to_offline(self) -> None:
        config = BraketExecutorConfig(device_arn="arn:aws:braket:::device/qpu/ionq/Aria-1", s3_bucket="unused")
        executor = BraketExecutor(config)
        mock_device = MagicMock()
        mock_device.status = "RETIRED"
        executor._device = mock_device
        status = await executor.get_status()
        assert status.status == "offline"

    @pytest.mark.asyncio
    async def test_unknown_status_maps_to_maintenance(self) -> None:
        config = BraketExecutorConfig(device_arn="arn:aws:braket:::device/qpu/ionq/Aria-1", s3_bucket="unused")
        executor = BraketExecutor(config)
        mock_device = MagicMock()
        mock_device.status = "UNKNOWN_STATUS"
        mock_device.queue_depth.return_value = MagicMock(quantum_tasks=None)
        executor._device = mock_device
        status = await executor.get_status()
        assert status.status == "maintenance"

    @pytest.mark.asyncio
    async def test_queue_depth_failure_returns_none(self) -> None:
        config = BraketExecutorConfig(device_arn="arn:aws:braket:::device/qpu/ionq/Aria-1", s3_bucket="unused")
        executor = BraketExecutor(config)
        mock_device = MagicMock()
        mock_device.status = "ONLINE"
        mock_device.queue_depth.side_effect = Exception("API error")
        executor._device = mock_device
        status = await executor.get_status()
        assert status.status == "online"
        assert status.queue_depth is None
        assert status.queue_time_seconds is None

    @pytest.mark.asyncio
    async def test_device_failure_returns_maintenance(self) -> None:
        config = BraketExecutorConfig(device_arn="arn:aws:braket:::device/qpu/ionq/Aria-1", s3_bucket="unused")
        executor = BraketExecutor(config)
        with patch.object(executor, '_get_device', side_effect=Exception("Connection failed")):
            status = await executor.get_status()
            assert status.status == "maintenance"
            assert status.queue_depth is None

class TestQuantinuumExecutorGetStatus:
    """Tests for QuantinuumExecutor.get_status()."""

    @pytest.mark.asyncio
    async def test_online_status(self) -> None:
        config = QuantinuumExecutorConfig(device_name="H2-1", simulator="state-vector", group="test-group", label="test-label")
        executor = QuantinuumExecutor(config)
        with patch.object(executor, 'get_device_status', new_callable=AsyncMock, return_value="online"):
            status = await executor.get_status()
            assert status.status == "online"
            assert status.queue_depth is None

    @pytest.mark.asyncio
    async def test_offline_status(self) -> None:
        config = QuantinuumExecutorConfig(device_name="H2-1", simulator="state-vector", group="test-group", label="test-label")
        executor = QuantinuumExecutor(config)
        with patch.object(executor, 'get_device_status', new_callable=AsyncMock, return_value="offline"):
            status = await executor.get_status()
            assert status.status == "offline"
            assert status.queue_depth is None
class TestAzureExecutorGetStatus:
    """Tests for AzureQuantumExecutor.get_status()."""

    @pytest.mark.asyncio
    async def test_available_maps_to_online(self) -> None:
        config = AzureQuantumExecutorConfig(
            subscription_id="sub", resource_group="rg",
            workspace_name="ws", location="eastus", target="ionq.simulator",
        )
        executor = AzureQuantumExecutor(config)
        with patch.object(executor, 'get_device_status', new_callable=AsyncMock, return_value="Available"):
            status = await executor.get_status()
            assert status.status == "online"
            assert status.queue_depth is None

    @pytest.mark.asyncio
    async def test_degraded_maps_to_maintenance(self) -> None:
        config = AzureQuantumExecutorConfig(
            subscription_id="sub", resource_group="rg",
            workspace_name="ws", location="eastus", target="ionq.simulator",
        )
        executor = AzureQuantumExecutor(config)
        with patch.object(executor, 'get_device_status', new_callable=AsyncMock, return_value="Degraded"):
            status = await executor.get_status()
            assert status.status == "maintenance"

    @pytest.mark.asyncio
    async def test_unavailable_maps_to_offline(self) -> None:
        config = AzureQuantumExecutorConfig(
            subscription_id="sub", resource_group="rg",
            workspace_name="ws", location="eastus", target="ionq.simulator",
        )
        executor = AzureQuantumExecutor(config)
        with patch.object(executor, 'get_device_status', new_callable=AsyncMock, return_value="Unavailable"):
            status = await executor.get_status()
            assert status.status == "offline"

    @pytest.mark.asyncio
    async def test_error_returns_maintenance(self) -> None:
        config = AzureQuantumExecutorConfig(
            subscription_id="sub", resource_group="rg",
            workspace_name="ws", location="eastus", target="ionq.simulator",
        )
        executor = AzureQuantumExecutor(config)
        with patch.object(executor, 'get_device_status', new_callable=AsyncMock, side_effect=Exception("API error")):
            status = await executor.get_status()
            assert status.status == "maintenance"
            assert status.queue_depth is None


class TestIBMExecutorGetStatus:
    """Tests for IBMExecutor.get_status()."""

    @pytest.mark.asyncio
    async def test_operational_maps_to_online(self) -> None:
        """IBM operational=True maps to 'online' with queue depth."""
        config = IBMExecutorConfig(backend_name="ibm_kingston")
        executor = IBMExecutor(config)
        with patch.object(executor, 'get_backend_status', new_callable=AsyncMock,
                          return_value={"operational": True, "pending_jobs": 5, "status_msg": "active"}):
            status = await executor.get_status()
            assert status.status == "online"
            assert status.queue_depth == 5
            assert status.queue_time_seconds == 300  # 5 * 60

    @pytest.mark.asyncio
    async def test_not_operational_maps_to_offline(self) -> None:
        """IBM operational=False maps to 'offline'."""
        config = IBMExecutorConfig(backend_name="ibm_kingston")
        executor = IBMExecutor(config)
        with patch.object(executor, 'get_backend_status', new_callable=AsyncMock,
                          return_value={"operational": False, "pending_jobs": 0, "status_msg": "down"}):
            status = await executor.get_status()
            assert status.status == "offline"
            assert status.queue_depth == 0

    @pytest.mark.asyncio
    async def test_error_returns_maintenance(self) -> None:
        """API failure returns maintenance."""
        config = IBMExecutorConfig(backend_name="ibm_kingston")
        executor = IBMExecutor(config)
        with patch.object(executor, 'get_backend_status', new_callable=AsyncMock,
                          side_effect=Exception("API error")):
            status = await executor.get_status()
            assert status.status == "maintenance"
            assert status.queue_depth is None

    @pytest.mark.asyncio
    async def test_none_pending_jobs(self) -> None:
        """None pending_jobs returns None queue_time."""
        config = IBMExecutorConfig(backend_name="ibm_kingston")
        executor = IBMExecutor(config)
        with patch.object(executor, 'get_backend_status', new_callable=AsyncMock,
                          return_value={"operational": True, "pending_jobs": None, "status_msg": "active"}):
            status = await executor.get_status()
            assert status.status == "online"
            assert status.queue_depth is None
            assert status.queue_time_seconds is None


class TestExecutorFactory:
    """Tests for ExecutorFactory provider routing."""

    def test_ionq_direct_is_supported(self) -> None:
        """IonQ Direct is listed as a supported provider."""
        assert ExecutorFactory.is_provider_supported("IonQ Direct")
        assert "IonQ Direct" in ExecutorFactory.get_supported_providers()

    def test_create_ionq_executor(self) -> None:
        """Factory builds an IonQExecutor for the IonQ Direct provider."""
        executor = ExecutorFactory.create_executor(
            "qpu.aria-1",
            {"provider": "IonQ Direct", "api_key": "test-key"},
        )
        assert isinstance(executor, IonQExecutor)
        assert executor.config.target == "qpu.aria-1"
        assert executor.config.api_key == "test-key"

    def test_create_ionq_executor_uses_explicit_target(self) -> None:
        """An explicit target overrides the backend slug."""
        executor = ExecutorFactory.create_executor(
            "ignored-slug",
            {"provider": "IonQ Direct", "target": "simulator"},
        )
        assert isinstance(executor, IonQExecutor)
        assert executor.config.target == "simulator"

    def test_missing_provider_raises(self) -> None:
        """A config without a provider raises ValueError."""
        with pytest.raises(ValueError, match="missing 'provider'"):
            ExecutorFactory.create_executor("sv1", {})

    def test_supported_providers_has_no_duplicates(self) -> None:
        """Regression: `IonQ Direct` was listed twice after concurrent unitaryHACK merges
        re-added it. get_supported_providers() must have no duplicates."""
        providers = ExecutorFactory.get_supported_providers()
        assert len(providers) == len(set(providers)), (
            f"duplicate provider(s) in get_supported_providers(): {providers}")

    def test_supported_providers_matches_the_registered_executors(self) -> None:
        """The list must be exactly the providers create_executor() dispatches to — so a new
        executor added without updating the list (or vice versa) trips this."""
        assert set(ExecutorFactory.get_supported_providers()) == {
            "Local", "AWS Braket", "IBM Quantum", "Quantinuum",
            "Azure Quantum", "IonQ Direct", "Rigetti QCS", "Quantum Brilliance",
            "CUDA-Q",
        }
