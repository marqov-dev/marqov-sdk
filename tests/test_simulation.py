"""Tests for marqov.simulation module."""

from unittest.mock import MagicMock, patch

import pytest

from marqov.circuits import Circuit, bell_state
from marqov.executors.base import ExecutionResult
from marqov.executors.factory import ExecutorFactory
from marqov.simulation.backends import SIMULATION_BACKENDS
from marqov.simulation.circuit_converter import convert_counts, count_qubits, ensure_measurements
from marqov.simulation.config import SimulationConfig
from marqov.simulation.executor import SimulationExecutor, _validate_qubit_limit


class TestCountQubits:
    """Tests for count_qubits function."""

    def test_single_qreg(self) -> None:
        """Parses single qreg declaration."""
        qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4];\n'
        assert count_qubits(qasm) == 4

    def test_multiple_qregs(self) -> None:
        """Sums across multiple qreg declarations."""
        qasm = 'OPENQASM 2.0;\nqreg a[2];\nqreg b[3];\n'
        assert count_qubits(qasm) == 5

    def test_no_qreg_raises(self) -> None:
        """Raises ValueError when no qreg found."""
        with pytest.raises(ValueError, match="No qreg declaration"):
            count_qubits('OPENQASM 2.0;\n')

    def test_qreg_with_whitespace(self) -> None:
        """Handles varying whitespace in qreg declaration."""
        qasm = 'OPENQASM 2.0;\nqreg   q[10];\n'
        assert count_qubits(qasm) == 10


class TestEnsureMeasurements:
    """Tests for ensure_measurements function."""

    def test_adds_measurements_when_missing(self) -> None:
        """Appends creg + measure instructions to QASM without measurements."""
        qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\ncx q[0],q[1];\n'
        result = ensure_measurements(qasm)
        assert "creg c_q[2];" in result
        assert "measure q[0] -> c_q[0];" in result
        assert "measure q[1] -> c_q[1];" in result

    def test_preserves_existing_measurements(self) -> None:
        """Does not modify QASM that already has measure statements."""
        qasm = 'OPENQASM 2.0;\nqreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];\n'
        assert ensure_measurements(qasm) == qasm

    def test_multiple_qregs(self) -> None:
        """Handles circuits with multiple qreg declarations."""
        qasm = 'OPENQASM 2.0;\nqreg a[1];\nqreg b[1];\nh a[0];\n'
        result = ensure_measurements(qasm)
        assert "creg c_a[1];" in result
        assert "creg c_b[1];" in result
        assert "measure a[0] -> c_a[0];" in result
        assert "measure b[0] -> c_b[0];" in result

    def test_no_qreg_returns_unchanged(self) -> None:
        """Returns QASM unchanged if no qreg declaration found."""
        qasm = 'OPENQASM 2.0;\n'
        assert ensure_measurements(qasm) == qasm


class TestConvertCounts:
    """Tests for convert_counts function."""

    def test_dict_like_input(self) -> None:
        """Converts dict-like results with bool-sequence keys."""
        mock_results = {(True, True): 500, (False, False): 500}
        counts = convert_counts(mock_results)
        assert counts == {"11": 500, "00": 500}

    def test_empty_results(self) -> None:
        """Empty input returns empty dict."""
        assert convert_counts({}) == {}

    def test_single_qubit(self) -> None:
        """Single-qubit results produce single-char bitstrings."""
        mock_results = {(False,): 700, (True,): 300}
        counts = convert_counts(mock_results)
        assert counts == {"0": 700, "1": 300}

    def test_three_qubit(self) -> None:
        """Three-qubit results produce three-char bitstrings."""
        mock_results = {(True, False, True): 42}
        counts = convert_counts(mock_results)
        assert counts == {"101": 42}

    def test_list_keys(self) -> None:
        """Handles list keys (not just tuple) as bool sequences."""
        mock_results = {tuple([False, True]): 100}
        counts = convert_counts(mock_results)
        assert counts == {"01": 100}

    def test_pybind11_like_map(self) -> None:
        """Converts pybind11-like map (no .items(), iteration yields keys only)."""

        class FakeVectorBool:
            """Mimics pybind11 VectorBool: iterable of bools."""

            def __init__(self, bits: list[bool]) -> None:
                self._bits = bits

            def __iter__(self):
                return iter(self._bits)

            def __len__(self):
                return len(self._bits)

        class FakeMapVectorBoolInt:
            """Mimics pybind11 MapVectorBoolInt: no .items(), iter yields keys."""

            def __init__(self, data: dict) -> None:
                self._data = data

            def __iter__(self):
                return iter(self._data.keys())

            def __getitem__(self, key):
                return self._data[key]

        k1 = FakeVectorBool([True, False])
        k2 = FakeVectorBool([False, True])
        mock_results = FakeMapVectorBoolInt({k1: 300, k2: 700})
        counts = convert_counts(mock_results)
        assert counts == {"10": 300, "01": 700}


class TestSimulationConfig:
    """Tests for SimulationConfig dataclass."""

    def test_from_backend_statevector(self) -> None:
        """Creates config from statevector backend record."""
        backend = {
            "slug": "qb-sim-statevector",
            "provider_target_id": "qpp",
        }
        config = SimulationConfig.from_backend(backend)
        assert config.backend_id == "qpp"
        assert config.backend_type == "statevector"
        assert config.max_bond_dimension is None

    def test_from_backend_tensor_network(self) -> None:
        """Creates config from tensor network backend record."""
        backend = {
            "slug": "qb-sim-tensor-network",
            "provider_target_id": "tnqvm",
            "max_bond_dimension": 512,
            "svd_cutoff": 1e-6,
        }
        config = SimulationConfig.from_backend(backend)
        assert config.backend_id == "tnqvm"
        assert config.backend_type == "tensor-network"
        assert config.max_bond_dimension == 512
        assert config.svd_cutoff == 1e-6

    def test_from_backend_with_seed(self) -> None:
        """Passes through seed for reproducibility."""
        backend = {
            "slug": "qb-sim-statevector",
            "provider_target_id": "qpp",
            "seed": 42,
        }
        config = SimulationConfig.from_backend(backend)
        assert config.seed == 42

    def test_from_backend_unknown_slug_defaults(self) -> None:
        """Unknown slug defaults to statevector type."""
        backend = {"slug": "qb-sim-unknown", "provider_target_id": "qpp"}
        config = SimulationConfig.from_backend(backend)
        assert config.backend_type == "statevector"


class TestBackendRegistry:
    """Tests for backend registry."""

    def test_statevector_backend_exists(self) -> None:
        """Registry contains qb-sim-statevector."""
        assert "qb-sim-statevector" in SIMULATION_BACKENDS
        sv = SIMULATION_BACKENDS["qb-sim-statevector"]
        assert sv["provider"] == "Quantum Brilliance"
        assert sv["provider_target_id"] == "qpp"
        assert sv["qubit_count"] == 28
        assert sv["pricing"]["perShot"] == 0

    def test_tensor_network_backend_exists(self) -> None:
        """Registry contains qb-sim-tensor-network."""
        assert "qb-sim-tensor-network" in SIMULATION_BACKENDS
        tn = SIMULATION_BACKENDS["qb-sim-tensor-network"]
        assert tn["provider_target_id"] == "tnqvm"
        assert tn["qubit_count"] == 100

    def test_all_backends_have_required_fields(self) -> None:
        """Every backend has slug, provider, device_type, pricing."""
        required = {"slug", "name", "provider", "device_type", "provider_target_id", "qubit_count", "pricing"}
        for slug, backend in SIMULATION_BACKENDS.items():
            missing = required - set(backend.keys())
            assert not missing, f"{slug} missing fields: {missing}"


class TestValidateQubitLimit:
    """Tests for _validate_qubit_limit."""

    def test_within_limit_passes(self) -> None:
        """No error when qubit count is within limit."""
        config = SimulationConfig(backend_id="qpp", backend_type="statevector", num_qubits=10)
        _validate_qubit_limit(config)  # Should not raise

    def test_at_limit_passes(self) -> None:
        """No error when qubit count equals limit."""
        config = SimulationConfig(backend_id="qpp", backend_type="statevector", num_qubits=28)
        _validate_qubit_limit(config)  # Should not raise

    def test_exceeds_limit_raises(self) -> None:
        """Raises ValueError when qubit count exceeds limit."""
        config = SimulationConfig(backend_id="qpp", backend_type="statevector", num_qubits=29)
        with pytest.raises(ValueError, match="supports up to 28 qubits"):
            _validate_qubit_limit(config)

    def test_tensor_network_higher_limit(self) -> None:
        """Tensor network has higher qubit limit (100)."""
        config = SimulationConfig(backend_id="tnqvm", backend_type="tensor-network", num_qubits=80)
        _validate_qubit_limit(config)  # Should not raise

    def test_density_matrix_14_qubit_limit(self) -> None:
        """Density matrix backend limited to 14 qubits."""
        config = SimulationConfig(backend_id="cudaq:dm", backend_type="density-matrix", num_qubits=15)
        with pytest.raises(ValueError, match="supports up to 14 qubits"):
            _validate_qubit_limit(config)

    def test_unknown_backend_defaults_to_28(self) -> None:
        """Unknown backend ID defaults to 28 qubit limit."""
        config = SimulationConfig(backend_id="unknown", backend_type="custom", num_qubits=29)
        with pytest.raises(ValueError):
            _validate_qubit_limit(config)


class TestGpuBackendRegistry:
    """Tests for GPU backend registry."""

    def test_gpu_backends_have_required_fields(self) -> None:
        """Every GPU backend has required fields."""
        from marqov.simulation.backends import GPU_SIMULATION_BACKENDS

        required = {"slug", "name", "provider", "device_type", "provider_target_id", "qubit_count", "pricing"}
        for slug, backend in GPU_SIMULATION_BACKENDS.items():
            missing = required - set(backend.keys())
            assert not missing, f"{slug} missing fields: {missing}"

    def test_density_matrix_in_gpu_backends(self) -> None:
        """Density matrix backend requires GPU (in GPU registry)."""
        from marqov.simulation.backends import GPU_SIMULATION_BACKENDS

        assert "qb-sim-density-matrix" in GPU_SIMULATION_BACKENDS


class TestSimulationExecutor:
    """Tests for SimulationExecutor."""

    @pytest.mark.asyncio
    async def test_execute_returns_execution_result(self) -> None:
        """Execute returns a properly structured ExecutionResult."""
        config = SimulationConfig(backend_id="qpp", backend_type="statevector")

        mock_session = MagicMock()
        mock_session.results = [[{(False, False): 500, (True, True): 500}]]
        mock_qristal = MagicMock()
        mock_qristal.session.return_value = mock_session

        with patch.dict("sys.modules", {"qristal": MagicMock(), "qristal.core": mock_qristal}):
            executor = SimulationExecutor(config)
            circuit = bell_state()
            result = await executor.execute(circuit, shots=1000)

        assert isinstance(result, ExecutionResult)
        assert result.backend == "qb-sim-statevector"
        assert result.shots == 1000
        assert result.counts == {"00": 500, "11": 500}

    @pytest.mark.asyncio
    async def test_execute_sets_session_params(self) -> None:
        """Execute configures session with correct backend and qubit count."""
        config = SimulationConfig(backend_id="qpp", backend_type="statevector", seed=42)

        mock_session = MagicMock()
        mock_session.results = [[{(False,): 1000}]]
        mock_qristal = MagicMock()
        mock_qristal.session.return_value = mock_session

        with patch.dict("sys.modules", {"qristal": MagicMock(), "qristal.core": mock_qristal}):
            executor = SimulationExecutor(config)
            circuit = Circuit().h(0)
            await executor.execute(circuit, shots=1000)

        mock_session.init.assert_called_once()
        assert mock_session.acc == "qpp"
        assert mock_session.sn == 1000
        assert mock_session.seed == 42
        mock_session.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_rejects_too_many_qubits(self) -> None:
        """Execute raises ValueError when circuit exceeds qubit limit."""
        config = SimulationConfig(backend_id="qpp", backend_type="statevector")
        executor = SimulationExecutor(config)

        circuit = MagicMock(spec=Circuit)
        circuit.to_openqasm.return_value = 'OPENQASM 2.0;\nqreg q[30];\n'

        with patch.object(executor, '_validate_circuit', return_value=circuit):
            with pytest.raises(ValueError, match="supports up to 28 qubits"):
                await executor.execute(circuit, shots=100)

    @pytest.mark.asyncio
    async def test_execute_passes_tensor_network_params(self) -> None:
        """Execute passes bond dimension and SVD cutoff to session."""
        config = SimulationConfig(
            backend_id="tnqvm",
            backend_type="tensor-network",
            max_bond_dimension=256,
            svd_cutoff=1e-8,
        )

        mock_session = MagicMock()
        mock_session.results = [[{(False,): 1000}]]
        mock_qristal = MagicMock()
        mock_qristal.session.return_value = mock_session

        with patch.dict("sys.modules", {"qristal": MagicMock(), "qristal.core": mock_qristal}):
            executor = SimulationExecutor(config)
            circuit = Circuit().h(0)
            await executor.execute(circuit, shots=100)

        assert mock_session.max_bond_dimension == 256
        mock_session.svd_cutoffs.__getitem__(0).__getitem__(0).__setitem__.assert_called_with(0, 1e-8)

    @pytest.mark.asyncio
    async def test_execute_passes_rel_svd_cutoff(self) -> None:
        """Execute passes rel_svd_cutoff to session via indexed access."""
        config = SimulationConfig(
            backend_id="tnqvm",
            backend_type="tensor-network",
            rel_svd_cutoff=1e-6,
        )

        mock_session = MagicMock()
        mock_session.results = [[{(False,): 1000}]]
        mock_qristal = MagicMock()
        mock_qristal.session.return_value = mock_session

        with patch.dict("sys.modules", {"qristal": MagicMock(), "qristal.core": mock_qristal}):
            executor = SimulationExecutor(config)
            circuit = Circuit().h(0)
            await executor.execute(circuit, shots=100)

        mock_session.rel_svd_cutoffs.__getitem__(0).__getitem__(0).__setitem__.assert_called_with(0, 1e-6)

    @pytest.mark.asyncio
    async def test_execute_empty_circuit(self) -> None:
        """Circuit with no gates produces valid (trivial) results."""
        config = SimulationConfig(backend_id="qpp", backend_type="statevector")

        mock_session = MagicMock()
        mock_session.results = [[{(False,): 1000}]]
        mock_qristal = MagicMock()
        mock_qristal.session.return_value = mock_session

        circuit = MagicMock(spec=Circuit)
        circuit.to_openqasm.return_value = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\n'

        with patch.dict("sys.modules", {"qristal": MagicMock(), "qristal.core": mock_qristal}):
            with patch.object(SimulationExecutor, '_validate_circuit', return_value=circuit):
                executor = SimulationExecutor(config)
                result = await executor.execute(circuit, shots=1000)

        assert isinstance(result, ExecutionResult)
        assert result.counts == {"0": 1000}

    @pytest.mark.asyncio
    async def test_execute_metadata_includes_simulator(self) -> None:
        """Result metadata includes simulator name and qubit count."""
        config = SimulationConfig(backend_id="qpp", backend_type="statevector")

        mock_session = MagicMock()
        mock_session.results = [[{(False,): 1000}]]
        mock_qristal = MagicMock()
        mock_qristal.session.return_value = mock_session

        with patch.dict("sys.modules", {"qristal": MagicMock(), "qristal.core": mock_qristal}):
            executor = SimulationExecutor(config)
            result = await executor.execute(Circuit().h(0), shots=100)

        assert result.metadata["simulator"] == "qpp"
        assert result.metadata["num_qubits"] >= 1


class TestFactoryIntegration:
    """Tests for ExecutorFactory Simulation support."""

    def test_create_simulation_executor(self) -> None:
        """Factory creates SimulationExecutor for Quantum Brilliance provider."""
        config = {
            "provider": "Quantum Brilliance",
            "slug": "qb-sim-statevector",
            "provider_target_id": "qpp",
        }
        executor = ExecutorFactory.create_executor("qb-sim-statevector", config)
        assert isinstance(executor, SimulationExecutor)

    def test_simulation_in_supported_providers(self) -> None:
        """Quantum Brilliance listed as a supported provider."""
        assert "Quantum Brilliance" in ExecutorFactory.get_supported_providers()
