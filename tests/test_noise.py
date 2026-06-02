"""Tests for marqov.simulation.noise module."""

import math

import pytest

from unittest.mock import MagicMock, patch

from marqov.circuits import Circuit
from marqov.simulation.backends import SIMULATION_BACKENDS
from marqov.simulation.config import SimulationConfig
from marqov.simulation.executor import SimulationExecutor
from marqov.simulation.noise import (
    AmplitudeDamping,
    Depolarizing,
    NoiseModel,
    PhaseDamping,
    ReadoutError,
)


class TestDepolarizing:
    """Tests for Depolarizing channel."""

    def test_creation(self) -> None:
        """Creates with probability parameter."""
        ch = Depolarizing(probability=0.01)
        assert ch.probability == 0.01

    def test_rejects_negative(self) -> None:
        """Rejects negative probability."""
        with pytest.raises(ValueError, match="probability"):
            Depolarizing(probability=-0.1)

    def test_rejects_above_one(self) -> None:
        """Rejects probability > 1."""
        with pytest.raises(ValueError, match="probability"):
            Depolarizing(probability=1.5)

    def test_repr(self) -> None:
        """Has readable repr."""
        ch = Depolarizing(probability=0.01)
        assert "0.01" in repr(ch)


class TestAmplitudeDamping:
    """Tests for AmplitudeDamping channel."""

    def test_creation(self) -> None:
        """Creates with gamma parameter."""
        ch = AmplitudeDamping(gamma=0.02)
        assert ch.gamma == 0.02

    def test_rejects_negative(self) -> None:
        """Rejects negative gamma."""
        with pytest.raises(ValueError, match="gamma"):
            AmplitudeDamping(gamma=-0.1)

    def test_rejects_above_one(self) -> None:
        """Rejects gamma > 1."""
        with pytest.raises(ValueError, match="gamma"):
            AmplitudeDamping(gamma=1.5)


class TestPhaseDamping:
    """Tests for PhaseDamping channel."""

    def test_creation(self) -> None:
        """Creates with gamma parameter."""
        ch = PhaseDamping(gamma=0.03)
        assert ch.gamma == 0.03

    def test_rejects_negative(self) -> None:
        """Rejects negative gamma."""
        with pytest.raises(ValueError, match="gamma"):
            PhaseDamping(gamma=-0.1)

    def test_rejects_above_one(self) -> None:
        """Rejects gamma > 1."""
        with pytest.raises(ValueError, match="gamma"):
            PhaseDamping(gamma=1.5)


class TestReadoutError:
    """Tests for ReadoutError channel."""

    def test_creation(self) -> None:
        """Creates with two probability parameters."""
        ch = ReadoutError(p0_given1=0.02, p1_given0=0.01)
        assert ch.p0_given1 == 0.02
        assert ch.p1_given0 == 0.01

    def test_rejects_negative(self) -> None:
        """Rejects negative probabilities."""
        with pytest.raises(ValueError, match="p0_given1"):
            ReadoutError(p0_given1=-0.1, p1_given0=0.01)

    def test_rejects_above_one(self) -> None:
        """Rejects probabilities > 1."""
        with pytest.raises(ValueError, match="p1_given0"):
            ReadoutError(p0_given1=0.01, p1_given0=1.5)


class TestNoiseModel:
    """Tests for NoiseModel container."""

    def test_empty_model(self) -> None:
        """New model has no entries."""
        model = NoiseModel()
        assert list(model.entries()) == []

    def test_add_channel(self) -> None:
        """Add a channel to specific qubits."""
        model = NoiseModel()
        model.add(Depolarizing(0.01), qubits=[0, 1])
        entries = list(model.entries())
        assert len(entries) == 1
        channel, qubits = entries[0]
        assert isinstance(channel, Depolarizing)
        assert qubits == [0, 1]

    def test_add_multiple_channels(self) -> None:
        """Multiple channels can be added."""
        model = NoiseModel()
        model.add(Depolarizing(0.01), qubits=[0])
        model.add(AmplitudeDamping(0.02), qubits=[1])
        model.add(ReadoutError(0.03, 0.01), qubits=[0, 1])
        assert len(list(model.entries())) == 3

    def test_add_validates_qubits_not_empty(self) -> None:
        """Add requires at least one qubit."""
        model = NoiseModel()
        with pytest.raises(ValueError, match="qubits"):
            model.add(Depolarizing(0.01), qubits=[])

    def test_add_validates_qubit_indices(self) -> None:
        """Add rejects negative qubit indices."""
        model = NoiseModel()
        with pytest.raises(ValueError, match="qubit"):
            model.add(Depolarizing(0.01), qubits=[-1])


class TestNoiseModelPresets:
    """Tests for NoiseModel convenience presets."""

    def test_depolarizing_uniform(self) -> None:
        """Creates uniform depolarizing noise on all qubits."""
        model = NoiseModel.depolarizing_uniform(p=0.01, num_qubits=3)
        entries = list(model.entries())
        assert len(entries) == 1
        channel, qubits = entries[0]
        assert isinstance(channel, Depolarizing)
        assert channel.probability == 0.01
        assert qubits == [0, 1, 2]

    def test_realistic_device(self) -> None:
        """Creates amplitude + phase damping from T1/T2/gate_time."""
        model = NoiseModel.realistic_device(
            t1=50e-6, t2=30e-6, gate_time=50e-9, num_qubits=2
        )
        entries = list(model.entries())
        # Should have amplitude damping + phase damping
        assert len(entries) == 2
        types = {type(ch) for ch, _ in entries}
        assert AmplitudeDamping in types
        assert PhaseDamping in types

    def test_realistic_device_gamma_values(self) -> None:
        """Gamma values are computed correctly from T1/T2."""
        t1 = 50e-6
        t2 = 30e-6
        gate_time = 50e-9
        model = NoiseModel.realistic_device(
            t1=t1, t2=t2, gate_time=gate_time, num_qubits=1
        )
        entries = list(model.entries())
        for channel, _ in entries:
            if isinstance(channel, AmplitudeDamping):
                expected = 1 - math.exp(-gate_time / t1)
                assert abs(channel.gamma - expected) < 1e-15
            elif isinstance(channel, PhaseDamping):
                expected = 1 - math.exp(-gate_time / t2)
                assert abs(channel.gamma - expected) < 1e-15



class TestNoisyAerBackend:
    """Tests for qb-sim-noisy-aer backend registry entry."""

    def test_noisy_aer_in_registry(self) -> None:
        """qb-sim-noisy-aer is in the backend registry."""
        assert "qb-sim-noisy-aer" in SIMULATION_BACKENDS

    def test_noisy_aer_uses_aer_backend(self) -> None:
        """qb-sim-noisy-aer maps to the aer provider target."""
        backend = SIMULATION_BACKENDS["qb-sim-noisy-aer"]
        assert backend["provider_target_id"] == "aer"

    def test_noisy_aer_has_required_fields(self) -> None:
        """qb-sim-noisy-aer has all required backend fields."""
        required = {"slug", "name", "provider", "device_type", "provider_target_id", "qubit_count", "pricing"}
        backend = SIMULATION_BACKENDS["qb-sim-noisy-aer"]
        missing = required - set(backend.keys())
        assert not missing, f"Missing fields: {missing}"



class TestSimulationConfigNoise:
    """Tests for noise_model field on SimulationConfig."""

    def test_default_noise_model_is_none(self) -> None:
        """Default config has no noise model."""
        config = SimulationConfig(backend_id="qpp", backend_type="statevector")
        assert config.noise_model is None

    def test_config_with_noise_model(self) -> None:
        """Config accepts a noise model."""
        noise = NoiseModel()
        noise.add(Depolarizing(0.01), qubits=[0])
        config = SimulationConfig(
            backend_id="aer", backend_type="noisy", noise_model=noise
        )
        assert config.noise_model is noise

    def test_from_backend_with_noise_model(self) -> None:
        """from_backend accepts noise_model kwarg."""
        noise = NoiseModel.depolarizing_uniform(p=0.01, num_qubits=2)
        backend = {"slug": "qb-sim-noisy-aer", "provider_target_id": "aer"}
        config = SimulationConfig.from_backend(backend, noise_model=noise)
        assert config.noise_model is noise
        assert config.backend_id == "aer"


class TestSimulationConfigStateVector:
    """Tests for extract_state_vector field on SimulationConfig."""

    def test_default_is_false(self) -> None:
        """Default config has extract_state_vector=False."""
        config = SimulationConfig(backend_id="qpp", backend_type="statevector")
        assert config.extract_state_vector is False

    def test_config_with_state_vector(self) -> None:
        """Config accepts extract_state_vector=True."""
        config = SimulationConfig(
            backend_id="qpp", backend_type="statevector",
            extract_state_vector=True,
        )
        assert config.extract_state_vector is True

    def test_from_backend_with_state_vector(self) -> None:
        """from_backend accepts extract_state_vector kwarg."""
        backend = {"slug": "qb-sim-statevector", "provider_target_id": "qpp"}
        config = SimulationConfig.from_backend(
            backend, extract_state_vector=True,
        )
        assert config.extract_state_vector is True
        assert config.backend_id == "qpp"

    def test_from_backend_default_state_vector(self) -> None:
        """from_backend defaults extract_state_vector to False."""
        backend = {"slug": "qb-sim-statevector", "provider_target_id": "qpp"}
        config = SimulationConfig.from_backend(backend)
        assert config.extract_state_vector is False



class TestExecutorNoiseIntegration:
    """Tests for noise model handling in SimulationExecutor."""

    @pytest.mark.asyncio
    async def test_execute_forces_aer_when_noise_set(self) -> None:
        """Executor forces acc='aer' when noise_model is present."""
        noise = NoiseModel()
        noise.add(Depolarizing(0.01), qubits=[0])
        config = SimulationConfig(
            backend_id="qpp", backend_type="statevector", noise_model=noise
        )

        # Use a dict to capture attribute assignments
        assigned = {}
        mock_session = MagicMock()
        mock_session.results = [[{(False,): 1000}]]

        original_setattr = type(mock_session).__setattr__

        def tracking_setattr(self, name, value):
            if not name.startswith("_"):
                assigned[name] = value
            original_setattr(self, name, value)

        mock_qristal = MagicMock()
        mock_qristal.session.return_value = mock_session

        with patch.dict("sys.modules", {"qristal": MagicMock(), "qristal.core": mock_qristal}):
            with patch.object(type(mock_session), "__setattr__", tracking_setattr):
                executor = SimulationExecutor(config)
                circuit = Circuit().h(0)
                await executor.execute(circuit, shots=100)

        # Should force aer, not qpp
        assert assigned.get("acc") == "aer"
        assert assigned.get("noise") is True

    @pytest.mark.asyncio
    async def test_execute_sets_noise_model_on_session(self) -> None:
        """Executor creates qristal NoiseModel and sets it on session."""
        noise = NoiseModel()
        noise.add(Depolarizing(0.01), qubits=[0])
        config = SimulationConfig(
            backend_id="aer", backend_type="noisy", noise_model=noise
        )

        assigned = {}
        mock_session = MagicMock()
        mock_session.results = [[{(False,): 1000}]]

        original_setattr = type(mock_session).__setattr__

        def tracking_setattr(self, name, value):
            if not name.startswith("_"):
                assigned[name] = value
            original_setattr(self, name, value)

        mock_qristal = MagicMock()
        mock_qristal.session.return_value = mock_session

        with patch.dict("sys.modules", {"qristal": MagicMock(), "qristal.core": mock_qristal}):
            with patch.object(type(mock_session), "__setattr__", tracking_setattr):
                executor = SimulationExecutor(config)
                circuit = Circuit().h(0)
                await executor.execute(circuit, shots=100)

        assert assigned.get("noise") is True
        assert "noise_model" in assigned

    @pytest.mark.asyncio
    async def test_execute_without_noise_skips_noise_setup(self) -> None:
        """Executor does not enable noise when noise_model is None."""
        config = SimulationConfig(backend_id="qpp", backend_type="statevector")

        assigned = {}
        mock_session = MagicMock()
        mock_session.results = [[{(False,): 1000}]]

        original_setattr = type(mock_session).__setattr__

        def tracking_setattr(self, name, value):
            if not name.startswith("_"):
                assigned[name] = value
            original_setattr(self, name, value)

        mock_qristal = MagicMock()
        mock_qristal.session.return_value = mock_session

        with patch.dict("sys.modules", {"qristal": MagicMock(), "qristal.core": mock_qristal}):
            with patch.object(type(mock_session), "__setattr__", tracking_setattr):
                executor = SimulationExecutor(config)
                circuit = Circuit().h(0)
                await executor.execute(circuit, shots=100)

        assert assigned.get("acc") == "qpp"
        assert "noise" not in assigned


class TestExecutorStateVector:
    """Tests for state vector extraction in SimulationExecutor."""

    @pytest.mark.asyncio
    async def test_state_vector_requested_sets_flag(self) -> None:
        """Executor sets session.get_state_vec when requested."""
        config = SimulationConfig(
            backend_id="qpp", backend_type="statevector",
            extract_state_vector=True,
        )

        assigned = {}
        mock_session = MagicMock()
        mock_session.results = [[{(False,): 1000}]]
        mock_session.get_state_vec_raw = [complex(1, 0), complex(0, 0)]

        original_setattr = type(mock_session).__setattr__

        def tracking_setattr(self, name, value):
            if not name.startswith("_"):
                assigned[name] = value
            original_setattr(self, name, value)

        mock_qristal = MagicMock()
        mock_qristal.session.return_value = mock_session

        with patch.dict("sys.modules", {"qristal": MagicMock(), "qristal.core": mock_qristal}):
            with patch.object(type(mock_session), "__setattr__", tracking_setattr):
                executor = SimulationExecutor(config)
                circuit = Circuit().h(0)
                result = await executor.execute(circuit, shots=100)

        assert assigned.get("get_state_vec") is True
        assert "state_vector" in result.metadata
        assert result.metadata["state_vector"] == [complex(1, 0), complex(0, 0)]

    @pytest.mark.asyncio
    async def test_state_vector_with_noise_raises(self) -> None:
        """State vector + noise model raises ValueError."""
        noise = NoiseModel()
        noise.add(Depolarizing(0.01), qubits=[0])
        config = SimulationConfig(
            backend_id="qpp", backend_type="statevector",
            noise_model=noise, extract_state_vector=True,
        )

        mock_qristal = MagicMock()
        mock_session = MagicMock()
        mock_qristal.session.return_value = mock_session

        with patch.dict("sys.modules", {"qristal": MagicMock(), "qristal.core": mock_qristal}):
            executor = SimulationExecutor(config)
            circuit = Circuit().h(0)
            with pytest.raises(ValueError, match="incompatible with noise"):
                await executor.execute(circuit, shots=100)

    @pytest.mark.asyncio
    async def test_state_vector_unsupported_backend_raises(self) -> None:
        """State vector on non-qpp backend raises ValueError."""
        config = SimulationConfig(
            backend_id="aer", backend_type="noisy",
            extract_state_vector=True,
        )

        mock_qristal = MagicMock()
        mock_session = MagicMock()
        mock_qristal.session.return_value = mock_session

        with patch.dict("sys.modules", {"qristal": MagicMock(), "qristal.core": mock_qristal}):
            executor = SimulationExecutor(config)
            circuit = Circuit().h(0)
            with pytest.raises(ValueError, match="not supported"):
                await executor.execute(circuit, shots=100)

    @pytest.mark.asyncio
    async def test_state_vector_not_requested_omitted(self) -> None:
        """Default config does not include state_vector in metadata."""
        config = SimulationConfig(backend_id="qpp", backend_type="statevector")

        mock_session = MagicMock()
        mock_session.results = [[{(False,): 1000}]]

        mock_qristal = MagicMock()
        mock_qristal.session.return_value = mock_session

        with patch.dict("sys.modules", {"qristal": MagicMock(), "qristal.core": mock_qristal}):
            executor = SimulationExecutor(config)
            circuit = Circuit().h(0)
            result = await executor.execute(circuit, shots=100)

        assert "state_vector" not in result.metadata
