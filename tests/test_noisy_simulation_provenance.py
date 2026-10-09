"""Validate actual noisy-backend limits and result provenance without native jobs."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from marqov import Circuit
from marqov.simulation.config import SimulationConfig
from marqov.simulation.executor import SimulationExecutor, _validate_qubit_limit
from marqov.simulation.noise import NoiseModel


class NativeSession:
    """Small vendor-shaped session; no simulator or external service involved."""

    def __init__(self, width, shots):
        self.results = {tuple(False for _ in range(width)): shots}
        self.acc_history = []
        self.run_calls = 0

    @property
    def acc(self):
        return self.acc_history[-1]

    @acc.setter
    def acc(self, value):
        self.acc_history.append(value)

    def run(self):
        self.run_calls += 1


class NativeNoiseModel:
    def add_gate_error(self, channel, gate, qubits):
        pass


def install_native_fake(monkeypatch, width, shots):
    session = NativeSession(width, shots)
    core = SimpleNamespace(
        session=lambda: session,
        NoiseModel=NativeNoiseModel,
        DepolarizingChannel=SimpleNamespace(Create=lambda qubit, probability: (qubit, probability)),
    )
    import_core = Mock(return_value=core)
    monkeypatch.setattr("marqov.simulation.executor._import_qristal_core", import_core)
    return session, import_core


@pytest.mark.asyncio
@pytest.mark.parametrize("target,mode", [
    ("qpp", "statevector"), ("tnqvm", "tensor-network"), ("aer", "noisy"),
])
async def test_noise_reports_requested_and_actual_backend(monkeypatch, target, mode):
    session, _ = install_native_fake(monkeypatch, 2, 100)
    noise = NoiseModel.depolarizing_uniform(p=0.01, num_qubits=2)
    config = SimulationConfig(backend_id=target, backend_type=mode, noise_model=noise, seed=7)
    result = await SimulationExecutor(config).execute(Circuit().h(0).cnot(0, 1), shots=100)

    assert result.backend == "qb-sim-noisy-aer"
    assert result.metadata["simulator"] == result.metadata["engine"] == "aer"
    assert result.metadata["requested_simulator"] == target
    assert session.acc_history == ["aer"]
    assert session.run_calls == 1
    assert session.noise is True
    assert session.seed == 7
    assert result.counts == {"00": 100}
    assert config.backend_id == target
    assert config.backend_type == mode
    assert config.num_qubits == 0
    assert config.noise_model is noise


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [29, 40])
async def test_noise_rejects_actual_aer_limit_before_native_import(monkeypatch, width):
    session, import_core = install_native_fake(monkeypatch, width, 100)
    config = SimulationConfig(
        backend_id="tnqvm", backend_type="tensor-network",
        noise_model=NoiseModel.depolarizing_uniform(p=0.01, num_qubits=width),
    )
    circuit = Circuit()
    for qubit in range(width):
        circuit.h(qubit)
    with pytest.raises(ValueError, match=f"Backend aer supports up to 28 qubits, got {width}"):
        await SimulationExecutor(config).execute(circuit, shots=100)
    import_core.assert_not_called()
    assert session.run_calls == 0


def test_noisy_limit_helper_uses_actual_backend():
    config = SimulationConfig(
        backend_id="tnqvm", backend_type="tensor-network", num_qubits=40,
        noise_model=NoiseModel(),
    )
    with pytest.raises(ValueError, match="Backend aer supports up to 28 qubits, got 40"):
        _validate_qubit_limit(config)


@pytest.mark.asyncio
@pytest.mark.parametrize("target,mode", [("qpp", "statevector"), ("tnqvm", "tensor-network")])
async def test_noiseless_execution_preserves_backend(monkeypatch, target, mode):
    session, _ = install_native_fake(monkeypatch, 2, 100)
    config = SimulationConfig(backend_id=target, backend_type=mode)
    result = await SimulationExecutor(config).execute(Circuit().h(0).cnot(0, 1), shots=100)
    assert result.backend == f"qb-sim-{mode}"
    assert result.metadata["simulator"] == result.metadata["engine"] == target
    assert result.metadata["requested_simulator"] == target
    assert session.acc_history == [target]
    assert not hasattr(session, "noise")


@pytest.mark.asyncio
@pytest.mark.parametrize("noisy,width,actual", [(True, 28, "aer"), (False, 40, "tnqvm")])
async def test_actual_backend_boundary_preserves_allowed_requests(monkeypatch, noisy, width, actual):
    session, _ = install_native_fake(monkeypatch, width, 100)
    config = SimulationConfig(
        backend_id="tnqvm", backend_type="tensor-network",
        noise_model=NoiseModel() if noisy else None,
    )
    circuit = Circuit()
    for qubit in range(width):
        circuit.h(qubit)
    result = await SimulationExecutor(config).execute(circuit, shots=100)
    assert session.acc_history == [actual]
    assert session.qn == width
    assert result.counts == {"0" * width: 100}
