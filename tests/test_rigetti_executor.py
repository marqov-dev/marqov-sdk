"""Unit tests for RigettiExecutor and Rigetti factory wiring."""

from __future__ import annotations

import types
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from marqov.circuits import bell_state
from marqov.executors.factory import ExecutorFactory
from marqov.executors.rigetti import RigettiExecutor, RigettiExecutorConfig


@pytest.fixture
def fake_pyquil_modules() -> dict[str, object]:
    """Provide minimal pyquil modules required by RigettiExecutor internals."""
    pyquil_module = types.ModuleType("pyquil")
    pyquil_module.get_qc = MagicMock(name="get_qc")

    gates_module = types.ModuleType("pyquil.gates")
    gates_module.MEASURE = MagicMock(name="MEASURE")

    class Program:
        def __init__(self, _src: str = "") -> None:
            self._wrapped_shots = 0

        def out(self) -> str:
            return ""

        def declare(self, _name: str, _kind: str, size: int):
            return list(range(size))

        def __iadd__(self, _other):
            return self

        def wrap_in_numshots_loop(self, shots: int) -> None:
            self._wrapped_shots = shots

    quil_module = types.ModuleType("pyquil.quil")
    quil_module.Program = Program

    return {
        "pyquil": pyquil_module,
        "pyquil.gates": gates_module,
        "pyquil.quil": quil_module,
    }


class TestRigettiExecutorConfig:
    def test_defaults(self) -> None:
        config = RigettiExecutorConfig()
        assert config.quantum_processor_id == "2q-qvm"
        assert config.poll_interval_seconds == 0.2
        assert config.timeout_seconds == 120.0


class TestRigettiExecutor:
    @pytest.mark.asyncio
    async def test_execute_normalizes_numpy_result(self, fake_pyquil_modules: dict[str, object]) -> None:
        with patch.dict("sys.modules", fake_pyquil_modules):
            executor = RigettiExecutor(RigettiExecutorConfig(quantum_processor_id="2q-qvm", as_qvm=True))

        mock_qc = MagicMock()
        mock_qc.compile.return_value = "compiled"
        mock_qc.run.return_value = np.array([[0, 0], [1, 1], [0, 0], [1, 1]], dtype=int)

        with patch.object(executor, "_get_quantum_computer", new=AsyncMock(return_value=mock_qc)):
            with patch.object(RigettiExecutor, "_prepare_program", return_value="prepared"):
                result = await executor.execute(bell_state(), shots=4)

        assert result.counts == {"00": 2, "11": 2}
        assert result.shots == 4
        assert result.backend == "rigetti:2q-qvm"
        assert "job_id" in result.metadata

    @pytest.mark.asyncio
    async def test_execute_raises_on_unexpected_result_shape(self, fake_pyquil_modules: dict[str, object]) -> None:
        with patch.dict("sys.modules", fake_pyquil_modules):
            executor = RigettiExecutor(RigettiExecutorConfig(quantum_processor_id="2q-qvm", as_qvm=True))

        mock_qc = MagicMock()
        mock_qc.compile.return_value = "compiled"
        mock_qc.run.return_value = "bad-result"

        with patch.object(executor, "_get_quantum_computer", new=AsyncMock(return_value=mock_qc)):
            with patch.object(RigettiExecutor, "_prepare_program", return_value="prepared"):
                with pytest.raises(RuntimeError, match="Unsupported Rigetti result format"):
                    await executor.execute(bell_state(), shots=10)

    @pytest.mark.asyncio
    async def test_get_status_qvm_online(self, fake_pyquil_modules: dict[str, object]) -> None:
        with patch.dict("sys.modules", fake_pyquil_modules):
            executor = RigettiExecutor(RigettiExecutorConfig(quantum_processor_id="2q-qvm", as_qvm=True))

        with patch.object(RigettiExecutor, "_port_open", side_effect=[True, True]):
            status = await executor.get_status()

        assert status.status == "online"
        assert status.queue_depth == 0
        assert status.queue_time_seconds == 0

    @pytest.mark.asyncio
    async def test_get_status_qvm_offline(self, fake_pyquil_modules: dict[str, object]) -> None:
        with patch.dict("sys.modules", fake_pyquil_modules):
            executor = RigettiExecutor(RigettiExecutorConfig(quantum_processor_id="2q-qvm", as_qvm=True))

        with patch.object(RigettiExecutor, "_port_open", side_effect=[True, False]):
            status = await executor.get_status()

        assert status.status == "offline"
        assert status.queue_depth is None
        assert status.queue_time_seconds is None


class TestRigettiFactoryIntegration:
    def test_factory_creates_rigetti_executor(self, fake_pyquil_modules: dict[str, object]) -> None:
        with patch.dict("sys.modules", fake_pyquil_modules):
            executor = ExecutorFactory.create_executor(
                "2q-qvm",
                {
                    "provider": "Rigetti QCS",
                    "quantum_processor_id": "2q-qvm",
                    "as_qvm": True,
                },
            )

        assert isinstance(executor, RigettiExecutor)

    def test_rigetti_provider_is_supported(self) -> None:
        assert "Rigetti QCS" in ExecutorFactory.get_supported_providers()
