"""Tests for marqov.executors.qilisdk (Qilimanjaro qilisdk local simulators)."""

import pytest

qilisdk = pytest.importorskip("qilisdk")

from marqov.circuits import Circuit, bell_state, ghz_state
from marqov.executors import ExecutionResult, ExecutorFactory, QiliSDKExecutor
from marqov.executors.qilisdk import QiliSDKExecutorConfig


class TestQiliSDKExecutor:
    """Tests for QiliSDKExecutor."""

    @pytest.mark.asyncio
    async def test_execute_returns_result(self) -> None:
        """Execute returns an ExecutionResult."""
        executor = QiliSDKExecutor()
        circuit = Circuit().h(0)
        result = await executor.execute(circuit, shots=100)

        assert isinstance(result, ExecutionResult)
        assert result.backend == "qilisdk-qilisim"
        assert result.shots == 100

    @pytest.mark.asyncio
    async def test_execute_bell_state(self) -> None:
        """Bell state produces expected measurement distribution."""
        executor = QiliSDKExecutor()
        circuit = bell_state()
        result = await executor.execute(circuit, shots=1000)

        assert set(result.counts.keys()).issubset({"00", "11"})
        assert sum(result.counts.values()) == 1000
        for count in result.counts.values():
            assert 400 < count < 600

    @pytest.mark.asyncio
    async def test_execute_ghz_state(self) -> None:
        """GHZ state on 3 qubits only produces all-0s or all-1s."""
        executor = QiliSDKExecutor()
        circuit = ghz_state(3)
        result = await executor.execute(circuit, shots=500)

        assert set(result.counts.keys()).issubset({"000", "111"})
        assert sum(result.counts.values()) == 500

    @pytest.mark.asyncio
    async def test_execute_all_supported_gates(self) -> None:
        """A circuit touching every canonical gate runs without error."""
        executor = QiliSDKExecutor()
        circuit = (
            Circuit()
            .h(0)
            .x(1)
            .y(0)
            .z(1)
            .s(0)
            .t(1)
            .rx(0.3, 0)
            .ry(0.6, 1)
            .rz(0.9, 0)
            .cnot(0, 1)
            .cz(0, 1)
            .swap(0, 1)
        )
        result = await executor.execute(circuit, shots=50)

        assert sum(result.counts.values()) == 50

    @pytest.mark.asyncio
    async def test_execute_metadata(self) -> None:
        """Execution includes simulator metadata."""
        executor = QiliSDKExecutor()
        circuit = Circuit().x(0)
        result = await executor.execute(circuit, shots=10)

        assert result.metadata["simulator"] == "qilisim"
        assert result.metadata["vendor"] == "Qilimanjaro"
        assert result.metadata["framework"] == "QiliSDK"
        assert result.metadata["access_path"] == "local"
        assert result.metadata["reproducibility"]["shots"] == 10

    @pytest.mark.asyncio
    async def test_execute_qutip_backend(self) -> None:
        """The qutip reference simulator produces the same-shaped result."""
        pytest.importorskip("qutip")

        executor = QiliSDKExecutor(QiliSDKExecutorConfig(simulator="qutip"))
        circuit = bell_state()
        result = await executor.execute(circuit, shots=200)

        assert result.backend == "qilisdk-qutip"
        assert set(result.counts.keys()).issubset({"00", "11"})
        assert sum(result.counts.values()) == 200

    @pytest.mark.asyncio
    @pytest.mark.parametrize("simulator", ["qilisim", "qutip"])
    async def test_provenance_matches_actual_engine_and_counts(self, simulator: str) -> None:
        import hashlib
        from importlib.metadata import version
        import json

        executor = QiliSDKExecutor(QiliSDKExecutorConfig(
            simulator=simulator, compute_provider="isolated-test-runtime"
        ))
        result = await executor.execute(Circuit().x(0).cz(0, 1), shots=32)
        assert result.counts == {"10": 32}
        assert result.metadata["compute_provider"] == "isolated-test-runtime"
        record = result.metadata["reproducibility"]
        assert record["packages"]["qilisdk"] == version("qilisdk")
        assert result.metadata["engine"] == (
            f"{type(executor._backend).__module__}.{type(executor._backend).__qualname__}"
        )
        assert record["counts_sha256"] == hashlib.sha256(
            json.dumps(result.counts, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        assert record["seed"] is None
        assert "no hardware execution" in record["scope"]

    def test_unsupported_simulator_raises(self) -> None:
        """An unknown simulator name raises immediately, not a silent fallback."""
        with pytest.raises(ValueError, match="Unknown qilisdk simulator"):
            QiliSDKExecutor(QiliSDKExecutorConfig(simulator="not-a-real-backend"))  # type: ignore[arg-type]


class TestQiliSDKExecutorFactory:
    """Tests for creating QiliSDKExecutor via ExecutorFactory."""

    def test_create_qilisdk_executor_default(self) -> None:
        """Factory creates a QiliSim-backed executor by default."""
        executor = ExecutorFactory.create_executor(
            "qilisdk-qilisim", {"provider": "Qilimanjaro"}
        )
        assert isinstance(executor, QiliSDKExecutor)
        assert executor.config.simulator == "qilisim"

    def test_create_qilisdk_executor_qutip(self) -> None:
        """Factory honors an explicit qutip simulator selection."""
        pytest.importorskip("qutip")

        executor = ExecutorFactory.create_executor(
            "qilisdk-qutip", {"provider": "Qilimanjaro", "simulator": "qutip"}
        )
        assert isinstance(executor, QiliSDKExecutor)
        assert executor.config.simulator == "qutip"

    def test_qilimanjaro_in_supported_providers(self) -> None:
        """Qilimanjaro appears in the supported-providers list."""
        assert "Qilimanjaro" in ExecutorFactory.get_supported_providers()
        assert ExecutorFactory.is_provider_supported("Qilimanjaro")
