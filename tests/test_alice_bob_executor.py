"""Tests for marqov.executors.alice_bob (Alice & Bob local cat-qubit emulators)."""

import pytest

pytest.importorskip("qiskit_alice_bob_provider")

from marqov.circuits import Circuit, bell_state
from marqov.executors import AliceBobExecutor, ExecutionResult, ExecutorFactory
from marqov.executors.alice_bob import (
    AliceBobExecutorConfig,
    UnsupportedCircuitError,
)


class TestAliceBobExecutor:
    """Tests for AliceBobExecutor."""

    @pytest.mark.asyncio
    async def test_execute_bell_state(self) -> None:
        """Bell state on the default (logical target) backend produces the expected distribution."""
        executor = AliceBobExecutor()
        result = await executor.execute(bell_state(), shots=1000)

        assert isinstance(result, ExecutionResult)
        assert set(result.counts.keys()).issubset({"00", "11"})
        assert sum(result.counts.values()) == 1000
        for count in result.counts.values():
            assert 400 < count < 600

    @pytest.mark.asyncio
    async def test_metadata_carries_required_attribution_fields(self) -> None:
        """metadata always identifies this as simulated, names the backend, and pins the provider version."""
        executor = AliceBobExecutor()
        result = await executor.execute(bell_state(), shots=100)

        assert result.metadata["simulated"] is True
        assert result.metadata["backend_name"] == "EMU:40Q:LOGICAL_TARGET"
        assert result.metadata["provider_package_version"]

    @pytest.mark.asyncio
    async def test_metadata_carries_resolved_noise_parameters(self) -> None:
        """metadata reports the noise-model params actually used, including provider defaults not explicitly passed."""
        executor = AliceBobExecutor()  # no config override — provider's own defaults apply
        result = await executor.execute(bell_state(), shots=10)

        # EMU:40Q:LOGICAL_TARGET's documented provider defaults (distance=15, kappa_1=100,
        # kappa_2=10_000_000, average_nb_photons=19) must show up even though we never set them.
        assert result.metadata["distance"] == 15
        assert result.metadata["kappa_1"] == 100
        assert result.metadata["kappa_2"] == 10_000_000
        assert result.metadata["average_nb_photons"] == 19

    @pytest.mark.asyncio
    async def test_oversized_circuit_raises_typed_error_not_silent_wrong_answer(self) -> None:
        """A circuit too large for the backend raises UnsupportedCircuitError, not a silently wrong result.

        Regression test for a confirmed provider gap: EMU:1Q:LESCANNE_2020 accepts
        a 2-qubit circuit with unsupported gates and silently returns counts for it
        instead of raising. The executor must catch this itself.
        """
        executor = AliceBobExecutor(
            AliceBobExecutorConfig(backend_name="EMU:1Q:LESCANNE_2020")
        )
        two_qubit_circuit = Circuit().h(0).cnot(0, 1)

        with pytest.raises(UnsupportedCircuitError):
            await executor.execute(two_qubit_circuit, shots=100)

    @pytest.mark.asyncio
    async def test_seeded_execution_is_deterministic(self) -> None:
        """Passing seed_simulator produces byte-identical counts across repeated runs."""
        executor = AliceBobExecutor(
            AliceBobExecutorConfig(backend_name="EMU:6Q:PHYSICAL_CATS")
        )
        circuit = bell_state()

        result_a = await executor.execute(circuit, shots=2000, seed_simulator=42)
        result_b = await executor.execute(circuit, shots=2000, seed_simulator=42)

        assert result_a.counts == result_b.counts

    def test_tier_inapplicable_config_kwarg_raises_loudly(self) -> None:
        """Passing a logical-only kwarg (distance) to a physical backend fails at construction, not silently."""
        config = AliceBobExecutorConfig(
            backend_name="EMU:6Q:PHYSICAL_CATS",
            distance=15,
        )

        with pytest.raises(TypeError):
            AliceBobExecutor(config)

    def test_registered_in_executor_factory(self) -> None:
        """The executor is reachable through ExecutorFactory, matching every other provider."""
        assert "Alice & Bob" in ExecutorFactory.get_supported_providers()
