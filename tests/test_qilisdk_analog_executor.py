"""Tests for QiliSDKExecutor.execute_analog (Qilimanjaro analog/annealing mode)."""

import pytest

qilisdk = pytest.importorskip("qilisdk")

from qilisdk.analog import Hamiltonian, PauliX, PauliZ, Schedule
from qilisdk.core.qtensor import InitialState

from marqov.executors import ExecutionResult, QiliSDKExecutor


class TestQiliSDKExecutorAnalog:
    """Tests for QiliSDKExecutor.execute_analog."""

    @pytest.mark.asyncio
    async def test_execute_analog_returns_result(self) -> None:
        """execute_analog returns an ExecutionResult tagged as analog."""
        executor = QiliSDKExecutor()
        driver = Hamiltonian({(PauliX(0),): 1.0})
        problem = Hamiltonian({(PauliZ(0),): 1.0})
        schedule = Schedule.linear(driver, problem, total_time=2.0, dt=0.1)

        result = await executor.execute_analog(schedule, shots=100)

        assert isinstance(result, ExecutionResult)
        assert result.backend == "qilisdk-qilisim"
        assert result.shots == 100
        assert result.metadata == {"simulator": "qilisim", "mode": "analog"}

    @pytest.mark.asyncio
    async def test_execute_analog_constant_schedule_is_deterministic(self) -> None:
        """A diagonal Hamiltonian on a computational basis state never flips it."""
        executor = QiliSDKExecutor()
        problem = Hamiltonian({(PauliZ(0), PauliZ(1)): 1.0})
        schedule = Schedule.constant(problem, total_time=3.0, dt=0.1)

        result = await executor.execute_analog(
            schedule, shots=200, initial_state=InitialState.ZERO
        )

        assert result.counts == {"00": 200}

    @pytest.mark.asyncio
    async def test_execute_analog_anneal_biases_toward_ground_state(self) -> None:
        """A transverse-field-driver -> ZZ-coupling anneal favors 00/11 over 01/10.

        Physical sanity check: annealing from a driver Hamiltonian (favors
        superposition) to a ferromagnetic ZZ problem Hamiltonian (favors
        aligned spins) should concentrate probability on the ground-state
        manifold {00, 11}, not the excited states {01, 10}.
        """
        executor = QiliSDKExecutor()
        driver = Hamiltonian({(PauliX(0),): 1.0, (PauliX(1),): 1.0})
        problem = Hamiltonian({(PauliZ(0), PauliZ(1)): 1.0})
        schedule = Schedule.linear(driver, problem, total_time=5.0, dt=0.05)

        result = await executor.execute_analog(
            schedule, shots=500, initial_state=InitialState.UNIFORM
        )

        ground_state_counts = result.counts.get("00", 0) + result.counts.get("11", 0)
        excited_state_counts = result.counts.get("01", 0) + result.counts.get("10", 0)
        assert ground_state_counts > excited_state_counts

    @pytest.mark.asyncio
    async def test_execute_analog_default_initial_state_is_uniform(self) -> None:
        """Omitting initial_state doesn't error and defaults to UNIFORM."""
        executor = QiliSDKExecutor()
        driver = Hamiltonian({(PauliX(0),): 1.0})
        problem = Hamiltonian({(PauliZ(0),): 1.0})
        schedule = Schedule.linear(driver, problem, total_time=1.0, dt=0.1)

        result = await executor.execute_analog(schedule, shots=50)

        assert sum(result.counts.values()) == 50

    @pytest.mark.asyncio
    async def test_execute_analog_qutip_backend(self) -> None:
        """The qutip reference simulator runs analog evolution too."""
        pytest.importorskip("qutip")
        from marqov.executors.qilisdk import QiliSDKExecutorConfig

        executor = QiliSDKExecutor(QiliSDKExecutorConfig(simulator="qutip"))
        problem = Hamiltonian({(PauliZ(0), PauliZ(1)): 1.0})
        schedule = Schedule.constant(problem, total_time=3.0, dt=0.1)

        result = await executor.execute_analog(
            schedule, shots=100, initial_state=InitialState.ZERO
        )

        assert result.backend == "qilisdk-qutip"
        assert result.counts == {"00": 100}
