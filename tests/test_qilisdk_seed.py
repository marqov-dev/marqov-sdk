"""Real QiliSim sampling: per-call seed replay and direct-backend agreement."""

import pytest

pytest.importorskip("qilisdk")

from qilisdk.analog import Hamiltonian, PauliZ, Schedule
from qilisdk.backends import ExecutionConfig, QiliSim
from qilisdk.core.qtensor import InitialState
from qilisdk.functionals import AnalogEvolution, DigitalPropagation
from qilisdk.readout import Readout

from marqov.circuits import Circuit
from marqov.executors.qilisdk import QiliSDKExecutor


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["digital", "analog"])
@pytest.mark.parametrize("seed", [0, 42, 2**31 - 1])
async def test_seeded_calls_replay_and_match_qilisim(mode, seed):
    executor = QiliSDKExecutor()
    original_backend = executor._backend
    shots = 1024
    if mode == "digital":
        program = Circuit().h(0).h(1).h(2).h(3)
        call = executor.execute
        functional = DigitalPropagation(executor._to_qilisdk_circuit(program))
    else:
        program = Schedule.constant(
            Hamiltonian({(PauliZ(0), PauliZ(1)): 1.0}), total_time=0.2, dt=0.1
        )
        call = executor.execute_analog
        functional = AnalogEvolution(schedule=program, initial_state=InitialState.UNIFORM)
    expected = (
        QiliSim(execution_config=ExecutionConfig(seed=seed, num_threads=1))
        .execute(functional, Readout().with_sampling(nshots=shots))
        .get_samples()
    )
    first = await call(program, shots=shots, seed=seed)
    # An intervening call must not advance or replace this call's random stream.
    await call(program, shots=37, seed=17)
    repeated = await call(program, shots=shots, seed=seed)
    assert first.counts == repeated.counts == expected
    assert len(first.counts) > 1  # Exercise sampling rather than a basis-state oracle.
    assert sum(first.counts.values()) == shots
    assert first.metadata["seed"] == seed
    assert first.metadata["num_threads"] == 1
    assert executor._backend is original_backend
    unseeded = await call(program, shots=31)
    assert sum(unseeded.counts.values()) == 31
    assert "seed" not in unseeded.metadata
    assert executor._backend is original_backend
