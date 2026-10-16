"""Exercise both real local executor paths with vendor imports stubbed.

The blocking fake stands in for QiliSim/QutipBackend.execute; these tests run
without installing QiliSDK and never submit a provider job.
"""

import asyncio
from dataclasses import dataclass
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import pytest

from marqov.circuits import Circuit
from marqov.executors.qilisdk import QiliSDKExecutor, QiliSDKExecutorConfig


@pytest.fixture
def local_executor(monkeypatch):
    started = threading.Event()
    finished = threading.Event()
    backends = []
    samples = {"0": 7}
    raw_result = SimpleNamespace(get_samples=lambda: samples)

    @dataclass
    class ExecutionConfig:
        seed: int
        num_threads: int

    class Backend:
        def __init__(self, execution_config=None):
            self.config = execution_config
            self.failure = None
            self.thread_id = None
            backends.append(self)

        def execute(self, functional, readout):
            self.thread_id = threading.get_ident()
            self.functional = functional
            self.readout = readout
            started.set()
            try:
                time.sleep(0.15)
                if self.failure is not None:
                    raise self.failure
                return raw_result
            finally:
                finished.set()

    class Readout:
        def with_sampling(self, nshots):
            self.nshots = nshots
            return self

    for name, attributes in {
        "qilisdk": {},
        "qilisdk.backends": {"QiliSim": Backend, "ExecutionConfig": ExecutionConfig},
        "qilisdk.functionals": {
            "DigitalPropagation": lambda **kw: SimpleNamespace(**kw),
            "AnalogEvolution": lambda **kw: SimpleNamespace(**kw),
        },
        "qilisdk.readout": {"Readout": Readout},
        "qilisdk.core": {},
        "qilisdk.core.qtensor": {"InitialState": SimpleNamespace(UNIFORM="uniform")},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setattr("marqov.executors.qilisdk.version", lambda name: "fixture-version")
    monkeypatch.setattr(QiliSDKExecutor, "_to_qilisdk_circuit", staticmethod(lambda c: c))
    executor = QiliSDKExecutor(QiliSDKExecutorConfig())
    return SimpleNamespace(executor=executor, started=started, finished=finished,
                           backends=backends, raw_result=raw_result, samples=samples)


def run_program(fixture, mode, *, seed=None):
    if mode == "digital":
        return fixture.executor.execute(Circuit().h(0), shots=7, seed=seed)
    return fixture.executor.execute_analog("schedule", shots=7, seed=seed)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["digital", "analog"])
@pytest.mark.parametrize("seed", [None, 42])
async def test_blocking_backend_leaves_event_loop_live(local_executor, mode, seed):
    ticks = 0

    async def ticker():
        nonlocal ticks
        while not local_executor.finished.is_set():
            if local_executor.started.is_set():
                ticks += 1
            await asyncio.sleep(0.005)

    heartbeat = asyncio.create_task(ticker())
    try:
        result = await run_program(local_executor, mode, seed=seed)
    finally:
        await heartbeat

    assert ticks >= 5, f"only {ticks} ticks while the simulator blocked"
    assert result.execution_time_ms >= 150
    assert result.counts == local_executor.samples
    assert result.raw_result is local_executor.raw_result
    assert result.shots == 7
    assert result.backend == "qilisdk-qilisim"
    backend = local_executor.backends[-1]
    assert backend.thread_id != threading.get_ident()
    assert backend.readout.nshots == 7
    assert result.metadata["reproducibility"]["seed"] == seed
    assert local_executor.executor._backend is local_executor.backends[0]
    if seed is not None:
        assert backend.config.seed == seed
        assert backend.config.num_threads == 1
        assert result.metadata["seed"] == seed
        assert result.metadata["num_threads"] == 1
    if mode == "analog":
        assert backend.functional.schedule == "schedule"
        assert backend.functional.initial_state == "uniform"
        assert result.metadata["mode"] == "analog"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["digital", "analog"])
async def test_backend_error_propagates(local_executor, mode):
    failure = RuntimeError("simulator failed")
    local_executor.executor._backend.failure = failure
    with pytest.raises(RuntimeError) as error:
        await run_program(local_executor, mode)
    assert error.value is failure


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["digital", "analog"])
async def test_cancellation_stops_waiting_but_not_simulator(local_executor, mode):
    task = asyncio.create_task(run_program(local_executor, mode))
    try:
        async with asyncio.timeout(2):
            while not local_executor.started.is_set():
                await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not local_executor.finished.is_set()
    finally:
        # Let the finite fake finish before teardown restores its module imports.
        assert await asyncio.to_thread(local_executor.finished.wait, 2)
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["digital", "analog"])
@pytest.mark.parametrize(
    "bad",
    [{"0": 6}, {"0": 8}, {"0": 3, "2": 4}, {"0": 4, "1": -1, "": 4}, {"0": 3.0, "1": 4}],
)
async def test_miscounted_or_malformed_samples_raise(local_executor, mode, bad):
    local_executor.samples.clear()
    local_executor.samples.update(bad)
    with pytest.raises(RuntimeError, match="invalid counts or shot accounting"):
        await run_program(local_executor, mode)


@pytest.mark.asyncio
async def test_digital_key_width_must_match_circuit(local_executor):
    local_executor.samples.clear()
    local_executor.samples.update({"00": 7})
    with pytest.raises(RuntimeError, match="invalid counts or shot accounting"):
        await run_program(local_executor, "digital")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["digital", "analog"])
async def test_provenance_shots_is_observed_total_and_bit_order(local_executor, mode):
    local_executor.samples.clear()
    local_executor.samples.update({"0": 3, "1": 4})
    result = await run_program(local_executor, mode)
    record = result.metadata["reproducibility"]
    assert record["shots"] == sum(result.counts.values()) == result.shots == 7
    assert record["bit_order"] == "QiliSDK sampling order; qubit 0 leftmost"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["digital", "analog"])
async def test_numpy_integer_counts_are_accepted_as_plain_ints(local_executor, mode):
    import numpy as np

    local_executor.samples.clear()
    local_executor.samples.update({"0": np.int64(3), "1": np.int64(4)})
    result = await run_program(local_executor, mode)
    assert result.counts == {"0": 3, "1": 4}
    assert all(type(v) is int for v in result.counts.values())
    assert result.metadata["reproducibility"]["shots"] == 7


@pytest.mark.asyncio
async def test_bool_counts_are_refused(local_executor):
    local_executor.samples.clear()
    local_executor.samples.update({"0": True, "1": 6})
    with pytest.raises(RuntimeError, match="invalid counts or shot accounting"):
        await run_program(local_executor, "digital")
