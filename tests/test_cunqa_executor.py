"""Tests for the CUNQA distributed-QC executor.

Uses an injected CunqaClient double so these tests never launch a real Slurm
job or need a cluster.
"""

import asyncio
import threading
import time
from typing import Any

import pytest

from marqov.circuits import bell_state
from marqov.executors import CUNQAExecutor, ExecutionResult
from marqov.executors.cunqa import CUNQAExecutorConfig

_FAST_POLL = {"startup_timeout_s": 5.0, "poll_interval_s": 0.01}


class _FakeCunqaResult:
    def __init__(self, counts: dict[str, int]) -> None:
        self.counts = counts


class _FakeCunqaClient:
    def __init__(self, counts_per_qpu: dict[str, int], *, ready_after_polls: int = 0) -> None:
        self._counts_per_qpu = counts_per_qpu
        self._ready_after_polls = ready_after_polls
        self._poll_count = 0
        self.qraise_calls: list[dict[str, Any]] = []
        self.get_qpus_calls: list[dict[str, Any]] = []
        self.run_calls: list[dict[str, Any]] = []
        self.submitted_circuits: list[Any] = []
        self.dropped: list[str] = []

    def qraise(self, n_qpus: int, walltime: str, **kwargs: Any) -> str:
        self.qraise_calls.append({"n_qpus": n_qpus, "walltime": walltime, **kwargs})
        # Real CUNQA echoes back the caller-supplied family (falling back to
        # the job id only when none is given) -- matched here since the
        # executor now always passes one explicitly.
        return kwargs.get("family") or "test-family"

    def get_QPUs(self, co_located: bool, family: str) -> list[str]:
        self.get_qpus_calls.append({"co_located": co_located, "family": family})
        self._poll_count += 1
        n_qpus = self.qraise_calls[-1]["n_qpus"]
        if self._poll_count <= self._ready_after_polls:
            return []
        return [f"qpu-{i}" for i in range(n_qpus)]

    def run(self, circuits: list[Any], qpus: list[Any], **run_args: Any) -> list[Any]:
        self.run_calls.append({"n_circuits": len(circuits), "n_qpus": len(qpus), **run_args})
        self.submitted_circuits = circuits
        return [f"job-{i}" for i in range(len(qpus))]

    def gather(self, jobs: list[Any]) -> list[_FakeCunqaResult]:
        return [_FakeCunqaResult(dict(self._counts_per_qpu)) for _ in jobs]

    def qdrop(self, family: str) -> None:
        self.dropped.append(family)


@pytest.mark.asyncio
async def test_execute_splits_shots_and_merges_counts_across_qpus() -> None:
    fake_client = _FakeCunqaClient(counts_per_qpu={"00": 128, "11": 122})
    config = CUNQAExecutorConfig(n_qpus=4, walltime="00:05:00", **_FAST_POLL)
    executor = CUNQAExecutor(config, client=fake_client)

    result = await executor.execute(bell_state(), shots=1000)

    assert isinstance(result, ExecutionResult)
    assert result.counts == {"00": 512, "11": 488}
    assert result.backend == "CUNQAExecutor"
    assert result.shots == 1000
    assert sum(result.counts.values()) == 1000
    assert result.metadata["per_qpu_counts"] == [{"00": 128, "11": 122}] * 4


@pytest.mark.asyncio
async def test_execute_submits_measured_circuits() -> None:
    fake_client = _FakeCunqaClient(counts_per_qpu={"00": 128, "11": 122})
    config = CUNQAExecutorConfig(n_qpus=4, walltime="00:05:00", **_FAST_POLL)
    executor = CUNQAExecutor(config, client=fake_client)

    await executor.execute(bell_state(), shots=1000)

    submitted = fake_client.submitted_circuits[0]
    assert submitted.num_clbits == submitted.num_qubits
    assert any(inst.operation.name == "measure" for inst in submitted.data)


@pytest.mark.asyncio
async def test_execute_passes_shots_per_qpu_not_total_to_run() -> None:
    fake_client = _FakeCunqaClient(counts_per_qpu={"00": 128, "11": 122})
    config = CUNQAExecutorConfig(n_qpus=4, walltime="00:05:00", **_FAST_POLL)
    executor = CUNQAExecutor(config, client=fake_client)

    await executor.execute(bell_state(), shots=1000)

    assert fake_client.run_calls[0]["shots"] == 250
    assert fake_client.run_calls[0]["n_circuits"] == 4


@pytest.mark.asyncio
async def test_execute_rejects_shots_not_evenly_divisible_by_n_qpus() -> None:
    fake_client = _FakeCunqaClient(counts_per_qpu={"0": 1})
    config = CUNQAExecutorConfig(n_qpus=3, walltime="00:05:00", **_FAST_POLL)
    executor = CUNQAExecutor(config, client=fake_client)

    with pytest.raises(ValueError, match="evenly divisible"):
        await executor.execute(bell_state(), shots=1000)

    assert fake_client.qraise_calls == []


@pytest.mark.asyncio
async def test_execute_forwards_config_to_qraise() -> None:
    fake_client = _FakeCunqaClient(counts_per_qpu={"00": 250})
    config = CUNQAExecutorConfig(
        n_qpus=2,
        walltime="00:07:00",
        simulator="Aer",
        co_located=False,
        classical_comm=True,
        mem_per_qpu_gb=8,
        **_FAST_POLL,
    )
    executor = CUNQAExecutor(config, client=fake_client)

    await executor.execute(bell_state(), shots=500)

    call = fake_client.qraise_calls[0]
    family = call.pop("family")
    assert family.startswith("marqov-")
    assert call == {
        "n_qpus": 2,
        "walltime": "00:07:00",
        "simulator": "Aer",
        "co_located": False,
        "classical_comm": True,
        "quantum_comm": False,
        "mem_per_qpu": 8,
    }


@pytest.mark.asyncio
async def test_execute_waits_for_qpus_to_register_before_running() -> None:
    fake_client = _FakeCunqaClient(counts_per_qpu={"00": 500}, ready_after_polls=2)
    config = CUNQAExecutorConfig(n_qpus=2, walltime="00:05:00", **_FAST_POLL)
    executor = CUNQAExecutor(config, client=fake_client)

    result = await executor.execute(bell_state(), shots=1000)

    assert fake_client.get_qpus_calls
    assert result.counts == {"00": 1000}


@pytest.mark.asyncio
async def test_execute_raises_timeout_if_qpus_never_register() -> None:
    fake_client = _FakeCunqaClient(counts_per_qpu={}, ready_after_polls=10**6)
    config = CUNQAExecutorConfig(n_qpus=2, walltime="00:05:00", startup_timeout_s=0.05, poll_interval_s=0.01)
    executor = CUNQAExecutor(config, client=fake_client)

    with pytest.raises(TimeoutError, match="registered"):
        await executor.execute(bell_state(), shots=1000)

    assert fake_client.dropped == [fake_client.qraise_calls[0]["family"]]


@pytest.mark.asyncio
async def test_execute_raises_shot_conservation_error_on_count_mismatch() -> None:
    fake_client = _FakeCunqaClient(counts_per_qpu={"00": 50, "11": 50})
    config = CUNQAExecutorConfig(n_qpus=2, walltime="00:05:00", **_FAST_POLL)
    executor = CUNQAExecutor(config, client=fake_client)

    with pytest.raises(RuntimeError, match="shot conservation"):
        await executor.execute(bell_state(), shots=1000)


@pytest.mark.asyncio
async def test_execute_always_drops_qpus_even_on_run_failure() -> None:
    class _RaisingClient(_FakeCunqaClient):
        def run(self, circuits: list[Any], qpus: list[Any], **run_args: Any) -> list[Any]:
            raise RuntimeError("simulated cluster failure")

    fake_client = _RaisingClient(counts_per_qpu={})
    config = CUNQAExecutorConfig(n_qpus=2, walltime="00:05:00", **_FAST_POLL)
    executor = CUNQAExecutor(config, client=fake_client)

    with pytest.raises(RuntimeError, match="simulated cluster failure"):
        await executor.execute(bell_state(), shots=1000)

    assert fake_client.dropped == [fake_client.qraise_calls[0]["family"]]


@pytest.mark.asyncio
async def test_execute_drops_qpus_on_cancellation() -> None:
    class _SlowRunClient(_FakeCunqaClient):
        def run(self, circuits: list[Any], qpus: list[Any], **run_args: Any) -> list[Any]:
            time.sleep(0.5)
            return super().run(circuits, qpus, **run_args)

    fake_client = _SlowRunClient(counts_per_qpu={"00": 500})
    config = CUNQAExecutorConfig(n_qpus=2, walltime="00:05:00", **_FAST_POLL)
    executor = CUNQAExecutor(config, client=fake_client)

    task = asyncio.create_task(executor.execute(bell_state(), shots=1000))
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert fake_client.dropped == [fake_client.qraise_calls[0]["family"]]


@pytest.mark.asyncio
async def test_execute_survives_repeated_cancellation_during_teardown() -> None:
    class _SlowRunAndDropClient(_FakeCunqaClient):
        def run(self, circuits: list[Any], qpus: list[Any], **run_args: Any) -> list[Any]:
            time.sleep(0.2)
            return super().run(circuits, qpus, **run_args)

        def qdrop(self, family: str) -> None:
            time.sleep(0.3)
            super().qdrop(family)

    fake_client = _SlowRunAndDropClient(counts_per_qpu={"00": 500})
    config = CUNQAExecutorConfig(n_qpus=2, walltime="00:05:00", **_FAST_POLL)
    executor = CUNQAExecutor(config, client=fake_client)

    task = asyncio.create_task(executor.execute(bell_state(), shots=1000))
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert fake_client.dropped == [fake_client.qraise_calls[0]["family"]]


@pytest.mark.asyncio
async def test_execute_qdrop_failure_does_not_mask_original_exception() -> None:
    class _DoubleFailingClient(_FakeCunqaClient):
        def run(self, circuits: list[Any], qpus: list[Any], **run_args: Any) -> list[Any]:
            raise RuntimeError("original failure: circuit rejected")

        def qdrop(self, family: str) -> None:
            raise RuntimeError("qdrop also failed: cluster unreachable")

    fake_client = _DoubleFailingClient(counts_per_qpu={})
    config = CUNQAExecutorConfig(n_qpus=2, walltime="00:05:00", **_FAST_POLL)
    executor = CUNQAExecutor(config, client=fake_client)

    with pytest.raises(RuntimeError, match="original failure: circuit rejected"):
        await executor.execute(bell_state(), shots=1000)


@pytest.mark.asyncio
async def test_execute_raises_timeout_if_qraise_itself_never_returns() -> None:
    """Real CUNQA's qraise() blocks internally (polling squeue until the job
    is RUNNING with every vQPU registered) with no timeout of its own —
    unlike this fake, which normally returns instantly. If the Slurm job
    never reaches RUNNING (a stuck node, insufficient capacity), that call
    can hang forever. This must time out via cfg.startup_timeout_s rather
    than hang the whole executor, and still attempt qdrop for cleanup using
    the family generated before the call (qraise's own return value is
    unusable here — it never returned one)."""

    release = threading.Event()

    class _HangingQraiseClient(_FakeCunqaClient):
        def qraise(self, n_qpus: int, walltime: str, **kwargs: Any) -> str:
            release.wait(5)
            return super().qraise(n_qpus, walltime, **kwargs)

    fake_client = _HangingQraiseClient(counts_per_qpu={})
    config = CUNQAExecutorConfig(n_qpus=2, walltime="00:05:00", startup_timeout_s=0.1, poll_interval_s=0.01)
    executor = CUNQAExecutor(config, client=fake_client)

    try:
        with pytest.raises(TimeoutError, match="qraise did not return"):
            await executor.execute(bell_state(), shots=1000)

        assert len(fake_client.dropped) == 1
        assert fake_client.dropped[0].startswith("marqov-")
    finally:
        release.set()


class _ShotTrackingClient(_FakeCunqaClient):
    """Derives counts from the shots ``run()`` was handed, so an extra vQPU
    shows up as extra shots instead of being hidden by a fixed counts dict."""

    def __init__(self, *, extra_qpus: int = 0) -> None:
        super().__init__(counts_per_qpu={})
        self._extra_qpus = extra_qpus
        self.registered: list[str] = []

    def get_QPUs(self, co_located: bool, family: str) -> list[str]:
        n_qpus = self.qraise_calls[-1]["n_qpus"]
        self.registered = [f"qpu-{i}" for i in range(n_qpus + self._extra_qpus)]
        return list(self.registered)

    def run(self, circuits: list[Any], qpus: list[Any], **run_args: Any) -> list[Any]:
        super().run(circuits, qpus, **run_args)
        return [run_args["shots"]] * len(circuits)

    def gather(self, jobs: list[Any]) -> list[_FakeCunqaResult]:
        return [_FakeCunqaResult({"00": shots}) for shots in jobs]


@pytest.mark.asyncio
async def test_execute_uses_only_requested_qpus_when_extra_register() -> None:
    fake_client = _ShotTrackingClient(extra_qpus=1)
    config = CUNQAExecutorConfig(n_qpus=4, walltime="00:05:00", **_FAST_POLL)
    executor = CUNQAExecutor(config, client=fake_client)

    result = await executor.execute(bell_state(), shots=1000)

    assert len(fake_client.registered) == 5
    assert fake_client.run_calls[0]["n_circuits"] == 4
    assert fake_client.run_calls[0]["n_qpus"] == 4
    assert sum(result.counts.values()) == 1000


@pytest.mark.parametrize("n_qpus", [0, -1])
def test_config_rejects_non_positive_n_qpus(n_qpus: int) -> None:
    with pytest.raises(ValueError, match=rf"n_qpus.*{n_qpus}"):
        CUNQAExecutorConfig(n_qpus=n_qpus)


@pytest.mark.asyncio
async def test_execution_time_excludes_teardown() -> None:
    drop_s = 0.3

    class _SlowDropClient(_FakeCunqaClient):
        def qdrop(self, family: str) -> None:
            time.sleep(drop_s)
            super().qdrop(family)

    fake_client = _SlowDropClient(counts_per_qpu={"00": 500})
    config = CUNQAExecutorConfig(n_qpus=2, walltime="00:05:00", **_FAST_POLL)
    executor = CUNQAExecutor(config, client=fake_client)

    start = time.monotonic()
    result = await executor.execute(bell_state(), shots=1000)
    wall_ms = (time.monotonic() - start) * 1000

    assert result.execution_time_ms <= wall_ms - drop_s * 1000


@pytest.mark.asyncio
async def test_startup_timeout_is_one_budget_for_qraise_and_registration() -> None:
    class _SlowQraiseNeverRegisters(_FakeCunqaClient):
        def qraise(self, n_qpus: int, walltime: str, **kwargs: Any) -> str:
            time.sleep(0.4)
            return super().qraise(n_qpus, walltime, **kwargs)

    fake_client = _SlowQraiseNeverRegisters(counts_per_qpu={}, ready_after_polls=10**6)
    config = CUNQAExecutorConfig(n_qpus=1, walltime="00:05:00", startup_timeout_s=0.5, poll_interval_s=0.01)
    executor = CUNQAExecutor(config, client=fake_client)

    start = time.monotonic()
    with pytest.raises(TimeoutError, match="registered"):
        await executor.execute(bell_state(), shots=1000)
    elapsed = time.monotonic() - start

    assert elapsed < 0.5 + 0.2


@pytest.mark.asyncio
async def test_timeout_error_raised_by_qraise_itself_keeps_its_message() -> None:
    class _QraiseTimesOut(_FakeCunqaClient):
        def qraise(self, n_qpus: int, walltime: str, **kwargs: Any) -> str:
            super().qraise(n_qpus, walltime, **kwargs)
            raise TimeoutError("sbatch submission timed out talking to the controller")

    fake_client = _QraiseTimesOut(counts_per_qpu={})
    config = CUNQAExecutorConfig(n_qpus=2, walltime="00:05:00", **_FAST_POLL)
    executor = CUNQAExecutor(config, client=fake_client)

    with pytest.raises(TimeoutError, match="sbatch submission timed out") as excinfo:
        await executor.execute(bell_state(), shots=1000)

    assert "may still be running" not in str(excinfo.value)
    assert fake_client.dropped == [fake_client.qraise_calls[0]["family"]]


@pytest.mark.asyncio
async def test_late_qraise_completion_is_dropped_again() -> None:
    release = threading.Event()
    late_drop = threading.Event()

    class _LateQraiseClient(_FakeCunqaClient):
        def qraise(self, n_qpus: int, walltime: str, **kwargs: Any) -> str:
            release.wait(5)
            return super().qraise(n_qpus, walltime, **kwargs)

        def qdrop(self, family: str) -> None:
            super().qdrop(family)
            if release.is_set():
                late_drop.set()

    fake_client = _LateQraiseClient(counts_per_qpu={})
    config = CUNQAExecutorConfig(n_qpus=2, walltime="00:05:00", startup_timeout_s=0.1, poll_interval_s=0.01)
    executor = CUNQAExecutor(config, client=fake_client)

    try:
        with pytest.raises(TimeoutError, match="qraise did not return"):
            await executor.execute(bell_state(), shots=1000)
        assert len(fake_client.dropped) == 1
    finally:
        release.set()

    assert late_drop.wait(5)
    family = fake_client.qraise_calls[0]["family"]
    assert fake_client.dropped == [family, family]


def test_abandoned_qraise_does_not_block_asyncio_run() -> None:
    release = threading.Event()

    class _HangingQraiseClient(_FakeCunqaClient):
        def qraise(self, n_qpus: int, walltime: str, **kwargs: Any) -> str:
            release.wait(3)
            return super().qraise(n_qpus, walltime, **kwargs)

    fake_client = _HangingQraiseClient(counts_per_qpu={})
    config = CUNQAExecutorConfig(n_qpus=2, walltime="00:05:00", startup_timeout_s=0.1, poll_interval_s=0.01)
    executor = CUNQAExecutor(config, client=fake_client)

    async def run() -> None:
        with pytest.raises(TimeoutError, match="qraise did not return"):
            await executor.execute(bell_state(), shots=1000)

    start = time.monotonic()
    try:
        asyncio.run(run())
        elapsed = time.monotonic() - start
    finally:
        release.set()

    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_raised_messages_contain_no_em_dash() -> None:
    release = threading.Event()

    class _HangingQraiseClient(_FakeCunqaClient):
        def qraise(self, n_qpus: int, walltime: str, **kwargs: Any) -> str:
            release.wait(5)
            return super().qraise(n_qpus, walltime, **kwargs)

    divisibility = CUNQAExecutor(
        CUNQAExecutorConfig(n_qpus=3, walltime="00:05:00", **_FAST_POLL),
        client=_FakeCunqaClient(counts_per_qpu={}),
    )
    with pytest.raises(ValueError) as value_error:
        await divisibility.execute(bell_state(), shots=1000)

    hanging = CUNQAExecutor(
        CUNQAExecutorConfig(n_qpus=2, walltime="00:05:00", startup_timeout_s=0.1, poll_interval_s=0.01),
        client=_HangingQraiseClient(counts_per_qpu={}),
    )
    try:
        with pytest.raises(TimeoutError, match="qraise did not return") as timeout_error:
            await hanging.execute(bell_state(), shots=1000)
    finally:
        release.set()

    assert "\u2014" not in str(value_error.value)
    assert "\u2014" not in str(timeout_error.value)


@pytest.mark.parametrize("n_qpus", [1.5, True, "2"])
def test_config_rejects_non_int_n_qpus(n_qpus: Any) -> None:
    with pytest.raises(ValueError, match="n_qpus must be an int"):
        CUNQAExecutorConfig(n_qpus=n_qpus)


@pytest.mark.asyncio
async def test_execute_revalidates_n_qpus_mutated_after_construction() -> None:
    fake_client = _FakeCunqaClient(counts_per_qpu={"00": 500})
    config = CUNQAExecutorConfig(n_qpus=2, walltime="00:05:00", **_FAST_POLL)
    executor = CUNQAExecutor(config, client=fake_client)
    config.n_qpus = 0

    with pytest.raises(ValueError, match="n_qpus must be >= 1, got 0"):
        await executor.execute(bell_state(), shots=1000)

    assert fake_client.qraise_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("shots", [0, -4, 2.0, True])
async def test_execute_rejects_non_positive_or_non_int_shots(shots: Any) -> None:
    fake_client = _FakeCunqaClient(counts_per_qpu={"00": 1})
    config = CUNQAExecutorConfig(n_qpus=2, walltime="00:05:00", **_FAST_POLL)
    executor = CUNQAExecutor(config, client=fake_client)

    with pytest.raises(ValueError, match=rf"CUNQA shots must be a positive integer, got {shots!r}"):
        await executor.execute(bell_state(), shots=shots)

    assert fake_client.qraise_calls == []
    assert fake_client.run_calls == []


def test_config_defaults() -> None:
    config = CUNQAExecutorConfig()
    assert config.n_qpus == 4
    assert config.walltime == "00:10:00"
    assert config.simulator == "Aer"
    assert config.co_located is True
    assert config.classical_comm is False
    assert config.quantum_comm is False
    assert config.mem_per_qpu_gb == 4
    assert config.startup_timeout_s == 60.0
    assert config.poll_interval_s == 1.0
    assert config.seed == 0



@pytest.mark.asyncio
async def test_numpy_integer_shots_and_n_qpus_are_accepted() -> None:
    """NumPy integers are accepted for shots and n_qpus, as elsewhere in the SDK."""
    import numpy as np

    config = CUNQAExecutorConfig(n_qpus=np.int64(2), walltime="00:05:00", **_FAST_POLL)
    assert config.n_qpus == 2 and type(config.n_qpus) is int
    fake_client = _FakeCunqaClient(counts_per_qpu={"00": 30, "11": 20})
    executor = CUNQAExecutor(config, client=fake_client)

    result = await executor.execute(bell_state(), shots=np.int64(100))

    assert result.counts == {"00": 60, "11": 40}
    assert result.shots == 100 and type(result.shots) is int
    assert fake_client.run_calls[-1]["shots"] == 50


_TIMING_FIELDS = ("startup_timeout_s", "poll_interval_s")
_BAD_TIMING_VALUES = [0, 0.0, -1, -0.5, float("nan"), float("inf"), float("-inf"), True, False, "1", None]


@pytest.mark.parametrize("field", _TIMING_FIELDS)
@pytest.mark.parametrize("value", _BAD_TIMING_VALUES)
def test_config_rejects_bad_timing_value(field: str, value: Any) -> None:
    with pytest.raises(ValueError, match=rf"{field} must be"):
        CUNQAExecutorConfig(**{field: value})


@pytest.mark.asyncio
@pytest.mark.parametrize("field", _TIMING_FIELDS)
@pytest.mark.parametrize("value", _BAD_TIMING_VALUES)
async def test_execute_revalidates_timing_mutated_after_construction(field: str, value: Any) -> None:
    fake_client = _FakeCunqaClient(counts_per_qpu={"00": 500})
    config = CUNQAExecutorConfig(n_qpus=2, walltime="00:05:00", **_FAST_POLL)
    executor = CUNQAExecutor(config, client=fake_client)
    setattr(config, field, value)

    with pytest.raises(ValueError, match=rf"{field} must be"):
        await executor.execute(bell_state(), shots=1000)

    assert fake_client.qraise_calls == []
    assert fake_client.run_calls == []


def test_config_accepts_numpy_timing_values_and_normalises_to_float() -> None:
    import numpy as np

    config = CUNQAExecutorConfig(startup_timeout_s=np.float32(2.5), poll_interval_s=np.int64(1))
    assert config.startup_timeout_s == 2.5 and type(config.startup_timeout_s) is float
    assert config.poll_interval_s == 1.0 and type(config.poll_interval_s) is float
