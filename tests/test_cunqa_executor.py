"""Tests for the CUNQA distributed-QC executor.

Uses an injected CunqaClient double so these tests never launch a real Slurm
job or need a cluster.
"""

import asyncio
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

    class _HangingQraiseClient(_FakeCunqaClient):
        def qraise(self, n_qpus: int, walltime: str, **kwargs: Any) -> str:
            time.sleep(2)
            return super().qraise(n_qpus, walltime, **kwargs)

    fake_client = _HangingQraiseClient(counts_per_qpu={})
    config = CUNQAExecutorConfig(n_qpus=2, walltime="00:05:00", startup_timeout_s=0.1, poll_interval_s=0.01)
    executor = CUNQAExecutor(config, client=fake_client)

    with pytest.raises(TimeoutError, match="qraise did not return"):
        await executor.execute(bell_state(), shots=1000)

    assert len(fake_client.dropped) == 1
    assert fake_client.dropped[0].startswith("marqov-")


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
