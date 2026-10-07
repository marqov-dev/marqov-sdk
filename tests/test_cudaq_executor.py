"""Tests for the NVIDIA CUDA-Q executor.

The gate-mapping and factory tests run anywhere (no ``cudaq`` needed) using a
recording kernel builder and a fake ``cudaq`` module. The final test exercises a
real CUDA-Q run and is skipped unless ``cudaq`` is importable (Linux-only).
"""

from __future__ import annotations

from typing import Any

import pytest

from marqov.circuits import Circuit, bell_state, ghz_state
from marqov.executors import CudaqExecutor, CudaqExecutorConfig, ExecutorFactory
from marqov.executors import cudaq as cudaq_module


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #


class _Register:
    """Indexable qubit register stand-in; ``reg[i]`` yields a hashable token."""

    def __init__(self, count: int) -> None:
        self.count = count

    def __getitem__(self, index: int) -> tuple[str, int]:
        return ("q", index)


class RecordingBuilder:
    """Records CUDA-Q builder calls so the transpiler can be asserted on."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def qalloc(self, count: int) -> _Register:
        self.calls.append(("qalloc", count))
        return _Register(count)

    def __getattr__(self, gate: str):  # h, x, rx, cx, mz, ...
        def record(*args: Any) -> None:
            self.calls.append((gate, *args))

        return record


class _FakeSampleResult:
    def __init__(self, counts: dict[str, int]) -> None:
        self._counts = counts

    def items(self):
        return self._counts.items()


class FakeCudaq:
    """Minimal fake of the ``cudaq`` module for executor tests."""

    def __init__(self, gpus: int = 0) -> None:
        self._gpus = gpus
        self.target: str | None = None
        self.target_kwargs: dict[str, Any] = {}
        self.seed: int | None = None
        self.sampled_shots: int | None = None

    def num_available_gpus(self) -> int:
        return self._gpus

    def set_target(self, target: str, **kwargs: Any) -> None:
        self.target = target
        self.target_kwargs = kwargs

    def set_random_seed(self, seed: int) -> None:
        self.seed = seed

    def make_kernel(self) -> RecordingBuilder:
        return RecordingBuilder()

    def sample(self, kernel: Any, shots_count: int) -> _FakeSampleResult:
        self.sampled_shots = shots_count
        half = shots_count // 2
        return _FakeSampleResult({"00": half, "11": shots_count - half})


@pytest.fixture
def fake_cudaq(monkeypatch: pytest.MonkeyPatch) -> FakeCudaq:
    """Patch ``_import_cudaq`` to return a FakeCudaq (0 GPUs by default)."""
    fake = FakeCudaq(gpus=0)
    monkeypatch.setattr(cudaq_module, "_import_cudaq", lambda: fake)
    return fake


# --------------------------------------------------------------------------- #
# Gate mapping (no cudaq required)
# --------------------------------------------------------------------------- #


class TestBuildKernel:
    def test_bell_state_maps_to_h_cx_mz(self) -> None:
        builder = RecordingBuilder()
        cudaq_module.build_kernel(builder, bell_state())

        assert builder.calls[0] == ("qalloc", 2)
        gate_names = [c[0] for c in builder.calls]
        assert "h" in gate_names
        assert "cx" in gate_names
        # Measurement is always applied last.
        assert builder.calls[-1][0] == "mz"

    def test_rotation_gate_carries_angle(self) -> None:
        builder = RecordingBuilder()
        cudaq_module.build_kernel(builder, Circuit().rx(0.5, 0))

        rx_calls = [c for c in builder.calls if c[0] == "rx"]
        assert len(rx_calls) == 1
        # ("rx", angle, qubit_token)
        assert rx_calls[0][1] == pytest.approx(0.5)

    def test_ghz_allocates_all_qubits(self) -> None:
        builder = RecordingBuilder()
        cudaq_module.build_kernel(builder, ghz_state(4))
        assert builder.calls[0] == ("qalloc", 4)

    def test_unsupported_gate_raises(self) -> None:
        class FakeInstr:
            name = "toffoli"
            params: list[float] = []

        class FakeQiskitCircuit:
            num_qubits = 3
            data = [type("I", (), {"operation": FakeInstr(), "qubits": []})()]

            def find_bit(self, q: Any) -> Any:  # pragma: no cover - not reached
                raise AssertionError

        class FakeMarqovCircuit:
            def to_qiskit(self) -> FakeQiskitCircuit:
                return FakeQiskitCircuit()

        with pytest.raises(NotImplementedError, match="toffoli"):
            cudaq_module.build_kernel(RecordingBuilder(), FakeMarqovCircuit())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Factory registration (no cudaq required)
# --------------------------------------------------------------------------- #


class TestFactory:
    @pytest.mark.parametrize(
        ("slug", "expected_target"),
        [("cudaq-cpu", "qpp-cpu"), ("cudaq-gpu", "nvidia"), ("cudaq-iqm", "iqm")],
    )
    def test_slug_selects_target(self, slug: str, expected_target: str) -> None:
        executor = ExecutorFactory.create_executor(slug, {"provider": "CUDA-Q"})
        assert isinstance(executor, CudaqExecutor)
        assert executor.config.target == expected_target

    def test_provider_registered(self) -> None:
        assert ExecutorFactory.is_provider_supported("CUDA-Q")

    def test_explicit_target_overrides_slug(self) -> None:
        executor = ExecutorFactory.create_executor(
            "cudaq-cpu", {"provider": "CUDA-Q", "target": "nvidia"}
        )
        assert executor.config.target == "nvidia"

    def test_iqm_config_forwarded(self) -> None:
        executor = ExecutorFactory.create_executor(
            "cudaq-iqm",
            {"provider": "CUDA-Q", "iqm_url": "https://example/garnet"},
        )
        assert executor.config.iqm_url == "https://example/garnet"


# --------------------------------------------------------------------------- #
# execute() with a fake cudaq (no real backend)
# --------------------------------------------------------------------------- #


class TestExecute:
    @pytest.mark.asyncio
    async def test_cpu_execute_returns_normalized_counts(self, fake_cudaq: FakeCudaq) -> None:
        executor = CudaqExecutor(CudaqExecutorConfig(target="qpp-cpu"))
        result = await executor.execute(bell_state(), shots=1000)

        assert result.counts == {"00": 500, "11": 500}
        assert result.shots == 1000
        assert result.backend == "cudaq:qpp-cpu"
        assert result.metadata["target"] == "qpp-cpu"
        assert fake_cudaq.target == "qpp-cpu"
        assert fake_cudaq.sampled_shots == 1000

    @pytest.mark.asyncio
    async def test_gpu_falls_back_to_cpu_when_no_gpu(self, fake_cudaq: FakeCudaq) -> None:
        executor = CudaqExecutor(CudaqExecutorConfig(target="nvidia"))
        result = await executor.execute(bell_state(), shots=100)

        # 0 GPUs available -> resolved target is qpp-cpu.
        assert fake_cudaq.target == "qpp-cpu"
        assert result.metadata["target"] == "qpp-cpu"
        assert result.metadata["requested_target"] == "nvidia"

    @pytest.mark.asyncio
    async def test_gpu_used_when_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeCudaq(gpus=1)
        monkeypatch.setattr(cudaq_module, "_import_cudaq", lambda: fake)
        executor = CudaqExecutor(CudaqExecutorConfig(target="nvidia"))
        await executor.execute(bell_state(), shots=100)
        assert fake.target == "nvidia"

    @pytest.mark.asyncio
    async def test_gpu_no_fallback_raises_nothing_but_sets_nvidia(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCudaq(gpus=0)
        monkeypatch.setattr(cudaq_module, "_import_cudaq", lambda: fake)
        executor = CudaqExecutor(
            CudaqExecutorConfig(target="nvidia", gpu_fallback_to_cpu=False)
        )
        await executor.execute(bell_state(), shots=10)
        assert fake.target == "nvidia"

    @pytest.mark.asyncio
    async def test_seed_forwarded(self, fake_cudaq: FakeCudaq) -> None:
        executor = CudaqExecutor(CudaqExecutorConfig(target="qpp-cpu", seed=7))
        await executor.execute(bell_state(), shots=10)
        assert fake_cudaq.seed == 7

    @pytest.mark.asyncio
    async def test_iqm_requires_url(self, fake_cudaq: FakeCudaq) -> None:
        executor = CudaqExecutor(CudaqExecutorConfig(target="iqm"))
        with pytest.raises(ValueError, match="iqm_url"):
            await executor.execute(bell_state(), shots=10)

    @pytest.mark.asyncio
    async def test_iqm_sets_target_with_url(self, fake_cudaq: FakeCudaq) -> None:
        executor = CudaqExecutor(
            CudaqExecutorConfig(target="iqm", iqm_url="https://example/garnet")
        )
        await executor.execute(bell_state(), shots=10)
        assert fake_cudaq.target == "iqm"
        assert fake_cudaq.target_kwargs["url"] == "https://example/garnet"

    def test_import_error_is_actionable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When ``import cudaq`` fails, _import_cudaq gives install guidance."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any):
            if name == "cudaq":
                raise ImportError("no cudaq")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match=r"marqov\[cudaq\]"):
            cudaq_module._import_cudaq()


# --------------------------------------------------------------------------- #
# Real CUDA-Q execution (Linux-only; skipped when cudaq is absent)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_real_cudaq_bell_state() -> None:
    pytest.importorskip("cudaq", reason="cudaq is Linux-only; not installed")
    executor = CudaqExecutor(CudaqExecutorConfig(target="qpp-cpu", seed=1))
    result = await executor.execute(bell_state(), shots=1000)

    assert sum(result.counts.values()) == 1000
    # Bell state -> only |00> and |11> outcomes.
    assert set(result.counts).issubset({"00", "11"})
    assert result.backend == "cudaq:qpp-cpu"


@pytest.mark.asyncio
async def test_concurrent_executes_preserve_target_seed_and_raw_result(monkeypatch):
    """Coordinate contention explicitly rather than relying on scheduler sleeps."""
    import asyncio
    import threading

    first_sampling = threading.Event()
    contender_attempted = threading.Event()

    class ObservedLock:
        def __init__(self):
            self.lock = threading.Lock()

        def __enter__(self):
            if self.lock.locked():
                contender_attempted.set()
            self.lock.acquire()

        def __exit__(self, *args):
            self.lock.release()

    class StatefulCudaq(FakeCudaq):
        def __init__(self):
            super().__init__(gpus=1)
            self.results = []

        def set_random_seed(self, seed):
            super().set_random_seed(seed)
            if seed == 22:
                contender_attempted.set()

        def sample(self, kernel, shots_count):
            if not self.results:
                first_sampling.set()
                assert contender_attempted.wait(5), "Second execution never reached the worker"
            result = _FakeSampleResult({"00": shots_count})
            result.sampled_target = self.target
            result.sampled_seed = self.seed
            self.results.append(result)
            return result

    fake = StatefulCudaq()
    monkeypatch.setattr(cudaq_module, "_EXECUTION_LOCK", ObservedLock(), raising=False)
    monkeypatch.setattr(cudaq_module, "_import_cudaq", lambda: fake)
    first = CudaqExecutor(CudaqExecutorConfig(target="qpp-cpu", seed=11))
    second = CudaqExecutor(CudaqExecutorConfig(target="nvidia", seed=22))
    first_task = asyncio.create_task(first.execute(bell_state(), shots=10))
    assert await asyncio.to_thread(first_sampling.wait, 5)
    results = await asyncio.gather(first_task, second.execute(bell_state(), shots=10))
    for result, target, seed, raw in zip(results, ["qpp-cpu", "nvidia"], [11, 22], fake.results):
        assert result.metadata["target"] == raw.sampled_target == target
        assert result.backend == f"cudaq:{raw.sampled_target}"
        assert raw.sampled_seed == seed
        assert result.raw_result is raw


@pytest.mark.asyncio
@pytest.mark.parametrize("previous", [None, "environment-secret", "configured-secret"])
@pytest.mark.parametrize("failure_stage", [None, "target", "kernel", "sample"])
async def test_iqm_token_override_is_temporary(monkeypatch, previous, failure_stage):
    import os
    import warnings

    if previous is None:
        monkeypatch.delenv("IQM_TOKEN", raising=False)
    else:
        monkeypatch.setenv("IQM_TOKEN", previous)
    observed = []

    class TokenCudaq(FakeCudaq):
        def set_target(self, target, **kwargs):
            observed.append(os.environ.get("IQM_TOKEN"))
            if failure_stage == "target":
                raise RuntimeError("target failed")
            super().set_target(target, **kwargs)

        def make_kernel(self):
            if failure_stage == "kernel":
                raise RuntimeError("kernel failed")
            return super().make_kernel()

        def sample(self, kernel, shots_count):
            observed.append(os.environ.get("IQM_TOKEN"))
            if failure_stage == "sample":
                raise RuntimeError("sample failed")
            return super().sample(kernel, shots_count)

    monkeypatch.setattr(cudaq_module, "_import_cudaq", lambda: TokenCudaq())
    executor = CudaqExecutor(
        CudaqExecutorConfig(
            target="iqm", iqm_url="https://example/garnet", iqm_token="configured-secret"
        )
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        if failure_stage:
            with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
                await executor.execute(bell_state(), shots=10)
        else:
            await executor.execute(bell_state(), shots=10)
    assert observed and set(observed) == {"configured-secret"}
    assert os.environ.get("IQM_TOKEN") == previous
    assert len(captured) == int(previous == "environment-secret")
    for warning in captured:
        message = str(warning.message)
        assert "overrides" in message
        assert "configured-secret" not in message
        assert "environment-secret" not in message


@pytest.mark.asyncio
async def test_iqm_without_configured_token_uses_environment(monkeypatch):
    import os

    monkeypatch.setenv("IQM_TOKEN", "environment-secret")
    observed = []

    class TokenCudaq(FakeCudaq):
        def set_target(self, target, **kwargs):
            observed.append(os.environ.get("IQM_TOKEN"))
            super().set_target(target, **kwargs)

    monkeypatch.setattr(cudaq_module, "_import_cudaq", lambda: TokenCudaq())
    await CudaqExecutor(
        CudaqExecutorConfig(target="iqm", iqm_url="https://example/garnet")
    ).execute(bell_state(), shots=10)
    assert observed == ["environment-secret"]
    assert os.environ["IQM_TOKEN"] == "environment-secret"


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["qpp-cpu", "nvidia", "nvidia-fp64", "nvidia-mgpu", "iqm"])
async def test_status_does_not_claim_unpolled_remote_is_online(target):
    status = await CudaqExecutor(CudaqExecutorConfig(target=target)).get_status()
    assert status.status == ("maintenance" if target == "iqm" else "online")
    if target == "iqm":
        assert status.queue_depth is None
        assert status.queue_time_seconds is None
