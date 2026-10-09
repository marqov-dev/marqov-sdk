"""Tests for marqov.executors.lightning (PennyLane Lightning, CPU engines).

Real Lightning runs, no mocks, except where a test must simulate an
environment this machine does not have (a missing or substituted device).
"""

from __future__ import annotations

import importlib.metadata as md
import math

import pytest

pytest.importorskip("pennylane_lightning")

from marqov.circuits import Circuit, bell_state
from marqov.executors import (
    ExecutionResult,
    ExecutorFactory,
    LightningExecutor,
    LightningExecutorConfig,
    LocalExecutor,
)
from marqov.executors.lightning import (
    LightningDeviceUnavailableError,
    probe_lightning_device,
)

KOKKOS = probe_lightning_device("lightning.kokkos")
needs_kokkos = pytest.mark.skipif(
    not KOKKOS.available, reason=f"lightning.kokkos unavailable: {KOKKOS.reason}"
)


def _spread_circuit() -> Circuit:
    """A 3-qubit parameterised circuit with many outcomes (so seeds matter)."""
    return Circuit().ry(0.7, 0).rx(1.3, 1).cnot(0, 1).ry(2.1, 2).cz(1, 2).h(0)


def _executor(device: str = "lightning.qubit", **kwargs) -> LightningExecutor:
    return LightningExecutor(LightningExecutorConfig(device=device, **kwargs))


class TestCountsAndBitOrder:
    @pytest.mark.asyncio
    async def test_bell_state_counts(self) -> None:
        result = await _executor().execute(bell_state(), shots=1000, seed=7)

        assert isinstance(result, ExecutionResult)
        assert result.backend == "lightning.qubit"
        assert result.shots == 1000
        assert sum(result.counts.values()) == 1000
        assert set(result.counts) <= {"00", "11"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "circuit",
        [
            Circuit().x(0).cz(0, 1),            # |10>: qubit 0 excited only
            Circuit().x(1).cz(0, 1),            # |01>
            Circuit().x(0).x(0).x(2),           # sparse wires {0, 2}: |01>
            Circuit().x(2).cnot(2, 0).x(2),     # sparse wires, reversed control
        ],
        ids=["q0", "q1", "sparse", "sparse-reversed"],
    )
    async def test_deterministic_outcomes_match_local_executor(self, circuit) -> None:
        """Same count keys as the SDK reference executor, sparse wires included."""
        lightning = await _executor().execute(circuit, shots=50, seed=1)
        local = await LocalExecutor().execute(circuit, shots=50)

        assert lightning.counts == local.counts

    @pytest.mark.asyncio
    async def test_distribution_matches_exact_probabilities(self) -> None:
        """Counts agree with the exact (QuantumFlow) distribution within 5 sigma."""
        circuit = _spread_circuit()
        shots = 20000
        result = await _executor().execute(circuit, shots=shots, seed=11)
        _assert_counts_match_exact(circuit, result.counts, shots)


def _assert_counts_match_exact(circuit: Circuit, counts: dict[str, int], shots: int) -> None:
    import numpy as np

    probs = np.abs(circuit.simulate().tensor.flatten()) ** 2
    n = circuit.num_qubits
    for index, p in enumerate(probs):
        key = format(index, f"0{n}b")
        sigma = math.sqrt(shots * p * (1 - p)) or 1.0
        assert abs(counts.get(key, 0) - shots * p) <= 5 * sigma, key


class TestReproducibility:
    @pytest.mark.asyncio
    async def test_same_seed_replays_identical_samples(self) -> None:
        executor = _executor()
        first = await executor.execute(_spread_circuit(), shots=1000, seed=1234)
        # An intervening call must not advance the next seeded call's stream.
        await executor.execute(_spread_circuit(), shots=1000, seed=99)
        again = await executor.execute(_spread_circuit(), shots=1000, seed=1234)

        rec1 = first.metadata["reproducibility"]
        rec2 = again.metadata["reproducibility"]
        assert first.counts == again.counts
        assert rec1["samples_sha256"] == rec2["samples_sha256"]

    @pytest.mark.asyncio
    async def test_different_seeds_give_different_samples(self) -> None:
        executor = _executor()
        a = await executor.execute(_spread_circuit(), shots=1000, seed=1)
        b = await executor.execute(_spread_circuit(), shots=1000, seed=2)

        assert (
            a.metadata["reproducibility"]["samples_sha256"]
            != b.metadata["reproducibility"]["samples_sha256"]
        )

    @pytest.mark.asyncio
    async def test_config_seed_is_used_when_no_call_seed(self) -> None:
        executor = _executor(seed=55)
        result = await executor.execute(_spread_circuit(), shots=200)
        record = result.metadata["reproducibility"]

        assert record["seed"] == 55
        assert record["seed_source"] == "config"

    @pytest.mark.asyncio
    async def test_unseeded_run_records_a_generated_seed_that_replays(self) -> None:
        executor = _executor()
        first = await executor.execute(_spread_circuit(), shots=1000)
        record = first.metadata["reproducibility"]
        assert record["seed_source"] == "generated"
        assert isinstance(record["seed"], int)

        replay = await executor.execute(_spread_circuit(), shots=1000, seed=record["seed"])
        assert replay.metadata["reproducibility"]["samples_sha256"] == record["samples_sha256"]


class TestRecord:
    @pytest.mark.asyncio
    async def test_record_fields(self) -> None:
        circuit = _spread_circuit()
        result = await _executor().execute(circuit, shots=300, seed=3)
        meta = result.metadata
        record = meta["reproducibility"]

        # Provenance: explicit fields, not a provider migration.
        assert meta["vendor"] == "Xanadu"
        assert meta["framework"] == "PennyLane"
        assert meta["engine"] == "lightning.qubit"
        assert meta["access_path"] == "local"
        assert meta["compute_provider"] == "local"
        assert meta["seed"] == 3

        assert record["device_requested"] == "lightning.qubit"
        assert record["device_name"] == "lightning.qubit"
        assert record["device_class"].endswith("LightningQubit")
        assert record["pennylane_version"] == md.version("pennylane")
        assert record["plugin_version"] == f"pennylane-lightning=={md.version('pennylane-lightning')}"
        assert record["wheel_tag"]
        assert record["binary"].startswith("lightning_qubit_ops")
        assert len(record["binary_sha256"]) == 64
        assert record["precision"] == "complex128"
        assert record["n_wires"] == 3
        assert record["wire_order"] == [0, 1, 2]
        assert record["measurements"] == [
            {"type": "sample", "basis": "computational", "wires": [0, 1, 2]}
        ]
        assert record["shots"] == 300
        assert record["seed"] == 3
        assert record["seed_kind"] == "int"
        assert record["seed_source"] == "caller"
        assert record["rng_policy"] == "fresh device per run; seed applied at construction"
        assert record["device_call_index"] == 0
        assert record["diff_method"] is None
        assert "omp_threads_env" in record
        for key in ("platform", "machine", "cpu", "cores", "python", "numpy", "scipy"):
            assert key in record["host"]
        for key in ("circuit_sha256", "input_sha256", "samples_sha256", "counts_sha256"):
            assert len(record[key]) == 64

    @pytest.mark.asyncio
    async def test_circuit_hash_tracks_the_circuit_not_the_run(self) -> None:
        executor = _executor()
        a = await executor.execute(Circuit().ry(0.5, 0), shots=10, seed=1)
        b = await executor.execute(Circuit().ry(0.5, 0), shots=10, seed=2)
        c = await executor.execute(Circuit().ry(0.6, 0), shots=10, seed=1)

        ra, rb, rc = (r.metadata["reproducibility"] for r in (a, b, c))
        assert ra["circuit_sha256"] == rb["circuit_sha256"] != rc["circuit_sha256"]
        # Input hash covers the executed program (ops + measurement + shots) only.
        assert ra["input_sha256"] == rb["input_sha256"] != rc["input_sha256"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("angle", ["float32", "float64", "int64"])
    async def test_numpy_scalar_angles_are_accepted_and_hashed(self, angle) -> None:
        import numpy as np

        value = getattr(np, angle)(1)
        result = await _executor().execute(Circuit().rx(value, 0), shots=10, seed=1)
        record = result.metadata["reproducibility"]
        assert len(record["circuit_sha256"]) == 64

    @pytest.mark.asyncio
    async def test_single_precision_is_recorded(self) -> None:
        result = await _executor(precision="single").execute(bell_state(), shots=10, seed=1)
        assert result.metadata["reproducibility"]["precision"] == "complex64"

    @pytest.mark.asyncio
    async def test_qualification_runs_per_precision(self, monkeypatch) -> None:
        monkeypatch.setattr(LightningExecutor, "_qualified", {})
        probed: list[str] = []
        original = LightningExecutor._sample

        def spy(self, qml, operations, wires, shots, seed):
            if seed == 0 and wires == [0, 1]:
                probed.append(self.config.precision)
            return original(self, qml, operations, wires, shots, seed)

        monkeypatch.setattr(LightningExecutor, "_sample", spy)
        await _executor().execute(bell_state(), shots=10, seed=1)
        result = await _executor(precision="single").execute(bell_state(), shots=10, seed=1)

        assert "single" in probed and "double" in probed
        qualification = result.metadata["reproducibility"]["qualification"]
        assert qualification["precision"] == "complex64"

    @pytest.mark.asyncio
    async def test_compute_provider_is_recorded_as_given(self) -> None:
        executor = _executor(compute_provider="aws-parallelcluster:eu-north-1")
        result = await executor.execute(bell_state(), shots=10, seed=1)
        assert result.metadata["compute_provider"] == "aws-parallelcluster:eu-north-1"


class TestUnsupportedInputs:
    @pytest.mark.asyncio
    async def test_unknown_option_is_refused(self) -> None:
        with pytest.raises(TypeError, match="noise_model"):
            await _executor().execute(bell_state(), shots=10, noise_model="x")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("shots", [0, -5])
    async def test_non_positive_shots_are_refused(self, shots) -> None:
        with pytest.raises(ValueError, match="shots"):
            await _executor().execute(bell_state(), shots=shots)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("shots", [True, 10.0, None])
    async def test_non_integer_shots_are_refused(self, shots) -> None:
        with pytest.raises(TypeError, match="shots"):
            await _executor().execute(bell_state(), shots=shots)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("seed", [True, -1, 1.5, "1", 2**63])
    async def test_invalid_seed_is_refused(self, seed) -> None:
        with pytest.raises((TypeError, ValueError), match="seed"):
            await _executor().execute(bell_state(), shots=10, seed=seed)

    @pytest.mark.asyncio
    async def test_empty_circuit_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no gates"):
            await _executor().execute(Circuit(), shots=10)

    @pytest.mark.asyncio
    async def test_unbound_parameter_is_refused(self) -> None:
        import sympy

        with pytest.raises(ValueError, match="unbound"):
            await _executor().execute(Circuit().rx(sympy.Symbol("t"), 0), shots=10)

    def test_non_lightning_device_is_refused(self) -> None:
        with pytest.raises(ValueError, match="lightning"):
            _executor(device="default.qubit")

    def test_unknown_precision_is_refused(self) -> None:
        with pytest.raises(ValueError, match="precision"):
            _executor(precision="half")


class TestDeviceAvailability:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("device", ["lightning.gpu", "lightning.tensor"])
    async def test_gpu_engines_report_unavailable_from_environment(self, device) -> None:
        probe = probe_lightning_device(device)
        if probe.available:
            pytest.skip(f"{device} is installed here")
        assert probe.reason  # the real error from constructing the device

        with pytest.raises(LightningDeviceUnavailableError, match=device) as exc_info:
            await _executor(device=device).execute(bell_state(), shots=10)
        assert probe.reason in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_availability_is_an_environment_check_not_a_catalogue(
        self, monkeypatch
    ) -> None:
        """If constructing lightning.qubit fails, it is reported unavailable too."""
        import pennylane as qml

        def broken_device(name, *args, **kwargs):
            raise RuntimeError("simulated missing binary")

        monkeypatch.setattr(qml, "device", broken_device)
        assert probe_lightning_device("lightning.qubit").available is False
        with pytest.raises(LightningDeviceUnavailableError, match="simulated missing binary"):
            await _executor().execute(bell_state(), shots=10)

    @pytest.mark.asyncio
    async def test_gpu_cpu_substitution_is_refused(self, monkeypatch) -> None:
        """A CPU device cannot satisfy a requested GPU execution."""
        import pennylane as qml

        real_device = qml.device

        def fake_device(name, *args, **kwargs):
            if name == "lightning.gpu":
                return real_device("lightning.qubit", *args, **kwargs)
            return real_device(name, *args, **kwargs)

        monkeypatch.setattr(qml, "device", fake_device)
        with pytest.raises(RuntimeError, match="lightning.qubit"):
            await _executor(device="lightning.gpu").execute(bell_state(), shots=10)

    @pytest.mark.asyncio
    async def test_substituted_device_is_refused(self, monkeypatch) -> None:
        """If PennyLane hands back a different device than requested, refuse."""
        import pennylane as qml

        real_device = qml.device
        monkeypatch.setattr(
            qml, "device", lambda name, **k: real_device("default.qubit", wires=k["wires"])
        )
        with pytest.raises(RuntimeError, match="default.qubit"):
            await _executor().execute(bell_state(), shots=10, seed=1)

    @pytest.mark.asyncio
    async def test_get_status(self) -> None:
        assert (await _executor().get_status()).status == "online"
        if not probe_lightning_device("lightning.gpu").available:
            assert (await _executor(device="lightning.gpu").get_status()).status == "offline"


@needs_kokkos
class TestKokkos:
    """lightning.kokkos is enabled only after it passes the same checks as qubit."""

    @pytest.mark.asyncio
    async def test_bit_order_matches_local_executor(self) -> None:
        circuit = Circuit().x(0).cz(0, 1)
        result = await _executor("lightning.kokkos").execute(circuit, shots=50, seed=1)
        assert result.counts == {"10": 50}

    @pytest.mark.asyncio
    async def test_distribution_matches_exact_probabilities(self) -> None:
        circuit = _spread_circuit()
        result = await _executor("lightning.kokkos").execute(circuit, shots=20000, seed=11)
        _assert_counts_match_exact(circuit, result.counts, 20000)

    @pytest.mark.asyncio
    async def test_seed_replays_and_record_names_kokkos(self) -> None:
        executor = _executor("lightning.kokkos")
        a = await executor.execute(_spread_circuit(), shots=1000, seed=5)
        b = await executor.execute(_spread_circuit(), shots=1000, seed=5)
        record = a.metadata["reproducibility"]

        assert a.counts == b.counts
        assert record["device_name"] == "lightning.kokkos"
        assert record["plugin_version"] == (
            f"pennylane-lightning-kokkos=={md.version('pennylane-lightning-kokkos')}"
        )
        assert record["binary"].startswith("lightning_kokkos_ops")
        assert record["qualification"]["passed"] is True
        assert "kokkos" in record["backend_info"]


class TestFactory:
    @pytest.mark.parametrize(
        ("slug", "device"),
        [("lightning-qubit", "lightning.qubit"), ("lightning-kokkos", "lightning.kokkos")],
    )
    def test_slug_selects_device(self, slug, device) -> None:
        executor = ExecutorFactory.create_executor(slug, {"provider": "PennyLane Lightning"})
        assert isinstance(executor, LightningExecutor)
        assert executor.config.device == device

    def test_config_is_forwarded(self) -> None:
        executor = ExecutorFactory.create_executor(
            "anything",
            {
                "provider": "PennyLane Lightning",
                "device": "lightning.qubit",
                "seed": 9,
                "precision": "single",
                "compute_provider": "hpc-node-17",
            },
        )
        assert executor.config == LightningExecutorConfig(
            device="lightning.qubit", seed=9, precision="single", compute_provider="hpc-node-17"
        )

    def test_unknown_slug_without_device_is_refused(self) -> None:
        with pytest.raises(ValueError, match="device"):
            ExecutorFactory.create_executor("lightning-foo", {"provider": "PennyLane Lightning"})

    def test_provider_is_registered(self) -> None:
        assert ExecutorFactory.is_provider_supported("PennyLane Lightning")


class TestGPUQualificationGate:
    @pytest.mark.parametrize("replay_matches", [True, False])
    def test_gpu_requires_bit_order_distribution_and_seed_replay(self, monkeypatch, replay_matches):
        import numpy as np
        import pennylane as qml
        executor = _executor(device="lightning.gpu")
        monkeypatch.setattr(LightningExecutor, "_qualified", {})
        calls = []
        def sample(qml, operations, wires, shots, seed):
            calls.append((shots, seed))
            if shots == 64:
                samples = np.tile([1, 0], (64, 1))
            else:
                samples = np.concatenate([np.zeros((2000, 2)), np.ones((2000, 2))])
                if len(calls) == 3 and not replay_matches:
                    samples = samples[::-1]
            return None, None, samples, 0
        monkeypatch.setattr(executor, "_sample", sample)
        result = executor._ensure_qualified(qml)
        assert result["passed"] is replay_matches
        assert result["checks"]["fresh_device_seed_replay"] is replay_matches
        assert calls == [(64, 0), (4000, 0), (4000, 0)]

    def test_gpu_constructor_keeps_device_seed_and_precision(self):
        from types import SimpleNamespace

        import numpy as np
        calls = []
        def device(name, **kwargs):
            calls.append((name, kwargs))
            return SimpleNamespace(name=name)
        executor = _executor(device="lightning.gpu", precision="single")
        executor._make_device(SimpleNamespace(device=device), [0, 3], 17)
        assert calls == [("lightning.gpu", {"wires": [0, 3], "seed": 17, "c_dtype": np.complex64})]
        _executor(device="lightning.tensor")._make_device(SimpleNamespace(device=device), [0], None)
        assert calls[-1] == ("lightning.tensor", {"wires": [0], "c_dtype": np.complex128,
                            "method": "tn", "backend": "cutensornet", "worksize_pref": "recommended"})
        assert len(calls) == 2


class TestUnseededTensorContract:
    @pytest.mark.parametrize("kwargs", [
        {"seed": 0}, {"tensor_method": "unknown"}, {"tensor_worksize_pref": "unknown"},
        {"tensor_max_bond_dim": 8}, {"tensor_method": "mps", "tensor_max_bond_dim": True},
        {"tensor_method": "mps", "tensor_cutoff": float("nan")},
        {"tensor_method": "mps", "tensor_cutoff": -1},
        {"tensor_method": "mps", "tensor_cutoff_mode": "unknown"},
    ])
    def test_invalid_options_fail_before_device(self, kwargs):
        with pytest.raises(ValueError):
            _executor(device="lightning.tensor", **kwargs)
        with pytest.raises(ValueError, match="Tensor options"):
            _executor(device="lightning.qubit", tensor_method="tn")

    @pytest.mark.asyncio
    async def test_tensor_record_and_no_seed_claim(self, monkeypatch):
        from types import SimpleNamespace

        import numpy as np
        executor = _executor(device="lightning.tensor", tensor_method="mps", tensor_max_bond_dim=16,
                             tensor_cutoff=0.01, tensor_cutoff_mode="rel")
        calls = []
        monkeypatch.setattr(executor, "_ensure_qualified", lambda qml: {"passed": True, "checks": {"fixture": True}})
        def sample(qml, operations, wires, shots, seed):
            calls.append(seed)
            dev = SimpleNamespace(name="lightning.tensor", c_dtype=np.complex128)
            tape = SimpleNamespace(shots=SimpleNamespace(total_shots=shots), operations=operations)
            return dev, tape, np.tile([1, 0], (shots, 1)), 0
        monkeypatch.setattr(executor, "_sample", sample)
        result = await executor.execute(bell_state(), shots=4)
        record = result.metadata["reproducibility"]
        assert result.counts == {"10": 4} and result.metadata["seed"] is None
        assert record["record_version"] == 2 and record["seed"] is None
        assert record["seed_kind"] == "unsupported" and "no seed interface" in record["rng_policy"]
        assert record["tensor_configuration"] == {"method": "mps", "backend": "cutensornet",
            "worksize_pref": "recommended", "max_bond_dim": 16, "cutoff": 0.01, "cutoff_mode": "rel", "approximation": "mps"}
        assert calls == [None]
        with pytest.raises(ValueError, match="no seed"):
            await executor.execute(bell_state(), seed=7)
        assert calls == [None]

    def test_cache_binds_resolved_options_without_seed_replay(self, monkeypatch):
        import numpy as np
        import pennylane as qml
        monkeypatch.setattr(LightningExecutor, "_qualified", {})
        calls = []
        def sample(self, qml, operations, wires, shots, seed):
            assert seed is None
            calls.append(shots)
            samples = np.tile([1, 0], (64, 1)) if shots == 64 else np.concatenate([np.zeros((2000, 2)), np.ones((2000, 2))])
            return None, None, samples, 0
        monkeypatch.setattr(LightningExecutor, "_sample", sample)
        for options in ({}, {"tensor_method": "mps"}, {"tensor_method": "mps", "tensor_max_bond_dim": 32},
                        {"tensor_method": "mps", "tensor_cutoff": 0.1}, {"tensor_method": "mps", "tensor_cutoff_mode": "rel"},
                        {"tensor_worksize_pref": "min"}):
            e = _executor(device="lightning.tensor", **options)
            q = e._ensure_qualified(qml)
            assert q["passed"] and "fresh_device_seed_replay" not in q["checks"]
            e._ensure_qualified(qml)
        assert len(calls) == 12
        assert len(LightningExecutor._qualified) == 6

    def test_factory_forwards_tensor_configuration(self):
        e = ExecutorFactory.create_executor("lightning-tensor", {"provider": "PennyLane Lightning",
             "tensor_method": "mps", "tensor_max_bond_dim": 32})
        assert e.config.tensor_method == "mps" and e.config.tensor_max_bond_dim == 32


@pytest.mark.asyncio
async def test_numpy_float32_rotation_executes_with_canonical_hash():
    import numpy as np

    angle = np.float32(0.37)
    native = await _executor().execute(Circuit().rx(angle, 0), shots=100, seed=7)
    canonical = await _executor().execute(Circuit().rx(float(angle), 0), shots=100, seed=7)
    assert native.counts == canonical.counts
    assert (
        native.metadata["reproducibility"]["circuit_sha256"]
        == canonical.metadata["reproducibility"]["circuit_sha256"]
    )
