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
    async def test_single_precision_is_recorded(self) -> None:
        result = await _executor(precision="single").execute(bell_state(), shots=10, seed=1)
        assert result.metadata["reproducibility"]["precision"] == "complex64"

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
    async def test_installed_gpu_engine_is_not_run_unvalidated(self, monkeypatch) -> None:
        """An installed GPU engine is refused as unvalidated, never substituted."""
        import pennylane as qml

        real_device = qml.device

        def fake_device(name, *args, **kwargs):
            if name == "lightning.gpu":
                return real_device("lightning.qubit", *args, **kwargs)
            return real_device(name, *args, **kwargs)

        monkeypatch.setattr(qml, "device", fake_device)
        with pytest.raises(NotImplementedError, match="lightning.gpu"):
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
