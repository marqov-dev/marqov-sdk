"""Alice & Bob local cat-qubit emulator executor.

Runs circuits on Alice & Bob's free, credential-less local emulators
(`AliceBobLocalProvider`) — six preset tiers spanning physical (uncorrected)
and logical (error-corrected) cat-qubit noise models, at 1-40 qubits:
"EMU:1Q:LESCANNE_2020", "EMU:6Q:PHYSICAL_CATS", "EMU:40Q:PHYSICAL_CATS",
"EMU:15Q:LOGICAL_EARLY", "EMU:40Q:LOGICAL_TARGET", "EMU:40Q:LOGICAL_NOISELESS".

No Felis Cloud account, API key, or network call required. Real-hardware
(Boson 4 QPU) / paid cloud-emulator access via `AliceBobRemoteProvider` is
deliberately out of scope for this executor — it needs cost-budget
guardrails this executor doesn't implement, so it's left for a future
executor rather than added here unguarded.

Provider quirk worth knowing before debugging an apparent circuit-rejection
issue on this backend: `qiskit.compiler.transpile()` is NOT a reliable
predictor of what `backend.run()` will accept. `transpile()` fails on
EMU:6Q:PHYSICAL_CATS for a circuit containing `h`/`initialize` (TranspilerError,
target-basis mismatch) that `backend.run()` executes correctly via its own
internal noise-pass + `decompose()` pipeline. Validate against the real
`run()` call, not a `transpile()` dry run — that's what `execute()` below does.

Every result's `metadata` carries `simulated=True` — these are Alice & Bob's
own emulator *models*, never a measurement of real hardware.

See https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider.
"""

from __future__ import annotations

import asyncio
import functools
import importlib.metadata
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from marqov.executors.base import BaseExecutor, ExecutionResult

if TYPE_CHECKING:
    from marqov.circuits import Circuit


class UnsupportedCircuitError(Exception):
    """Raised when a circuit can't run on the configured Alice & Bob backend.

    Covers two cases the underlying provider does NOT reject itself:
    a circuit with more qubits than the backend supports (checked directly
    by `execute()` before ever calling the provider — confirmed empirically
    that e.g. EMU:1Q:LESCANNE_2020 otherwise silently executes an oversized
    circuit and returns misleading counts instead of raising), and any other
    execution-time rejection surfaced by the provider (wrapped from whatever
    exception `backend.run()` raises).
    """


@dataclass
class AliceBobExecutorConfig:
    """Configuration for the Alice & Bob local-emulator executor.

    All fields beyond `backend_name` are optional pass-through overrides to
    the underlying `AliceBobLocalProvider` builder for the chosen tier — any
    left as None use that builder's own defaults. Passing a kwarg the chosen
    tier's builder doesn't accept (e.g. `distance` with a physical backend)
    raises TypeError at construction — the underlying provider's own
    keyword-argument behavior, not something this executor adds or catches.

    Attributes:
        backend_name: One of the six local emulator tiers: "EMU:1Q:LESCANNE_2020",
            "EMU:6Q:PHYSICAL_CATS", "EMU:40Q:PHYSICAL_CATS", "EMU:15Q:LOGICAL_EARLY",
            "EMU:40Q:LOGICAL_TARGET" (default), "EMU:40Q:LOGICAL_NOISELESS".
        average_nb_photons: Cat size — physical tiers and non-noiseless
            logical tiers only.
        kappa_1: One-photon dissipation rate (Hz) — physical tiers and
            non-noiseless logical tiers only.
        kappa_2: Two-photon dissipation rate (Hz) — physical tiers and
            non-noiseless logical tiers only.
        distance: Repetition-code distance — logical tiers only. The knob
            behind the exponential error-suppression story: sweeping this
            is what "watch QEC work" actually demonstrates.
    """

    backend_name: str = "EMU:40Q:LOGICAL_TARGET"
    average_nb_photons: float | None = None
    kappa_1: float | None = None
    kappa_2: float | None = None
    distance: int | None = None

    def to_provider_kwargs(self) -> dict[str, Any]:
        """Non-None override fields, ready to forward to `AliceBobLocalProvider.get_backend()`."""
        kwargs = {
            "average_nb_photons": self.average_nb_photons,
            "kappa_1": self.kappa_1,
            "kappa_2": self.kappa_2,
            "distance": self.distance,
        }
        return {k: v for k, v in kwargs.items() if v is not None}


class AliceBobExecutor(BaseExecutor):
    """Execute circuits on Alice & Bob's local cat-qubit emulators.

    Free, credential-less, entirely local — no Felis Cloud account, API key,
    or network call needed for any of the six preset tiers.

    Example:
        >>> executor = AliceBobExecutor()
        >>> result = await executor.execute(bell_state(), shots=1000)
        >>> print(result.counts)  # {"00": ~500, "11": ~500}

    Run the raw, uncorrected physical tier instead of the default
    error-corrected logical target:
        >>> executor = AliceBobExecutor(AliceBobExecutorConfig(backend_name="EMU:6Q:PHYSICAL_CATS"))

    Sweep code distance on the logical tier — the flagship "watch QEC work"
    demonstration (see the integration spec for the traced suppression math):
        >>> for d in (3, 5, 7, 9, 11, 13, 15):
        ...     executor = AliceBobExecutor(AliceBobExecutorConfig(distance=d))
        ...     result = await executor.execute(circuit, shots=2000, seed_simulator=42)
    """

    def __init__(self, config: AliceBobExecutorConfig | None = None) -> None:
        """Initialize the Alice & Bob local-emulator executor.

        Args:
            config: Configuration options. Uses the logical-target tier with
                the provider's own default noise parameters if not provided.

        Raises:
            ImportError: If qiskit-alice-bob-provider is not installed.
            TypeError: If a config field doesn't apply to the chosen backend
                tier (e.g. `distance` with a physical backend) — raised by
                the underlying provider's builder signature.
        """
        self.config = config or AliceBobExecutorConfig()
        try:
            from qiskit_alice_bob_provider import AliceBobLocalProvider
        except ImportError as exc:
            raise ImportError(
                "qiskit-alice-bob-provider is required for AliceBobExecutor. "
                "Install with: pip install marqov[alicebob]"
            ) from exc

        self._provider = AliceBobLocalProvider()
        self._backend = self._provider.get_backend(
            self.config.backend_name, **self.config.to_provider_kwargs()
        )

    async def execute(
        self,
        circuit: Circuit,
        shots: int = 1000,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Run circuit on the configured Alice & Bob local emulator.

        Args:
            circuit: The circuit to execute.
            shots: Number of measurement shots.
            **kwargs: Forwarded to the underlying AerSimulator run call
                (e.g. `seed_simulator=42` for reproducible results,
                `max_parallel_threads=1`).

        Returns:
            ExecutionResult with measurement counts and metadata identifying
            this as a simulated result (`metadata["simulated"] is True`).

        Raises:
            UnsupportedCircuitError: If the circuit has more qubits than the
                backend supports, or the provider rejects it at execution time.
        """
        circuit = self._validate_circuit(circuit)

        if circuit.num_qubits > self._backend.num_qubits:
            raise UnsupportedCircuitError(
                f"Circuit has {circuit.num_qubits} qubits, but "
                f"{self.config.backend_name} only supports "
                f"{self._backend.num_qubits}."
            )

        qiskit_circuit = circuit.to_qiskit()
        if not qiskit_circuit.cregs:
            qiskit_circuit.measure_all()
        start_time = time.perf_counter()

        loop = asyncio.get_event_loop()
        try:
            counts, job = await loop.run_in_executor(
                None,
                functools.partial(self._run_sync, qiskit_circuit, shots, kwargs),
            )
        except UnsupportedCircuitError:
            raise
        except Exception as exc:
            raise UnsupportedCircuitError(
                f"{self.config.backend_name} could not execute this circuit: {exc}"
            ) from exc

        execution_time_ms = (time.perf_counter() - start_time) * 1000

        return ExecutionResult(
            counts=counts,
            backend=f"alice_bob-{self.config.backend_name}",
            execution_time_ms=execution_time_ms,
            shots=shots,
            raw_result=job,
            metadata=self._build_metadata(),
        )

    def _run_sync(
        self, qiskit_circuit: Any, shots: int, run_kwargs: dict[str, Any]
    ) -> tuple[dict[str, int], Any]:
        """Blocking submit-and-wait, run off the event loop by `execute()`.

        Deliberately calls `backend.run()` directly on the untranspiled
        circuit rather than pre-flight-checking it with `transpile()` or
        `backend.target` membership — see the module docstring for why
        both of those disagree with what `run()` itself actually accepts.
        """
        job = self._backend.run(qiskit_circuit, shots=shots, **run_kwargs)
        counts = job.result().get_counts()
        return counts, job

    def _build_metadata(self) -> dict[str, Any]:
        """The only code path allowed to build result metadata — always sets attribution fields.

        Reads back the *resolved* noise-model parameters from the built
        backend's processor object, not just what was explicitly passed in
        config — so a result run entirely on provider defaults still records
        what those defaults actually were. `_processor` and its attributes
        are private provider internals (best-effort introspection, not a
        documented API) — fall back to None rather than raising if a future
        provider release renames them.
        """
        processor = getattr(self._backend, "_processor", None)
        return {
            "simulated": True,
            "backend_name": self.config.backend_name,
            "provider_package_version": importlib.metadata.version("qiskit-alice-bob-provider"),
            "average_nb_photons": getattr(processor, "_average_nb_photons", None),
            "kappa_1": getattr(processor, "_kappa_1", None),
            "kappa_2": getattr(processor, "_kappa_2", None),
            "distance": getattr(processor, "_distance", None),
        }
