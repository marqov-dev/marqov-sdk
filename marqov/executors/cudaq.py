"""NVIDIA CUDA-Q executor for GPU-accelerated simulation and direct IQM.

This executor runs circuits on NVIDIA's CUDA-Q toolkit, giving Marqov two
capabilities its other backends don't cover:

- **GPU statevector simulation** via the ``nvidia`` target (scales past what the
  CPU ``LocalExecutor`` / cloud ``SV1`` reach), and
- **Direct IQM Resonance** via the ``iqm`` target (a lower-latency route than
  IQM-through-Braket).

The CPU target (``qpp-cpu``) needs no GPU and is the default, so this executor
is useful even without accelerator hardware.

Backend slugs handled by the factory:
    - ``cudaq-cpu``  -> CUDA-Q ``qpp-cpu`` target (CPU statevector)
    - ``cudaq-gpu``  -> CUDA-Q ``nvidia`` target (GPU statevector; CPU fallback)
    - ``cudaq-iqm``  -> CUDA-Q ``iqm`` target (direct IQM Resonance QPU)

Example:
    >>> from marqov.circuits import bell_state
    >>> from marqov.executors import CudaqExecutor, CudaqExecutorConfig
    >>>
    >>> executor = CudaqExecutor(CudaqExecutorConfig(target="qpp-cpu"))
    >>> result = await executor.execute(bell_state(), shots=1000)
    >>> print(result.counts)  # {"00": ~500, "11": ~500}

Note:
    ``cudaq`` ships Linux-only wheels (no macOS build). Install the backend with
    ``pip install marqov[cudaq]`` on Linux, or run inside the official
    ``nvcr.io/nvidia/quantum/cuda-quantum`` container.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from marqov.executors.base import BaseExecutor, DeviceStatus, ExecutionResult

if TYPE_CHECKING:
    from marqov.circuits import Circuit


# Targets that never need remote credentials / can run without a GPU present.
_LOCAL_TARGETS = frozenset({"qpp-cpu", "nvidia", "nvidia-fp64", "nvidia-mgpu"})

# Slug -> CUDA-Q target name.
_SLUG_TO_TARGET: dict[str, str] = {
    "cudaq-cpu": "qpp-cpu",
    "cudaq-gpu": "nvidia",
    "cudaq-iqm": "iqm",
}


class _KernelBuilder(Protocol):
    """Structural type for the object returned by ``cudaq.make_kernel()``.

    Declaring only the methods we call lets the gate-mapping logic be unit
    tested with a recording double, so the transpiler is verifiable without a
    CUDA-Q install (which is Linux-only).
    """

    def qalloc(self, count: int) -> Any: ...
    def h(self, q: Any) -> None: ...
    def x(self, q: Any) -> None: ...
    def y(self, q: Any) -> None: ...
    def z(self, q: Any) -> None: ...
    def s(self, q: Any) -> None: ...
    def t(self, q: Any) -> None: ...
    def rx(self, angle: float, q: Any) -> None: ...
    def ry(self, angle: float, q: Any) -> None: ...
    def rz(self, angle: float, q: Any) -> None: ...
    def cx(self, ctrl: Any, tgt: Any) -> None: ...
    def cz(self, ctrl: Any, tgt: Any) -> None: ...
    def swap(self, a: Any, b: Any) -> None: ...
    def mz(self, q: Any) -> None: ...


@dataclass
class CudaqExecutorConfig:
    """Configuration for the CUDA-Q executor.

    Attributes:
        target: CUDA-Q target name — ``qpp-cpu`` (CPU), ``nvidia`` (GPU), or
            ``iqm`` (IQM Resonance QPU).
        iqm_url: IQM server URL (required when ``target == "iqm"``).
        iqm_token: IQM API token. Falls back to the ``IQM_TOKEN`` environment
            variable when not set.
        gpu_fallback_to_cpu: If a GPU target is requested but no GPU is
            available, fall back to ``qpp-cpu`` instead of raising. Matches the
            behaviour of the reference tutorial notebooks.
        seed: Optional CUDA-Q random seed for reproducible sampling.
        target_options: Extra keyword args forwarded verbatim to
            ``cudaq.set_target`` (e.g. IQM-specific options).
    """

    target: str = "qpp-cpu"
    iqm_url: str | None = None
    iqm_token: str | None = None
    gpu_fallback_to_cpu: bool = True
    seed: int | None = None
    target_options: dict[str, Any] = field(default_factory=dict)


def _import_cudaq() -> Any:
    """Import ``cudaq`` with an actionable error if it isn't installed."""
    try:
        import cudaq  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "CUDA-Q is required for CudaqExecutor. Install with: "
            "pip install marqov[cudaq]\n"
            "Note: cudaq ships Linux-only wheels — there is no macOS build. "
            "On macOS, run inside the nvcr.io/nvidia/quantum/cuda-quantum container."
        ) from exc
    return cudaq


def build_kernel(builder: _KernelBuilder, circuit: Circuit) -> Any:
    """Populate a CUDA-Q kernel from a Marqov circuit and return the qubit register.

    Transpiles via Qiskit (a stable, well-documented instruction API) and maps
    each instruction onto a CUDA-Q builder call. Kept free of any ``cudaq``
    import so it can be unit tested with a recording ``builder`` double.

    Args:
        builder: A ``cudaq.make_kernel()`` result (or a structural stand-in).
        circuit: The Marqov circuit to transpile.

    Returns:
        The allocated qubit register (``builder.qalloc(n)``), with a terminal
        ``mz`` applied for measurement.

    Raises:
        NotImplementedError: If the circuit contains a gate with no CUDA-Q
            builder equivalent.
    """
    qiskit_circuit = circuit.to_qiskit()
    num_qubits = qiskit_circuit.num_qubits
    qubits = builder.qalloc(num_qubits)

    # name -> (arity, is_parametric); mapped onto builder methods below.
    for instruction in qiskit_circuit.data:
        op = instruction.operation
        name = op.name.lower()
        qargs = [qiskit_circuit.find_bit(q).index for q in instruction.qubits]
        params = [float(p) for p in op.params]

        if name in ("h", "x", "y", "z", "s", "t"):
            getattr(builder, name)(qubits[qargs[0]])
        elif name in ("rx", "ry", "rz"):
            getattr(builder, name)(params[0], qubits[qargs[0]])
        elif name in ("cx", "cnot"):
            builder.cx(qubits[qargs[0]], qubits[qargs[1]])
        elif name == "cz":
            builder.cz(qubits[qargs[0]], qubits[qargs[1]])
        elif name == "swap":
            builder.swap(qubits[qargs[0]], qubits[qargs[1]])
        elif name in ("measure", "barrier"):
            # Measurement is applied uniformly below; barriers are no-ops.
            continue
        else:
            raise NotImplementedError(
                f"Gate '{name}' has no CUDA-Q builder mapping. Supported: "
                f"h, x, y, z, s, t, rx, ry, rz, cx, cz, swap."
            )

    builder.mz(qubits)
    return qubits


class CudaqExecutor(BaseExecutor):
    """Execute circuits on NVIDIA CUDA-Q (CPU/GPU statevector or direct IQM).

    Example:
        >>> executor = CudaqExecutor(CudaqExecutorConfig(target="nvidia"))
        >>> result = await executor.execute(circuit, shots=2000)
    """

    def __init__(self, config: CudaqExecutorConfig | None = None) -> None:
        """Initialize the CUDA-Q executor.

        Args:
            config: Configuration options. Defaults to the CPU (``qpp-cpu``)
                target when not provided.
        """
        self.config = config or CudaqExecutorConfig()

    async def execute(
        self,
        circuit: Circuit,
        shots: int = 1000,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Run a circuit on the configured CUDA-Q target.

        Args:
            circuit: The circuit to execute.
            shots: Number of measurement shots.
            **kwargs: Ignored; present for interface compatibility.

        Returns:
            ExecutionResult with normalized measurement counts.
        """
        circuit = self._validate_circuit(circuit)
        cudaq = _import_cudaq()

        start_time = time.perf_counter()
        # CUDA-Q's set_target / sample are synchronous and CPU/GPU-bound; run off
        # the event loop so concurrent executes don't block each other.
        counts, resolved_target = await asyncio.to_thread(
            self._run_sync, cudaq, circuit, shots
        )
        execution_time_ms = (time.perf_counter() - start_time) * 1000

        return ExecutionResult(
            counts=counts,
            backend=f"cudaq:{resolved_target}",
            execution_time_ms=execution_time_ms,
            shots=shots,
            raw_result=None,
            metadata={
                "framework": "cudaq",
                "target": resolved_target,
                "requested_target": self.config.target,
            },
        )

    def _run_sync(
        self, cudaq: Any, circuit: Circuit, shots: int
    ) -> tuple[dict[str, int], str]:
        """Synchronous CUDA-Q execution (set target, build kernel, sample)."""
        resolved_target = self._select_target(cudaq)

        if self.config.seed is not None:
            cudaq.set_random_seed(self.config.seed)

        kernel = cudaq.make_kernel()
        build_kernel(kernel, circuit)

        sample_result = cudaq.sample(kernel, shots_count=shots)
        counts = {str(bitstring): int(count) for bitstring, count in sample_result.items()}
        return counts, resolved_target

    def _select_target(self, cudaq: Any) -> str:
        """Set the CUDA-Q target, applying the requested config and GPU fallback.

        Returns:
            The target name actually set (may differ from the request when a GPU
            target falls back to CPU).
        """
        target = self.config.target

        if target == "iqm":
            if not self.config.iqm_url:
                raise ValueError("CudaqExecutorConfig.iqm_url is required for the 'iqm' target.")
            options = dict(self.config.target_options)
            if self.config.iqm_token is not None:
                import os

                os.environ.setdefault("IQM_TOKEN", self.config.iqm_token)
            cudaq.set_target("iqm", url=self.config.iqm_url, **options)
            return "iqm"

        # GPU targets: optionally fall back to CPU when no GPU is present.
        if target.startswith("nvidia") and self.config.gpu_fallback_to_cpu:
            if cudaq.num_available_gpus() == 0:
                cudaq.set_target("qpp-cpu")
                return "qpp-cpu"

        cudaq.set_target(target, **self.config.target_options)
        return target

    async def get_status(self) -> DeviceStatus:
        """Local CUDA-Q targets are always available; IQM status is not polled here."""
        if self.config.target in _LOCAL_TARGETS:
            return DeviceStatus.always_online()
        return DeviceStatus(status="online", queue_depth=None, queue_time_seconds=None)
