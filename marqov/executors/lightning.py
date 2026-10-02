"""PennyLane Lightning executor (CPU engines), with a per-run reproducibility record.

Runs Marqov circuits locally on Xanadu's PennyLane Lightning simulators:

- ``lightning.qubit`` (``pip install "marqov[pennylane]"``), and
- ``lightning.kokkos`` (``pip install "marqov[lightning-kokkos]"``; wheels for
  Linux x86_64/aarch64 and macOS arm64).

``lightning.gpu`` and ``lightning.tensor`` are not run by this version. Asking
for them constructs the device to find out whether this environment has it: if
construction fails, the error says so (``LightningDeviceUnavailableError``,
carrying PennyLane's own message); if it succeeds, the run is refused as not
validated (``NotImplementedError``). Availability is never read from a list of
names, and another device or simulator is never substituted.

Scope: bound gate circuits (the canonical gate set), finite shots,
computational-basis samples of every qubit the circuit uses. Count keys follow
the SDK convention: qubit indices in ascending order, lowest index leftmost
(the same keys ``LocalExecutor`` returns, including for sparse indices).

Seeds and replay: each run builds a fresh device and passes the seed at
construction, so no RNG state carries over between calls. A run with no seed
gets a generated one, which is recorded. Within one device, PennyLane and
Lightning versions and platform, the same seed replays the same samples.
Samples are not expected to match across devices (``lightning.qubit`` vs
``lightning.kokkos``) or across versions; seed behaviour is version-specific.

Example:
    >>> from marqov.circuits import Circuit
    >>> from marqov.executors import LightningExecutor, LightningExecutorConfig
    >>>
    >>> executor = LightningExecutor(LightningExecutorConfig(device="lightning.qubit"))
    >>> result = await executor.execute(Circuit().h(0).cnot(0, 1), shots=1000, seed=42)
    >>> result.counts                               # {"00": ~500, "11": ~500}
    >>> result.metadata["reproducibility"]["seed"]  # 42
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import importlib.metadata as md
import importlib.util
import json
import math
import os
import platform
import re
import secrets
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from marqov.executors.base import BaseExecutor, DeviceStatus, ExecutionResult

if TYPE_CHECKING:
    from marqov.circuits import Circuit

# Devices this executor runs. Everything else is probed and then refused.
_CPU_DEVICES = frozenset({"lightning.qubit", "lightning.kokkos"})

# Device -> (installed distribution, compiled module inside pennylane_lightning).
_PLUGINS: dict[str, tuple[str, str]] = {
    "lightning.qubit": ("pennylane-lightning", "lightning_qubit_ops"),
    "lightning.kokkos": ("pennylane-lightning-kokkos", "lightning_kokkos_ops"),
    "lightning.gpu": ("pennylane-lightning-gpu", "lightning_gpu_ops"),
    "lightning.tensor": ("pennylane-lightning-tensor", "lightning_tensor_ops"),
}

_PRECISIONS = {"double": "complex128", "single": "complex64"}

# numpy.random.default_rng accepts any non-negative int; cap at 63 bits so the
# value round-trips through JSON and signed 64-bit stores unchanged.
_MAX_SEED = 2**63 - 1

RNG_POLICY = "fresh device per run; seed applied at construction"
RECORD_VERSION = 1

_OMP_NOTE = (
    "Recorded as found. PennyLane does not read it; whether the compiled engine "
    "honours it depends on the build (no effect was observed on the macOS arm64 "
    "0.45 wheels)."
)


class LightningDeviceUnavailableError(RuntimeError):
    """The requested Lightning device cannot be constructed in this environment."""


@dataclass(frozen=True)
class LightningAvailability:
    """Outcome of actually constructing a Lightning device here.

    Attributes:
        device: The PennyLane device name probed.
        available: True if ``qml.device(device, wires=1)`` succeeded.
        reason: PennyLane's error when it did not, else None.
    """

    device: str
    available: bool
    reason: str | None = None


def probe_lightning_device(device: str) -> LightningAvailability:
    """Report whether ``device`` can be constructed in this environment.

    This is an environment check (it builds a one-wire device), not a lookup.

    Args:
        device: PennyLane device name, e.g. ``"lightning.gpu"``.

    Returns:
        LightningAvailability with PennyLane's error message when unavailable.
    """
    try:
        import pennylane as qml  # type: ignore[import-untyped]

        qml.device(device, wires=1)
    except Exception as exc:  # any construction failure means "not here"
        return LightningAvailability(device, False, f"{type(exc).__name__}: {exc}")
    return LightningAvailability(device, True)


@dataclass
class LightningExecutorConfig:
    """Configuration for the PennyLane Lightning executor.

    Attributes:
        device: PennyLane device name. ``"lightning.qubit"`` or
            ``"lightning.kokkos"`` run; ``"lightning.gpu"`` and
            ``"lightning.tensor"`` are probed and refused (see module docs).
        seed: Default sampling seed for calls that pass none. None means each
            such call gets a generated seed, recorded in the result.
        precision: ``"double"`` (complex128, default) or ``"single"`` (complex64).
        compute_provider: Where the simulation actually runs, recorded verbatim
            (e.g. ``"local"``, ``"aws-parallelcluster:eu-north-1"``).
    """

    device: str = "lightning.qubit"
    seed: int | None = None
    precision: Literal["double", "single"] = "double"
    compute_provider: str = "local"


def _import_pennylane() -> Any:
    try:
        import pennylane as qml
    except ImportError as exc:
        raise ImportError(
            'PennyLane Lightning is required for LightningExecutor. Install with: '
            'pip install "marqov[pennylane]"'
        ) from exc
    return qml


def _check_seed(seed: Any) -> int:
    if type(seed) is not int:
        raise TypeError(f"seed must be an integer or None, got {type(seed).__name__}")
    if not 0 <= seed <= _MAX_SEED:
        raise ValueError(f"seed must be between 0 and 2**63-1, got {seed}")
    return seed


def _json_scalar(value: Any) -> Any:
    # numpy scalars (float32, int64, ...) are not JSON-serialisable; hash their Python value.
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _sha256_json(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_scalar)
    return hashlib.sha256(text.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=None)
def _plugin_info(device: str) -> dict[str, Any]:
    """Installed-distribution facts for a device's plugin (cached per process)."""
    dist_name, module = _PLUGINS[device]
    info: dict[str, Any] = {
        "plugin_version": None,
        "wheel_tag": None,
        "binary": None,
        "binary_sha256": None,
    }
    try:
        dist = md.distribution(dist_name)
    except md.PackageNotFoundError:
        return info
    info["plugin_version"] = f"{dist_name}=={dist.version}"
    tags = [
        line.split(":", 1)[1].strip()
        for line in (dist.read_text("WHEEL") or "").splitlines()
        if line.startswith("Tag:")
    ]
    info["wheel_tag"] = tags[0] if tags else None
    spec = importlib.util.find_spec("pennylane_lightning")
    for root in (spec.submodule_search_locations or []) if spec else []:
        for path in sorted(Path(root).iterdir()):
            if path.name.startswith(module) and path.suffix in (".so", ".pyd"):
                info["binary"] = path.name
                info["binary_sha256"] = _sha256_file(path)
                break
    return info


def _cpu_name() -> str:
    try:
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            if out:
                return out
        elif sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor()


@lru_cache(maxsize=1)
def _host_info() -> dict[str, Any]:
    import numpy
    import scipy  # type: ignore[import-untyped]

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _cpu_name(),
        "cores": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
    }


def _backend_info(device: str) -> dict[str, Any]:
    """Compiled-engine facts. Call only after the device has executed.

    Kokkos aborts the process if its info functions run before Kokkos is
    initialised, so they are read only once ``kokkos_is_initialized()``.
    """
    info: dict[str, Any] = {}
    try:
        info["scipy_openblas32"] = md.version("scipy-openblas32")
    except md.PackageNotFoundError:
        info["scipy_openblas32"] = None
    try:
        ops = importlib.import_module(f"pennylane_lightning.{_PLUGINS[device][1]}")
    except Exception:
        return info
    if device == "lightning.kokkos":
        try:
            if not ops.kokkos_is_initialized():
                return info
            text = str(ops.print_configuration())
            version = re.search(r"Kokkos Version:\s*(\S+)", text)
            space = re.search(r"Default Device:\s*(\S+)", text)
            info["kokkos"] = {
                "version": version.group(1) if version else None,
                "default_execution_space": space.group(1) if space else None,
            }
        except Exception:
            pass
    for name in ("compile_info", "runtime_info"):
        try:
            info[name] = {str(k): v for k, v in dict(getattr(ops, name)()).items()}
        except Exception:
            pass
    return info


def _counts_from_samples(samples: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in samples:
        key = "".join("1" if bit else "0" for bit in row)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


class LightningExecutor(BaseExecutor):
    """Execute circuits on PennyLane Lightning CPU simulators.

    Every result carries ``metadata["reproducibility"]``, a JSON-serialisable
    record of what ran: circuit/input/output hashes, wire order, measurement,
    device and plugin versions with wheel tag and binary hash, precision,
    shots, seed and RNG policy, thread environment and host.

    Example:
        >>> executor = LightningExecutor()
        >>> result = await executor.execute(circuit, shots=1000, seed=7)
    """

    # Per-process qualification results, keyed by (device name, precision).
    _qualified: dict[tuple[str, str], dict[str, Any]] = {}
    _qualify_lock = threading.Lock()

    def __init__(self, config: LightningExecutorConfig | None = None) -> None:
        """Initialize the executor.

        Args:
            config: Configuration options. Defaults to ``lightning.qubit``.

        Raises:
            ValueError: If the device is not a Lightning device or the
                precision is unknown.
            TypeError, ValueError: If ``config.seed`` is not a valid seed.
        """
        self.config = config or LightningExecutorConfig()
        if not self.config.device.startswith("lightning."):
            raise ValueError(
                f"LightningExecutor runs PennyLane Lightning devices only "
                f"(lightning.*), got '{self.config.device}'."
            )
        if self.config.precision not in _PRECISIONS:
            raise ValueError(
                f"Unknown precision '{self.config.precision}'. Supported: 'double', 'single'."
            )
        if self.config.seed is not None:
            _check_seed(self.config.seed)

    async def execute(
        self,
        circuit: Circuit,
        shots: int = 1000,
        *,
        seed: int | None = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Sample a circuit on the configured Lightning device.

        Args:
            circuit: The circuit to execute. Must have at least one gate and
                only bound angles.
            shots: Number of samples; a positive integer.
            seed: Sampling seed for this call, an integer in ``[0, 2**63-1]``.
                Falls back to ``config.seed``, then to a generated seed. The
                seed used is always recorded. Replay holds for the same device,
                versions and platform only.
            **kwargs: Unsupported; any option raises TypeError.

        Returns:
            ExecutionResult with counts (lowest qubit index leftmost), the raw
            ``(shots, n_wires)`` sample array as ``raw_result``, and the record
            in ``metadata["reproducibility"]``.

        Raises:
            TypeError: For unknown options or non-integer shots/seed.
            ValueError: For non-positive shots, out-of-range seeds, an empty
                circuit or an unbound angle.
            NotImplementedError: For gates outside the canonical set, or a
                GPU/tensor engine that is installed but not validated here.
            LightningDeviceUnavailableError: If the device cannot be
                constructed here, or failed its qualification probe.
        """
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"LightningExecutor.execute() got unsupported options: {names}")
        if type(shots) is not int:
            raise TypeError(f"shots must be a positive integer, got {type(shots).__name__}")
        if shots <= 0:
            raise ValueError(f"shots must be a positive integer, got {shots}")

        if seed is not None:
            run_seed, seed_source = _check_seed(seed), "caller"
        elif self.config.seed is not None:
            run_seed, seed_source = self.config.seed, "config"
        else:
            run_seed, seed_source = secrets.randbits(32), "generated"

        circuit = self._validate_circuit(circuit)
        if circuit.num_qubits == 0:
            raise ValueError("LightningExecutor: the circuit has no gates, so nothing to measure.")
        script = circuit.to_pennylane()  # the one Circuit -> PennyLane boundary
        circuit_sha256 = _sha256_json(circuit.to_dict())

        qml = _import_pennylane()
        return await asyncio.to_thread(
            self._run_sync, qml, script, shots, run_seed, seed_source, circuit_sha256
        )

    def _make_device(self, qml: Any, wires: list[int], seed: int) -> Any:
        """Construct the requested device or explain precisely why not."""
        import numpy as np

        name = self.config.device
        dtype = np.complex128 if self.config.precision == "double" else np.complex64
        try:
            dev = qml.device(name, wires=wires, seed=seed, c_dtype=dtype)
        except Exception as exc:
            raise LightningDeviceUnavailableError(
                f"{name} is unavailable in this environment: {type(exc).__name__}: {exc}"
            ) from exc
        if name not in _CPU_DEVICES:
            raise NotImplementedError(
                f"{name} is installed here, but LightningExecutor has only been "
                f"validated on {', '.join(sorted(_CPU_DEVICES))}. Not running it."
            )
        if getattr(dev, "name", None) != name:
            raise RuntimeError(
                f"Requested {name} but PennyLane constructed {getattr(dev, 'name', dev)!r}; "
                "refusing to run on a substituted device."
            )
        return dev

    def _sample(
        self, qml: Any, operations: list[Any], wires: list[int], shots: int, seed: int
    ) -> tuple[Any, Any, Any, float]:
        import numpy as np

        dev = self._make_device(qml, wires, seed)
        tape = qml.tape.QuantumScript(operations, [qml.sample(wires=wires)], shots=shots)
        start = time.perf_counter()
        (raw,) = qml.execute([tape], dev, diff_method=None)
        elapsed_ms = (time.perf_counter() - start) * 1000
        samples = np.asarray(raw).astype(np.uint8).reshape(shots, len(wires))
        return dev, tape, samples, elapsed_ms

    def _ensure_qualified(self, qml: Any) -> dict[str, Any]:
        """Run once per process and precision: bit order and a known distribution on this device.

        Installed is not qualified: the device must reproduce the SDK's count
        keys for an asymmetric basis state, and a Bell state's distribution
        within 5 sigma, before any user circuit runs on it.
        """
        name = self.config.device
        key = (name, self.config.precision)
        with self._qualify_lock:
            if key in self._qualified:
                return self._qualified[key]
            checks: dict[str, bool] = {}
            _, _, samples, _ = self._sample(
                qml, [qml.PauliX(0), qml.CZ(wires=[0, 1])], [0, 1], 64, seed=0
            )
            checks["bit_order_q0_leftmost"] = _counts_from_samples(samples) == {"10": 64}
            shots = 4000
            _, _, samples, _ = self._sample(
                qml, [qml.Hadamard(0), qml.CNOT(wires=[0, 1])], [0, 1], shots, seed=0
            )
            counts = _counts_from_samples(samples)
            tolerance = 5 * math.sqrt(shots * 0.25)
            checks["bell_distribution"] = set(counts) <= {"00", "11"} and all(
                abs(counts.get(k, 0) - shots / 2) <= tolerance for k in ("00", "11")
            )
            result = {
                "passed": all(checks.values()),
                "checks": checks,
                "precision": "complex128" if self.config.precision == "double" else "complex64",
                "scope": "once per process and precision, before the first user circuit",
            }
            self._qualified[key] = result
            return result

    def _run_sync(
        self,
        qml: Any,
        script: Any,
        shots: int,
        seed: int,
        seed_source: str,
        circuit_sha256: str,
    ) -> ExecutionResult:
        import numpy as np

        name = self.config.device
        wires = sorted(int(w) for w in script.wires.tolist())

        qualification = self._ensure_qualified(qml)
        if not qualification["passed"]:
            raise LightningDeviceUnavailableError(
                f"{name} is installed but failed its qualification probe "
                f"{qualification['checks']}; not running user circuits on it."
            )

        dev, tape, samples, elapsed_ms = self._sample(
            qml, list(script.operations), wires, shots, seed
        )
        counts = _counts_from_samples(samples)
        executed_shots = tape.shots.total_shots
        measurements = [{"type": "sample", "basis": "computational", "wires": wires}]
        program = {
            "operations": [
                [op.name, [int(w) for w in op.wires], [float(p) for p in op.parameters]]
                for op in tape.operations
            ],
            "measurements": measurements,
            "shots": executed_shots,
        }
        samples_digest = hashlib.sha256(
            f"{samples.shape}".encode() + samples.tobytes()
        ).hexdigest()

        plugin = _plugin_info(name)
        record: dict[str, Any] = {
            "record_version": RECORD_VERSION,
            "pennylane_version": qml.version(),
            "device_requested": name,
            "device_name": dev.name,
            "device_class": f"{type(dev).__module__}.{type(dev).__qualname__}",
            "plugin_version": plugin["plugin_version"],
            "wheel_tag": plugin["wheel_tag"],
            "binary": plugin["binary"],
            "binary_sha256": plugin["binary_sha256"],
            "backend_info": _backend_info(name),
            "precision": np.dtype(dev.c_dtype).name,
            "n_wires": len(wires),
            "wire_order": wires,
            "bit_order": "bitstring position i is wire_order[i] (lowest index leftmost)",
            "measurements": measurements,
            "shots": executed_shots,
            "seed": seed,
            "seed_kind": "int",
            "seed_source": seed_source,
            "rng_policy": RNG_POLICY,
            "device_call_index": 0,
            "diff_method": None,
            "omp_threads_env": os.environ.get("OMP_NUM_THREADS"),
            "omp_threads_note": _OMP_NOTE,
            "qualification": {**qualification, "checks": dict(qualification["checks"])},
            "host": dict(_host_info()),
            "circuit_sha256": circuit_sha256,
            "input_sha256": _sha256_json(program),
            "samples_sha256": samples_digest,
            "counts_sha256": _sha256_json(counts),
        }

        return ExecutionResult(
            counts=counts,
            backend=name,
            execution_time_ms=elapsed_ms,
            shots=executed_shots,
            raw_result=samples,
            metadata={
                "vendor": "Xanadu",
                "framework": "PennyLane",
                "engine": name,
                "access_path": "local",
                "compute_provider": self.config.compute_provider,
                "seed": seed,
                "reproducibility": record,
            },
        )

    async def get_status(self) -> DeviceStatus:
        """Online if the configured device can be constructed here, else offline."""
        probe = await asyncio.to_thread(probe_lightning_device, self.config.device)
        if probe.available and self.config.device in _CPU_DEVICES:
            return DeviceStatus.always_online()
        return DeviceStatus(status="offline", queue_depth=None, queue_time_seconds=None)
