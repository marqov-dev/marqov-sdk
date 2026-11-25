"""Clifft executor: exact near-Clifford simulation, with a per-run reproducibility record.

Runs Marqov circuits locally on Clifft (Unitary Foundation), an exact
simulator for Clifford circuits with a small number of non-Clifford gates.
Install with ``pip install "marqov[clifft]"`` (also in ``[all]``); the pin is
one tested Clifft release because Clifft's API and seeded sampling change
between 0.x releases.

Clifft keeps a dense state only over the qubits the non-Clifford gates have
made "active". It pays off on wide circuits whose compiled peak active width
stays well below the qubit count. When that width approaches the qubit count,
or on small circuits, ``LocalExecutor`` or ``LightningExecutor`` is usually
faster. The peak active width of each run is in the record so the choice can
be checked.

Scope: bound gate circuits in the canonical gate set, finite shots,
computational-basis samples of every qubit below ``Circuit.num_qubits``
(identity wires included). Count keys follow the SDK
convention: qubit indices in ascending order, lowest index leftmost (the same
keys ``LocalExecutor`` returns). A ``Circuit`` carries no noise, detectors or
mid-circuit measurements, so this executor runs none; error-correction work
that needs them calls Clifft directly inside a ``@task``.

Seeds and replay: every run compiles a fresh program and passes the seed to
``clifft.sample``, so no RNG state carries over between calls. A run with no
seed gets a generated one, which is recorded. With the same Clifft version,
platform and circuit, the same seed and shot count replay the same samples.
Sampling always uses one thread and Clifft's automatic execution choice
(``batch_size="auto"``, ``sharing="auto"``). Clifft documents that rows for one
seed can differ between execution configurations, so the configuration Clifft
selected (scalar, packed or shared execution, and its batch size) is recorded
under ``execution``.

Example:
    >>> from marqov.circuits import Circuit
    >>> from marqov.executors.clifft import ClifftExecutor
    >>>
    >>> result = await ClifftExecutor().execute(Circuit().h(0).cnot(0, 1), shots=1000, seed=42)
    >>> result.counts                               # {"00": ~500, "11": ~500}
    >>> result.metadata["reproducibility"]["seed"]  # 42
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import platform
import secrets
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from marqov.circuits import Circuit
from marqov.executors.base import BaseExecutor, DeviceStatus, ExecutionResult, reject_unsupported_options

# The release the executor and its tests were qualified against. Kept equal to
# the pin in pyproject.toml; a different installed version refuses to run.
CLIFFT_VERSION = "0.14.0"

RECORD_VERSION = 1
RNG_POLICY = "fresh program per run; seed passed to clifft.sample"

# clifft.sample takes an unsigned 64-bit seed; cap at 63 bits so the recorded
# value round-trips through JSON and signed 64-bit stores unchanged.
_MAX_SEED = 2**63 - 1

# Fixed so a seed replays: Clifft's rows depend on the execution configuration,
# and with automatic batching on the thread count as well.
_THREADS = 1
_BATCH_SIZE = "auto"
_SHARING = "auto"

# QuantumFlow gate name -> Clifft (Stim-format) instruction.
_ONE_QUBIT = {"H": "H", "X": "X", "Y": "Y", "Z": "Z", "S": "S", "T": "T"}
_ROTATIONS = {"Rx": "R_X", "Ry": "R_Y", "Rz": "R_Z"}
_TWO_QUBIT = {"CNot": "CX", "CZ": "CZ", "Swap": "SWAP"}
SUPPORTED_GATES = frozenset(
    {*_ONE_QUBIT, *_ROTATIONS, *_TWO_QUBIT, "CCNot", "CSwap", "I"}
)


class ClifftUnavailableError(RuntimeError):
    """Clifft is not installed, or the installed version is not the qualified one."""


@dataclass
class ClifftExecutorConfig:
    """Configuration for the Clifft executor.

    Attributes:
        seed: Default sampling seed for calls that pass none. None means each
            such call gets a generated seed, recorded in the result.
    """

    seed: int | None = None


def _check_seed(seed: Any) -> int:
    if type(seed) is not int:
        raise TypeError(f"seed must be an integer or None, got {type(seed).__name__}")
    if not 0 <= seed <= _MAX_SEED:
        raise ValueError(f"seed must be between 0 and 2**63-1, got {seed}")
    return seed


def _import_clifft() -> Any:
    try:
        import clifft
    except ImportError as exc:
        raise ClifftUnavailableError(
            'Clifft is required for ClifftExecutor. Install with: pip install "marqov[clifft]"'
        ) from exc
    installed = getattr(clifft, "__version__", None)
    if installed != CLIFFT_VERSION:
        raise ClifftUnavailableError(
            f"ClifftExecutor is qualified against clifft {CLIFFT_VERSION}, but {installed} is "
            f'installed. Clifft\'s API changes between releases; install with: pip install "marqov[clifft]"'
        )
    return clifft


def to_clifft_text(circuit: Circuit) -> str:
    """Translate a circuit's gates into Clifft's Stim-format program text, without measurement.

    Rotations become ``R_X``/``R_Y``/``R_Z`` with the angle in units of pi
    (Clifft's convention); every gate keeps the same matrix up to global
    phase, which sampling cannot observe. Clifft has no controlled-SWAP, so
    ``CSwap c a b`` is emitted as ``CX b a; CCX c a b; CX b a``. Identity gates
    are dropped, and a final ``I`` on every qubit below ``circuit.num_qubits``
    keeps identity wires in the program's width.

    Args:
        circuit: The circuit to translate.

    Returns:
        Program text, one instruction per line.

    Raises:
        NotImplementedError: For a gate outside the mapped set.
        ValueError: For an unbound (symbolic) or non-real rotation angle.
    """
    lines: list[str] = []
    for index, gate in enumerate(circuit.to_dict()["gates"]):
        name, qubits, params = gate["gate"], [int(q) for q in gate["qubits"]], gate["params"]
        if name in _ONE_QUBIT:
            lines.append(f"{_ONE_QUBIT[name]} {qubits[0]}")
        elif name in _ROTATIONS:
            angle = Circuit._bound_angle(params[0], name, index, "ClifftExecutor")
            lines.append(f"{_ROTATIONS[name]}({angle / math.pi!r}) {qubits[0]}")
        elif name in _TWO_QUBIT:
            lines.append(f"{_TWO_QUBIT[name]} {qubits[0]} {qubits[1]}")
        elif name == "CCNot":
            lines.append(f"CCX {qubits[0]} {qubits[1]} {qubits[2]}")
        elif name == "CSwap":
            control, a, b = qubits
            lines += [f"CX {b} {a}", f"CCX {control} {a} {b}", f"CX {b} {a}"]
        elif name == "I":
            continue
        else:
            raise NotImplementedError(
                f"ClifftExecutor: no Clifft translation for gate '{name}' at position {index}. "
                f"Supported: {', '.join(sorted(SUPPORTED_GATES))}."
            )
    if circuit.num_qubits:
        lines.append("I " + " ".join(str(q) for q in range(circuit.num_qubits)))
    return "\n".join(lines)


def _sha256_json(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def _execution_record(execution: Any) -> dict[str, Any]:
    """The execution configuration Clifft selected, as JSON-friendly values."""
    fields = (
        "executor",
        "batch_size",
        "sharing",
        "shot_workers",
        "intra_shot_workers",
        "noise_preparation",
    )
    record: dict[str, Any] = {}
    for name in fields:
        value = getattr(execution, name, None)
        if value is not None and not isinstance(value, (bool, int, float, str)):
            value = str(value)
        record[name] = value
    return record


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


class ClifftExecutor(BaseExecutor):
    """Execute circuits on Clifft's exact near-Clifford sampler.

    Every result carries ``metadata["reproducibility"]``, a JSON-serialisable
    record of what ran: Clifft and NumPy versions, the CPU kernel Clifft
    selected, platform, the program text and hashes of the circuit, program,
    samples and counts, shots, seed and its source, the requested and the
    selected execution configuration, and the compiled peak active width.

    Example:
        >>> executor = ClifftExecutor()
        >>> result = await executor.execute(circuit, shots=1000, seed=7)
    """

    def __init__(self, config: ClifftExecutorConfig | None = None) -> None:
        """Initialize the executor.

        Args:
            config: Configuration options. Uses defaults if not provided.

        Raises:
            TypeError, ValueError: If ``config.seed`` is not a valid seed.
        """
        self.config = config or ClifftExecutorConfig()
        if self.config.seed is not None:
            _check_seed(self.config.seed)

    async def execute(
        self,
        circuit: Circuit,
        shots: int = 1000,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Sample every qubit of a circuit with Clifft.

        Args:
            circuit: The circuit to execute. Must have at least one qubit
                and only bound angles.
            shots: Number of samples; a positive integer.
            **kwargs: Only ``seed`` (an integer in ``[0, 2**63-1]``, or None)
                is accepted. It falls back to ``config.seed``, then to a
                generated seed; the seed used is recorded.

        Returns:
            ExecutionResult with counts (lowest qubit index leftmost), the raw
            ``(shots, num_qubits)`` uint8 sample array as ``raw_result``, and
            the record in ``metadata["reproducibility"]``.

        Raises:
            TypeError: For unknown options or non-integer shots/seed.
            ValueError: For non-positive shots, out-of-range seeds, a circuit
                with no qubits, or an unbound angle.
            NotImplementedError: For a gate the adapter cannot map.
            ClifftUnavailableError: If Clifft is missing or not the pinned version.
        """
        reject_unsupported_options("ClifftExecutor.execute()", kwargs, ("seed",))
        if type(shots) is not int:
            raise TypeError(f"shots must be a positive integer, got {type(shots).__name__}")
        if shots <= 0:
            raise ValueError(f"shots must be a positive integer, got {shots}")

        seed = kwargs.get("seed")
        if seed is not None:
            run_seed, seed_source = _check_seed(seed), "caller"
        elif self.config.seed is not None:
            run_seed, seed_source = self.config.seed, "config"
        else:
            run_seed, seed_source = secrets.randbits(32), "generated"

        circuit = self._validate_circuit(circuit)
        width = circuit.num_qubits
        if width == 0:
            raise ValueError("ClifftExecutor: the circuit has no qubits, so nothing to measure.")
        text = to_clifft_text(circuit) + "\nM " + " ".join(str(q) for q in range(width))
        circuit_sha256 = _sha256_json(circuit.to_dict())

        clifft = _import_clifft()
        return await asyncio.to_thread(
            self._run_sync, clifft, text, width, shots, run_seed, seed_source, circuit_sha256
        )

    def _run_sync(
        self,
        clifft: Any,
        text: str,
        width: int,
        shots: int,
        seed: int,
        seed_source: str,
        circuit_sha256: str,
    ) -> ExecutionResult:
        import numpy as np

        start = time.perf_counter()
        program = clifft.compile(text)
        compile_ms = (time.perf_counter() - start) * 1000
        result = clifft.sample(
            program,
            shots=shots,
            seed=seed,
            threads=_THREADS,
            batch_size=_BATCH_SIZE,
            sharing=_SHARING,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        samples = np.ascontiguousarray(result.measurements, dtype=np.uint8)
        if samples.shape != (shots, width):
            raise RuntimeError(
                f"Clifft returned samples of shape {samples.shape}, expected {(shots, width)}"
            )
        keys, frequencies = np.unique(samples, axis=0, return_counts=True)
        counts = {
            "".join("1" if bit else "0" for bit in key): int(n)
            for key, n in zip(keys, frequencies)
        }

        record: dict[str, Any] = {
            "record_version": RECORD_VERSION,
            "clifft_version": clifft.__version__,
            "clifft_runtime_isa": str(clifft.runtime_isa()),
            "numpy_version": np.__version__,
            "marqov_version": _package_version("marqov"),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "n_qubits": width,
            "bit_order": "bitstring position i is qubit i (lowest index leftmost)",
            "measurements": [
                {"type": "sample", "basis": "computational", "qubits": list(range(width))}
            ],
            "shots": shots,
            "seed": seed,
            "seed_source": seed_source,
            "rng_policy": RNG_POLICY,
            "threads": _THREADS,
            "batch_size": _BATCH_SIZE,
            "sharing": _SHARING,
            "execution": _execution_record(result.execution),
            "peak_active_width": int(program.peak_active_width),
            "program_text": text,
            "circuit_sha256": circuit_sha256,
            "program_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "samples_sha256": hashlib.sha256(
                f"{samples.shape}".encode() + samples.tobytes()
            ).hexdigest(),
            "counts_sha256": _sha256_json(dict(sorted(counts.items()))),
            "compile_ms": compile_ms,
        }

        return ExecutionResult(
            counts=counts,
            backend="clifft",
            execution_time_ms=elapsed_ms,
            shots=shots,
            raw_result=samples,
            metadata={
                "vendor": "Unitary Foundation",
                "engine": "clifft",
                "access_path": "local",
                "seed": seed,
                "reproducibility": record,
            },
        )

    async def get_status(self) -> DeviceStatus:
        """Online if the pinned Clifft version is importable here, else offline."""
        try:
            _import_clifft()
        except ClifftUnavailableError:
            return DeviceStatus(status="offline", queue_depth=None, queue_time_seconds=None)
        return DeviceStatus.always_online()
