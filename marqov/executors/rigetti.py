"""Rigetti QCS executor for running circuits on Rigetti QPUs and the local QVM.

This module provides ``RigettiExecutor`` for executing quantum circuits through
Rigetti's stack. Circuits are converted with ``circuit.to_pyquil()`` (already
implemented in :mod:`marqov.circuits`), measurements are added for every qubit,
the program is compiled with ``quilc`` and run on a ``QuantumComputer`` obtained
from ``pyquil.get_qc``.

The same code path targets two backends, selected by ``quantum_processor_id``:

- A **local QVM** (a name ending in ``-qvm`` such as ``"2q-qvm"``). Rigetti's
  Quantum Virtual Machine is fully open source and runs locally via Docker, so it
  needs no cloud account or credits. This is the path the tests exercise.
- A **real QCS QPU** (e.g. ``"Ankaa-3"``). The executor submits through the
  native QPU interface, retains its response and waits for results separately.
  Interrupted result waits request best-effort cancellation.

Example:
    >>> from marqov.circuits import bell_state
    >>> from marqov.executors import RigettiExecutor, RigettiExecutorConfig
    >>>
    >>> config = RigettiExecutorConfig(quantum_processor_id="2q-qvm")
    >>> executor = RigettiExecutor(config)
    >>> result = await executor.execute(bell_state(), shots=1000)
    >>> print(result.counts)  # {"00": ~500, "11": ~500}

``pyquil`` is an optional dependency. Install it with ``pip install marqov[rigetti]``
and follow ``CONTRIBUTING.md`` §4 to start the ``quilc`` and ``qvm`` containers
before running against a local QVM.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import Counter, OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, replace
from threading import Lock
from typing import TYPE_CHECKING, Any

from marqov.executors._blocking import BlockingCalls
from marqov.executors.base import BaseExecutor, DeviceStatus, ExecutionResult

if TYPE_CHECKING:
    from marqov.circuits import Circuit


@dataclass
class RigettiExecutorConfig:
    """Configuration for the Rigetti QCS executor.

    Attributes:
        quantum_processor_id: The pyquil quantum-computer name. Use a QVM name
            ending in ``-qvm`` (e.g. ``"2q-qvm"``, ``"9q-square-qvm"``) for local
            simulation, or a real QCS QPU id (e.g. ``"Ankaa-3"``) for hardware.
        as_qvm: Force QVM execution (``True``) or QPU execution (``False``). When
            ``None``, pyquil infers it from ``quantum_processor_id`` (names ending
            in ``-qvm`` run on the QVM).
        compiler_timeout_seconds: Timeout passed to ``quilc`` when compiling.
        execution_timeout_seconds: Per-job execution timeout passed to pyquil.
        timeout_seconds: Optional overall wall-clock budget for ``execute`` around
            compilation and execution. ``None`` means no extra cap (the pyquil
            timeouts above still apply).
    """

    quantum_processor_id: str = "2q-qvm"
    as_qvm: bool | None = None
    compiler_timeout_seconds: float = 30.0
    execution_timeout_seconds: float = 30.0
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.timeout_seconds is not None and (
            type(self.timeout_seconds) not in (int, float)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")


class RigettiExecutor(BaseExecutor):
    """Execute circuits on Rigetti QPUs or the local QVM via pyquil.

    Converts a circuit with ``to_pyquil()``, measures every qubit into a ``ro``
    register, compiles with ``quilc`` and runs the program on a pyquil
    ``QuantumComputer``. Counts use the SDK convention where qubit 0 is the
    leftmost bit, matching :class:`~marqov.executors.local.LocalExecutor`.

    A ``QuantumComputer`` may be injected for unit testing without a running QVM
    or QCS credentials.

    Example:
        >>> config = RigettiExecutorConfig(quantum_processor_id="Ankaa-3")
        >>> executor = RigettiExecutor(config)
        >>> if (await executor.get_status()).status == "online":
        ...     result = await executor.execute(circuit, shots=1000)
    """

    def __init__(
        self,
        config: RigettiExecutorConfig,
        *,
        qc: Any = None,
        list_processors: Callable[[], Any] | None = None,
    ) -> None:
        """Initialize RigettiExecutor.

        Args:
            config: Executor configuration including the target processor id.
            qc: Optional pyquil ``QuantumComputer`` (or a compatible test double
                exposing ``compile(program)`` and ``run(executable)``). When
                ``None``, it is created lazily on first use via ``pyquil.get_qc``.
            list_processors: Optional zero-argument callable returning the live
                list of available QCS quantum-processor ids, used by
                ``get_status`` for real QPUs. When ``None``, it defaults to
                ``qcs_sdk.qpu.list_quantum_processors``. Injecting it keeps
                ``get_status`` testable without QCS credentials.
        """
        self.config = config
        self._connection_config = replace(config)
        self._active_jobs: dict[str, tuple[Any, Any]] = {}
        self._interrupted_jobs: OrderedDict[str, tuple[Any, Any]] = OrderedDict()
        self._jobs_lock = Lock()
        self._qc = qc
        self._list_processors = list_processors

    def _is_qvm(self) -> bool:
        """Return whether this executor targets a local QVM.

        Returns:
            ``config.as_qvm`` when set, otherwise inferred from the processor id
            (pyquil treats names ending in ``-qvm`` as QVMs).
        """
        if self.config.as_qvm is not None:
            return self.config.as_qvm
        return self.config.quantum_processor_id.endswith("-qvm")

    def _get_qc_sync(self) -> Any:
        """Get or lazily create the pyquil ``QuantumComputer`` (synchronous).

        Returns:
            The cached or newly created ``QuantumComputer``.
        """
        if self._qc is None:
            from pyquil import get_qc

            self._qc = get_qc(
                self._connection_config.quantum_processor_id,
                as_qvm=self._connection_config.as_qvm,
                compiler_timeout=self._connection_config.compiler_timeout_seconds,
                execution_timeout=self._connection_config.execution_timeout_seconds,
            )
        return self._qc

    @staticmethod
    def _build_measured_program(program: Any, num_qubits: int, shots: int) -> Any:
        """Add a readout register, measurements and a shot loop to a program.

        ``Circuit.to_pyquil()`` returns the gate sequence only, so the executor
        appends a ``MEASURE`` of every qubit into a fresh ``ro`` register and wraps
        the whole program in a ``shots`` loop, ready to compile and run.

        Args:
            program: The pyquil ``Program`` produced by ``to_pyquil()``.
            num_qubits: Number of qubits to measure (qubit ``i`` -> ``ro[i]``).
            shots: Number of shots to run.

        Returns:
            A new pyquil ``Program`` with declarations, gates, measurements and
            the shot loop.
        """
        from pyquil import Program
        from pyquil.gates import MEASURE

        measured = Program()
        ro = measured.declare("ro", "BIT", num_qubits)
        measured += program
        for qubit in range(num_qubits):
            measured += MEASURE(qubit, ro[qubit])
        measured.wrap_in_numshots_loop(shots)
        return measured

    @staticmethod
    def _result_to_counts(result: Any, num_qubits: int) -> dict[str, int]:
        """Convert a pyquil execution result into measurement counts.

        Reads the ``ro`` register (a ``shots`` x ``num_qubits`` array of bits) and
        bins the per-shot bitstrings. Qubit 0 is the leftmost character, matching
        the SDK's :class:`~marqov.executors.local.LocalExecutor` convention.

        Args:
            result: The object returned by ``QuantumComputer.run`` (exposes
                ``get_register_map()``).
            num_qubits: Number of measured qubits, used only as a guard.

        Returns:
            Mapping of bitstrings to integer counts. Empty when there is no
            readout data or no qubits.
        """
        if num_qubits == 0:
            return {}

        register_map = result.get_register_map()
        readout = register_map.get("ro")
        if readout is None:
            return {}

        counts: Counter[str] = Counter("".join(str(int(bit)) for bit in shot) for shot in readout)
        return dict(counts)

    async def execute(
        self,
        circuit: Circuit,
        shots: int = 1000,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute a circuit on a Rigetti QPU or the local QVM.

        Args:
            circuit: The quantum circuit to execute.
            shots: Number of measurement shots.
            **kwargs: Additional backend-specific options (currently unused).

        Returns:
            ExecutionResult with measurement counts and metadata.

        Raises:
            RuntimeError: If compilation or execution fails on the backend.
            TimeoutError: If execution exceeds ``config.timeout_seconds``.
        """
        circuit = self._validate_circuit(circuit)

        start_time = time.perf_counter()

        num_qubits = circuit.num_qubits

        # An empty circuit has nothing to measure; return early without touching
        # the backend so a missing QVM does not turn into a spurious failure.
        if num_qubits == 0:
            return ExecutionResult(
                counts={},
                backend=self.config.quantum_processor_id,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
                shots=shots,
                raw_result=None,
                metadata=self._metadata(0, shots, 0.0),
            )

        program = circuit.to_pyquil()  # type: ignore[no-untyped-call]
        measured = self._build_measured_program(program, num_qubits, shots)

        identity: dict[str, Any] = {
            "provider": "rigetti",
            "quantum_processor_id": self._connection_config.quantum_processor_id,
            "requested_quantum_processor_id": self._connection_config.quantum_processor_id,
            "as_qvm": self._connection_config.as_qvm
            if self._connection_config.as_qvm is not None
            else self._connection_config.quantum_processor_id.endswith("-qvm"),
            "job_id": None,
            "submission_status": "not_submitted",
            "phase": "initialization",
            "cancellation_supported": False,
        }
        try:
            with BlockingCalls() as calls:
                async with asyncio.timeout(self.config.timeout_seconds) as deadline:
                    qc = await calls.call(self._get_qc_sync)
                    identity["phase"] = "compilation"
                    executable = await calls.call(qc.compile, measured)
                    qam: Any = getattr(qc, "qam", None)
                    from pyquil.api import QPU

                    native_cancel = isinstance(qam, QPU)
                    identity["phase"] = "submission"
                    identity["submission_status"] = "unknown"
                    if native_cancel:
                        identity["cancellation_supported"] = True
                        identity["as_qvm"] = False
                        identity["quantum_processor_id"] = qam.quantum_processor_id
                        response = await calls.call(qam.execute, executable)
                        identity["job_id"] = response.job_id
                        identity["submission_status"] = "submitted"
                        identity["phase"] = "result"
                        with self._jobs_lock:
                            self._active_jobs[response.job_id] = (qam, response)
                        # Use the remaining overall budget for the result wait.
                        # Its TimeoutError must survive a second cancellation
                        # during best-effort cleanup.
                        expires = deadline.when()
                        remaining = (
                            None
                            if expires is None
                            else max(0.0, expires - asyncio.get_running_loop().time())
                        )
                        deadline.reschedule(None)
                        try:
                            result = await calls.wait(
                                lambda: qam.get_result(response),
                                lambda: qam.cancel(response),
                                job_id=response.job_id,
                                timeout=remaining,
                            )
                        except BaseException:
                            with self._jobs_lock:
                                self._interrupted_jobs[response.job_id] = (qam, response)
                                while len(self._interrupted_jobs) > 32:
                                    self._interrupted_jobs.popitem(last=False)
                            raise
                        finally:
                            with self._jobs_lock:
                                self._active_jobs.pop(response.job_id, None)
                    else:
                        # QVM and compatible injected run-only backends expose no remote handle.
                        identity["phase"] = "execution"
                        result = await calls.call(qc.run, executable)
            wall_time = time.perf_counter() - start_time
            counts = self._result_to_counts(result, num_qubits)
        except (TimeoutError, asyncio.CancelledError) as error:
            error.remote_job = dict(identity)  # type: ignore[union-attr]
            error.add_note(
                f"Rigetti execution interrupted in {identity['phase']}; native job "
                f"{identity['job_id']!r}, acceptance {identity['submission_status']}. "
                "Cancellation is best effort when a handle exists; provider state is unconfirmed."
            )
            raise
        except Exception as exc:
            failure = RuntimeError(
                f"Rigetti execution failed on {identity['quantum_processor_id']}: {exc}"
            )
            failure.remote_job = dict(identity)  # type: ignore[attr-defined]
            raise failure from exc

        return ExecutionResult(
            counts=counts,
            backend=identity["quantum_processor_id"],
            execution_time_ms=wall_time * 1000,
            shots=shots,
            raw_result=result,
            metadata={**self._metadata(num_qubits, shots, wall_time * 1000), **identity},
        )

    async def cancel(self, job_id: str) -> bool:
        """Request cancellation for an owned active or recently interrupted native job.

        QVM/run-only backends and IDs without a retained handle return
        False. The last 32 interrupted handles are retained; successes are removed.
        A successful request does not confirm terminal cancellation.
        """
        with self._jobs_lock:
            owned = self._active_jobs.get(job_id) or self._interrupted_jobs.get(job_id)
        if owned is None:
            return False
        qam, response = owned
        try:
            with BlockingCalls() as calls:
                await asyncio.wait_for(calls.call(qam.cancel, response), 1.0)
            return True
        except Exception:  # noqa: BLE001 - public best-effort cancellation contract
            return False

    def _metadata(self, num_qubits: int, shots: int, wall_time_ms: float) -> dict[str, Any]:
        """Build the ExecutionResult metadata for a run.

        Args:
            num_qubits: Number of qubits measured.
            shots: Number of shots executed.
            wall_time_ms: Measured wall time in milliseconds.

        Returns:
            Metadata dictionary.
        """
        return {
            "provider": "rigetti",
            "quantum_processor_id": self.config.quantum_processor_id,
            "as_qvm": self._is_qvm(),
            "num_qubits": num_qubits,
            "shots": shots,
            "wall_time_ms": wall_time_ms,
        }

    async def get_status(self) -> DeviceStatus:
        """Get live device availability.

        A local QVM is a simulator with no queue, so it always reports online. For
        a real QCS QPU the live list of available processors is queried and the
        target is reported online only when it appears in that list, offline
        otherwise. Any error degrades to ``maintenance`` so callers can back off.
        """
        if self._is_qvm():
            return DeviceStatus(status="online", queue_depth=0, queue_time_seconds=0)

        try:
            loop = asyncio.get_running_loop()
            processors = await loop.run_in_executor(None, self._list_processors_or_default)
            available = {str(processor) for processor in processors}
            status = "online" if self.config.quantum_processor_id in available else "offline"
            return DeviceStatus(status=status, queue_depth=None, queue_time_seconds=None)
        except Exception:
            return DeviceStatus(status="maintenance", queue_depth=None, queue_time_seconds=None)

    def _list_processors_or_default(self) -> Any:
        """Return the live QCS processor list via the injected or default source.

        Returns:
            Whatever the processor lister returns (iterable of processor ids).
        """
        if self._list_processors is not None:
            return self._list_processors()
        from qcs_sdk.qpu import list_quantum_processors

        return list_quantum_processors()
