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
- A **real QCS QPU** (e.g. ``"Ankaa-3"``). ``QuantumComputer.run`` submits to QCS
  and blocks until the job finishes, so the job lifecycle (queued, running,
  completed) is handled internally by pyquil.

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
import time
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

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
                self.config.quantum_processor_id,
                as_qvm=self.config.as_qvm,
                compiler_timeout=self.config.compiler_timeout_seconds,
                execution_timeout=self.config.execution_timeout_seconds,
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

        loop = asyncio.get_running_loop()
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

        def _compile_and_run() -> Any:
            qc = self._get_qc_sync()
            executable = qc.compile(measured)
            return qc.run(executable)

        try:
            coro = loop.run_in_executor(None, _compile_and_run)
            if self.config.timeout_seconds is not None:
                result = await asyncio.wait_for(coro, timeout=self.config.timeout_seconds)
            else:
                result = await coro
        except asyncio.TimeoutError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalised to RuntimeError below
            raise RuntimeError(
                f"Rigetti execution failed on {self.config.quantum_processor_id}: {exc}"
            ) from exc

        wall_time = time.perf_counter() - start_time
        counts = self._result_to_counts(result, num_qubits)

        return ExecutionResult(
            counts=counts,
            backend=self.config.quantum_processor_id,
            execution_time_ms=wall_time * 1000,
            shots=shots,
            raw_result=result,
            metadata=self._metadata(num_qubits, shots, wall_time * 1000),
        )

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
