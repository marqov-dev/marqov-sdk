"""Quantinuum executor using pytket-quantinuum.

This module keeps Quantinuum integration behind lazy imports so the core SDK
remains importable without pytket installed. Unit tests can mock the backend
without hardware credentials; real hardware smoke tests are expected post-merge.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any

from marqov.executors.base import BaseExecutor, DeviceStatus, ExecutionResult

if TYPE_CHECKING:
    from marqov.circuits import Circuit


@dataclass
class QuantinuumExecutorConfig:
    """Configuration for Quantinuum execution through pytket.

    Attributes:
        device_name: Quantinuum target name (for example, "H1-1E" or "H2-1E").
        username: Optional Quantinuum username. If omitted, pytket's configured
            credentials/environment are used.
        machine_debug: Use Quantinuum's emulator/debug mode when supported.
        optimisation_level: pytket compilation optimisation level.
        poll_interval_seconds: Delay between status polls.
        timeout_seconds: Maximum time to wait for a job. None waits indefinitely.
        backend_options: Extra keyword arguments passed to QuantinuumBackend.
    """

    device_name: str
    username: str | None = None
    machine_debug: bool = False
    optimisation_level: int = 2
    poll_interval_seconds: float = 1.0
    timeout_seconds: float | None = None
    backend_options: dict[str, Any] = field(default_factory=dict)


class QuantinuumExecutor(BaseExecutor):
    """Execute Marqov circuits on Quantinuum via pytket-quantinuum."""

    _ONLINE_STATUSES = {"ONLINE", "AVAILABLE", "ACTIVE", "ONLINE_IDLE"}
    _OFFLINE_STATUSES = {"OFFLINE", "UNAVAILABLE", "RETIRED"}
    _DONE_STATUSES = {"COMPLETED", "DONE", "SUCCESS"}
    _FAILED_STATUSES = {"CANCELLED", "CANCELED", "ERROR", "FAILED"}

    def __init__(self, config: QuantinuumExecutorConfig) -> None:
        self.config = config
        self._backend: Any | None = None
        self._current_handle: Any | None = None

    def _create_backend_sync(self):
        try:
            from pytket.extensions.quantinuum import QuantinuumBackend
        except ImportError:
            raise ImportError(
                "pytket-quantinuum is required for QuantinuumExecutor. "
                "Install with: pip install marqov[pytket,quantinuum]"
            )

        options = dict(self.config.backend_options)
        options.setdefault("machine_debug", self.config.machine_debug)
        if self.config.username is not None:
            options.setdefault("username", self.config.username)
        return QuantinuumBackend(self.config.device_name, **options)

    async def _get_backend(self):
        if self._backend is None:
            loop = asyncio.get_running_loop()
            self._backend = await loop.run_in_executor(None, self._create_backend_sync)
        return self._backend

    @staticmethod
    def _normalize_bitstring(key: Any) -> str:
        if isinstance(key, str):
            return key.replace(" ", "")
        if isinstance(key, (tuple, list)):
            return "".join(str(bit) for bit in key)
        readable = getattr(key, "to_readable", None)
        if callable(readable):
            return str(readable()).replace(" ", "")
        return str(key).replace(" ", "")

    @classmethod
    def _normalize_counts(cls, counts: Any) -> dict[str, int]:
        return {cls._normalize_bitstring(key): int(value) for key, value in dict(counts).items()}

    @staticmethod
    def _status_name(raw_status: Any) -> str:
        status = getattr(raw_status, "status", raw_status)
        name = getattr(status, "name", status)
        return str(name).upper()

    def _compile_circuit_sync(self, backend: Any, tk_circuit: Any) -> Any:
        if hasattr(backend, "get_compiled_circuit"):
            return backend.get_compiled_circuit(
                tk_circuit,
                optimisation_level=self.config.optimisation_level,
            )
        if hasattr(backend, "compile_circuit"):
            return backend.compile_circuit(
                tk_circuit,
                optimisation_level=self.config.optimisation_level,
            )
        return tk_circuit

    def _submit_sync(self, backend: Any, tk_circuit: Any, shots: int, kwargs: dict[str, Any]) -> Any:
        return backend.process_circuit(tk_circuit, n_shots=shots, **kwargs)

    def _wait_for_result_sync(self, backend: Any, handle: Any) -> Any:
        start = time.perf_counter()
        while True:
            if hasattr(backend, "circuit_status"):
                status_name = self._status_name(backend.circuit_status(handle))
                if status_name in self._FAILED_STATUSES:
                    raise RuntimeError(f"Quantinuum job failed with status {status_name}")
                if status_name not in self._DONE_STATUSES:
                    if (
                        self.config.timeout_seconds is not None
                        and time.perf_counter() - start > self.config.timeout_seconds
                    ):
                        raise TimeoutError(
                            f"Quantinuum job did not complete within "
                            f"{self.config.timeout_seconds} seconds"
                        )
                    time.sleep(self.config.poll_interval_seconds)
                    continue
            return backend.get_result(handle)

    async def execute(
        self,
        circuit: Circuit,
        shots: int = 1000,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute a circuit on Quantinuum and return normalized counts."""
        circuit = self._validate_circuit(circuit)

        loop = asyncio.get_running_loop()
        start_time = time.perf_counter()
        backend = await self._get_backend()

        tk_circuit = circuit.to_pytket()
        compiled = await loop.run_in_executor(None, self._compile_circuit_sync, backend, tk_circuit)
        handle = await loop.run_in_executor(
            None,
            partial(self._submit_sync, backend, compiled, shots, kwargs),
        )
        self._current_handle = handle
        result = await loop.run_in_executor(None, self._wait_for_result_sync, backend, handle)

        raw_counts = result.get_counts() if hasattr(result, "get_counts") else {}
        counts = self._normalize_counts(raw_counts)
        wall_time_ms = (time.perf_counter() - start_time) * 1000

        return ExecutionResult(
            counts=counts,
            backend=self.config.device_name,
            execution_time_ms=wall_time_ms,
            shots=shots,
            raw_result=result,
            metadata={
                "handle": handle,
                "device_name": self.config.device_name,
                "optimisation_level": self.config.optimisation_level,
                "wall_time_ms": wall_time_ms,
            },
        )

    async def cancel(self, job_id: str) -> bool:
        """Cancel a running Quantinuum job when pytket exposes cancellation."""
        try:
            backend = await self._get_backend()
            cancel = getattr(backend, "cancel", None) or getattr(backend, "cancel_circuit", None)
            if cancel is None:
                return False
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, cancel, job_id)
            return True
        except Exception:
            return False

    async def get_status(self) -> DeviceStatus:
        """Map Quantinuum backend availability to Marqov DeviceStatus."""
        try:
            backend = await self._get_backend()
            raw_status = None
            if hasattr(backend, "device_state"):
                raw_status = backend.device_state()
            elif hasattr(backend, "status"):
                raw_status = backend.status

            status_name = self._status_name(raw_status)
            if status_name in self._ONLINE_STATUSES:
                status = "online"
            elif status_name in self._OFFLINE_STATUSES:
                status = "offline"
            else:
                status = "maintenance"

            queue_depth = getattr(backend, "queue_depth", None)
            if callable(queue_depth):
                queue_depth = queue_depth()

            return DeviceStatus(status=status, queue_depth=queue_depth, queue_time_seconds=None)
        except Exception:
            return DeviceStatus(status="maintenance", queue_depth=None, queue_time_seconds=None)
