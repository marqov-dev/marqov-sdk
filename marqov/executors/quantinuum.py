"""Quantinuum executor for running circuits via pytket-quantinuum."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

from marqov.executors.base import BaseExecutor, DeviceStatus, ExecutionResult

if TYPE_CHECKING:
    from marqov.circuits import Circuit


@dataclass
class QuantinuumExecutorConfig:
    """Configuration for Quantinuum executor.

    Attributes:
        device_name: Quantinuum target name, e.g. "H2-1" or "H2-1E".
        label: Default job label used by pytket-quantinuum.
        simulator: Simulator mode for emulator devices.
        group: Optional Quantinuum job group identifier.
        provider: Optional federated authentication provider.
        machine_debug: Use pytket-quantinuum's debug/offline machine mode.
        optimisation_level: pytket compilation optimisation level.
        timeout_seconds: Maximum time to wait for result retrieval.
        api_handler: Optional pytket-quantinuum API handler for tests/offline use.
        options: Optional Quantinuum request options.
    """

    device_name: str
    label: str = "marqov"
    simulator: str = "state-vector"
    group: str | None = None
    provider: str | None = None
    machine_debug: bool = False
    optimisation_level: int = 2
    timeout_seconds: float | None = None
    api_handler: Any | None = None
    options: dict[str, Any] | None = None


class QuantinuumExecutor(BaseExecutor):
    """Execute circuits on Quantinuum systems through pytket-quantinuum."""

    _STATUS_MAP = {
        "online": "online",
        "available": "online",
        "active": "online",
        "offline": "offline",
        "unavailable": "offline",
        "retired": "offline",
    }

    def __init__(self, config: QuantinuumExecutorConfig) -> None:
        """Initialize a Quantinuum executor."""
        self.config = config
        self._backend = None
        self._current_handle: str | None = None

    def _create_backend_sync(self):
        """Create a pytket-quantinuum backend."""
        try:
            from pytket.extensions.quantinuum import QuantinuumBackend
        except ImportError:
            raise ImportError(
                "pytket-quantinuum is required for QuantinuumExecutor. "
                "Install with: pip install marqov[quantinuum]"
            )

        kwargs: dict[str, Any] = {
            "device_name": self.config.device_name,
            "label": self.config.label,
            "simulator": self.config.simulator,
            "machine_debug": self.config.machine_debug,
        }
        if self.config.group is not None:
            kwargs["group"] = self.config.group
        if self.config.provider is not None:
            kwargs["provider"] = self.config.provider
        if self.config.api_handler is not None:
            kwargs["api_handler"] = self.config.api_handler
        if self.config.options is not None:
            kwargs["options"] = self.config.options

        return QuantinuumBackend(**kwargs)

    async def _get_backend(self):
        """Get or create the pytket-quantinuum backend."""
        if self._backend is None:
            loop = asyncio.get_running_loop()
            self._backend = await loop.run_in_executor(None, self._create_backend_sync)
        return self._backend

    async def execute(
        self,
        circuit: Circuit,
        shots: int = 1000,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute a circuit on Quantinuum via pytket-quantinuum."""
        circuit = self._validate_circuit(circuit)

        loop = asyncio.get_running_loop()
        start_time = time.perf_counter()
        backend = await self._get_backend()

        pytket_circuit = circuit.to_pytket()
        if getattr(pytket_circuit, "n_bits", 0) == 0:
            pytket_circuit.measure_all()

        compiled_circuit = await loop.run_in_executor(
            None,
            partial(
                backend.get_compiled_circuit,
                pytket_circuit,
                optimisation_level=self.config.optimisation_level,
            ),
        )
        handle = await loop.run_in_executor(
            None,
            partial(backend.process_circuit, compiled_circuit, n_shots=shots, **kwargs),
        )
        self._current_handle = str(handle)

        result_future = loop.run_in_executor(None, partial(backend.get_result, handle))
        if self.config.timeout_seconds is not None:
            raw_result = await asyncio.wait_for(result_future, timeout=self.config.timeout_seconds)
        else:
            raw_result = await result_future

        wall_time_ms = (time.perf_counter() - start_time) * 1000
        counts = self._normalise_counts(raw_result.get_counts())

        return ExecutionResult(
            counts=counts,
            backend=self.config.device_name,
            execution_time_ms=wall_time_ms,
            shots=shots,
            raw_result=raw_result,
            metadata={
                "handle": self._current_handle,
                "device_name": self.config.device_name,
                "optimisation_level": self.config.optimisation_level,
                "wall_time_ms": wall_time_ms,
            },
        )

    @staticmethod
    def _normalise_counts(raw_counts: dict[Any, Any]) -> dict[str, int]:
        """Convert pytket count keys into Marqov bitstring counts."""
        counts: dict[str, int] = {}
        for outcome, count in dict(raw_counts).items():
            if isinstance(outcome, str):
                bitstring = outcome
            else:
                values = outcome.tolist() if hasattr(outcome, "tolist") else outcome
                bitstring = "".join(str(int(bit)) for bit in values)
            counts[bitstring] = int(count)
        return counts

    async def cancel(self, job_id: str) -> bool:
        """Cancel a running Quantinuum job by ResultHandle string."""
        try:
            from pytket.backends import ResultHandle

            backend = await self._get_backend()
            handle = ResultHandle.from_str(job_id)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, partial(backend.cancel, handle))
            return True
        except Exception:
            return False

    async def get_device_status(self) -> str:
        """Get raw Quantinuum device state."""
        try:
            from pytket.extensions.quantinuum import QuantinuumBackend
        except ImportError:
            raise ImportError(
                "pytket-quantinuum is required for QuantinuumExecutor. "
                "Install with: pip install marqov[quantinuum]"
            )

        kwargs: dict[str, Any] = {}
        if self.config.api_handler is not None:
            kwargs["api_handler"] = self.config.api_handler

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            partial(QuantinuumBackend.device_state, self.config.device_name, **kwargs),
        )

    async def is_device_available(self) -> bool:
        """Check whether the configured Quantinuum device is online."""
        status = await self.get_status()
        return status.status == "online"

    async def get_status(self) -> DeviceStatus:
        """Map Quantinuum device state to the shared DeviceStatus shape."""
        try:
            raw_status = (await self.get_device_status()).lower()
            status = self._STATUS_MAP.get(raw_status, "maintenance")
            return DeviceStatus(status=status, queue_depth=None, queue_time_seconds=None)
        except Exception:
            return DeviceStatus(status="maintenance", queue_depth=None, queue_time_seconds=None)
