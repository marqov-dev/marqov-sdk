"""Quantinuum executor for running circuits on Quantinuum devices.

This module provides QuantinuumExecutor for executing quantum circuits on Quantinuum devices.

"""

from __future__ import annotations

import asyncio
import time
from functools import partial
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

from marqov.executors.base import BaseExecutor, DeviceStatus, ExecutionResult

if TYPE_CHECKING:
    from marqov.circuits import Circuit
    from pytket import Circuit as TketCircuit
    from pytket.extensions.quantinuum.backends.api_wrappers import QuantinuumAPI
    from pytket.extensions.quantinuum import QuantinuumBackend
    from pytket.extensions.quantinuum.backends.quantinuum import (
        QuantinuumBackendCompilationConfig,
    )

@dataclass
class QuantinuumExecutorConfig:
    """Configuration for Quantinuum executor.

    Attributes: 
        device_name: Name of the Quantinuum device.
        label: Label for the job.
        simulator: Simulator type.
        group: Group for the job.
        provider: Provider for the job.
        machine_debug: Whether to enable machine debug.
        api_handler: API handler for the job.
        compilation_config: Compilation configuration for the job.
        options: Options for the job.
        poll_interval_seconds: Polling interval for the job.
        timeout_seconds: Timeout for the job.
        optimisation_level: Optimisation level for the job.
    """

    device_name: str
    label: str = "job"  # not str | None — pytket expects str
    simulator: str = "state-vector"
    group: str | None = None
    provider: str | None = None
    machine_debug: bool = False
    api_handler: QuantinuumAPI | None = None  # optional, use default
    compilation_config: QuantinuumBackendCompilationConfig | None = None
    options: dict[str, Any] = field(default_factory=dict)

    poll_interval_seconds: float = 2.0
    timeout_seconds: float | None = 300.0
    optimisation_level: int = 2


class QuantinuumExecutor(BaseExecutor):
    """Execute circuits on Quantinuum devices.

    Supports both state-vector, stabilizer.

    Example:
        >>> config = QuantinuumExecutorConfig(
        ...     device_name="H2-1",
        ...     simulator="state-vector",
        ... )
        >>> executor = QuantinuumExecutor(config)
        >>> result = await executor.execute(circuit, shots=1000)
        >>> print(result.counts)  # {"00": ~500, "11": ~500}
    """

    def __init__(self, config: QuantinuumExecutorConfig) -> None:
        """Initialize QuantinuumExecutor.

        Args:
            config: Executur configuration including device settings.
        """
        self.config = config
        self._api_handler = self.config.api_handler 
        self._backend: QuantinuumBackend | None = None
        self._current_job_id: str | None = None

    def _get_backend_sync(self) -> QuantinuumBackend:
        """Get or create the Quantinuum backend (synchronous).

        Returns:
            QuantinuumBackend instance.
        """
        from pytket.extensions.quantinuum import QuantinuumBackend
        from pytket.extensions.quantinuum.backends.api_wrappers import QuantinuumAPI
        
        if self._backend is None:
            api_handler = self._api_handler or QuantinuumAPI()
            self._backend = QuantinuumBackend(
                device_name=self.config.device_name,
                label=self.config.label,
                simulator=self.config.simulator,
                group=self.config.group,
                provider=self.config.provider,
                machine_debug=self.config.machine_debug,
                api_handler=api_handler,
                compilation_config=self.config.compilation_config,
                **self.config.options,
            )
        return self._backend

    async def _get_backend(self) -> QuantinuumBackend:
        """Get or create the Quantinuum backend (async wrapper).

        Returns:
            QuantinuumBackend instance.
        """

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self._get_backend_sync))

    def _run_sync(
        self,
        backend: QuantinuumBackend,
        tket_circuit: TketCircuit,
        shots: int,
        **kwargs: Any,
    ) -> tuple[Any, str]:
        """Compile, submit, and retrieve results (synchronous)."""
        compiled = backend.get_compiled_circuit(
            tket_circuit, optimisation_level=self.config.optimisation_level
        )
        from pytket.extensions.quantinuum import QuantinuumBackend

        handle = backend.process_circuit(compiled, n_shots=shots, **kwargs)
        job_id = QuantinuumBackend.get_jobid(handle)
        get_result_kwargs: dict[str, Any] = {}
        if self.config.timeout_seconds is not None:
            get_result_kwargs["timeout"] = self.config.timeout_seconds
        result = backend.get_result(handle, **get_result_kwargs)
        return result, job_id

    @staticmethod
    def _pytket_counts_to_bitstring(counts_dict: Mapping[Any, int]) -> dict[str, int]:
        """Convert tuples or bitstrings to a dictionary of bitstrings.
        Args:
            counts_dict: Dictionary of tuples to convert.
        Returns:
            Dictionary of bitstrings to counts.
        """     
        return {
            key if isinstance(key, str) else "".join(str(b) for b in key): count
            for key, count in counts_dict.items()
        }

        

    async def execute(
        self,
        circuit: Circuit,
        shots: int = 1000,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute a circuit on the Quantinuum backend.

        Args:
            circuit: The circuit to execute.
            shots: Number of measurement shots.
            **kwargs: Additional backend-specific options.

        Returns:
            ExecutionResult with measurement counts and metadata.
        """
        circuit = self._validate_circuit(circuit)
        loop = asyncio.get_running_loop()
        start_time = time.perf_counter()

        backend = await self._get_backend()
        tket_circuit = circuit.to_pytket()
        result, job_id = await loop.run_in_executor(
            None, self._run_sync, backend, tket_circuit, shots, **kwargs
        )
        wall_time = time.perf_counter() - start_time
        self._current_job_id = job_id
        counts = self._pytket_counts_to_bitstring(result.get_counts())
        return ExecutionResult(
            backend=self.config.device_name,
            counts=counts,
            execution_time_ms=wall_time * 1000,
            shots=shots,
            raw_result=result,
            metadata={
                "job_id": job_id,
                "device_name": self.config.device_name,
                "wall_time_ms": wall_time * 1000,
            },
        )

    _QUANTINUUM_STATUS_MAP = {
        "online": "online",
        "available": "online",
        "active": "online",
        "offline": "offline",
        "unavailable": "offline",
        "retired": "offline",
    }

    def _get_device_state_sync(self) -> str:
        """Get raw device state from Quantinuum API."""
        from pytket.extensions.quantinuum import QuantinuumBackend
        from pytket.extensions.quantinuum.backends.api_wrappers import QuantinuumAPI

        api_handler = self._api_handler or QuantinuumAPI()
        return QuantinuumBackend.device_state(self.config.device_name, api_handler=api_handler)


    async def get_device_status(self) -> str:
        """Get current device status.

        Returns:
            Device status string (e.g., "online", "offline").
        """
        
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_device_state_sync)

    async def is_device_available(self) -> bool:
        """Check if device is available for execution.

        Returns:
            True if device is online, False otherwise.
        """
        status = await self.get_device_status()
        return status.lower() == "online"

    async def get_status(self) -> DeviceStatus:
        """Get live device status from Quantinuum."""
        try:
            raw = await self.get_device_status()
            status = self._QUANTINUUM_STATUS_MAP.get(raw.lower(), "maintenance")
            return DeviceStatus(
                status=status, queue_depth=None, queue_time_seconds=None
            )
        except Exception:
            return DeviceStatus(
                status="maintenance", queue_depth=None, queue_time_seconds=None
            )
