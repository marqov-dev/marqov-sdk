"""IBM Quantum executor for running circuits on IBM Quantum Platform.

This module provides IBMExecutor for executing quantum circuits on IBM Quantum
processors via the Qiskit Runtime service, using the SamplerV2 primitive.

Supports IBM Quantum Open Plan (free tier) and paid plans. Key backends
include ibm_kingston (Heron r2, 133 qubits, 340k CLOPS).

Example:
    >>> from marqov.circuits import bell_state
    >>> from marqov.executors import IBMExecutor, IBMExecutorConfig
    >>>
    >>> config = IBMExecutorConfig(backend_name="ibm_kingston")
    >>> executor = IBMExecutor(config)
    >>> result = await executor.execute(bell_state(), shots=1000)
    >>> print(result.counts)  # {"00": ~500, "11": ~500}
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
class IBMExecutorConfig:
    """Configuration for IBM Quantum executor.

    Attributes:
        backend_name: IBM Quantum backend name (e.g., "ibm_kingston").
        channel: Service channel — "ibm_quantum" for Open/paid plans,
                 "ibm_cloud" for IBM Cloud Quantum instances.
        instance: IBM Quantum instance in "hub/group/project" format.
                  Defaults to "ibm-q/open/main" (Open Plan).
        token: IBM Quantum API token. If None, uses saved credentials
               from QiskitRuntimeService.save_account().
        optimization_level: Transpiler optimization level (0-3).
        resilience_level: Error mitigation level for Sampler (0-2).
        poll_interval_seconds: Polling interval for job completion.
        timeout_seconds: Maximum time to wait for job completion.
    """

    backend_name: str
    channel: str = "ibm_quantum"
    instance: str = "ibm-q/open/main"
    token: str | None = None
    optimization_level: int = 1
    resilience_level: int = 1
    poll_interval_seconds: float = 2.0
    timeout_seconds: float | None = None


class IBMExecutor(BaseExecutor):
    """Execute circuits on IBM Quantum backends via Qiskit Runtime.

    Uses the SamplerV2 primitive for circuit execution. Supports both
    the IBM Quantum Open Plan (free, up to 10-180 min runtime) and paid plans.

    Example:
        >>> config = IBMExecutorConfig(
        ...     backend_name="ibm_kingston",
        ...     token="your-ibm-quantum-token",
        ... )
        >>> executor = IBMExecutor(config)
        >>> if await executor.is_device_available():
        ...     result = await executor.execute(circuit, shots=1000)
    """

    def __init__(self, config: IBMExecutorConfig) -> None:
        self.config = config
        self._service = None
        self._backend = None
        self._current_job_id: str | None = None

    def _get_service_sync(self):
        """Get or create the QiskitRuntimeService (synchronous).

        Returns:
            QiskitRuntimeService instance.
        """
        if self._service is None:
            from qiskit_ibm_runtime import QiskitRuntimeService

            kwargs: dict[str, Any] = {
                "channel": self.config.channel,
                "instance": self.config.instance,
            }
            if self.config.token:
                kwargs["token"] = self.config.token

            self._service = QiskitRuntimeService(**kwargs)
        return self._service

    def _get_backend_sync(self):
        """Get or create the IBM backend (synchronous).

        Returns:
            IBMBackend instance for the configured backend name.
        """
        if self._backend is None:
            service = self._get_service_sync()
            self._backend = service.backend(self.config.backend_name)
        return self._backend

    async def _get_backend(self):
        """Get or create the IBM backend (async wrapper).

        Returns:
            IBMBackend instance.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_backend_sync)

    def _transpile_sync(self, qiskit_circuit, backend):
        """Transpile a circuit for the target backend (synchronous).

        Args:
            qiskit_circuit: Qiskit QuantumCircuit to transpile.
            backend: Target IBM backend.

        Returns:
            Transpiled QuantumCircuit.
        """
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        pm = generate_preset_pass_manager(
            optimization_level=self.config.optimization_level,
            backend=backend,
        )
        return pm.run(qiskit_circuit)

    def _run_sampler_sync(self, transpiled_circuit, backend, shots: int) -> Any:
        """Run the SamplerV2 primitive (synchronous).

        Args:
            transpiled_circuit: Transpiled circuit ready for execution.
            backend: Target IBM backend.
            shots: Number of measurement shots.

        Returns:
            SamplerV2 result object.
        """
        from qiskit_ibm_runtime import SamplerV2

        sampler = SamplerV2(mode=backend)
        job = sampler.run([transpiled_circuit], shots=shots)
        self._current_job_id = job.job_id()
        return job.result()

    @staticmethod
    def _extract_counts(result) -> dict[str, int]:
        """Extract measurement counts from a SamplerV2 result.

        Args:
            result: SamplerV2 PrimitiveResult.

        Returns:
            Dictionary mapping bitstrings to counts.
        """
        pub_result = result[0]
        data_bin = pub_result.data

        # SamplerV2 returns BitArray in classical register fields
        # Get the first classical register (typically "meas" or "c")
        creg_names = [attr for attr in dir(data_bin) if not attr.startswith("_")]
        if not creg_names:
            return {}

        bit_array = getattr(data_bin, creg_names[0])
        return dict(bit_array.get_counts())

    async def execute(
        self,
        circuit: Circuit,
        shots: int = 1000,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute a circuit on an IBM Quantum backend.

        Args:
            circuit: The quantum circuit to execute.
            shots: Number of measurement shots.
            **kwargs: Additional options (e.g., optimization_level override).

        Returns:
            ExecutionResult with measurement counts and metadata.

        Raises:
            RuntimeError: If job fails or backend is unavailable.
        """
        circuit = self._validate_circuit(circuit)

        loop = asyncio.get_running_loop()
        start_time = time.perf_counter()

        # Get backend (lazy initialization)
        backend = await self._get_backend()

        # Convert to Qiskit and add measurements if needed
        qiskit_circuit = circuit.to_qiskit()
        if not qiskit_circuit.cregs:
            qiskit_circuit.measure_all()

        # Transpile for target backend
        transpiled = await loop.run_in_executor(
            None,
            partial(self._transpile_sync, qiskit_circuit, backend),
        )

        # Execute via SamplerV2
        if self.config.timeout_seconds is not None:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    partial(self._run_sampler_sync, transpiled, backend, shots),
                ),
                timeout=self.config.timeout_seconds,
            )
        else:
            result = await loop.run_in_executor(
                None,
                partial(self._run_sampler_sync, transpiled, backend, shots),
            )

        wall_time = time.perf_counter() - start_time

        counts = self._extract_counts(result)

        return ExecutionResult(
            counts=counts,
            backend=self.config.backend_name,
            execution_time_ms=wall_time * 1000,
            shots=shots,
            raw_result=result,
            metadata={
                "job_id": self._current_job_id,
                "backend_name": self.config.backend_name,
                "channel": self.config.channel,
                "instance": self.config.instance,
                "optimization_level": self.config.optimization_level,
                "resilience_level": self.config.resilience_level,
                "transpiled_depth": transpiled.depth(),
                "transpiled_gate_count": transpiled.size(),
                "wall_time_ms": wall_time * 1000,
            },
        )

    async def cancel(self, job_id: str) -> bool:
        """Cancel a running IBM Quantum job.

        Args:
            job_id: The job ID to cancel.

        Returns:
            True if cancellation was successful, False otherwise.
        """
        try:
            loop = asyncio.get_running_loop()
            service = await loop.run_in_executor(None, self._get_service_sync)
            job = service.job(job_id)
            await loop.run_in_executor(None, job.cancel)
            return True
        except Exception:
            return False

    async def get_status(self) -> DeviceStatus:
        """Get live device status from IBM Quantum.

        Maps IBM operational state to standard DeviceStatus.
        Uses pending_jobs for queue depth, estimates ~60s per job.
        """
        try:
            raw = await self.get_backend_status()
            status = "online" if raw.get("operational") else "offline"
            pending = raw.get("pending_jobs")
            queue_time = pending * 60 if pending is not None else None
            return DeviceStatus(
                status=status,
                queue_depth=pending,
                queue_time_seconds=queue_time,
            )
        except Exception:
            return DeviceStatus(status="maintenance", queue_depth=None, queue_time_seconds=None)

    async def get_backend_status(self) -> dict[str, Any]:
        """Get current backend status.

        Returns:
            Dictionary with status info including operational state
            and pending jobs.
        """
        backend = await self._get_backend()
        loop = asyncio.get_running_loop()
        status = await loop.run_in_executor(None, backend.status)
        return {
            "operational": status.operational,
            "pending_jobs": status.pending_jobs,
            "status_msg": status.status_msg,
        }

    async def is_device_available(self) -> bool:
        """Check if backend is available for execution.

        Returns:
            True if backend is operational, False otherwise.
        """
        status = await self.get_backend_status()
        return status.get("operational", False)
