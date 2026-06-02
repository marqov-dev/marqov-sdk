"""Azure Quantum executor for running circuits on Azure Quantum devices.

This module provides AzureQuantumExecutor for executing quantum circuits on
Azure Quantum backends including Quantinuum, PASQAL, IonQ, and Rigetti.

Supports both Qiskit and Cirq frameworks for maximum backend compatibility.

Example (Qiskit):
    >>> from marqov.circuits import bell_state
    >>> from marqov.executors import AzureQuantumExecutor, AzureQuantumExecutorConfig
    >>>
    >>> config = AzureQuantumExecutorConfig(
    ...     subscription_id="your-subscription-id",
    ...     resource_group="your-resource-group",
    ...     workspace_name="your-workspace",
    ...     location="eastus",
    ...     target="ionq.simulator",
    ...     framework="qiskit",
    ... )
    >>> executor = AzureQuantumExecutor(config)
    >>> result = await executor.execute(bell_state(), shots=1000)
    >>> print(result.counts)  # {"00": ~500, "11": ~500}

Example (Cirq):
    >>> config = AzureQuantumExecutorConfig(
    ...     subscription_id="your-subscription-id",
    ...     resource_group="your-resource-group",
    ...     workspace_name="your-workspace",
    ...     location="eastus",
    ...     target="quantinuum.sim.h1-1e",
    ...     framework="cirq",
    ... )
    >>> executor = AzureQuantumExecutor(config)
    >>> result = await executor.execute(circuit, shots=1000)
"""

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
class AzureQuantumExecutorConfig:
    """Configuration for Azure Quantum executor.

    Attributes:
        subscription_id: Azure subscription ID.
        resource_group: Azure resource group name.
        workspace_name: Azure Quantum workspace name.
        location: Azure region (e.g., "eastus", "westus").
        target: Target device name (e.g., "ionq.simulator", "quantinuum.qpu.h2-1").
        framework: Quantum framework to use ("qiskit" or "cirq").
        timeout_seconds: Maximum time to wait for job completion. None for no timeout.
        poll_interval_seconds: Polling interval for job completion.
    """

    subscription_id: str
    resource_group: str
    workspace_name: str
    location: str
    target: str
    framework: str = "qiskit"
    timeout_seconds: float | None = 300.0
    poll_interval_seconds: float = 2.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.framework not in ("qiskit", "cirq"):
            raise ValueError(f"Unsupported framework: {self.framework}. Must be 'qiskit' or 'cirq'.")


class AzureQuantumExecutor(BaseExecutor):
    """Execute circuits on Azure Quantum devices.

    Supports Quantinuum H2 (20-32 qubits), PASQAL Fresnel (100 qubits),
    IonQ, and Rigetti QPUs via Azure Quantum. Uses Qiskit or Cirq framework
    depending on backend compatibility.

    Example:
        >>> config = AzureQuantumExecutorConfig(
        ...     subscription_id="abc123",
        ...     resource_group="quantum-rg",
        ...     workspace_name="my-workspace",
        ...     location="eastus",
        ...     target="quantinuum.qpu.h2-1",
        ...     framework="qiskit",
        ... )
        >>> executor = AzureQuantumExecutor(config)
        >>> result = await executor.execute(circuit, shots=1000)
    """

    def __init__(self, config: AzureQuantumExecutorConfig) -> None:
        """Initialize AzureQuantumExecutor.

        Args:
            config: Executor configuration including workspace and target details.
        """
        self.config = config
        self._workspace = None
        self._backend = None
        self._current_job_id: str | None = None

    def _create_workspace_sync(self):
        """Create Azure Quantum workspace connection (synchronous).

        Returns:
            Workspace instance connected to Azure Quantum.
        """
        from azure.quantum import Workspace

        return Workspace(
            subscription_id=self.config.subscription_id,
            resource_group=self.config.resource_group,
            name=self.config.workspace_name,
            location=self.config.location,
        )

    async def _get_workspace(self):
        """Get or create workspace connection (async wrapper).

        Returns:
            Workspace instance.
        """
        if self._workspace is None:
            loop = asyncio.get_running_loop()
            self._workspace = await loop.run_in_executor(None, self._create_workspace_sync)
        return self._workspace

    def _get_qiskit_backend_sync(self, workspace):
        """Get Qiskit backend from workspace (synchronous).

        Args:
            workspace: Azure Quantum workspace.

        Returns:
            Qiskit backend instance.
        """
        from azure.quantum.qiskit import AzureQuantumProvider

        provider = AzureQuantumProvider(workspace=workspace)
        return provider.get_backend(self.config.target)

    async def _get_qiskit_backend(self):
        """Get or create Qiskit backend (async wrapper).

        Returns:
            Qiskit backend instance.
        """
        if self._backend is None:
            workspace = await self._get_workspace()
            loop = asyncio.get_running_loop()
            self._backend = await loop.run_in_executor(
                None, partial(self._get_qiskit_backend_sync, workspace)
            )
        return self._backend

    def _get_cirq_service_sync(self, workspace):
        """Get Cirq service from workspace (synchronous).

        Args:
            workspace: Azure Quantum workspace.

        Returns:
            Cirq service instance.
        """
        from azure.quantum.cirq import AzureQuantumService

        service = AzureQuantumService(
            workspace=workspace,
            default_target=self.config.target,
        )
        return service

    async def _get_cirq_service(self):
        """Get or create Cirq service (async wrapper).

        Returns:
            Cirq service instance.
        """
        if self._backend is None:
            workspace = await self._get_workspace()
            loop = asyncio.get_running_loop()
            self._backend = await loop.run_in_executor(
                None, partial(self._get_cirq_service_sync, workspace)
            )
        return self._backend

    async def execute(
        self,
        circuit: Circuit,
        shots: int = 1000,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute a circuit on the Azure Quantum device.

        Args:
            circuit: The quantum circuit to execute.
            shots: Number of measurement shots.
            **kwargs: Additional backend-specific options.

        Returns:
            ExecutionResult with measurement counts and metadata.

        Raises:
            RuntimeError: If job fails or device is unavailable.
            NotImplementedError: If framework is not supported.
        """
        circuit = self._validate_circuit(circuit)

        loop = asyncio.get_running_loop()
        start_time = time.perf_counter()

        if self.config.framework == "qiskit":
            result = await self._execute_qiskit(circuit, shots, **kwargs)
        elif self.config.framework == "cirq":
            result = await self._execute_cirq(circuit, shots, **kwargs)
        else:
            raise NotImplementedError(f"Framework not supported: {self.config.framework}")

        wall_time = time.perf_counter() - start_time

        # Add wall time to metadata
        result.metadata["wall_time_ms"] = wall_time * 1000

        return result

    async def _execute_qiskit(
        self,
        circuit: Circuit,
        shots: int,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute circuit using Qiskit framework.

        Args:
            circuit: The circuit to execute.
            shots: Number of shots.
            **kwargs: Additional options.

        Returns:
            ExecutionResult with counts and metadata.
        """
        loop = asyncio.get_running_loop()

        # Get Qiskit backend
        backend = await self._get_qiskit_backend()

        # Convert circuit to Qiskit
        qiskit_circuit = circuit.to_qiskit()

        # Azure Quantum requires explicit measurements for QIR compilation
        # Add measurement to all qubits if not already present
        if not qiskit_circuit.cregs:
            # No classical registers, need to add measurements
            qiskit_circuit.measure_all()

        # Submit job
        job = await loop.run_in_executor(
            None,
            partial(backend.run, qiskit_circuit, shots=shots),
        )
        self._current_job_id = job.job_id()

        # Wait for result
        if self.config.timeout_seconds is not None:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, job.result),
                timeout=self.config.timeout_seconds,
            )
        else:
            result = await loop.run_in_executor(None, job.result)

        # Check if job succeeded
        if not result.success:
            error_msg = "Job failed"
            if hasattr(result, "status"):
                error_msg = f"Job failed with status: {result.status}"
            if hasattr(result, "results") and result.results:
                exp_result = result.results[0]
                if hasattr(exp_result, "status"):
                    error_msg += f", experiment status: {exp_result.status}"
                if hasattr(exp_result, "header") and hasattr(exp_result.header, "status_msg"):
                    error_msg += f", message: {exp_result.header.status_msg}"
            raise RuntimeError(error_msg)

        # Extract counts — reverse bitstrings from Qiskit's little-endian
        # (qubit 0 = rightmost) to Marqov's big-endian (qubit 0 = leftmost).
        raw_counts = dict(result.get_counts())
        counts = {k.replace(" ", "")[::-1]: v for k, v in raw_counts.items()}

        # Extract metadata
        metadata = {
            "job_id": job.job_id(),
            "backend": self.config.target,
            "framework": "qiskit",
            "provider": "Azure Quantum",
        }

        # Try to get cost and timing info
        try:
            # Azure job metadata may have cost information
            job_metadata = await loop.run_in_executor(None, lambda: job.properties())
            if hasattr(job_metadata, "cost_estimate"):
                metadata["cost_estimate"] = job_metadata.cost_estimate
            if hasattr(job_metadata, "execution_time"):
                metadata["quantum_time_ms"] = job_metadata.execution_time * 1000
        except Exception:
            pass

        execution_time_ms = metadata.get("quantum_time_ms", 0.0)

        return ExecutionResult(
            counts=counts,
            backend=self.config.target,
            execution_time_ms=execution_time_ms,
            shots=shots,
            raw_result=result,
            metadata=metadata,
        )

    async def _execute_cirq(
        self,
        circuit: Circuit,
        shots: int,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute circuit using Cirq framework.

        Args:
            circuit: The circuit to execute.
            shots: Number of shots.
            **kwargs: Additional options.

        Returns:
            ExecutionResult with counts and metadata.
        """
        loop = asyncio.get_running_loop()

        # Get Cirq service
        service = await self._get_cirq_service()

        # Convert circuit to Cirq
        cirq_circuit = circuit.to_cirq()

        # Azure Quantum requires explicit measurements
        # Check if circuit has measurements, add if not
        import cirq
        has_measurements = any(
            isinstance(op.gate, cirq.MeasurementGate)
            for moment in cirq_circuit
            for op in moment
        )
        if not has_measurements:
            # Add measurement to all qubits
            qubits = sorted(cirq_circuit.all_qubits())
            cirq_circuit.append(cirq.measure(*qubits, key="result"))

        # Submit job
        job = await loop.run_in_executor(
            None,
            partial(service.run, cirq_circuit, repetitions=shots),
        )
        self._current_job_id = str(job.id()) if hasattr(job, "id") else None

        # Wait for result
        if self.config.timeout_seconds is not None:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: job.results()),
                timeout=self.config.timeout_seconds,
            )
        else:
            result = await loop.run_in_executor(None, lambda: job.results())

        # Convert Cirq result to counts
        # Cirq results are typically a list of measurement results
        counts = {}
        if hasattr(result, "histogram"):
            # Cirq histogram returns integer keys. Convert to bitstrings
            # with Marqov's big-endian convention (qubit 0 = leftmost).
            raw_histogram = dict(result.histogram(key="result"))
            num_qubits = len(sorted(cirq_circuit.all_qubits()))
            # format() places bit 0 at rightmost (little-endian).
            # Reverse to get big-endian (qubit 0 = leftmost).
            counts = {
                format(k, f"0{num_qubits}b")[::-1]: v
                for k, v in raw_histogram.items()
            }
        else:
            # Convert measurement results to counts
            measurements = result.measurements if hasattr(result, "measurements") else result
            for measurement in measurements:
                bitstring = "".join(str(int(b)) for b in measurement)
                counts[bitstring] = counts.get(bitstring, 0) + 1

        metadata = {
            "job_id": self._current_job_id,
            "backend": self.config.target,
            "framework": "cirq",
            "provider": "Azure Quantum",
        }

        return ExecutionResult(
            counts=counts,
            backend=self.config.target,
            execution_time_ms=0.0,  # Cirq doesn't provide this easily
            shots=shots,
            raw_result=result,
            metadata=metadata,
        )

    async def cancel(self, job_id: str) -> bool:
        """Cancel a running Azure Quantum job.

        Args:
            job_id: The job ID to cancel.

        Returns:
            True if cancellation was successful, False otherwise.
        """
        try:
            workspace = await self._get_workspace()
            loop = asyncio.get_running_loop()

            def cancel_sync():
                # Use Azure SDK to cancel job
                job = workspace.get_job(job_id)
                job.cancel()

            await loop.run_in_executor(None, cancel_sync)
            return True
        except Exception:
            return False

    _AZURE_STATUS_MAP = {
        "Available": "online",
        "Degraded": "maintenance",
    }

    async def get_status(self) -> DeviceStatus:
        """Get live device status from Azure Quantum."""
        try:
            raw_status = await self.get_device_status()
            status = self._AZURE_STATUS_MAP.get(raw_status, "offline")
            return DeviceStatus(status=status, queue_depth=None, queue_time_seconds=None)
        except Exception:
            return DeviceStatus(status="maintenance", queue_depth=None, queue_time_seconds=None)

    async def get_device_status(self) -> str:
        """Get current device status.

        Returns:
            Device status string (e.g., "Available", "Unavailable").
        """
        try:
            workspace = await self._get_workspace()
            loop = asyncio.get_running_loop()

            def get_status_sync():
                targets = workspace.get_targets()
                for target in targets:
                    if target.name == self.config.target:
                        return target.current_availability

            status = await loop.run_in_executor(None, get_status_sync)
            return status or "Unknown"
        except Exception:
            return "Unknown"

    async def is_device_available(self) -> bool:
        """Check if device is available for execution.

        Returns:
            True if device is available, False otherwise.
        """
        status = await self.get_device_status()
        return "available" in status.lower() if status else False
