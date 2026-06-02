"""AWS Braket executor for running circuits on Braket devices.

This module provides BraketExecutor for executing quantum circuits on AWS Braket
simulators (SV1, DM1, TN1) and QPUs (IQM, Rigetti, IonQ, QuEra).

Example:
    >>> from marqov.circuits import bell_state
    >>> from marqov.executors import BraketExecutor, BraketExecutorConfig
    >>>
    >>> config = BraketExecutorConfig(
    ...     device_arn="arn:aws:braket:::device/quantum-simulator/amazon/sv1",
    ...     s3_bucket="amazon-braket-my-bucket",
    ... )
    >>> executor = BraketExecutor(config)
    >>> result = await executor.execute(bell_state(), shots=1000)
    >>> print(result.counts)  # {"00": ~500, "11": ~500}
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import boto3
from braket.aws import AwsDevice, AwsSession
from braket.circuits import Circuit as BraketCircuit

from marqov.executors.base import BaseExecutor, DeviceStatus, ExecutionResult

if TYPE_CHECKING:
    from marqov.circuits import Circuit


def _extract_region_from_arn(arn: str) -> str:
    """Extract AWS region from a Braket device ARN.

    ARN format: arn:aws:braket:{region}::device/{type}/{provider}/{name}
    Simulator ARNs use empty region (:::) which means us-east-1.

    Args:
        arn: Braket device ARN.

    Returns:
        AWS region string (e.g., "us-east-1", "eu-north-1").
    """
    parts = arn.split(":")
    if len(parts) >= 4:
        region = parts[3]
        return region if region else "us-east-1"
    return "us-east-1"


@dataclass
class BraketExecutorConfig:
    """Configuration for AWS Braket executor.

    Attributes:
        device_arn: ARN of the Braket device (simulator or QPU).
        s3_bucket: S3 bucket for task results (must be Braket-enabled).
        s3_prefix: Prefix for S3 objects. Defaults to "marqov".
        aws_profile: AWS profile name. None uses default credentials.
        aws_region: AWS region. Inferred from device_arn if not provided.
        poll_interval_seconds: Polling interval for task completion.
        timeout_seconds: Maximum time to wait for task completion. None for no timeout.
    """

    device_arn: str
    s3_bucket: str
    s3_prefix: str = "marqov"
    aws_profile: str | None = None
    aws_region: str | None = None
    poll_interval_seconds: float = 1.0
    timeout_seconds: float | None = None


class BraketExecutor(BaseExecutor):
    """Execute circuits on AWS Braket devices.

    Supports both on-demand simulators (SV1, DM1, TN1) and QPUs
    (IonQ, IQM, Rigetti, QuEra). Uses lazy device initialization
    and handles cross-region access automatically.

    Example:
        >>> config = BraketExecutorConfig(
        ...     device_arn="arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet",
        ...     s3_bucket="amazon-braket-my-bucket-eu",
        ...     aws_profile="my-profile",
        ... )
        >>> executor = BraketExecutor(config)
        >>> if await executor.is_device_available():
        ...     result = await executor.execute(circuit, shots=1000)
    """

    def __init__(self, config: BraketExecutorConfig) -> None:
        """Initialize BraketExecutor.

        Args:
            config: Executor configuration including device ARN and S3 settings.
        """
        self.config = config
        self._device: AwsDevice | None = None
        self._aws_session: AwsSession | None = None
        self._current_task_arn: str | None = None

    def _create_aws_session(self) -> AwsSession:
        """Create AWS session for Braket access.

        Returns:
            AwsSession configured for the target region.
        """
        region = self.config.aws_region or _extract_region_from_arn(self.config.device_arn)

        boto_session = boto3.Session(
            profile_name=self.config.aws_profile,
            region_name=region,
        )
        return AwsSession(boto_session=boto_session)

    def _get_device_sync(self) -> AwsDevice:
        """Get or create the AWS device (synchronous).

        Returns:
            AwsDevice instance for cloud simulators and QPUs.
        """
        if self._device is None:
            if self._aws_session is None:
                self._aws_session = self._create_aws_session()
            self._device = AwsDevice(self.config.device_arn, aws_session=self._aws_session)
        return self._device

    async def _get_device(self) -> AwsDevice:
        """Get or create the AWS device (async wrapper).

        Returns:
            AwsDevice instance for cloud simulators and QPUs.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_device_sync)

    async def execute(
        self,
        circuit: Circuit,
        shots: int = 1000,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute a circuit on the Braket device.

        Args:
            circuit: The quantum circuit to execute.
            shots: Number of measurement shots.
            **kwargs: Additional backend-specific options.

        Returns:
            ExecutionResult with measurement counts and metadata.

        Raises:
            RuntimeError: If task fails or device is unavailable.
        """
        circuit = self._validate_circuit(circuit)

        loop = asyncio.get_running_loop()
        start_time = time.perf_counter()

        # Get device (lazy initialization)
        device = await self._get_device()

        # Convert circuit to Braket format
        braket_circuit = circuit.to_braket()

        # Wrap in verbatim box for devices that require it (e.g. Rigetti QPUs).
        # The circuit must already use only native gates — use
        # clifford_to_circuit_native() / SRBConfig.use_native_gates=True for SRB.
        #
        # Allowed gate set is Rigetti-specific (Ankaa-3, Cepheus):
        #   1Q: Rx(θ), Rz(θ)
        #   2Q: CZ, XY(θ)
        #   Measurement: Measure (explicit; Braket also handles it implicitly via shots)
        #
        # NOTE: This set is hardcoded for Rigetti and will need to become a
        # device-aware lookup when IQM or other verbatim providers are added.
        # IQM native set is {Rz, SX, CZ} — different from Rigetti.
        if kwargs.get("verbatim"):
            _RIGETTI_VERBATIM_ALLOWED = {"rx", "rz", "cz", "xy", "measure"}
            non_native = [
                instr.operator.name
                for instr in braket_circuit.instructions
                if instr.operator.name.lower() not in _RIGETTI_VERBATIM_ALLOWED
            ]
            if non_native:
                raise ValueError(
                    f"verbatim=True requires Rigetti native gates only "
                    f"(1Q: Rx/Rz, 2Q: CZ/XY, plus Measure). "
                    f"Found non-native gates: {sorted(set(non_native))}. "
                    f"Use clifford_to_circuit_native() or set SRBConfig.use_native_gates=True."
                )
            braket_circuit = BraketCircuit().add_verbatim_box(braket_circuit)

        # Submit task to AWS Braket
        task = await loop.run_in_executor(
            None,
            partial(
                device.run,
                braket_circuit,
                s3_destination_folder=(self.config.s3_bucket, self.config.s3_prefix),
                shots=shots,
            ),
        )
        self._current_task_arn = task.id

        # Wait for result (blocking call in thread pool)
        if self.config.timeout_seconds is not None:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, task.result),
                timeout=self.config.timeout_seconds,
            )
        else:
            result = await loop.run_in_executor(None, task.result)

        wall_time = time.perf_counter() - start_time

        # Extract timing from task metadata
        execution_duration_ms = 0
        queue_time_ms = None

        try:
            metadata = await loop.run_in_executor(None, task.metadata)
            if metadata:
                execution_duration_ms = metadata.get("executionDuration", 0)
                queue_time_ms = (wall_time * 1000) - execution_duration_ms if execution_duration_ms else None
        except AttributeError:
            pass

        counts = dict(result.measurement_counts)
        if not counts:
            # Some QPU backends (e.g. IonQ Forte-1) return measurementProbabilities
            # instead of raw shot counts. Convert to synthetic counts using shots.
            probs = getattr(result, 'measurement_probabilities', {}) or {}
            if probs:
                counts = {bs: round(float(prob) * shots) for bs, prob in dict(probs).items()}

        return ExecutionResult(
            counts=counts,
            backend=self.config.device_arn,
            execution_time_ms=execution_duration_ms if execution_duration_ms else wall_time * 1000,
            shots=shots,
            raw_result=result,
            metadata={
                "task_arn": task.id,
                "device_name": device.name,
                "s3_location": f"s3://{self.config.s3_bucket}/{self.config.s3_prefix}",
                "execution_duration_ms": execution_duration_ms,
                "queue_time_ms": queue_time_ms,
                "wall_time_ms": wall_time * 1000,
            },
        )

    async def cancel(self, job_id: str) -> bool:
        """Cancel a running Braket task.

        Args:
            job_id: The task ARN to cancel.

        Returns:
            True if cancellation was successful, False otherwise.
        """
        try:
            loop = asyncio.get_running_loop()
            if self._aws_session is None:
                self._aws_session = self._create_aws_session()

            client = self._aws_session.braket_client
            await loop.run_in_executor(
                None,
                partial(client.cancel_quantum_task, quantumTaskArn=job_id),
            )
            return True
        except Exception:
            return False

    async def get_device_status(self) -> str:
        """Get current device status.

        Returns:
            Device status string (e.g., "ONLINE", "OFFLINE", "RETIRED").
        """
        device = await self._get_device()
        return str(device.status)

    async def is_device_available(self) -> bool:
        """Check if device is available for execution.

        Returns:
            True if device is ONLINE, False otherwise.
        """
        status = await self.get_device_status()
        return status == "ONLINE"

    _BRAKET_STATUS_MAP = {
        "ONLINE": "online",
        "OFFLINE": "offline",
        "RETIRED": "offline",
    }

    async def get_status(self) -> DeviceStatus:
        """Get live device status from AWS Braket."""
        try:
            device = await self._get_device()
            raw_status = str(device.status)
            status = self._BRAKET_STATUS_MAP.get(raw_status, "maintenance")

            queue_depth = None
            queue_time_seconds = None
            try:
                loop = asyncio.get_running_loop()
                queue_info = await loop.run_in_executor(None, device.queue_depth)
                if queue_info and queue_info.quantum_tasks:
                    queue_depth = sum(queue_info.quantum_tasks.values())
                    queue_time_seconds = queue_depth * 30
            except Exception:
                pass

            return DeviceStatus(status=status, queue_depth=queue_depth, queue_time_seconds=queue_time_seconds)
        except Exception:
            return DeviceStatus(status="maintenance", queue_depth=None, queue_time_seconds=None)

    async def get_queue_depth(self) -> dict[str, int]:
        """Get queue depth information for the device.

        Returns:
            Dictionary with queue types and their depths.
            Empty dict if queue info is not available.
        """
        try:
            device = await self._get_device()
            loop = asyncio.get_running_loop()
            queue_info = await loop.run_in_executor(None, device.queue_depth)
            return dict(queue_info.quantum_tasks) if queue_info.quantum_tasks else {}
        except Exception:
            return {}
