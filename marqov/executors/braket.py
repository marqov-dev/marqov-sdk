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
import logging
import math
import time
from dataclasses import dataclass
from functools import partial
from numbers import Integral
from typing import TYPE_CHECKING, Any

try:
    import boto3
    from braket.aws import AwsDevice, AwsSession
    from braket.circuits import Circuit as BraketCircuit
except ModuleNotFoundError as error:
    if error.name not in {"boto3", "braket"}:
        raise
    boto3 = AwsDevice = AwsSession = BraketCircuit = None

from marqov._braket_native import prepare_braket, validate_options
from marqov.executors._blocking import BlockingCalls
from marqov.executors._braket_provenance import measurement_provenance
from marqov.executors._counts import allocate_counts
from marqov.executors.base import BaseExecutor, DeviceStatus, ExecutionResult

if TYPE_CHECKING:
    from marqov.circuits import Circuit


logger = logging.getLogger(__name__)


def _simulator_execution_duration_ms(result: Any) -> int | None:
    """Read the documented simulator duration, never infer QPU or queue timing."""
    try:
        additional = getattr(result, "additional_metadata", None)
        simulator = getattr(additional, "simulatorMetadata", None)
        duration = getattr(simulator, "executionDuration", None)
        if isinstance(duration, Integral) and not isinstance(duration, bool) and duration >= 0:
            return int(duration)
    except Exception:
        logger.debug("Optional Braket result timing metadata could not be read", exc_info=True)
    return None


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
        poll_interval_seconds: Vendor polling interval when timeout_seconds is
            configured, capped at that timeout. Otherwise the vendor default is used.
        timeout_seconds: Maximum result wait after submission. Also bounds the
            vendor polling budget. None keeps the vendor default. Interruption
            adds at most one second waiting for a best-effort cancellation request.
    """

    device_arn: str
    s3_bucket: str
    s3_prefix: str = "marqov"
    aws_profile: str | None = None
    aws_region: str | None = None
    poll_interval_seconds: float = 1.0
    timeout_seconds: float | None = None


def _serialize_execution_windows(device: "AwsDevice") -> list[dict[str, str]] | None:
    """Serialize a Braket device's executionWindows into a portable, JSON-friendly shape.

    Shape: ``[{"executionDay": <ExecutionDay .value, e.g. "Everyday">, "windowStartHour": "HH:MM:SS",
    "windowEndHour": "HH:MM:SS"}]`` — always ``.value`` (never ``.name``), sub-second truncated via
    strftime, times UTC. ``[]`` (device reports no windows) is preserved distinct from ``None``.

    Fail-safe: ANY failure (missing property, unexpected shape) returns ``None``, so a malformed payload
    can never blank a device's availability downstream. Reads the already-fetched ``device`` — no extra
    GetDevice call. Deliberately does NOT feed ``is_device_available`` (advisory data only).
    """
    try:
        return [
            {
                "executionDay": w.executionDay.value,
                "windowStartHour": w.windowStartHour.strftime("%H:%M:%S"),
                "windowEndHour": w.windowEndHour.strftime("%H:%M:%S"),
            }
            for w in device.properties.service.executionWindows
        ]
    except Exception:
        return None


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
        if AwsDevice is None:
            raise ImportError('AWS Braket requires pip install "marqov[braket]"')
        if config.timeout_seconds is not None:
            for name, value in (
                ("timeout_seconds", config.timeout_seconds),
                ("poll_interval_seconds", config.poll_interval_seconds),
            ):
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{name} must be finite and positive")
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
        *,
        preserve_qubit_labels: bool = False,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute a circuit on the Braket device.

        Args:
            circuit: The quantum circuit to execute.
            shots: Number of measurement shots.
            preserve_qubit_labels: Opt in to Rigetti native export without wire
                compaction. Requires verbatim=True and disable_qubit_rewiring=True.
            **kwargs: Braket options verbatim and disable_qubit_rewiring must be
                bools. Explicit rewiring is forwarded; omission preserves the
                provider default. Physical placement/capability is unqualified.

        Returns:
            ExecutionResult with measurement counts and metadata.

        Raises:
            RuntimeError: If task fails or device is unavailable.
        """
        circuit = self._validate_circuit(circuit)

        start_time = time.perf_counter()

        validate_options(preserve_qubit_labels, kwargs, self.config.device_arn)
        braket_circuit = prepare_braket(
            circuit, preserve_qubit_labels=preserve_qubit_labels, options=kwargs,
            device_arn=self.config.device_arn, circuit_factory=BraketCircuit,
        )
        device = await self._get_device()
        run_options = {}
        if "disable_qubit_rewiring" in kwargs:
            run_options["disable_qubit_rewiring"] = kwargs["disable_qubit_rewiring"]
        if self.config.timeout_seconds is not None:
            run_options["poll_timeout_seconds"] = self.config.timeout_seconds
            run_options["poll_interval_seconds"] = min(
                self.config.poll_interval_seconds, self.config.timeout_seconds
            )

        with BlockingCalls() as calls:
            task = await calls.call(
                device.run,
                braket_circuit,
                s3_destination_folder=(self.config.s3_bucket, self.config.s3_prefix),
                shots=shots,
                **run_options,
            )
            task_arn = task.id
            self._current_task_arn = task_arn  # Tracking only, never cleanup ownership.
            result = await calls.wait(
                partial(self._task_result_sync, task),
                partial(self._cancel_task_sync, task_arn),
                job_id=task_arn,
                timeout=self.config.timeout_seconds,
            )

            wall_time = time.perf_counter() - start_time

            # Auxiliary metadata cannot invalidate a successfully retrieved result.
            try:
                await calls.call(task.metadata, use_cached_value=True)
            except Exception:
                logger.debug("Optional Braket task metadata lookup failed after result retrieval", exc_info=True)

        execution_duration_ms = _simulator_execution_duration_ms(result)
        # Local wall time includes conversion, polling and download, so its
        # difference from execution duration cannot establish measured queue time.
        queue_time_ms = None

        counts = dict(result.measurement_counts)
        probability_fallback = False
        if not counts:
            # Some QPU backends (e.g. IonQ Forte-1) return measurementProbabilities
            # instead of raw shot counts. Convert to synthetic counts using shots.
            probs = getattr(result, 'measurement_probabilities', {}) or {}
            if probs:
                # Largest-remainder allocation, not round(): naive rounding does
                # not conserve the shot total (three bins at 1/3 of 1000 shots
                # round to 333 each = 999), and downstream code divides by the
                # total assuming it equals `shots`.
                counts = allocate_counts(dict(probs), shots)
                probability_fallback = True

        return ExecutionResult(
            counts=counts,
            backend=self.config.device_arn,
            execution_time_ms=execution_duration_ms if execution_duration_ms is not None else wall_time * 1000,
            shots=shots,
            raw_result=result,
            metadata={
                "measurement_provenance": measurement_provenance(
                    result, counts, shots, probability_fallback=probability_fallback,
                ),
                "task_arn": task.id,
                "device_name": device.name,
                "s3_location": f"s3://{self.config.s3_bucket}/{self.config.s3_prefix}",
                "execution_duration_ms": execution_duration_ms,
                "queue_time_ms": queue_time_ms,
                "wall_time_ms": wall_time * 1000,
            },
        )

    def _task_result_sync(self, task: Any) -> Any:
        """Own and close the event loop used by Braket's synchronous poller."""
        worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(worker_loop)
        try:
            result = task.result()
            if result is None:
                status = task.metadata(use_cached_value=True).get("status")
                if status in {"FAILED", "CANCELLED"}:
                    raise RuntimeError(f"Braket task {task.id} finished with status {status}")
                if self.config.timeout_seconds is not None:
                    raise TimeoutError(f"Braket task {task.id} returned no result within its polling budget")
                raise RuntimeError(f"Braket task {task.id} returned no result")
            return result
        finally:
            worker_loop.close()
            asyncio.set_event_loop(None)

    def _cancel_task_sync(self, job_id: str) -> Any:
        if self._aws_session is None:
            self._aws_session = self._create_aws_session()
        return self._aws_session.braket_client.cancel_quantum_task(quantumTaskArn=job_id)

    async def cancel(self, job_id: str) -> bool:
        """Cancel a running Braket task.

        Args:
            job_id: The task ARN to cancel.

        Returns:
            True if cancellation was successful, False otherwise.
        """
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                partial(self._cancel_task_sync, job_id),
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

            return DeviceStatus(
                status=status,
                queue_depth=queue_depth,
                queue_time_seconds=queue_time_seconds,
                execution_windows=_serialize_execution_windows(device),
            )
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
