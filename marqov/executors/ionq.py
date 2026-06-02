"""IonQ Direct API executor.

This module provides an executor for submitting Marqov circuits directly to
IonQ's API through the official ``ionq-core`` client.
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


_IONQ_EXTRA_MESSAGE = (
    "IonQ Direct execution requires ionq-core. Install with: pip install marqov[ionq]"
)


@dataclass
class IonQExecutorConfig:
    """Configuration for IonQ Direct API execution.

    Attributes:
        backend: IonQ backend name, e.g. "simulator" or a QPU backend slug.
        api_key: IonQ API key. None lets ionq-core read its default environment.
        base_url: IonQ API base URL.
        max_retries: Optional ionq-core retry override.
        poll_interval_seconds: Polling interval while waiting for completion.
        timeout_seconds: Maximum time to wait for job completion.
        sharpen_probabilities: Whether IonQ should sharpen returned probabilities.
    """

    backend: str = "simulator"
    api_key: str | None = None
    base_url: str = "https://api.ionq.co/v0.4"
    max_retries: int | None = None
    poll_interval_seconds: float = 1.0
    timeout_seconds: float = 300.0
    sharpen_probabilities: bool = False


class IonQExecutor(BaseExecutor):
    """Execute circuits through IonQ's direct API."""

    _SINGLE_QUBIT_GATES = {
        "H": "h",
        "X": "x",
        "Y": "y",
        "Z": "z",
        "S": "s",
        "T": "t",
    }
    _ROTATION_GATES = {
        "Rx": "rx",
        "Ry": "ry",
        "Rz": "rz",
    }
    _STATUS_MAP = {
        "available": "online",
        "unavailable": "offline",
        "retired": "offline",
    }

    def __init__(self, config: IonQExecutorConfig | None = None) -> None:
        """Initialize IonQExecutor.

        Args:
            config: Executor configuration. Defaults target the IonQ simulator.
        """
        self.config = config or IonQExecutorConfig()
        self._client: Any = None
        self._current_job_id: str | None = None

    @staticmethod
    def _coerce_float(value: Any) -> float:
        """Convert numeric SDK values, including zero-imaginary complex values."""
        if isinstance(value, complex):
            if abs(value.imag) > 1e-12:
                raise ValueError(f"Expected a real rotation angle, got {value!r}")
            return float(value.real)
        return float(value)

    @staticmethod
    def _require_qubits(gate_name: str, qubits: list[Any], expected: int) -> list[int]:
        if len(qubits) != expected:
            raise ValueError(f"IonQ gate {gate_name!r} requires {expected} qubit(s)")
        return [int(qubit) for qubit in qubits]

    @staticmethod
    def _require_rotation(gate_name: str, params: list[Any]) -> float:
        if not params:
            raise ValueError(f"IonQ rotation gate {gate_name!r} requires an angle parameter")
        return IonQExecutor._coerce_float(params[0])

    def _create_client_sync(self) -> Any:
        """Create an ionq-core client."""
        try:
            from ionq_core import IonQClient
        except ImportError as exc:
            raise ImportError(_IONQ_EXTRA_MESSAGE) from exc

        return IonQClient(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            max_retries=self.config.max_retries,
            additional_user_agent="marqov/0.5.0",
        )

    async def _get_client(self) -> Any:
        """Get or create the IonQ API client."""
        if self._client is None:
            loop = asyncio.get_running_loop()
            self._client = await loop.run_in_executor(None, self._create_client_sync)
        return self._client

    def _build_qis_gates(self, circuit: Circuit) -> list[Any]:
        """Translate Marqov's serialized gate list to IonQ QIS gate models."""
        try:
            from ionq_core.models import GateQisGate
        except ImportError as exc:
            raise ImportError(_IONQ_EXTRA_MESSAGE) from exc

        ionq_gates = []
        for gate_data in circuit.to_dict().get("gates", []):
            gate_name = gate_data.get("gate")
            qubits = gate_data.get("qubits", [])
            params = gate_data.get("params", [])

            if gate_name in self._SINGLE_QUBIT_GATES:
                target = self._require_qubits(gate_name, list(qubits), 1)[0]
                ionq_gates.append(
                    GateQisGate(gate=self._SINGLE_QUBIT_GATES[gate_name], target=target)
                )
                continue

            if gate_name in self._ROTATION_GATES:
                target = self._require_qubits(gate_name, list(qubits), 1)[0]
                rotation = self._require_rotation(gate_name, list(params))
                ionq_gates.append(
                    GateQisGate(
                        gate=self._ROTATION_GATES[gate_name],
                        target=target,
                        rotation=rotation,
                    )
                )
                continue

            if gate_name == "CNot":
                control, target = self._require_qubits(gate_name, list(qubits), 2)
                ionq_gates.append(GateQisGate(gate="cnot", control=control, target=target))
                continue

            if gate_name == "CZ":
                control, target = self._require_qubits(gate_name, list(qubits), 2)
                ionq_gates.extend(
                    [
                        GateQisGate(gate="h", target=target),
                        GateQisGate(gate="cnot", control=control, target=target),
                        GateQisGate(gate="h", target=target),
                    ]
                )
                continue

            if gate_name == "Swap":
                targets = self._require_qubits(gate_name, list(qubits), 2)
                ionq_gates.append(GateQisGate(gate="swap", targets=targets))
                continue

            raise NotImplementedError(
                f"IonQ Direct does not support Marqov gate {gate_name!r}. "
                "Supported gates: H, X, Y, Z, S, T, Rx, Ry, Rz, CNot, CZ, Swap."
            )

        return ionq_gates

    def _build_job_payload(self, circuit: Circuit, shots: int) -> Any:
        """Build an IonQ circuit job payload from a Marqov circuit."""
        try:
            from ionq_core.models import CircuitJobCreationPayload, QisCircuitInput
        except ImportError as exc:
            raise ImportError(_IONQ_EXTRA_MESSAGE) from exc

        return CircuitJobCreationPayload(
            backend=self.config.backend,
            type_="ionq.circuit.v1",
            input_=QisCircuitInput(
                qubits=int(circuit.num_qubits),
                gateset="qis",
                circuit=self._build_qis_gates(circuit),
            ),
            shots=shots,
        )

    @staticmethod
    def _state_key_to_bitstring(state: Any, num_qubits: int) -> str:
        state_str = str(state)
        if state_str.isdigit():
            decimal_bits = format(int(state_str), f"0{num_qubits}b")
            if len(decimal_bits) <= num_qubits:
                return decimal_bits

        if set(state_str) <= {"0", "1"} and len(state_str) <= num_qubits:
            return state_str.zfill(num_qubits)

        raise ValueError(f"Cannot convert IonQ result state {state!r} to a bitstring")

    def _probabilities_to_counts(
        self,
        probabilities: dict[str, float],
        shots: int,
        num_qubits: int,
    ) -> dict[str, int]:
        """Convert IonQ probability results to shot counts."""
        counts = {
            self._state_key_to_bitstring(state, num_qubits): round(float(probability) * shots)
            for state, probability in probabilities.items()
            if float(probability) > 0
        }

        if counts:
            diff = shots - sum(counts.values())
            if diff != 0:
                largest_key = max(counts, key=counts.get)
                counts[largest_key] += diff

        return {state: count for state, count in counts.items() if count > 0}

    @staticmethod
    def _extract_probabilities(probabilities_response: Any) -> dict[str, float]:
        if probabilities_response is None:
            return {}

        if isinstance(probabilities_response, dict):
            return {str(key): float(value) for key, value in probabilities_response.items()}

        additional_properties = getattr(probabilities_response, "additional_properties", None)
        if additional_properties:
            return {
                str(key): float(value)
                for key, value in dict(additional_properties).items()
            }

        to_dict = getattr(probabilities_response, "to_dict", None)
        if callable(to_dict):
            return {
                str(key): float(value)
                for key, value in to_dict().items()
                if key != "url"
            }

        return {}

    @staticmethod
    def _status_value(status: Any) -> str:
        return str(getattr(status, "value", status)).lower()

    def _create_job_sync(self, client: Any, payload: Any) -> Any:
        from ionq_core.api.default import create_job

        return create_job.sync(client=client, body=payload)

    def _wait_for_job_sync(self, client: Any, job_id: str) -> Any:
        from ionq_core import wait_for_job

        return wait_for_job(
            client,
            job_id,
            poll_interval=self.config.poll_interval_seconds,
            timeout=self.config.timeout_seconds,
            raise_on_failure=True,
        )

    def _get_probabilities_sync(self, client: Any, job_id: str) -> Any:
        from ionq_core.api.default import get_job_probabilities

        return get_job_probabilities.sync(
            uuid=job_id,
            client=client,
            sharpen=self.config.sharpen_probabilities,
        )

    def _cancel_job_sync(self, client: Any, job_id: str) -> Any:
        from ionq_core.api.default import cancel_job

        return cancel_job.sync(uuid=job_id, client=client)

    def _get_backend_sync(self, client: Any) -> Any:
        from ionq_core.api.backends import get_backend

        return get_backend.sync(backend=self.config.backend, client=client)

    async def execute(
        self,
        circuit: Circuit,
        shots: int = 1000,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute a circuit through IonQ's direct API."""
        circuit = self._validate_circuit(circuit)
        loop = asyncio.get_running_loop()
        start_time = time.perf_counter()

        client = await self._get_client()
        payload = self._build_job_payload(circuit, shots)
        job = await loop.run_in_executor(None, partial(self._create_job_sync, client, payload))

        job_id = getattr(job, "id", None)
        if not job_id:
            raise RuntimeError("IonQ job creation did not return a job id")
        self._current_job_id = str(job_id)

        completed_job = await loop.run_in_executor(
            None,
            partial(self._wait_for_job_sync, client, self._current_job_id),
        )
        probabilities_response = await loop.run_in_executor(
            None,
            partial(self._get_probabilities_sync, client, self._current_job_id),
        )
        wall_time_ms = (time.perf_counter() - start_time) * 1000

        probabilities = self._extract_probabilities(probabilities_response)
        counts = self._probabilities_to_counts(probabilities, shots, int(circuit.num_qubits))
        if not counts:
            raise RuntimeError("IonQ job completed but did not return probabilities")

        execution_duration_ms = getattr(completed_job, "execution_duration_ms", None)
        predicted_wait_time_ms = getattr(completed_job, "predicted_wait_time_ms", None)

        return ExecutionResult(
            counts=counts,
            backend=self.config.backend,
            execution_time_ms=execution_duration_ms or wall_time_ms,
            shots=shots,
            raw_result=completed_job,
            metadata={
                "job_id": self._current_job_id,
                "status": self._status_value(getattr(completed_job, "status", "")),
                "execution_duration_ms": execution_duration_ms,
                "predicted_wait_time_ms": predicted_wait_time_ms,
                "wall_time_ms": wall_time_ms,
            },
        )

    async def cancel(self, job_id: str) -> bool:
        """Cancel a running IonQ job."""
        try:
            client = await self._get_client()
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                partial(self._cancel_job_sync, client, job_id),
            )
            status = self._status_value(getattr(response, "status", ""))
            return response is not None and status in {"canceled", "cancelled"}
        except Exception:
            return False

    @staticmethod
    def _queue_time_seconds(backend: Any) -> int | None:
        queue_time_ms = getattr(backend, "average_queue_time", None)
        if queue_time_ms is None:
            return None
        return round(float(queue_time_ms) / 1000)

    def _backend_to_device_status(self, backend: Any) -> DeviceStatus:
        raw_status = str(getattr(backend, "status", "")).lower()
        degraded = bool(getattr(backend, "degraded", False))
        status = self._STATUS_MAP.get(raw_status, "maintenance")
        if status == "online" and degraded:
            status = "maintenance"

        return DeviceStatus(
            status=status,
            queue_depth=None,
            queue_time_seconds=self._queue_time_seconds(backend),
        )

    async def get_status(self) -> DeviceStatus:
        """Get live device status from IonQ."""
        if self.config.backend == "simulator":
            return DeviceStatus.always_online()

        try:
            client = await self._get_client()
            loop = asyncio.get_running_loop()
            backend = await loop.run_in_executor(None, partial(self._get_backend_sync, client))
            if backend is None:
                return DeviceStatus(status="maintenance", queue_depth=None, queue_time_seconds=None)
            return self._backend_to_device_status(backend)
        except Exception:
            return DeviceStatus(status="maintenance", queue_depth=None, queue_time_seconds=None)
