"""IonQ Direct API executor.

This module provides an executor for submitting Marqov circuits directly to the
IonQ API instead of routing through Braket or Azure.
"""

from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

from marqov.executors.base import BaseExecutor, DeviceStatus, ExecutionResult

if TYPE_CHECKING:
    from marqov.circuits import Circuit


class _IonQRestClient:
    """Minimal IonQ v0.4 REST client."""

    def __init__(self, api_key: str, base_url: str = "https://api.ionq.co/v0.4") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            urllib.parse.urljoin(f"{self.base_url}/", path.lstrip("/")),
            data=body,
            method=method,
            headers={
                "Authorization": f"apiKey {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request) as response:
                response_body = response.read()
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"IonQ API request failed: {exc.code} {error_body}") from exc

        if not response_body:
            return {}
        return json.loads(response_body.decode("utf-8"))

    def create_job(self, payload: dict[str, Any]) -> Any:
        return self._request("POST", "/jobs", IonQExecutor._to_ionq_api_payload(payload))

    def get_job(self, job_id: str) -> Any:
        return self._request("GET", f"/jobs/{job_id}")

    def cancel_job(self, job_id: str) -> Any:
        return self._request("PUT", f"/jobs/{job_id}/status/cancel")

    def get_backend(self, backend: str) -> Any:
        return self._request("GET", f"/backends/{backend}")

    def get_job_probabilities(self, job_id: str) -> dict[str, float]:
        result = self._request("GET", f"/jobs/{job_id}/results/probabilities")
        return dict(result)


@dataclass
class IonQExecutorConfig:
    """Configuration for the IonQ Direct executor."""

    backend: str
    api_key: str | None = None
    base_url: str = "https://api.ionq.co/v0.4"
    poll_interval_seconds: float = 1.0
    timeout_seconds: float | None = None
    client: Any | None = None


class IonQExecutor(BaseExecutor):
    """Execute circuits through IonQ's direct API."""

    _SUCCESS_STATES = {"completed", "complete", "succeeded", "success", "done"}
    _FAILURE_STATES = {"failed", "failure", "error", "cancelled", "canceled"}

    def __init__(self, config: IonQExecutorConfig) -> None:
        self.config = config
        self._client: Any | None = None
        self._current_job_id: str | None = None

    def _create_client_sync(self) -> Any:
        if self.config.client is not None:
            return self.config.client

        if not self.config.api_key:
            raise ValueError("IonQ Direct executor requires api_key when no client is injected")

        return _IonQRestClient(api_key=self.config.api_key, base_url=self.config.base_url)

    def _get_client_sync(self) -> Any:
        if self._client is None:
            self._client = self._create_client_sync()
        return self._client

    async def _get_client(self) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._get_client_sync)

    @staticmethod
    def _qasm_from_qiskit(qiskit_circuit: Any) -> str:
        try:
            from qiskit import qasm3

            return qasm3.dumps(qiskit_circuit)
        except Exception:
            try:
                from qiskit import qasm2

                return qasm2.dumps(qiskit_circuit)
            except Exception:
                if hasattr(qiskit_circuit, "qasm"):
                    return str(qiskit_circuit.qasm())
                raise

    @staticmethod
    def _maybe_measure_all(qiskit_circuit: Any) -> Any:
        if hasattr(qiskit_circuit, "measure_all"):
            cregs = getattr(qiskit_circuit, "cregs", None)
            if cregs is not None and len(cregs) == 0:
                qiskit_circuit.measure_all()
        return qiskit_circuit

    @staticmethod
    def _extract_job_id(job: Any) -> str:
        if isinstance(job, dict):
            job_id = job.get("id") or job.get("job_id")
        else:
            job_id = getattr(job, "id", None) or getattr(job, "job_id", None)

        if not job_id:
            raise RuntimeError("IonQ job response did not include a job id")
        return str(job_id)

    @staticmethod
    def _status_from_job(job: Any) -> str:
        if isinstance(job, dict):
            return str(job.get("status") or job.get("state") or "").lower()
        return str(getattr(job, "status", "") or getattr(job, "state", "")).lower()

    @staticmethod
    def _value(data: Any, *keys: str) -> Any:
        for key in keys:
            if isinstance(data, dict) and key in data:
                return data[key]
            if hasattr(data, key):
                return getattr(data, key)
        return None

    def _create_job_sync(self, client: Any, payload: dict[str, Any]) -> Any:
        if hasattr(client, "create_job"):
            return client.create_job(payload)
        if hasattr(client, "submit_job"):
            return client.submit_job(payload)
        if hasattr(client, "jobs") and hasattr(client.jobs, "create"):
            return client.jobs.create(**payload)
        raise RuntimeError("IonQ client does not expose create_job(), submit_job(), or jobs.create()")

    def _get_job_sync(self, client: Any, job_id: str) -> Any:
        if hasattr(client, "get_job"):
            return client.get_job(job_id)
        if hasattr(client, "jobs") and hasattr(client.jobs, "get_job"):
            return client.jobs.get_job(job_id)
        if hasattr(client, "jobs") and hasattr(client.jobs, "get"):
            return client.jobs.get(job_id)
        raise RuntimeError("IonQ client does not expose get_job() or jobs.get()")

    def _cancel_job_sync(self, client: Any, job_id: str) -> Any:
        if hasattr(client, "cancel_job"):
            return client.cancel_job(job_id)
        if hasattr(client, "jobs") and hasattr(client.jobs, "cancel_job"):
            return client.jobs.cancel_job(job_id)
        if hasattr(client, "jobs") and hasattr(client.jobs, "cancel"):
            return client.jobs.cancel(job_id)
        raise RuntimeError("IonQ client does not expose cancel_job() or jobs.cancel()")

    def _get_backend_sync(self, client: Any) -> Any:
        if hasattr(client, "get_backend"):
            return client.get_backend(self.config.backend)
        if hasattr(client, "get_device"):
            return client.get_device(self.config.backend)
        if hasattr(client, "backends") and hasattr(client.backends, "get"):
            return client.backends.get(self.config.backend)
        raise RuntimeError("IonQ client does not expose backend status methods")

    @staticmethod
    def _to_ionq_api_payload(payload: dict[str, Any]) -> dict[str, Any]:
        api_payload: dict[str, Any] = {
            "type": "ionq.circuit.v1",
            "backend": payload["backend"],
            "shots": payload["shots"],
            "input": {
                "qasm": payload["input"]["data"],
            },
        }
        if "metadata" in payload:
            api_payload["metadata"] = payload["metadata"]
        return api_payload

    async def _wait_for_job(self, client: Any, job_id: str) -> Any:
        loop = asyncio.get_running_loop()
        start_time = time.perf_counter()

        while True:
            job = await loop.run_in_executor(None, partial(self._get_job_sync, client, job_id))
            status = self._status_from_job(job)

            if status in self._SUCCESS_STATES:
                return job
            if status in self._FAILURE_STATES:
                raise RuntimeError(f"IonQ job {job_id} failed with status: {status}")

            if (
                self.config.timeout_seconds is not None
                and time.perf_counter() - start_time >= self.config.timeout_seconds
            ):
                raise TimeoutError(f"Timed out waiting for IonQ job {job_id}")

            await asyncio.sleep(self.config.poll_interval_seconds)

    @staticmethod
    def _extract_counts(job: Any, shots: int) -> dict[str, int]:
        result = IonQExecutor._value(job, "result", "results", "data") or job
        counts = IonQExecutor._value(result, "counts", "measurement_counts")
        if counts:
            return {str(key): int(value) for key, value in dict(counts).items()}

        probabilities = IonQExecutor._value(
            result,
            "probabilities",
            "measurement_probabilities",
            "histogram",
        )
        if probabilities:
            return {
                str(bitstring): round(float(probability) * shots)
                for bitstring, probability in dict(probabilities).items()
            }

        return {}

    async def execute(
        self,
        circuit: Circuit,
        shots: int = 1000,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute a circuit through IonQ Direct."""
        circuit = self._validate_circuit(circuit)

        loop = asyncio.get_running_loop()
        client = await self._get_client()
        start_time = time.perf_counter()

        qiskit_circuit = self._maybe_measure_all(circuit.to_qiskit())
        qasm = self._qasm_from_qiskit(qiskit_circuit)
        backend = str(kwargs.get("backend", kwargs.get("target", self.config.backend)))
        payload = {
            "target": backend,
            "backend": backend,
            "shots": shots,
            "input": {
                "format": "qasm",
                "data": qasm,
            },
        }
        if "metadata" in kwargs:
            payload["metadata"] = kwargs["metadata"]

        job = await loop.run_in_executor(None, partial(self._create_job_sync, client, payload))
        job_id = self._extract_job_id(job)
        self._current_job_id = job_id

        completed_job = await self._wait_for_job(client, job_id)
        wall_time_ms = (time.perf_counter() - start_time) * 1000
        execution_time_ms = self._value(
            completed_job,
            "execution_time_ms",
            "executionTimeMs",
            "runtime_ms",
        )

        counts = self._extract_counts(completed_job, shots)
        if not counts and hasattr(client, "get_job_probabilities"):
            probabilities = await loop.run_in_executor(
                None,
                partial(client.get_job_probabilities, job_id),
            )
            completed_job = {"job": completed_job, "probabilities": probabilities}
            counts = self._extract_counts(completed_job, shots)

        return ExecutionResult(
            counts=counts,
            backend=backend,
            execution_time_ms=float(execution_time_ms or wall_time_ms),
            shots=shots,
            raw_result=completed_job,
            metadata={
                "job_id": job_id,
                "provider": "IonQ Direct",
                "target": backend,
                "wall_time_ms": wall_time_ms,
            },
        )

    async def cancel(self, job_id: str) -> bool:
        """Cancel an IonQ job."""
        try:
            loop = asyncio.get_running_loop()
            client = await self._get_client()
            await loop.run_in_executor(None, partial(self._cancel_job_sync, client, job_id))
            return True
        except Exception:
            return False

    async def get_status(self) -> DeviceStatus:
        """Get live IonQ device status."""
        try:
            loop = asyncio.get_running_loop()
            client = await self._get_client()
            backend = await loop.run_in_executor(None, partial(self._get_backend_sync, client))
            raw_status = str(self._value(backend, "status", "state", "availability") or "").lower()

            if raw_status in {"online", "available", "ready", "running"}:
                status = "online"
            elif raw_status in {"offline", "unavailable", "down", "retired"}:
                status = "offline"
            else:
                status = "maintenance"

            queue_depth = self._value(backend, "queue_depth", "queueDepth", "queue")
            queue_time = self._value(
                backend,
                "queue_time_seconds",
                "queueTimeSeconds",
                "estimated_queue_time",
            )
            return DeviceStatus(
                status=status,
                queue_depth=int(queue_depth) if queue_depth is not None else None,
                queue_time_seconds=int(queue_time) if queue_time is not None else None,
            )
        except Exception:
            return DeviceStatus(status="maintenance", queue_depth=None, queue_time_seconds=None)
