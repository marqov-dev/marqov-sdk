"""IonQ Direct API executor for running circuits on IonQ hardware and simulators.

This module provides IonQExecutor for executing quantum circuits directly
against IonQ's REST API (https://api.ionq.co/v0.3), bypassing the AWS Braket
routing used by BraketExecutor. Talking to IonQ directly gives a shorter
round-trip, IonQ-native error messages (job ``failure`` payloads), and access
to IonQ-specific options such as noise models and error mitigation that the
Braket wrapper does not expose.

Example:
    >>> from marqov.circuits import bell_state
    >>> from marqov.executors import IonQExecutor, IonQExecutorConfig
    >>>
    >>> config = IonQExecutorConfig(api_key="...", backend="simulator")
    >>> executor = IonQExecutor(config)
    >>> result = await executor.execute(bell_state(), shots=1000)
    >>> print(result.counts)  # {"00": ~500, "11": ~500}
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import requests

from marqov.executors.base import BaseExecutor, DeviceStatus, ExecutionResult

if TYPE_CHECKING:
    from marqov.circuits import Circuit


_DEFAULT_BASE_URL = "https://api.ionq.co/v0.3"

# IonQ backend-level status values -> DeviceStatus.status.
#
# NOTE: these are the values documented for the v0.3 /backends endpoint.
# IonQ's v0.4 API restructures backend responses, so if/when the SDK moves
# to v0.4 this map (and the field names read in get_status()) will need a
# follow-up pass -- see https://docs.ionq.com/api-reference/v0.4/migration-from-v0.3
_IONQ_BACKEND_STATUS_MAP = {
    "available": "online",
    "running": "online",
    "calibrating": "maintenance",
    "unavailable": "offline",
    "reserved": "offline",
    "offline": "offline",
}

# IonQ job-level terminal statuses (https://docs.ionq.com/api-reference/v0.3/jobs)
_IONQ_TERMINAL_STATUSES = frozenset({"completed", "failed", "canceled"})
_IONQ_FAILURE_STATUSES = frozenset({"failed", "canceled"})


class _IonQRestClient:
    """Thin wrapper around the IonQ v0.3 REST API.

    Deliberately small and easy to substitute in tests via
    ``IonQExecutorConfig.client`` -- each method maps 1:1 onto an endpoint
    documented at https://docs.ionq.com/api-reference/v0.3.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = _DEFAULT_BASE_URL,
        session: requests.Session | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._session = session or requests.Session()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"apiKey {self._api_key}",
            "Content-Type": "application/json",
        }

    def create_job(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /jobs"""
        resp = self._session.post(f"{self._base_url}/jobs", json=body, headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    def get_job(self, job_id: str) -> dict[str, Any]:
        """GET /jobs/{id}"""
        resp = self._session.get(f"{self._base_url}/jobs/{job_id}", headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    def get_results(self, job_id: str) -> dict[str, Any]:
        """GET /jobs/{id}/results"""
        resp = self._session.get(f"{self._base_url}/jobs/{job_id}/results", headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        """DELETE /jobs/{id}"""
        resp = self._session.delete(f"{self._base_url}/jobs/{job_id}", headers=self._headers())
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    def get_backend(self, name: str) -> dict[str, Any]:
        """GET /backends/{name}"""
        resp = self._session.get(f"{self._base_url}/backends/{name}", headers=self._headers())
        resp.raise_for_status()
        return resp.json()


@dataclass
class IonQExecutorConfig:
    """Configuration for the IonQ Direct API executor.

    Attributes:
        api_key: IonQ API key. Falls back to the ``IONQ_API_KEY`` environment
            variable if not provided and ``client`` is not given. Generate one
            at https://cloud.ionq.com/settings/keys.
        backend: IonQ target/backend name (e.g. "simulator", "qpu.aria-1").
        base_url: Base URL for the IonQ REST API. Defaults to v0.3.
        poll_interval_seconds: Polling interval while waiting for job completion.
        timeout_seconds: Maximum time to wait for job completion. None for no timeout.
        noise_model: Optional IonQ noise model name for simulator jobs
            (e.g. "aria-1"). Passed through as ``{"noise": {"model": ...}}``.
        error_mitigation: If True, request IonQ's debiasing error mitigation.
        client: Optional pre-configured client implementing the
            ``_IonQRestClient`` interface. Mainly for tests -- if omitted,
            IonQExecutor builds one from ``api_key``/``base_url``.
    """

    api_key: str | None = None
    backend: str = "simulator"
    base_url: str = _DEFAULT_BASE_URL
    poll_interval_seconds: float = 0.5
    timeout_seconds: float | None = None
    noise_model: str | None = None
    error_mitigation: bool = False
    client: Any = None


def _circuit_to_qasm(qiskit_circuit: Any) -> str:
    """Convert a Qiskit circuit to OpenQASM 2.0 text for IonQ's "qasm" input format."""
    try:
        from qiskit import qasm2

        return qasm2.dumps(qiskit_circuit)
    except ImportError:
        # Older Qiskit (<1.0) exposes QuantumCircuit.qasm() directly.
        return qiskit_circuit.qasm()


def _probabilities_to_counts(results: dict[str, Any], num_qubits: int, shots: int) -> dict[str, int]:
    """Convert IonQ's probability map into Marqov-style measurement counts.

    IonQ's /results endpoint returns a mapping of decimal state indices to
    probabilities, e.g. ``{"0": 0.5, "6": 0.5}`` for a 3-qubit circuit.
    ExecutionResult expects bitstring counts (e.g. ``{"000": 500, "110": 500}``),
    matching BraketExecutor's output. Bit 0 (qubit 0) is the least-significant
    bit, so the decimal index is rendered as a fixed-width binary string with
    qubit (num_qubits - 1) on the left -- the same convention Qiskit uses when
    printing counts.
    """
    counts: dict[str, int] = {}
    for state, probability in results.items():
        bitstring = format(int(state), f"0{num_qubits}b")
        counts[bitstring] = round(float(probability) * shots)
    return counts


class IonQExecutor(BaseExecutor):
    """Execute circuits directly against the IonQ Quantum Cloud REST API.

    Unlike BraketExecutor (which can also reach IonQ QPUs via AWS Braket),
    this executor talks to ``api.ionq.co`` directly: no AWS account or S3
    bucket required, IonQ-native failure messages, and access to IonQ-specific
    options (noise models, error mitigation) on the simulator.

    Example:
        >>> config = IonQExecutorConfig(api_key="...", backend="qpu.aria-1")
        >>> executor = IonQExecutor(config)
        >>> status = await executor.get_status()
        >>> if status.status == "online":
        ...     result = await executor.execute(circuit, shots=1000)
    """

    def __init__(self, config: IonQExecutorConfig) -> None:
        """Initialize IonQExecutor.

        Args:
            config: Executor configuration including API key and target backend.
        """
        self.config = config
        self._client: _IonQRestClient | None = config.client
        self._current_job_id: str | None = None

    def _get_client(self) -> _IonQRestClient:
        """Get or create the IonQ REST client (lazy initialization)."""
        if self._client is None:
            api_key = self.config.api_key or os.environ.get("IONQ_API_KEY")
            if not api_key:
                raise ValueError(
                    "IonQExecutor requires an IonQ API key. Pass api_key= in "
                    "IonQExecutorConfig, set the IONQ_API_KEY environment "
                    "variable, or provide a pre-configured client=. Generate "
                    "a key at https://cloud.ionq.com/settings/keys."
                )
            self._client = _IonQRestClient(api_key=api_key, base_url=self.config.base_url)
        return self._client

    async def execute(
        self,
        circuit: Circuit,
        shots: int = 1000,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute a circuit on the IonQ Direct API.

        The circuit is converted via ``circuit.to_qiskit()`` and submitted as
        OpenQASM 2.0 using IonQ's ``"qasm"`` input format.

        Args:
            circuit: The quantum circuit to execute.
            shots: Number of measurement shots.
            **kwargs: Additional options. Supports ``backend`` (override the
                configured target for this call), ``noise_model``, and
                ``error_mitigation``.

        Returns:
            ExecutionResult with measurement counts and metadata.

        Raises:
            ValueError: If no API key is configured.
            RuntimeError: If the IonQ job fails or is canceled.
            TimeoutError: If ``timeout_seconds`` elapses before completion.
        """
        circuit = self._validate_circuit(circuit)
        loop = asyncio.get_running_loop()
        client = self._get_client()

        qiskit_circuit = circuit.to_qiskit()
        num_qubits = qiskit_circuit.num_qubits
        qasm = _circuit_to_qasm(qiskit_circuit)

        body: dict[str, Any] = {
            "target": kwargs.get("backend", self.config.backend),
            "shots": shots,
            "input": {"format": "qasm", "data": qasm},
        }

        noise_model = kwargs.get("noise_model", self.config.noise_model)
        if noise_model:
            body["noise"] = {"model": noise_model}

        if kwargs.get("error_mitigation", self.config.error_mitigation):
            body["error_mitigation"] = {"debias": True}

        start_time = time.perf_counter()
        job = await loop.run_in_executor(None, partial(client.create_job, body))
        job_id = job["id"]
        self._current_job_id = job_id

        status = job.get("status", "submitted")
        while status not in _IONQ_TERMINAL_STATUSES:
            if (
                self.config.timeout_seconds is not None
                and (time.perf_counter() - start_time) > self.config.timeout_seconds
            ):
                raise TimeoutError(
                    f"IonQ job {job_id} did not complete within "
                    f"{self.config.timeout_seconds}s (last status: {status})"
                )
            await asyncio.sleep(self.config.poll_interval_seconds)
            job = await loop.run_in_executor(None, partial(client.get_job, job_id))
            status = job.get("status")

        wall_time_ms = (time.perf_counter() - start_time) * 1000

        if status in _IONQ_FAILURE_STATUSES:
            failure = job.get("failure") or {}
            raise RuntimeError(
                f"IonQ job {job_id} ended with status '{status}': "
                f"{failure.get('error', 'no error message provided')} "
                f"(code: {failure.get('code', 'unknown')})"
            )

        results = await loop.run_in_executor(None, partial(client.get_results, job_id))
        counts = _probabilities_to_counts(results, num_qubits, shots)

        return ExecutionResult(
            counts=counts,
            backend=job.get("target", self.config.backend),
            execution_time_ms=float(job.get("execution_time", wall_time_ms)),
            shots=shots,
            raw_result=job,
            metadata={
                "job_id": job_id,
                "cost_usd": job.get("cost_usd"),
                "predicted_execution_time_ms": job.get("predicted_execution_time"),
                "wall_time_ms": wall_time_ms,
                "gate_counts": job.get("gate_counts"),
            },
        )

    async def cancel(self, job_id: str) -> bool:
        """Cancel a running IonQ job.

        Args:
            job_id: The IonQ job ID to cancel. Callers obtain this from
                ``ExecutionResult.metadata["job_id"]``.

        Returns:
            True if the cancellation request succeeded, False otherwise.
        """
        try:
            loop = asyncio.get_running_loop()
            client = self._get_client()
            await loop.run_in_executor(None, partial(client.cancel_job, job_id))
            return True
        except Exception:
            return False

    async def get_status(self) -> DeviceStatus:
        """Get live IonQ backend status.

        Maps IonQ's backend-level status (available/running/calibrating/
        unavailable/reserved/offline) onto Marqov's DeviceStatus, per
        CONTRIBUTING.md \u00a72 (device-level, not job-level, status).
        """
        try:
            loop = asyncio.get_running_loop()
            client = self._get_client()
            backend_info = await loop.run_in_executor(None, partial(client.get_backend, self.config.backend))

            raw_status = str(backend_info.get("status", "")).lower()
            status = _IONQ_BACKEND_STATUS_MAP.get(raw_status, "maintenance")

            queue_depth = backend_info.get("backlog") or backend_info.get("queue_depth")
            avg_queue_time = backend_info.get("average_queue_time")
            queue_time_seconds = int(avg_queue_time) if isinstance(avg_queue_time, (int, float)) else None

            return DeviceStatus(
                status=status,
                queue_depth=queue_depth,
                queue_time_seconds=queue_time_seconds,
            )
        except Exception:
            return DeviceStatus(status="maintenance", queue_depth=None, queue_time_seconds=None)
