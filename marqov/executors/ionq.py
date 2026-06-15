"""IonQ Direct API executor for running circuits on IonQ hardware.

This module provides IonQExecutor for executing quantum circuits directly against
IonQ's REST API (https://api.ionq.co), bypassing the AWS Braket intermediary.
Talking to IonQ directly shortens the round-trip, surfaces richer error messages,
and exposes IonQ-specific features (e.g. simulator noise models) without requiring
an AWS account or S3 bucket.

Circuits are converted with ``circuit.to_qiskit()`` and dumped to OpenQASM, then
submitted using IonQ's ``qasm`` input format.

Note on the official ``ionq`` client:
    The official ``ionq`` Python client is intentionally not used. Its only release
    (``0.0.0a15``) pins ``pydantic<2``, which conflicts with marqov's core
    ``pydantic>=2`` requirement, making ``marqov[ionq]`` impossible to install while
    that client is a dependency. We therefore call IonQ's REST API directly with
    ``requests`` (the same HTTP library the official client uses under the hood).

Example:
    >>> from marqov.circuits import bell_state
    >>> from marqov.executors import IonQExecutor, IonQExecutorConfig
    >>>
    >>> config = IonQExecutorConfig(target="simulator", api_key="your-ionq-key")
    >>> executor = IonQExecutor(config)
    >>> result = await executor.execute(bell_state(), shots=1000)
    >>> print(result.counts)  # {"00": ~500, "11": ~500}
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from marqov.executors.base import BaseExecutor, DeviceStatus, ExecutionResult

if TYPE_CHECKING:
    from marqov.circuits import Circuit

# Job statuses that mean the job has finished successfully.
_SUCCESS_STATUSES = frozenset({"completed"})
# Job statuses that mean the job has finished without usable results.
_FAILURE_STATUSES = frozenset({"failed", "canceled", "cancelled"})

# Per-request HTTP timeout (seconds). Overall job completion is bounded separately
# by IonQExecutorConfig.timeout_seconds around polling.
_HTTP_TIMEOUT_SECONDS = 30


@dataclass
class IonQExecutorConfig:
    """Configuration for the IonQ Direct API executor.

    Attributes:
        target: IonQ backend target (e.g. "simulator", "qpu.aria-1",
            "qpu.forte-1").
        api_key: IonQ API key. If None, falls back to the ``IONQ_API_KEY``
            environment variable at request time.
        base_url: Base URL for the IonQ REST API.
        poll_interval_seconds: Polling interval while waiting for a job to finish.
        timeout_seconds: Maximum time to wait for job completion. None for no
            timeout.
        noise_model: Optional simulator noise model (e.g. "aria-1", "forte-1").
            Only applied when ``target`` is the simulator.
    """

    target: str = "simulator"
    api_key: str | None = None
    base_url: str = "https://api.ionq.co/v0.3"
    poll_interval_seconds: float = 1.0
    timeout_seconds: float | None = None
    noise_model: str | None = None


class IonQExecutor(BaseExecutor):
    """Execute circuits on IonQ hardware via IonQ's direct REST API.

    Converts circuits with ``to_qiskit()`` and submits them as OpenQASM, polls for
    completion, and converts IonQ's probability histogram into measurement counts.
    An HTTP session may be injected for testing without network access or credentials.

    Example:
        >>> config = IonQExecutorConfig(
        ...     target="qpu.aria-1",
        ...     api_key="your-ionq-key",
        ... )
        >>> executor = IonQExecutor(config)
        >>> if (await executor.get_status()).status == "online":
        ...     result = await executor.execute(circuit, shots=1000)
    """

    # IonQ backend availability → standard DeviceStatus state.
    _IONQ_STATUS_MAP = {
        "available": "online",
        "running": "online",
        "unavailable": "offline",
        "offline": "offline",
        "reserved": "maintenance",
        "calibrating": "maintenance",
    }

    def __init__(self, config: IonQExecutorConfig, *, session: Any = None) -> None:
        """Initialize IonQExecutor.

        Args:
            config: Executor configuration including target and credentials.
            session: Optional HTTP session/transport exposing
                ``request(method, url, **kwargs)`` (e.g. ``requests.Session`` or a
                test double). If None, each request uses a fresh ``requests`` call,
                which keeps the executor safe to share across concurrent coroutines
                (``requests.Session`` is not guaranteed thread-safe).
        """
        self.config = config
        self._session = session
        self._current_job_id: str | None = None

    def _do_request(self, method: str, url: str, **kwargs: Any) -> Any:
        """Perform a single synchronous HTTP request.

        Uses the injected session if provided, otherwise a fresh ``requests`` call
        per invocation. Avoiding a shared, cached session means concurrent calls
        (each run in a worker thread via ``run_in_executor``) don't race on a single
        non-thread-safe ``requests.Session``.

        Args:
            method: HTTP method.
            url: Fully-qualified request URL.
            **kwargs: Extra arguments forwarded to the transport.

        Returns:
            The HTTP response object.
        """
        if self._session is not None:
            return self._session.request(method, url, **kwargs)
        import requests

        return requests.request(method, url, **kwargs)

    def _auth_headers(self) -> dict[str, str]:
        """Build IonQ authorization headers.

        Returns:
            Headers dict with the IonQ API key.

        Raises:
            ValueError: If no API key is available from config or environment.
        """
        api_key = self.config.api_key or os.environ.get("IONQ_API_KEY")
        if not api_key:
            raise ValueError(
                "IonQ API key not found. Set it via IonQExecutorConfig(api_key=...) "
                "or the IONQ_API_KEY environment variable."
            )
        return {"Authorization": f"apiKey {api_key}"}

    @staticmethod
    def _circuit_to_qasm(circuit: Circuit) -> tuple[str, int]:
        """Convert a Marqov circuit to OpenQASM via the Qiskit path.

        This implements the explicit ``to_qiskit()`` → QASM conversion: the circuit
        is first converted to a Qiskit ``QuantumCircuit`` and then dumped to QASM 2.0.

        Args:
            circuit: The Marqov circuit to convert.

        Returns:
            A tuple of (QASM string, number of qubits).
        """
        from qiskit import qasm2  # type: ignore[import-untyped]

        qiskit_circuit = circuit.to_qiskit()  # type: ignore[no-untyped-call]
        qasm: str = qasm2.dumps(qiskit_circuit)
        return qasm, qiskit_circuit.num_qubits

    @staticmethod
    def _extract_histogram(payload: dict[str, Any]) -> dict[str, float]:
        """Extract the probability histogram from a results response.

        The IonQ results endpoint may return the histogram directly
        (``{"0": 0.5, ...}``) or wrapped (``{"histogram": {...}}`` or
        ``{"data": {"histogram": {...}}}``, and sometimes under ``probabilities``).
        This normalizes those shapes to the bare ``{state_index: probability}`` map.

        Args:
            payload: The parsed JSON results response.

        Returns:
            The probability histogram mapping.
        """
        data = payload.get("data")
        if isinstance(data, dict) and isinstance(data.get("histogram"), dict):
            nested: dict[str, float] = data["histogram"]
            return nested
        if isinstance(payload.get("histogram"), dict):
            wrapped: dict[str, float] = payload["histogram"]
            return wrapped
        if isinstance(payload.get("probabilities"), dict):
            probabilities: dict[str, float] = payload["probabilities"]
            return probabilities
        # Assume the payload is already a bare histogram mapping.
        return payload

    @staticmethod
    def _histogram_to_counts(
        histogram: dict[str, float],
        shots: int,
        num_qubits: int,
    ) -> dict[str, int]:
        """Convert an IonQ probability histogram into measurement counts.

        IonQ returns a sparse histogram mapping big-endian state indices (as
        strings) to probabilities, where the leftmost bit corresponds to qubit 0.

        Counts are allocated with the largest-remainder (Hamilton) method so the
        totals sum exactly to ``shots`` — naive per-bin rounding can drift above or
        below ``shots`` and break downstream "total == shots" assumptions.

        Args:
            histogram: Mapping of state index strings to probabilities.
            shots: Number of shots, used to scale probabilities to counts.
            num_qubits: Number of qubits, used to zero-pad bitstrings.

        Returns:
            Mapping of bitstrings to integer counts that sum to ``shots``.
        """
        if not histogram:
            return {}

        # Floor each bin and remember its fractional remainder.
        counts: dict[str, int] = {}
        remainders: dict[str, float] = {}
        allocated = 0
        for index, probability in histogram.items():
            bitstring = format(int(index), f"0{num_qubits}b")
            exact = float(probability) * shots
            base = int(exact)  # floor (probabilities are non-negative)
            counts[bitstring] = base
            remainders[bitstring] = exact - base
            allocated += base

        # Distribute (or reclaim) the leftover shots so the total equals `shots`.
        leftover = shots - allocated
        if leftover > 0:
            # Hand extra shots to the largest fractional remainders first.
            ordered = sorted(remainders, key=lambda b: remainders[b], reverse=True)
            for i in range(leftover):
                counts[ordered[i % len(ordered)]] += 1
        elif leftover < 0:
            # Reclaim over-allocated shots from the smallest remainders first.
            ordered = sorted(remainders, key=lambda b: remainders[b])
            i = 0
            while leftover < 0 and i < len(ordered) * (-leftover + 1):
                bitstring = ordered[i % len(ordered)]
                if counts[bitstring] > 0:
                    counts[bitstring] -= 1
                    leftover += 1
                i += 1

        return {bitstring: count for bitstring, count in counts.items() if count > 0}

    async def execute(
        self,
        circuit: Circuit,
        shots: int = 1000,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute a circuit on an IonQ backend via the direct REST API.

        Args:
            circuit: The quantum circuit to execute.
            shots: Number of measurement shots.
            **kwargs: Additional backend-specific options (currently unused).

        Returns:
            ExecutionResult with measurement counts and metadata.

        Raises:
            RuntimeError: If the job fails or is canceled by IonQ.
            ValueError: If no API key is available.
            TimeoutError: If the job does not finish within ``timeout_seconds``.
        """
        circuit = self._validate_circuit(circuit)

        start_time = time.perf_counter()

        qasm, num_qubits = self._circuit_to_qasm(circuit)

        payload: dict[str, Any] = {
            "target": self.config.target,
            "shots": shots,
            "input": {"format": "qasm", "data": qasm},
        }
        if self.config.noise_model and self.config.target == "simulator":
            payload["noise"] = {"model": self.config.noise_model}

        # Submit the job and record its id for cancellation/tracking.
        submit_response = await self._request("POST", "/jobs", json=payload)
        job_id = submit_response["id"]
        self._current_job_id = job_id

        # Poll until the job reaches a terminal state.
        if self.config.timeout_seconds is not None:
            job = await asyncio.wait_for(
                self._poll_until_done(job_id),
                timeout=self.config.timeout_seconds,
            )
        else:
            job = await self._poll_until_done(job_id)

        wall_time = time.perf_counter() - start_time

        status = job.get("status")
        if status in _FAILURE_STATUSES:
            message = job.get("failure", {}).get("error") or f"job {status}"
            raise RuntimeError(f"IonQ job {job_id} {status}: {message}")

        histogram = job.get("data", {}).get("histogram")
        if histogram is None:
            results = await self._request("GET", f"/jobs/{job_id}/results")
            histogram = self._extract_histogram(results)

        counts = self._histogram_to_counts(histogram, shots, num_qubits)

        # Prefer IonQ's reported execution time; fall back to measured wall time.
        # Use an explicit None check so a valid 0 is preserved (not treated as missing).
        reported_time = job.get("execution_time")
        execution_time_ms = reported_time if reported_time is not None else wall_time * 1000
        return ExecutionResult(
            counts=counts,
            backend=self.config.target,
            execution_time_ms=execution_time_ms,
            shots=shots,
            raw_result=job,
            metadata={
                "job_id": job_id,
                "target": self.config.target,
                "provider": "ionq",
                "noise_model": self.config.noise_model,
                "wall_time_ms": wall_time * 1000,
            },
        )

    async def _poll_until_done(self, job_id: str) -> dict[str, Any]:
        """Poll a job until it reaches a terminal state.

        Args:
            job_id: The IonQ job id to poll.

        Returns:
            The terminal job object.
        """
        while True:
            job = await self._request("GET", f"/jobs/{job_id}")
            status = job.get("status")
            if status in _SUCCESS_STATUSES or status in _FAILURE_STATUSES:
                return job
            await asyncio.sleep(self.config.poll_interval_seconds)

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """Make an authenticated request to the IonQ API (off the event loop).

        Args:
            method: HTTP method ("GET", "POST", "PUT").
            path: API path relative to ``base_url`` (e.g. "/jobs").
            **kwargs: Extra arguments forwarded to the HTTP session (e.g. ``json``).

        Returns:
            The parsed JSON response body.
        """
        loop = asyncio.get_running_loop()
        url = f"{self.config.base_url}{path}"

        # Merge any caller-provided headers with auth headers so they don't collide
        # with the explicit headers= argument below (auth takes precedence).
        headers = {**kwargs.pop("headers", {}), **self._auth_headers()}

        def _call() -> dict[str, Any]:
            response = self._do_request(
                method, url, headers=headers, timeout=_HTTP_TIMEOUT_SECONDS, **kwargs
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            return data

        return await loop.run_in_executor(None, _call)

    async def cancel(self, job_id: str) -> bool:
        """Cancel a running IonQ job.

        Args:
            job_id: The job id to cancel.

        Returns:
            True if cancellation succeeded, False otherwise.
        """
        try:
            await self._request("PUT", f"/jobs/{job_id}/status/cancel")
            return True
        except Exception:
            return False

    async def get_status(self) -> DeviceStatus:
        """Get live device status from the IonQ API.

        Maps IonQ backend availability to the standard DeviceStatus states.
        Returns "maintenance" on any error so callers can degrade gracefully.
        """
        try:
            backend = await self._request("GET", f"/backends/{self.config.target}")
            raw_status = str(backend.get("status", "")).lower()
            status = self._IONQ_STATUS_MAP.get(raw_status, "maintenance")

            # Explicit None check so a valid queue_depth of 0 is not discarded.
            queue_depth = backend.get("queue_depth")
            if queue_depth is None:
                queue_depth = backend.get("jobs_queued")
            avg_queue = backend.get("average_queue_time")
            queue_time_seconds = int(avg_queue) if avg_queue is not None else None

            return DeviceStatus(
                status=status,
                queue_depth=queue_depth,
                queue_time_seconds=queue_time_seconds,
            )
        except Exception:
            return DeviceStatus(status="maintenance", queue_depth=None, queue_time_seconds=None)
