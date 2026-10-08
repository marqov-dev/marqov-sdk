"""IonQ Direct API executor for running circuits on IonQ hardware.

This module provides IonQExecutor for executing quantum circuits directly against
IonQ's REST API (https://api.ionq.co), bypassing the AWS Braket intermediary.
Talking to IonQ directly shortens the round-trip, surfaces richer error messages,
and exposes IonQ-specific features (e.g. simulator noise models) without requiring
an AWS account or S3 bucket.

Circuits are converted with ``circuit.to_qiskit()`` and dumped to OpenQASM, then
submitted using IonQ's ``qasm`` input format by default. The explicit v0.4
ideal-simulator route uses QASM3 and retains probability artifact provenance.

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
import base64
import contextlib
import hashlib
import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from marqov.executors._counts import allocate_counts
from marqov.executors.base import BaseExecutor, DeviceStatus, ExecutionResult

if TYPE_CHECKING:
    from marqov.circuits import Circuit

# Job statuses that mean the job has finished successfully.
_SUCCESS_STATUSES = frozenset({"completed"})
# Job statuses that mean the job has finished without usable results.
_FAILURE_STATUSES = frozenset({"failed", "canceled", "cancelled"})
# Job statuses that mean the job is still in progress and worth polling again.
# Anything outside these three sets is treated as unknown and raises, so an
# unmodelled terminal state (e.g. "deleted") surfaces as a failure instead of an
# endless poll. A status IonQ legitimately uses for in-progress work belongs
# here, as a one-line addition.
_PENDING_STATUSES = frozenset({"submitted", "ready", "running"})

# Per-request HTTP timeout (seconds). Overall job completion is bounded separately
# by IonQExecutorConfig.timeout_seconds around polling.
_HTTP_TIMEOUT_SECONDS = 30
_LOGGER = logging.getLogger(__name__)

# Default overall budget for a single job (seconds). Longer than any IonQ queue
# the SDK has been used against, and still finite: an unconfigured executor must
# not be able to wait forever. Set timeout_seconds=None to opt back out.
_DEFAULT_TIMEOUT_SECONDS = 3600.0


@dataclass
class IonQExecutorConfig:
    """Configuration for the IonQ Direct API executor.

    Attributes:
        target: IonQ backend target (e.g. "simulator", "qpu.aria-1",
            "qpu.forte-1").
        api_key: IonQ API key. If None, falls back to the ``IONQ_API_KEY``
            environment variable at request time.
        base_url: Base URL for the IonQ REST API; defaults to the selected version.
        api_version: Explicit protocol version: legacy 0.3 (default) or opt-in
            0.4 QASM3/probabilities-v2 ideal simulator route.
        poll_interval_seconds: Polling interval while waiting for a job to finish.
        timeout_seconds: Maximum time to wait for job completion, one hour by
            default. On timeout the executor issues a best-effort cancel for the
            submitted job. Set to None for an unbounded wait, which leaves a
            stuck job with nothing to stop it.
        noise_model: Optional simulator noise model (e.g. "aria-1", "forte-1").
            Only applied when ``target`` is the simulator.
    """

    target: str = "simulator"
    api_key: str | None = None
    base_url: str | None = None
    poll_interval_seconds: float = 1.0
    timeout_seconds: float | None = _DEFAULT_TIMEOUT_SECONDS
    noise_model: str | None = None
    api_version: str = "0.3"

    def __post_init__(self) -> None:
        if self.api_version not in {"0.3", "0.4"}:
            raise ValueError("IonQ api_version must be 0.3 or 0.4")
        if self.base_url is None:
            self.base_url = f"https://api.ionq.co/v{self.api_version}"
        self.base_url = self.base_url.rstrip("/")
        for version in ("0.3", "0.4"):
            if self.base_url.endswith(f"/v{version}") and version != self.api_version:
                raise ValueError("IonQ base_url and api_version disagree")
        if self.api_version == "0.4" and (self.target != "simulator" or self.noise_model):
            raise ValueError("IonQ v0.4 route currently supports the ideal simulator only")


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

        IonQ returns a sparse histogram mapping state indices (as strings) to
        probabilities. This conversion formats each index with ``format()`` and
        does not reverse it, so the most significant bit of the index becomes the
        leftmost character, which under Marqov's convention (qubit 0 leftmost)
        reads as qubit 0.

        Unverified against hardware, and disputed by the vendor documentation.
        IonQ's Direct API guide
        (https://docs.ionq.com/guides/direct-api-submission) states that "the
        output keys are little-endian integers: qubit i from the submitted
        program occupies the bit with value 2^i, so the rightmost bit of the
        key's binary form is qubit zero", which would require reversing the
        formatted string. The ordering is deliberately left unchanged here: no
        live IonQ run is on record either way, and flipping it on a documentation
        reading alone would silently change every existing caller's results. The
        current behaviour is pinned by an asymmetric test
        (``TestHistogramToCounts::test_asymmetric_index_pins_current_bit_order``)
        so a future change is deliberate, and resolving it needs a live run
        against a known asymmetric circuit.

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

        # Map IonQ's state indices to bitstrings, then delegate the
        # shot-conserving allocation to the shared helper (see _counts.py).
        probabilities = {
            format(int(index), f"0{num_qubits}b"): float(probability)
            for index, probability in histogram.items()
        }
        return allocate_counts(probabilities, shots)

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
            RuntimeError: If the job fails, is canceled by IonQ, or reports a
                status the executor does not recognize.
            ValueError: If no API key is available or the returned job ID is unusable.
            TimeoutError: If the job does not finish within ``timeout_seconds``.
                The remote job is sent a best-effort cancel first.

        Cancellation during submission stops waiting without replaying the POST.
        The request worker retains per-call ownership and requests cancellation
        if it obtains a usable job ID. Cancellation notes and logs report
        uncertain acceptance or failed cleanup. HTTP socket waits have finite
        timeouts, but DNS, a continuing response or worker shutdown can take
        longer; this is not a hard process deadline or guaranteed provider cancel.
        """
        circuit = self._validate_circuit(circuit)

        if self.config.api_version == "0.4":
            owned = IonQExecutor(
                replace(self.config, api_key=self._auth_headers()["Authorization"][7:]),
                session=self._session,
            )
            return await owned._execute_v04(circuit, shots)

        start_time = time.perf_counter()

        qasm, num_qubits = self._circuit_to_qasm(circuit)

        payload: dict[str, Any] = {
            "target": self.config.target,
            "shots": shots,
            "input": {"format": "qasm", "data": qasm},
        }
        if self.config.noise_model and self.config.target == "simulator":
            payload["noise"] = {"model": self.config.noise_model}

        # The submission worker owns cleanup if its caller stops waiting.
        submit_response = await self._submit_job(payload)
        job_id = submit_response["id"]

        job_artifact: dict[str, Any] = {}
        # Poll until the job reaches a terminal state. If the wait is cut short,
        # the job is still queued or running on IonQ's side, and billable, so ask
        # IonQ to stop it before propagating.
        try:
            if self.config.timeout_seconds is not None:
                job = await asyncio.wait_for(
                    self._poll_until_done(job_id, artifact=job_artifact),
                    timeout=self.config.timeout_seconds,
                )
            else:
                job = await self._poll_until_done(job_id, artifact=job_artifact)
        except (TimeoutError, asyncio.CancelledError):
            await self._cancel_interrupted_job(job_id)
            raise

        wall_time = time.perf_counter() - start_time

        status = job.get("status")
        if status in _FAILURE_STATUSES:
            message = job.get("failure", {}).get("error") or f"job {status}"
            raise RuntimeError(f"IonQ job {job_id} {status}: {message}")

        artifact: dict[str, Any] = {}
        histogram = job.get("data", {}).get("histogram")
        if histogram is None:
            results = await self._request("GET", f"/jobs/{job_id}/results", artifact=artifact)
            histogram = self._extract_histogram(results)

        if not artifact:
            artifact = job_artifact or {
                "payload": job,
                "body_base64": None,
                "sha256": None,
                "source": "inline-job",
            }
        artifact.update({"job_id": job_id, "format": "legacy-probability-histogram"})
        source_probabilities = {
            format(int(k), f"0{num_qubits}b"): float(v) for k, v in histogram.items()
        }
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
                "api_version": "0.3",
                "result_artifact": artifact,
                "source_probabilities": source_probabilities,
                "counts_kind": "probability-derived",
                "counts_allocation": "hamilton",
                "raw_shot_eligible": False,
                "wire_order": "legacy-msb-first-unqualified",
                "wall_time_ms": wall_time * 1000,
            },
        )

    async def _execute_v04(self, circuit: Circuit, shots: int) -> ExecutionResult:
        """Explicit ideal-simulator QASM3 / probabilities-v2 route.

        This instance has a per-call configuration/credential snapshot. The
        legacy route and its historical integer ordering remain unchanged.
        """
        from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qasm3

        if type(shots) is not int or not 1 <= shots <= 1_000_000:
            raise ValueError("IonQ shots must be an integer between 1 and 1000000")
        labels = {q for gate in circuit.to_dict()["gates"] for q in gate["qubits"]}
        if labels != set(range(circuit.num_qubits)) or any(type(q) is not int for q in labels):
            raise ValueError("IonQ v0.4 requires dense integer wires starting at zero")
        qc = circuit.to_qiskit()  # type: ignore[no-untyped-call]
        if not qc.num_qubits:
            raise ValueError("IonQ v0.4 requires at least one qubit")
        supported = {"h", "x", "y", "z", "s", "t", "rx", "ry", "rz", "cx", "cz", "swap"}
        if qc.parameters or any(op.operation.name not in supported for op in qc.data):
            raise ValueError("IonQ v0.4 route supports concrete canonical builder gates only")
        if any(not math.isfinite(float(p)) for op in qc.data for p in op.operation.params):
            raise ValueError("IonQ v0.4 rotation parameters must be finite")
        canonical = QuantumCircuit(
            QuantumRegister(qc.num_qubits, "q"), ClassicalRegister(qc.num_qubits, "c")
        )
        canonical.compose(qc, inplace=True)
        qc = canonical
        for i in range(qc.num_qubits):
            qc.measure(i, i)
        program = qasm3.dumps(qc)
        payload = {
            "type": "ionq.qasm3.v1",
            "backend": "simulator",
            "shots": shots,
            "input": {"data": program},
        }
        start = time.perf_counter()
        submitted = await self._submit_job(payload)
        job_id = submitted["id"]
        identity = {
            "provider": "ionq",
            "api_version": "0.4",
            "job_id": job_id,
            "target": "simulator",
            "base_url": self.config.base_url,
        }
        artifact: dict[str, Any] = {}
        try:
            async with asyncio.timeout(self.config.timeout_seconds):
                job = await self._poll_until_done(job_id)
                if job.get("id") != job_id:
                    raise ValueError("IonQ completed job identity mismatch")
                if job["status"] in _FAILURE_STATUSES:
                    message = f"IonQ job {job_id} {job['status']}"
                    if job.get("failure"):
                        message += f": {job['failure']}"
                    raise RuntimeError(message)
                if job.get("backend") != "simulator" or job.get("type") != "ionq.qasm3.v1":
                    raise ValueError("IonQ completed job type/backend mismatch")
                fmt = "ionq.result.probabilities.json.v2"
                results = job.get("results")
                descriptor = results.get(fmt) if isinstance(results, dict) else None
                if (
                    not isinstance(descriptor, dict)
                    or descriptor.get("format") != fmt
                    or descriptor.get("media_type") != "application/json"
                ):
                    raise ValueError("IonQ job has no supported probabilities-v2 artifact")
                artifact_id = descriptor.get("id")
                if not isinstance(artifact_id, str) or not artifact_id:
                    raise ValueError("IonQ result artifact has no usable ID")
                artifact.update({"format": fmt, "id": artifact_id, "job_id": job_id})
                result = await self._request(
                    "GET",
                    f"/jobs/{quote(job_id, safe='')}/artifacts/{quote(artifact_id, safe='')}",
                    artifact=artifact,
                )
                if artifact["body_base64"] is None:
                    raise ValueError("IonQ artifact transport must expose original response bytes")
                if not isinstance(result, dict):
                    raise ValueError("IonQ artifact must be a JSON object")  # noqa: TRY004 - invalid provider JSON
                probability_data = result.get("probabilities", {})
                if not isinstance(probability_data, dict):
                    raise ValueError("IonQ artifact probabilities must be a JSON object")  # noqa: TRY004 - invalid provider JSON
                registers = probability_data.get("registers", {})
                if not isinstance(registers, dict):
                    raise ValueError("IonQ artifact registers must be a JSON object")  # noqa: TRY004 - invalid provider JSON
                probabilities = registers.get("output_all")
                if not isinstance(probabilities, dict) or not probabilities:
                    raise ValueError("IonQ artifact has no output_all probability distribution")
                for bits, probability in probabilities.items():
                    if (
                        not isinstance(bits, str)
                        or len(bits) != qc.num_qubits
                        or set(bits) - {"0", "1"}
                    ):
                        raise ValueError("IonQ output_all bitstring width/alphabet mismatch")
                    if (
                        type(probability) not in (int, float)
                        or not math.isfinite(probability)
                        or not 0 <= probability <= 1
                    ):
                        raise ValueError("IonQ artifact contains invalid probability")
                if not math.isclose(
                    math.fsum(probabilities.values()), 1.0, rel_tol=0, abs_tol=1e-10
                ):
                    raise ValueError("IonQ probabilities must sum to one")
        except (TimeoutError, asyncio.CancelledError) as error:
            await self._cancel_interrupted_job(job_id)
            error.remote_job = identity  # type: ignore[union-attr]
            raise
        except Exception as error:
            error.remote_job = identity  # type: ignore[attr-defined]
            error.result_artifact = artifact  # type: ignore[attr-defined]
            raise
        wall_ms = (time.perf_counter() - start) * 1000
        reported_ms = job.get("execution_duration_ms")
        return ExecutionResult(
            counts=allocate_counts(probabilities, shots),
            backend="simulator",
            shots=shots,
            execution_time_ms=reported_ms if reported_ms is not None else wall_ms,
            raw_result=job,
            metadata={
                **identity,
                "wall_time_ms": wall_ms,
                "input_type": "ionq.qasm3.v1",
                "input_sha256": hashlib.sha256(program.encode()).hexdigest(),
                "result_artifact": artifact,
                "source_probabilities": probabilities,
                "counts_kind": "probability-derived",
                "counts_allocation": "hamilton",
                "raw_shot_eligible": False,
                "wire_order": "q0-leftmost",
                "wire_order_evidence": "vendor-contract/offline-tests; live-controls-required",
                "result_register": "output_all",
                "num_qubits": qc.num_qubits,
                "wire_labels": list(range(qc.num_qubits)),
                "ideal_shots_ignored": True,
            },
        )

    async def _poll_until_done(
        self, job_id: str, *, artifact: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Poll a job until it reaches a terminal state.

        Returns as soon as the job reports a success or failure status. A status
        that is neither terminal nor a known in-progress one is treated as an
        unmodelled terminal state and raises, because polling it again would
        never finish.

        Args:
            job_id: The IonQ job id to poll.

        Returns:
            The terminal job object.

        Raises:
            RuntimeError: If IonQ reports a status outside the success, failure
                and pending sets.
        """
        while True:
            job = await self._request(
                "GET", f"/jobs/{job_id}", **({"artifact": artifact} if artifact is not None else {})
            )
            status = job.get("status")
            if status in _SUCCESS_STATUSES or status in _FAILURE_STATUSES:
                return job
            pending = (
                _PENDING_STATUSES | {"started"}
                if self.config.api_version == "0.4"
                else _PENDING_STATUSES
            )
            if status not in pending:
                raise RuntimeError(
                    f"IonQ job {job_id} reported unknown status {status!r}. "
                    "Polling stopped because this status is not a known "
                    "in-progress state; if IonQ uses it for work still in "
                    "flight, add it to marqov.executors.ionq._PENDING_STATUSES."
                )
            await asyncio.sleep(self.config.poll_interval_seconds)

    async def _cancel_interrupted_job(self, job_id: str) -> None:
        """Issue a best-effort cancel for a job whose wait was cut short.

        Called when the executor stops waiting on a job that has not reached a
        terminal state (a timeout, or cancellation of the awaiting task), so the
        remote job does not keep running and billing with nobody watching it.
        Failures are swallowed: the original TimeoutError or CancelledError is
        the interesting one, and a failed cancel must not mask it.

        Args:
            job_id: The IonQ job id to cancel.
        """
        # cancel() already swallows request failures; the suppress here covers a
        # further cancellation arriving while the cancel request is in flight.
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await self.cancel(job_id)

    async def _submit_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Keep per-call ownership across cancellation of a blocking POST.

        Cancellation returns promptly. The worker finishes its existing request
        and attempts one cancellation if it obtains a job ID. No POST is replayed.
        HTTP timeouts bound socket waits, not total thread/process lifetime.
        """
        loop = asyncio.get_running_loop()
        base_url = self.config.base_url
        headers = self._auth_headers()
        lock = threading.Lock()
        interrupted = False
        job_id: str | None = None
        cancel_claimed = False

        def cancel_submitted(owned_id: str) -> None:
            try:
                self._request_sync("PUT", f"{base_url}/jobs/{owned_id}/status/cancel", headers)
            except Exception:  # noqa: BLE001 - cleanup failures must not mask cancellation
                _LOGGER.warning(
                    "IonQ cancellation request failed for interrupted submission %r; "
                    "the job may still be running",
                    owned_id,
                )
            else:
                _LOGGER.info("IonQ cancellation requested for interrupted submission %r", owned_id)

        def submit() -> dict[str, Any]:
            nonlocal job_id, cancel_claimed
            try:
                response = self._request_sync("POST", f"{base_url}/jobs", headers, json=payload)
                owned_id = response["id"]
                if not isinstance(owned_id, str) or not owned_id:
                    raise ValueError("IonQ submission response has no usable job ID")
            except Exception:
                with lock:
                    was_interrupted = interrupted
                if was_interrupted:
                    _LOGGER.warning(
                        "IonQ submission ended without a usable job ID after cancellation; "
                        "acceptance is unknown and the POST must not be replayed"
                    )
                raise
            with lock:
                job_id = owned_id
                # Compatibility tracking only; cleanup uses this call's ID.
                self._current_job_id = owned_id
                needs_cancel = interrupted and not cancel_claimed
                if needs_cancel:
                    cancel_claimed = True
            if needs_cancel:
                cancel_submitted(owned_id)
            return response

        future = loop.run_in_executor(None, submit)
        try:
            return await future
        except asyncio.CancelledError as error:
            with lock:
                interrupted = True
                owned_id = job_id
                needs_cancel = owned_id is not None and not cancel_claimed
                if needs_cancel:
                    cancel_claimed = True
            if needs_cancel:
                # The worker may have returned before the cancellation reached
                # this coroutine. A new worker owns cleanup in that race.
                try:
                    loop.run_in_executor(None, cancel_submitted, owned_id)
                except RuntimeError:
                    _LOGGER.warning(
                        "IonQ cleanup could not be scheduled for interrupted submission %r; "
                        "the job may still be running",
                        owned_id,
                    )
            error.add_note(
                "IonQ submission was interrupted. Acceptance may be unresolved; "
                "cleanup requests cancellation if a job ID is obtained. "
                "Do not automatically resubmit. Cleanup does not guarantee cancellation."
            )
            _LOGGER.warning(
                "IonQ submission interrupted (job ID %r); cancellation is best effort "
                "and the POST must not be replayed",
                owned_id,
            )
            raise

    def _request_sync(
        self, method: str, url: str, headers: dict[str, str], **kwargs: Any
    ) -> dict[str, Any]:
        """Perform one request without retries; used by request/cleanup workers."""
        artifact = kwargs.pop("artifact", None)
        response = self._do_request(
            method, url, headers=headers, timeout=_HTTP_TIMEOUT_SECONDS, **kwargs
        )
        response.raise_for_status()
        body = getattr(response, "content", None)
        if artifact is not None:
            artifact.update(
                {
                    "source": url,
                    "body_base64": base64.b64encode(body).decode("ascii")
                    if isinstance(body, bytes)
                    else None,
                    "sha256": hashlib.sha256(body).hexdigest() if isinstance(body, bytes) else None,
                }
            )
        data: dict[str, Any] = (
            json.loads(body)
            if artifact is not None and isinstance(body, bytes)
            else response.json()
        )
        if artifact is not None:
            artifact["payload"] = data
        return data

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
            return self._request_sync(method, url, headers, **kwargs)

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
