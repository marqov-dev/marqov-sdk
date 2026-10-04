"""Explicit QB QCStack transport; never uses Qristal's retrying hardware session.

Wire schema: Qristal a5c3e5f qdk.cpp/visitor_CZ.cpp. Direct access remains
unqualified until the endpoint/model/account mapping is connected-qualified.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter

from marqov.executors.base import BaseExecutor, ExecutionResult


@dataclass(frozen=True)
class QBRemoteConfig:
    endpoint: str
    target: str
    account: str
    token: str = field(repr=False)
    model: str = "QB-QDK2-CZ"
    timeout_seconds: float = 120
    request_timeout_seconds: float = 5
    poll_interval_seconds: float = 2

    def __post_init__(self) -> None:
        u = urlsplit(self.endpoint)
        if (
            u.scheme != "https"
            or not u.hostname
            or u.username
            or u.password
            or u.query
            or u.fragment
        ):
            raise ValueError("Explicit HTTPS QB endpoint required")
        if self.model != "QB-QDK2-CZ":
            raise ValueError("Only QB-QDK2-CZ native workload schema is supported")
        if any(
            not isinstance(x, str) or not x.strip() for x in (self.target, self.account, self.token)
        ):
            raise ValueError("Explicit QB target/account/token required")
        for x in (self.timeout_seconds, self.request_timeout_seconds, self.poll_interval_seconds):
            if type(x) not in (int, float) or not math.isfinite(x) or x <= 0:
                raise ValueError("Finite positive QB time budgets required")


@dataclass(frozen=True)
class QBRemoteJob:
    job_id: int
    endpoint: str
    target: str
    account: str
    model: str
    payload: str

    def __post_init__(self) -> None:
        if type(self.job_id) is not int or self.job_id < 0:
            raise ValueError("Invalid QB circuit identity")
        if any(
            not isinstance(x, str) or not x
            for x in (self.endpoint, self.target, self.account, self.model, self.payload)
        ):
            raise ValueError("Invalid QB retained request")


class QBRemoteExecutionError(RuntimeError):
    def __init__(self, job_id: int | None, *, acceptance_unknown: bool = False):
        self.job_id = job_id
        self.acceptance_unknown = acceptance_unknown
        super().__init__(
            "QB acceptance unknown; do not resubmit"
            if acceptance_unknown
            else "QB result unresolved; inspect retained circuit"
        )


def _json_object(content: str | bytes | bytearray) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate QB JSON field")
            result[key] = value
        return result

    def number(value: str) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Nonfinite QB JSON value")
        return result

    def invalid(value: str) -> Any:
        raise ValueError("Invalid QB JSON constant")

    result = json.loads(
        content, object_pairs_hook=pairs, parse_float=number, parse_constant=invalid
    )
    if not isinstance(result, dict):
        raise TypeError("QB JSON object required")
    return result


def _payload(circuit: Any, shots: int) -> dict[str, Any]:
    if type(shots) is not int or not 0 < shots <= 100000:
        raise ValueError("QB shots must be between 1 and 100000")
    width = circuit.num_qubits
    if type(width) is not int or not 0 < width <= 28:
        raise ValueError("QB native workload width must be between 1 and 28")
    gates = circuit.to_dict()["gates"]
    if not gates or len(gates) > 10000:
        raise ValueError("QB native workload needs 1 to 10000 gates")
    out = []
    for gate in gates:
        name, qubits, params = gate["gate"], gate["qubits"], gate["params"]
        if any(type(q) is not int or not 0 <= q < width for q in qubits):
            raise ValueError("Invalid QB qubit")
        if name in ("Rx", "Ry") and len(qubits) == 1 and len(params) == 1:
            angle = params[0]
            if (
                isinstance(angle, bool)
                or not isinstance(angle, (int, float))
                or not math.isfinite(angle)
            ):
                raise ValueError("QB requires finite numeric angles")
            # Vendor XASM grammar requires fixed-point angles, not exponent notation.
            angle = math.remainder(angle, 2 * math.pi)
            out.append(f"{name}(q[{qubits[0]}],{angle:.16f})")
        elif name == "CZ" and len(qubits) == 2 and len(set(qubits)) == 2 and not params:
            out.append(f"CZ(q[{qubits[0]}],q[{qubits[1]}])")
        else:
            raise ValueError("QB remote supports native Rx, Ry and CZ only")
    return {
        "command": "circuit",
        "settings": {"shots": shots, "results": "normal", "shot_fulfilment_strategy": "exact"},
        "init": [0] * width,
        "circuit": out,
        "measure": [[q, q] for q in range(width)],
    }


def _retained_body(payload: str) -> dict[str, Any]:
    if len(payload.encode()) > 2 * 1024 * 1024:
        raise ValueError("QB retained request too large")
    body = _json_object(payload)
    if not isinstance(body, dict) or set(body) != {
        "command",
        "settings",
        "init",
        "circuit",
        "measure",
    }:
        raise ValueError("Invalid QB retained request")
    init, settings, gates = body["init"], body["settings"], body["circuit"]
    if (
        not isinstance(init, list)
        or not 0 < len(init) <= 28
        or any(type(x) is not int or x != 0 for x in init)
    ):
        raise ValueError("Invalid QB initial state")
    width = len(init)
    if (
        body["command"] != "circuit"
        or not isinstance(settings, dict)
        or set(settings) != {"shots", "results", "shot_fulfilment_strategy"}
        or type(settings["shots"]) is not int
        or not 0 < settings["shots"] <= 100000
        or settings["results"] != "normal"
        or settings["shot_fulfilment_strategy"] != "exact"
        or body["measure"] != [[q, q] for q in range(width)]
        or not isinstance(gates, list)
        or not 0 < len(gates) <= 10000
    ):
        raise ValueError("Invalid QB retained settings")
    for gate in gates:
        if not isinstance(gate, str):
            raise TypeError("Invalid QB retained gate")
        rotation = re.fullmatch(r"R[xy]\(q\[(\d+)\],(-?\d+\.\d{16})\)", gate)
        cz = re.fullmatch(r"CZ\(q\[(\d+)\],q\[(\d+)\]\)", gate)
        if (
            rotation
            and int(rotation[1]) < width
            and math.isfinite(float(rotation[2]))
            and abs(float(rotation[2])) <= math.pi
        ):
            continue
        if cz and int(cz[1]) < width and int(cz[2]) < width and cz[1] != cz[2]:
            continue
        raise ValueError("Invalid QB retained native gate")
    if json.dumps(body, sort_keys=True, separators=(",", ":")) != payload:
        raise ValueError("QB retained request must be canonical")
    return body


class QBRemoteExecutor(BaseExecutor):
    """One circuit POST; retained readback sends GET only. No reservation writes."""

    max_response_bytes = 2 * 1024 * 1024

    def __init__(self, config: QBRemoteConfig):
        self.config = config
        self.last_job: QBRemoteJob | None = None

    def _request_unbounded(
        self, method: str, path: str, deadline: float, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            with requests.Session() as session:
                session.trust_env = False
                session.mount("https://", HTTPAdapter(max_retries=0))
                with session.request(
                    method,
                    self.config.endpoint.rstrip("/") + "/" + path,
                    headers={"Authorization": "Bearer " + self.config.token},
                    json=body,
                    allow_redirects=False,
                    verify=True,
                    stream=True,
                    timeout=min(remaining, self.config.request_timeout_seconds),
                ) as response:
                    if response.status_code == 425 and method == "GET":
                        return {"data": None}
                    if response.status_code != 200:
                        raise RuntimeError
                    content = bytearray()
                    for chunk in response.iter_content(chunk_size=1):
                        if time.monotonic() >= deadline:
                            raise TimeoutError
                        content.extend(chunk)
                        if len(content) > self.max_response_bytes:
                            raise ValueError
            result = _json_object(content)
            if not isinstance(result, dict):
                raise TypeError
            return result
        except Exception:  # noqa: BLE001 — never leak credential-bearing requests
            raise RuntimeError("QB HTTP request unresolved") from None

    def _request(
        self, method: str, path: str, deadline: float, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        # Bound caller wait even if headers/DNS trickle. A timed-out one-send
        # worker may still complete, so callers must retain unknown acceptance.
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("QB HTTP deadline elapsed")
        results: queue.Queue[Any] = queue.Queue(maxsize=1)

        def send() -> None:
            try:
                results.put((True, self._request_unbounded(method, path, deadline, body)))
            except Exception:  # noqa: BLE001 — sanitize worker failures
                results.put((False, None))

        threading.Thread(target=send, daemon=True).start()
        try:
            success, result = results.get(timeout=remaining)
        except queue.Empty:
            raise RuntimeError("QB HTTP deadline elapsed; request may still complete") from None
        if not success or not isinstance(result, dict):
            raise RuntimeError("QB HTTP request unresolved")
        return result

    async def execute(self, circuit: Any, shots: int = 1000, **kwargs: Any) -> ExecutionResult:
        if kwargs:
            raise TypeError("QB remote supports no extra execution options")
        body = _payload(self._validate_circuit(circuit), shots)
        return await asyncio.to_thread(self._submit, body)

    def _submit(self, body: dict[str, Any]) -> ExecutionResult:
        deadline = time.monotonic() + self.config.timeout_seconds
        self.last_job = None
        try:
            reply = self._request("POST", "api/v1/circuits", deadline, body)
            job_id = reply["id"]
            if type(job_id) is not int or job_id < 0:
                raise ValueError
        except Exception:  # noqa: BLE001 — write acceptance may be unknown
            raise QBRemoteExecutionError(None, acceptance_unknown=True) from None
        job = QBRemoteJob(
            job_id,
            self.config.endpoint.rstrip("/"),
            self.config.target,
            self.config.account,
            self.config.model,
            json.dumps(body, sort_keys=True, separators=(",", ":")),
        )
        self.last_job = job
        return self._poll(job, deadline)

    async def readback(self, job: QBRemoteJob) -> ExecutionResult:
        if type(job) is not QBRemoteJob or (job.endpoint, job.target, job.account, job.model) != (
            self.config.endpoint.rstrip("/"),
            self.config.target,
            self.config.account,
            self.config.model,
        ):
            raise ValueError("QB readback endpoint/target/account/model mismatch")
        _retained_body(job.payload)
        return await asyncio.to_thread(
            self._poll, job, time.monotonic() + self.config.timeout_seconds
        )

    def _poll(self, job: QBRemoteJob, deadline: float) -> ExecutionResult:
        start = time.monotonic()
        try:
            body = _retained_body(job.payload)
            width, shots = len(body["init"]), body["settings"]["shots"]
            while True:
                result = self._request("GET", f"api/v1/circuits/{job.job_id}", deadline)
                if "id" in result and (type(result["id"]) is not int or result["id"] != job.job_id):
                    raise ValueError("QB circuit identity mismatch")
                if "data" not in result:
                    raise ValueError("QB result schema mismatch")
                rows = result["data"]
                if rows is not None:
                    if not isinstance(rows, list) or len(rows) != shots:
                        raise ValueError("QB shot accounting mismatch")
                    counts: dict[str, int] = {}
                    for row in rows:
                        if (
                            not isinstance(row, list)
                            or len(row) != width
                            or any(type(x) is not int or x not in (0, 1) for x in row)
                        ):
                            raise ValueError("QB invalid sample")
                        key = "".join(str(x) for x in row)
                        counts[key] = counts.get(key, 0) + 1
                    return ExecutionResult(
                        counts=counts,
                        backend=job.target,
                        shots=shots,
                        execution_time_ms=(time.monotonic() - start) * 1000,
                        metadata={
                            "vendor": "Quantum Brilliance",
                            "framework": "QCStack",
                            "access_path": "direct",
                            "compute_provider": "remote",
                            "endpoint": job.endpoint,
                            "target": job.target,
                            "account": job.account,
                            "model": job.model,
                            "job_id": job.job_id,
                            "payload_sha256": hashlib.sha256(job.payload.encode()).hexdigest(),
                            "bit_order": "measurement slots in ascending qubit order",
                            "target_binding": "configured endpoint/model; API does not echo account/target/payload",
                            "scope": "direct SDK; connected qualification required",
                        },
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                time.sleep(min(remaining, self.config.poll_interval_seconds))
        except Exception:  # noqa: BLE001 — retain known identity without replay
            raise QBRemoteExecutionError(job.job_id) from None

    async def get_status(self) -> Any:
        raise NotImplementedError("Pinned QB circuit API exposes no device status contract")

    async def cancel(self, job_id: str) -> bool:
        return False
