"""Direct SpeQtrum execution using the pinned QiliSDK 0.3.0 wire schema.

No keyring, ambient credentials, write retries, or managed billing authority.
An uncertain submission must be reconciled by the caller, never resubmitted.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import Any, cast
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter

from marqov.executors.base import BaseExecutor, DeviceStatus, ExecutionResult
from marqov.executors.qilisdk import QiliSDKExecutor


@dataclass(frozen=True)
class SpeQtrumExecutorConfig:
    device_code: str
    username: str
    api_key: str = field(repr=False)
    api_url: str = "https://qilimanjaro.ddns.net/public-api/api/v1"
    audience: str = "urn:qilimanjaro.tech:public-api:beren"
    timeout_seconds: float = 120
    request_timeout_seconds: float = 10
    poll_interval_seconds: float = 2

    def __post_init__(self) -> None:
        for value in (self.device_code, self.username, self.api_key, self.audience):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Explicit SpeQtrum device, account, key and audience required")
        url = urlsplit(self.api_url)
        if (
            url.scheme != "https"
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("SpeQtrum API URL must be an explicit HTTPS origin/path")
        for budget in (
            self.timeout_seconds,
            self.request_timeout_seconds,
            self.poll_interval_seconds,
        ):
            if type(budget) not in (float, int) or not math.isfinite(budget) or budget <= 0:
                raise ValueError("SpeQtrum time budgets must be finite and positive")


class SpeQtrumExecutionError(RuntimeError):
    def __init__(self, job_id: int | None, *, acceptance_unknown: bool = False):
        self.job_id = job_id
        self.acceptance_unknown = acceptance_unknown
        super().__init__(
            "SpeQtrum submission acceptance unknown; do not resubmit"
            if acceptance_unknown
            else "SpeQtrum result unresolved or unsuccessful; inspect retained job"
        )


@dataclass(frozen=True)
class SpeQtrumJob:
    """Retain this account-bound readback record across caller restarts."""

    job_id: int
    device_code: str
    username: str
    api_url: str
    audience: str
    payload: str
    execute_type: str
    shots: int
    num_qubits: int

    def __post_init__(self) -> None:
        if any(type(x) is not int or x <= 0 for x in (self.job_id, self.shots, self.num_qubits)):
            raise ValueError("Invalid SpeQtrum retained job identity")
        if self.execute_type not in ("digital_propagation", "analog_evolution"):
            raise ValueError("Invalid SpeQtrum retained functional type")
        if any(
            not isinstance(x, str) or not x
            for x in (self.device_code, self.username, self.api_url, self.audience, self.payload)
        ):
            raise ValueError("Invalid SpeQtrum retained request")


class SpeQtrumHTTPClient:
    """Concrete one-send HTTP client. Login is explicit and has no auth replay."""

    max_response_bytes = 2 * 1024 * 1024

    def __init__(self, config: SpeQtrumExecutorConfig):
        self.config = config

    def _request(
        self,
        method: str,
        path: str,
        deadline: float,
        *,
        token: str | None = None,
        body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("SpeQtrum deadline elapsed")
        # A fresh session cannot inherit cached auth, cookies, proxies or retry policy.
        with requests.Session() as session:
            session.trust_env = False
            session.mount("https://", HTTPAdapter(max_retries=0))
            headers = {"User-Agent": "marqov-speqtrum/qilisdk-0.3.0"}
            if token is not None:
                headers["Authorization"] = f"Bearer {token}"
            with session.request(
                method,
                self.config.api_url.rstrip("/") + path,
                headers=headers,
                json=body,
                params=params,
                stream=True,
                allow_redirects=False,
                timeout=min(remaining, self.config.request_timeout_seconds),
            ) as response:
                if not 200 <= response.status_code < 300:
                    raise RuntimeError("SpeQtrum HTTP request unsuccessful")
                content = bytearray()
                for chunk in response.iter_content(chunk_size=1):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("SpeQtrum deadline elapsed")
                    content.extend(chunk)
                    if len(content) > self.max_response_bytes:
                        raise ValueError("SpeQtrum response too large")
        result = json.loads(content)
        if not isinstance(result, dict):
            raise TypeError("SpeQtrum response must be an object")
        return result

    def request(self, method: str, path: str, deadline: float, **kwargs: Any) -> dict[str, Any]:
        try:
            return self._request(method, path, deadline, **kwargs)
        except Exception:  # noqa: BLE001 — never expose requests containing credential assertions
            raise RuntimeError("SpeQtrum HTTP request unsuccessful or unresolved") from None

    def login(self, deadline: float) -> str:
        assertion = {
            "username": self.config.username,
            "api_key": self.config.api_key,
            "audience": self.config.audience,
            "iat": int(time.time()),
        }
        encoded = base64.urlsafe_b64encode(json.dumps(assertion, indent=2).encode()).decode()
        result = self.request(
            "POST",
            "/authorisation-tokens",
            deadline,
            body={
                "grantType": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "assertion": encoded,
                "scope": "user profile",
            },
        )
        token = result.get("accessToken")
        if not isinstance(token, str) or not token:
            raise ValueError("SpeQtrum authentication unsuccessful")
        return token


def _sampling_counts(text: str) -> dict[str, int]:
    """Parse only inert QiliSDK sampling nodes; never invoke vendor constructors."""
    from ruamel.yaml import YAML
    from ruamel.yaml.events import AliasEvent, CollectionEndEvent, CollectionStartEvent
    from ruamel.yaml.nodes import MappingNode, ScalarNode, SequenceNode

    if not isinstance(text, str) or len(text.encode()) > 1024 * 1024:
        raise ValueError("Invalid SpeQtrum sampling document")
    yaml = YAML(typ="safe", pure=True)
    allowed = {"!FunctionalResult", "!ReadoutCompositeResults", "!SamplingReadoutResult"}
    standard = "tag:yaml.org,2002:"
    depth = 0
    for count, event in enumerate(yaml.parse(text), 1):
        if isinstance(event, CollectionStartEvent):
            depth += 1
        elif isinstance(event, CollectionEndEvent):
            depth -= 1
        if depth > 12:
            raise ValueError("SpeQtrum YAML excessive depth")
        if count > 20000 or isinstance(event, AliasEvent) or getattr(event, "anchor", None):
            raise ValueError("SpeQtrum YAML aliases/anchors or excessive nodes refused")
        tag = getattr(event, "tag", None)
        if (
            tag
            and tag not in allowed
            and tag not in {standard + k for k in ("map", "seq", "str", "int", "float", "null")}
        ):
            raise ValueError("SpeQtrum YAML executable/unknown tags refused")

    def inert(node: Any, depth: int = 0) -> Any:
        if depth > 12:
            raise ValueError("SpeQtrum YAML excessive depth")
        if isinstance(node, MappingNode):
            values = {}
            for key, value in node.value:
                if (
                    not isinstance(key, ScalarNode)
                    or key.tag != standard + "str"
                    or key.value in values
                ):
                    raise ValueError("SpeQtrum YAML duplicate or invalid keys")
                values[key.value] = inert(value, depth + 1)
            return (node.tag, values)
        if isinstance(node, SequenceNode) and node.tag == standard + "seq":
            return [inert(x, depth + 1) for x in node.value]
        if isinstance(node, ScalarNode):
            if node.tag == standard + "str":
                return node.value
            if node.tag == standard + "null":
                return None
            if node.tag == standard + "int" and node.value.isdecimal():
                return int(node.value)
            if node.tag == standard + "float":
                number = float(node.value)
                if math.isfinite(number):
                    return number
        raise ValueError("SpeQtrum YAML unsupported node")

    def mapping(value: Any, tag: str, keys: set[str]) -> dict[str, Any]:
        if not isinstance(value, tuple) or value[0] != tag or set(value[1]) != keys:
            raise ValueError("SpeQtrum sampling schema mismatch")
        return cast(dict[str, Any], value[1])

    root = mapping(
        inert(yaml.compose(text)),
        "!FunctionalResult",
        {"_execution_time", "_intermediate_results", "_readout_results"},
    )
    if (
        type(root["_execution_time"]) not in (int, float)
        or root["_execution_time"] < 0
        or root["_intermediate_results"] != []
    ):
        raise ValueError("SpeQtrum unsupported execution result")
    composite = mapping(
        root["_readout_results"],
        "!ReadoutCompositeResults",
        {"expectation_values", "sampling", "state_tomography"},
    )
    if composite["expectation_values"] is not None or composite["state_tomography"] is not None:
        raise ValueError("SpeQtrum requires sampling-only result")
    sampling = mapping(
        composite["sampling"], "!SamplingReadoutResult", {"_samples", "_probabilities"}
    )
    samples = sampling["_samples"]
    if not isinstance(samples, tuple) or samples[0] != standard + "map":
        raise ValueError("SpeQtrum invalid sample map")
    probabilities = sampling["_probabilities"]
    if probabilities is not None:
        probabilities = mapping(probabilities, standard + "map", set(samples[1]))
        if any(type(x) not in (int, float) or not 0 <= x <= 1 for x in probabilities.values()):
            raise ValueError("SpeQtrum invalid probabilities")
    return cast(dict[str, int], samples[1])


class SpeQtrumExecutor(BaseExecutor):
    def __init__(self, config: SpeQtrumExecutorConfig):
        self.config = config
        self._client = SpeQtrumHTTPClient(config)
        self.last_job: SpeQtrumJob | None = None

    @staticmethod
    def _require_qilisdk() -> None:
        if version("qilisdk") != "0.3.0":
            raise ImportError("SpeQtrum adapter requires qilisdk==0.3.0")

    async def execute(self, circuit: Any, shots: int = 1000, **kwargs: Any) -> ExecutionResult:
        if kwargs:
            raise TypeError("SpeQtrum execution does not support seeds or extra options")
        self._require_qilisdk()
        from qilisdk.functionals import DigitalPropagation

        circuit = self._validate_circuit(circuit)
        # Conversion is shared; do not construct the local simulator.
        functional = DigitalPropagation(circuit=QiliSDKExecutor._to_qilisdk_circuit(circuit))
        return await asyncio.to_thread(
            self._run, functional, "digital_propagation", shots, circuit.num_qubits
        )

    async def execute_analog(
        self, schedule: Any, shots: int = 1000, initial_state: Any = None, **kwargs: Any
    ) -> ExecutionResult:
        if kwargs:
            raise TypeError("SpeQtrum execution does not support seeds or extra options")
        self._require_qilisdk()
        from qilisdk.analog import Schedule
        from qilisdk.core.qtensor import InitialState
        from qilisdk.functionals import AnalogEvolution

        if not isinstance(schedule, Schedule):
            raise TypeError("SpeQtrum analog execution requires a QiliSDK Schedule")
        functional = AnalogEvolution(
            schedule=schedule,
            initial_state=initial_state if initial_state is not None else InitialState.UNIFORM,
        )
        return await asyncio.to_thread(
            self._run, functional, "analog_evolution", shots, schedule.nqubits
        )

    def _run(self, functional: Any, kind: str, shots: int, width: int) -> ExecutionResult:
        from qilisdk.readout import Readout
        from qilisdk.utils.serialization import serialize

        if type(shots) is not int or shots <= 0:
            raise ValueError("shots must be a positive integer")
        if type(width) is not int or width <= 0:
            raise ValueError("SpeQtrum workload requires positive qubit width")
        readout = Readout().with_sampling(nshots=shots)
        payload: dict[str, Any] = {
            "type": kind,
            "digital_propagation_payload": None,
            "analog_evolution_payload": None,
            "quantum_reservoir_payload": None,
            "variational_program_payload": None,
            "experiment_payload": None,
        }
        payload[kind + "_payload"] = {kind: serialize(functional), "readout": serialize(readout)}
        wire_payload = json.dumps(payload, separators=(",", ":"))
        deadline = time.monotonic() + self.config.timeout_seconds
        token = self._client.login(deadline)
        self._device(token, deadline, kind, width)
        body: dict[str, Any] = {
            "device_code": self.config.device_code,
            "payload": wire_payload,
            "job_type": "digital" if kind == "digital_propagation" else "analog",
            "meta": {},
        }
        # There is no server idempotency guarantee. Any failure after POST begins is unknown.
        self.last_job = None
        try:
            reply = self._client.request("POST", "/execute", deadline, token=token, body=body)
            job_id = reply["id"]
            if type(job_id) is not int or job_id <= 0:
                raise ValueError("invalid job identity")
        except Exception:  # noqa: BLE001 — preserve unknown contact and sanitize provider failures
            raise SpeQtrumExecutionError(None, acceptance_unknown=True) from None
        job = SpeQtrumJob(
            job_id,
            self.config.device_code,
            self.config.username,
            self.config.api_url.rstrip("/"),
            self.config.audience,
            wire_payload,
            kind,
            shots,
            width,
        )
        self.last_job = job
        return self._readback(job, token, deadline)

    def _device(
        self, token: str, deadline: float, kind: str | None = None, width: int = 0
    ) -> dict[str, Any]:
        rows = self._client.request("GET", "/devices", deadline, token=token).get("items")
        if not isinstance(rows, list):
            raise TypeError("Invalid SpeQtrum device list")
        matches = [
            r for r in rows if isinstance(r, dict) and r.get("code") == self.config.device_code
        ]
        if len(matches) != 1:
            raise ValueError("Configured SpeQtrum target unavailable or ambiguous")
        device = matches[0]
        if kind is not None:
            target_type = "qpu.digital" if kind == "digital_propagation" else "qpu.analog"
            if device.get("status") != "online" or device.get("type") not in (
                target_type,
                "simulator",
            ):
                raise ValueError("Configured SpeQtrum target incompatible or offline")
            if type(device.get("nqubits")) is not int or width > device["nqubits"]:
                raise ValueError("SpeQtrum target width exceeded")
        return device

    async def readback(self, job: SpeQtrumJob) -> ExecutionResult:
        self._require_qilisdk()
        if type(job) is not SpeQtrumJob:
            raise TypeError("SpeQtrum readback requires retained job record")
        if (job.device_code, job.username, job.api_url, job.audience) != (
            self.config.device_code,
            self.config.username,
            self.config.api_url.rstrip("/"),
            self.config.audience,
        ):
            raise ValueError("SpeQtrum readback account/target mismatch")
        deadline = time.monotonic() + self.config.timeout_seconds
        return await asyncio.to_thread(self._readback_login, job, deadline)

    def _readback_login(self, job: SpeQtrumJob, deadline: float) -> ExecutionResult:
        try:
            return self._readback(job, self._client.login(deadline), deadline)
        except Exception:  # noqa: BLE001 — preserve unknown contact and sanitize provider failures
            raise SpeQtrumExecutionError(job.job_id) from None

    def _readback(self, job: SpeQtrumJob, token: str, deadline: float) -> ExecutionResult:

        start = time.monotonic()
        observed_device_id = None
        try:
            while True:
                detail = self._client.request(
                    "GET",
                    f"/jobs/{job.job_id}",
                    deadline,
                    token=token,
                    params={"payload": True, "result": True},
                )
                if type(detail.get("id")) is not int or detail["id"] != job.job_id:
                    raise ValueError("SpeQtrum job identity mismatch")
                device_id = detail.get("device_id")
                if type(device_id) is not int or device_id <= 0:
                    raise ValueError("SpeQtrum invalid provider device identity")
                if observed_device_id is not None and device_id != observed_device_id:
                    raise ValueError("SpeQtrum provider device identity changed")
                observed_device_id = device_id
                saved = detail.get("payload")
                if isinstance(saved, str):
                    saved = json.loads(saved)
                if saved != json.loads(job.payload):
                    raise ValueError("SpeQtrum workload readback mismatch")
                status = detail.get("status")
                if status == "completed":
                    result = detail.get("result")
                    if isinstance(result, str):
                        result = json.loads(base64.b64decode(result, validate=True))
                    if not isinstance(result, dict) or result.get("type") != job.execute_type:
                        raise ValueError("SpeQtrum result type mismatch")
                    counts = _sampling_counts(result["functional_result"])
                    if (
                        not isinstance(counts, dict)
                        or any(
                            not isinstance(k, str)
                            or len(k) != job.num_qubits
                            or set(k) - {"0", "1"}
                            or type(v) is not int
                            or v < 0
                            for k, v in counts.items()
                        )
                        or sum(counts.values()) != job.shots
                    ):
                        raise ValueError("SpeQtrum invalid counts or shot accounting")
                    return ExecutionResult(
                        counts=counts,
                        backend=self.config.device_code,
                        shots=job.shots,
                        execution_time_ms=(time.monotonic() - start) * 1000,
                        metadata={
                            "vendor": "Qilimanjaro",
                            "framework": "QiliSDK",
                            "access_path": "direct",
                            "engine": "SpeQtrum",
                            "job_id": job.job_id,
                            "qilisdk_version": "0.3.0",
                            "api_url": job.api_url,
                            "audience": job.audience,
                            "provider_device_id": observed_device_id,
                            "target_binding": "configured code sent; numeric device continuity only",
                            "payload_sha256": hashlib.sha256(job.payload.encode()).hexdigest(),
                            "bit_order": "QiliSDK sampling order; qubit 0 leftmost",
                            "scope": "direct SDK; no hosted attestation or managed authorization",
                        },
                    )
                if status in ("error", "cancelled", "timeout"):
                    raise SpeQtrumExecutionError(job.job_id)
                if status not in ("pending", "validating", "queued", "running"):
                    raise ValueError("SpeQtrum unknown status")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("SpeQtrum deadline elapsed")
                time.sleep(min(remaining, self.config.poll_interval_seconds))
        except Exception:  # noqa: BLE001 — preserve unknown contact and sanitize provider failures
            raise SpeQtrumExecutionError(job.job_id) from None

    async def get_status(self) -> DeviceStatus:
        return await asyncio.to_thread(self._status)

    def _status(self) -> DeviceStatus:
        deadline = time.monotonic() + self.config.timeout_seconds
        token = self._client.login(deadline)
        device = self._device(token, deadline)
        status = device.get("status")
        if status not in ("online", "offline", "maintenance"):
            raise ValueError("Invalid SpeQtrum device status")
        return DeviceStatus(status=status, queue_depth=None, queue_time_seconds=None)

    async def cancel(self, job_id: str) -> bool:
        # Pinned vendor client exposes no cancellation endpoint.
        return False
