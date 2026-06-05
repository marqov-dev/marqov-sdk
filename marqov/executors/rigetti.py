from __future__ import annotations

import asyncio
import socket
import time
import uuid
from dataclasses import dataclass
from functools import partial
from typing import Any
from urllib.parse import urlparse

import numpy as np

from marqov.executors.base import (
    BaseExecutor,
    DeviceStatus,
    ExecutionResult,
)

try:
    from qcs_sdk.client import (
        AuthServer,
        OAuthSession,
        QCSClient,
        RefreshToken,
    )
except ImportError:
    AuthServer = None
    OAuthSession = None
    QCSClient = None
    RefreshToken = None


DEFAULT_GRPC_API_URL = "https://grpc.qcs.rigetti.com"
DEFAULT_QUILC_URL = "tcp://127.0.0.1:5555"
DEFAULT_QVM_URL = "http://127.0.0.1:5000"


@dataclass
class RigettiExecutorConfig:
    quantum_processor_id: str = "2q-qvm"

    as_qvm: bool | None = None

    refresh_token: str | None = None
    client_id: str | None = None
    issuer: str | None = None

    grpc_api_url: str = DEFAULT_GRPC_API_URL
    quilc_url: str = DEFAULT_QUILC_URL
    qvm_url: str = DEFAULT_QVM_URL

    poll_interval_seconds: float = 0.2
    timeout_seconds: float | None = 120.0


def _is_qvm_target(
    quantum_processor_id: str,
    as_qvm: bool | None,
) -> bool:
    if as_qvm is not None:
        return as_qvm

    return "qvm" in quantum_processor_id.lower()


def build_oauth_session(
    refresh_token: str,
    client_id: str | None = None,
    issuer: str | None = None,
):
    if client_id and issuer:
        auth_server = AuthServer(
            client_id=client_id,
            issuer=issuer,
        )
    else:
        auth_server = AuthServer.default()

    return OAuthSession(
        RefreshToken(refresh_token=refresh_token),
        auth_server,
    )


def build_qcs_client(
    oauth_session,
    *,
    grpc_api_url: str | None = None,
    quilc_url: str | None = None,
    qvm_url: str | None = None,
):
    kwargs = {
        "oauth_session": oauth_session,
    }

    if grpc_api_url is not None:
        kwargs["grpc_api_url"] = grpc_api_url

    if quilc_url is not None:
        kwargs["quilc_url"] = quilc_url

    if qvm_url is not None:
        kwargs["qvm_url"] = qvm_url

    return QCSClient(**kwargs)


class RigettiExecutor(BaseExecutor):
    _STATUS_MAP = {
        "online": "online",
        "available": "online",
        "offline": "offline",
        "retired": "offline",
        "maintenance": "maintenance",
        "degraded": "maintenance",
    }

    def __init__(
        self,
        config: RigettiExecutorConfig | None = None,
    ) -> None:
        self.config = config or RigettiExecutorConfig()

        self._qc = None
        self._qcs_client = None

        self._current_job_id: str | None = None
        self._active_future = None

        try:
            from pyquil import get_qc

            self._get_qc = get_qc
        except ImportError as exc:
            raise ImportError(
                "pyQuil is required. "
                "Install with: pip install marqov[rigetti]"
            ) from exc

    def _get_qcs_client(self):
        if self._qcs_client is not None:
            return self._qcs_client

        if self.config.refresh_token:
            oauth = build_oauth_session(
                refresh_token=self.config.refresh_token,
                client_id=self.config.client_id,
                issuer=self.config.issuer,
            )

            self._qcs_client = build_qcs_client(
                oauth,
                grpc_api_url=self.config.grpc_api_url,
                quilc_url=self.config.quilc_url,
                qvm_url=self.config.qvm_url,
            )
        else:
            self._qcs_client = QCSClient.load()

        return self._qcs_client

    async def execute(
        self,
        circuit,
        shots: int = 1000,
        **kwargs,
    ) -> ExecutionResult:
        circuit = self._validate_circuit(circuit)

        if _is_qvm_target(
            self.config.quantum_processor_id,
            self.config.as_qvm,
        ):
            return await self._execute_qvm(
                circuit,
                shots,
            )

        return await self._execute_qcs(
            circuit,
            shots,
        )

    async def _execute_qvm(
        self,
        circuit,
        shots: int,
    ) -> ExecutionResult:
        qc = await self._get_quantum_computer()

        start = time.perf_counter()

        program = self._prepare_program(
            circuit.to_pyquil(),
            circuit.num_qubits,
            shots,
        )

        loop = asyncio.get_running_loop()

        compiled = await loop.run_in_executor(
            None,
            partial(qc.compile, program),
        )

        self._current_job_id = str(uuid.uuid4())

        run_result = await loop.run_in_executor(
            None,
            partial(qc.run, compiled),
        )

        counts = self._normalize_counts(run_result)

        return ExecutionResult(
            counts=counts,
            backend=f"rigetti:{self.config.quantum_processor_id}",
            execution_time_ms=(time.perf_counter() - start) * 1000,
            shots=shots,
            raw_result=run_result,
            metadata={
                "job_id": self._current_job_id,
                "mode": "qvm",
            },
        )

    async def _execute_qcs(
        self,
        circuit,
        shots: int,
    ) -> ExecutionResult:
        from qcs_sdk.compiler.quilc import (
            QuilcClient,
            TargetDevice,
            compile_program,
        )

        from qcs_sdk.qpu.isa import (
            get_instruction_set_architecture,
        )

        from qcs_sdk.qpu.api import (
            submit,
            retrieve_results,
        )

        from qcs_sdk.qpu.translation import translate

        start = time.perf_counter()

        client = self._get_qcs_client()

        loop = asyncio.get_running_loop()

        program = self._prepare_program(
            circuit.to_pyquil(),
            circuit.num_qubits,
            shots,
        )

        isa = await loop.run_in_executor(
            None,
            partial(
                get_instruction_set_architecture,
                quantum_processor_id=self.config.quantum_processor_id,
                client=client,
            ),
        )

        quilc_client = QuilcClient.new_rpcq(client.quilc_url)

        compiled = await loop.run_in_executor(
            None,
            partial(
                compile_program,
                quil=program.out(),
                target=TargetDevice.from_isa(isa),
                client=quilc_client,
            ),
        )

        translation_result = await loop.run_in_executor(
            None,
            partial(
                translate,
                native_quil=compiled.program.to_quil(),
                num_shots=shots,
                quantum_processor_id=self.config.quantum_processor_id,
                client=client,
            ),
        )

        job_id = await loop.run_in_executor(
            None,
            partial(
                submit,
                program=translation_result.program,
                patch_values={},
                quantum_processor_id=self.config.quantum_processor_id,
                client=client,
            ),
        )

        self._current_job_id = job_id

        raw_result = await loop.run_in_executor(
            None,
            partial(
                retrieve_results,
                job_id=job_id,
                quantum_processor_id=self.config.quantum_processor_id,
                client=client,
            ),
        )

        counts = self._normalize_qcs_result(
            raw_result,
            translation_result.ro_sources,
        )

        return ExecutionResult(
            counts=counts,
            backend=f"rigetti:{self.config.quantum_processor_id}",
            execution_time_ms=(time.perf_counter() - start) * 1000,
            shots=shots,
            raw_result=raw_result,
            metadata={
                "job_id": job_id,
                "mode": "qpu",
                "quantum_processor_id": self.config.quantum_processor_id,
            },
        )

    async def cancel(
        self,
        job_id: str,
    ) -> bool:
        try:
            from qcs_sdk.qpu.api import cancel_job

            client = self._get_qcs_client()

            loop = asyncio.get_running_loop()

            await loop.run_in_executor(
                None,
                partial(
                    cancel_job,
                    job_id=job_id,
                    quantum_processor_id=self.config.quantum_processor_id,
                    client=client,
                ),
            )

            return True

        except Exception:
            return False

    @staticmethod
    def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
        """Return True if the TCP port is accepting connections."""
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    async def get_status(self) -> DeviceStatus:
        if _is_qvm_target(self.config.quantum_processor_id, self.config.as_qvm):
            qvm_parsed = urlparse(self.config.qvm_url)
            quilc_parsed = urlparse(self.config.quilc_url)

            qvm_host = qvm_parsed.hostname or "127.0.0.1"
            qvm_port = qvm_parsed.port or 5000
            quilc_host = quilc_parsed.hostname or "127.0.0.1"
            quilc_port = quilc_parsed.port or 5555

            loop = asyncio.get_running_loop()
            qvm_up = await loop.run_in_executor(
                None, partial(self._port_open, qvm_host, qvm_port)
            )
            quilc_up = await loop.run_in_executor(
                None, partial(self._port_open, quilc_host, quilc_port)
            )

            if qvm_up and quilc_up:
                return DeviceStatus(
                    status="online",
                    queue_depth=0,
                    queue_time_seconds=0,
                )
            return DeviceStatus(
                status="offline",
                queue_depth=None,
                queue_time_seconds=None,
            )

        return DeviceStatus(
            status="online",
            queue_depth=None,
            queue_time_seconds=None,
        )

    async def _get_quantum_computer(self):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._get_quantum_computer_sync,
        )

    def _get_quantum_computer_sync(self):
        if self._qc is None:
            self._qc = self._get_qc(
                self.config.quantum_processor_id,
                as_qvm=_is_qvm_target(
                    self.config.quantum_processor_id,
                    self.config.as_qvm,
                ),
            )

        return self._qc

    @staticmethod
    def _prepare_program(program, num_qubits, shots):
        from pyquil.gates import MEASURE
        from pyquil.quil import Program

        prepared = Program(program.out())

        ro = prepared.declare(
            "ro",
            "BIT",
            num_qubits,
        )

        for qubit in range(num_qubits):
            prepared += MEASURE(
                qubit,
                ro[qubit],
            )

        prepared.wrap_in_numshots_loop(shots)

        return prepared

    @staticmethod
    def _counts_from_ndarray(readout):
        counts = {}

        for row in readout:
            key = "".join(str(int(bit)) for bit in row)
            counts[key] = counts.get(key, 0) + 1

        return counts

    @classmethod
    def _normalize_counts(cls, result):
        if isinstance(result, np.ndarray):
            return cls._counts_from_ndarray(result)

        if hasattr(result, "readout_data"):
            data = result.readout_data

            if isinstance(data, dict):
                data = data.get("ro")

            if isinstance(data, np.ndarray):
                return cls._counts_from_ndarray(data)

        raise RuntimeError(
            "Unsupported Rigetti result format"
        )

    @staticmethod
    def _normalize_qcs_result(result, ro_sources: dict) -> dict:
        import re
        from qcs_sdk import RegisterMap
        from qcs_sdk.qpu import QPUResultData, ReadoutValues

        readout_values = {
            key: ReadoutValues(value.data) for key, value in result.buffers.items()
        }
        qpu_result_data = QPUResultData(
            mappings=dict(ro_sources),
            readout_values=readout_values,
            memory_values=result.memory,
        )
        register_map = qpu_result_data.to_register_map()

        declared_registers: set[str] = set()
        for mem_ref in ro_sources:
            match = re.match(r"^(.+)\[\d+\]$", mem_ref)
            if match:
                declared_registers.add(match.group(1))

        register_arrays = []
        for reg_name in sorted(declared_registers):
            matrix = register_map.get_register_matrix(reg_name)
            if matrix is not None:
                register_arrays.append(matrix.to_ndarray().astype(int))

        if not register_arrays:
            raise RuntimeError(
                f"No data found for declared registers {declared_registers}"
            )

        all_bits = np.hstack(register_arrays)
        counts: dict[str, int] = {}
        for row in all_bits:
            key = "".join(str(b) for b in row)
            counts[key] = counts.get(key, 0) + 1
        return counts
