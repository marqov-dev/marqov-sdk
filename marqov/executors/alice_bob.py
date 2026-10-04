"""Alice & Bob cat-qubit models and explicit direct-provider execution.

Native Qiskit input preserves initialization, delays and measurement semantics.
This SDK adapter is not a hosted gateway or a managed billing authorization.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import math
import time
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import Any, Literal

from marqov.executors.base import BaseExecutor, DeviceStatus, ExecutionResult


@dataclass
class AliceBobExecutorConfig:
    backend_name: str = "EMU:40Q:LOGICAL_NOISELESS"
    mode: Literal["local", "remote"] = "local"
    api_key: str | None = field(default=None, repr=False)
    backend_options: dict[str, int | float] = field(default_factory=dict)
    compute_provider: str | None = None
    optimization_level: int = 0
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("local", "remote"):
            raise ValueError("Alice & Bob mode must be local or remote")
        if not isinstance(self.backend_name, str) or not self.backend_name:
            raise ValueError("backend_name is required")
        if self.mode == "local" and self.api_key is not None:
            raise ValueError("A local model must not receive provider credentials")
        if self.mode == "remote" and not self.api_key:
            raise ValueError("Remote Alice & Bob execution requires an explicit API key")
        if type(self.optimization_level) is not int or not 0 <= self.optimization_level <= 3:
            raise ValueError("optimization_level must be an integer from 0 to 3")
        self.backend_options = dict(self.backend_options)
        if any(
            type(v) not in (int, float) or not math.isfinite(v)
            for v in self.backend_options.values()
        ):
            raise ValueError("backend_options must contain finite numeric model parameters")
        if self.timeout_seconds is not None and (
            type(self.timeout_seconds) not in (int, float)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        if self.mode == "local" and self.timeout_seconds is not None:
            raise ValueError("The local vendor job does not support a result timeout")


class AliceBobExecutionError(RuntimeError):
    """A submitted job failed or has an unresolved result; never resubmit blindly."""

    def __init__(self, job_id: str, message: str) -> None:
        self.job_id = job_id
        super().__init__(f"Alice & Bob job {job_id}: {message}")


class AliceBobExecutor(BaseExecutor):
    def __init__(self, config: AliceBobExecutorConfig | None = None) -> None:
        self.config = config or AliceBobExecutorConfig()
        try:
            from qiskit_alice_bob_provider import AliceBobLocalProvider, AliceBobRemoteProvider
        except ImportError as exc:
            raise ImportError(
                "Install the isolated Alice & Bob environment: pip install 'marqov[alice-bob]'"
            ) from exc
        if self.config.mode == "local":
            self._provider = AliceBobLocalProvider()
        else:
            self._provider = AliceBobRemoteProvider(self.config.api_key)
        # Missing or unsupported models propagate; no replacement provider/device.
        self._backend = self._provider.get_backend(
            self.config.backend_name, **self.config.backend_options
        )

    async def execute(
        self, circuit: Any, shots: int = 1000, *, seed: int | None = None, **kwargs: Any
    ) -> ExecutionResult:
        """Execute a canonical zero-initialized gate circuit with full measurement."""
        from qiskit import QuantumCircuit

        circuit = self._validate_circuit(circuit)
        gates = circuit.to_qiskit()
        native = QuantumCircuit(gates.num_qubits)
        native.initialize("0" * gates.num_qubits)
        native.compose(gates, inplace=True)
        native.measure_all()
        return await self.execute_native(native, shots, seed=seed, **kwargs)

    async def execute_native(
        self, circuit: Any, shots: int = 1000, *, seed: int | None = None, **kwargs: Any
    ) -> ExecutionResult:
        """Preserve Qiskit initialization/delays; require full terminal measurement.

        Partial or permuted classical mappings are refused rather than reporting
        misleading canonical counts. Seeds are for local models only.
        """
        from qiskit import QuantumCircuit

        if kwargs:
            raise TypeError(
                f"Unsupported Alice & Bob execution options: {', '.join(sorted(kwargs))}"
            )
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError("execute_native requires a Qiskit QuantumCircuit")
        if type(shots) is not int or shots <= 0:
            raise ValueError("shots must be a positive integer")
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
            raise ValueError("seed must be a uint32 integer or None")
        if seed is not None and self.config.mode != "local":
            raise ValueError("Simulator seeds are not supported for remote execution")
        native = circuit.copy()
        if (
            native.num_qubits == 0
            or native.num_clbits != native.num_qubits
            or len(native.cregs) != 1
        ):
            raise NotImplementedError("Require one classical register measuring every qubit")
        measured: set[int] = set()
        measuring = False
        for instruction in native.data:
            if instruction.operation.name == "measure":
                measuring = True
                q = native.find_bit(instruction.qubits[0]).index
                c = native.find_bit(instruction.clbits[0]).index
                if q != c or q in measured:
                    raise NotImplementedError(
                        "Require one terminal measurement per qubit, q[i] to c[i]"
                    )
                measured.add(q)
            elif measuring and instruction.operation.name != "barrier":
                raise NotImplementedError("Mid-circuit measurement is unsupported")
        if measured != set(range(native.num_qubits)):
            raise NotImplementedError("Require terminal measurement of every qubit")
        return await asyncio.to_thread(self._run_sync, native, shots, seed)

    def _run_sync(self, native: Any, shots: int, seed: int | None) -> ExecutionResult:
        from qiskit import qpy, transpile

        source = io.BytesIO()
        qpy.dump(native, source)
        start = time.perf_counter()
        compiled = transpile(
            native,
            self._backend,
            optimization_level=self.config.optimization_level,
            seed_transpiler=seed,
        )
        options: dict[str, Any] = {"shots": shots}
        if seed is not None:
            options["seed_simulator"] = seed
        job = self._backend.run(compiled, **options)
        job_id = job.job_id()
        try:
            result = (
                job.result()
                if self.config.mode == "local"
                else job.result(timeout=self.config.timeout_seconds)
            )
            raw_counts = result.get_counts()
        except Exception as exc:
            raise AliceBobExecutionError(
                job_id, "result unresolved; inspect this job before retrying"
            ) from exc
        if not result.success:
            raise AliceBobExecutionError(job_id, "provider reported an unsuccessful result")
        n = native.num_qubits
        if (
            not isinstance(raw_counts, dict)
            or any(
                not isinstance(k, str)
                or len(k) != n
                or set(k) - {"0", "1"}
                or type(v) is not int
                or v < 0
                for k, v in raw_counts.items()
            )
            or sum(raw_counts.values()) != shots
        ):
            raise AliceBobExecutionError(job_id, "invalid counts or shot accounting")
        counts = {bits[::-1]: count for bits, count in raw_counts.items()}
        digest = hashlib.sha256(
            json.dumps(counts, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return ExecutionResult(
            counts=counts,
            backend=self._backend.name,
            shots=shots,
            execution_time_ms=(time.perf_counter() - start) * 1000,
            raw_result=result,
            metadata={
                "vendor": "Alice & Bob",
                "framework": "Qiskit",
                "engine": self._backend.name,
                "access_path": "local" if self.config.mode == "local" else "direct",
                "compute_provider": self.config.compute_provider
                or ("local" if self.config.mode == "local" else "vendor-managed"),
                "job_id": job_id,
                "execution_kind": "local cat-qubit model"
                if self.config.mode == "local"
                else "remote provider execution",
                "reproducibility": {
                    "record_version": 1,
                    "packages": {
                        p: version(p) for p in ("qiskit-alice-bob-provider", "qiskit", "numpy")
                    },
                    "model_parameters": dict(self.config.backend_options),
                    "seed": seed,
                    "shots": shots,
                    "optimization_level": self.config.optimization_level,
                    "circuit_qpy_sha256": hashlib.sha256(source.getvalue()).hexdigest(),
                    "counts_sha256": digest,
                    "bit_order": "qubit 0 leftmost; terminal q[i] to c[i]",
                },
            },
        )

    async def get_status(self) -> DeviceStatus:
        if self.config.mode == "local":
            return DeviceStatus.always_online()
        status = await asyncio.to_thread(self._backend.status)
        return DeviceStatus(
            status="online" if status.operational else "offline",
            queue_depth=status.pending_jobs,
            queue_time_seconds=None,
        )
