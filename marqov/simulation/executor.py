"""Simulation executor for C++ quantum backends."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import time
from collections.abc import Mapping
from typing import Any

from marqov.circuits import Circuit
from marqov.executors.base import BaseExecutor, ExecutionResult
from marqov.simulation.backends import GPU_SIMULATION_BACKENDS, SIMULATION_BACKENDS
from marqov.simulation.circuit_converter import (
    convert_counts,
    count_qubits,
    ensure_measurements,
)
from marqov.simulation.config import SimulationConfig
from marqov.simulation.noise import (
    AmplitudeDamping,
    Depolarizing,
    PhaseDamping,
    ReadoutError,
)


class SimulationExecutor(BaseExecutor):
    """High-performance quantum simulation via C++ backends.

    Supports state vector (up to 28 qubits), tensor network (up to 100+
    qubits), density matrix (up to 14 qubits), and GPU-accelerated variants.

    Cancellation is not supported — C++ execution is a blocking call in a
    thread pool. The default cancel() returning False applies.
    """

    def __init__(self, config: SimulationConfig):
        _require_local_backend(config)
        self.config = config

    async def execute(
        self,
        circuit: Circuit,
        shots: int = 1000,
        **kwargs: Any,
    ) -> ExecutionResult:
        if kwargs:
            raise TypeError("Simulation execution options belong in SimulationConfig")
        if type(shots) is not int or shots <= 0:
            raise ValueError("shots must be a positive integer")
        circuit = self._validate_circuit(circuit)
        qasm_str = circuit.to_openqasm()
        qasm_str = ensure_measurements(qasm_str)
        num_qubits = count_qubits(qasm_str)

        config = dataclasses.replace(self.config, num_qubits=num_qubits)
        _validate_qubit_limit(config)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, _run_simulation, qasm_str, shots, config
        )


def _require_local_backend(config: SimulationConfig) -> None:
    supported = {
        entry["provider_target_id"]
        for entry in (*SIMULATION_BACKENDS.values(), *GPU_SIMULATION_BACKENDS.values())
    }
    # Preserve the documented single-precision simulator accepted by direct configs.
    supported.add("cudaq:custatevec_fp32")
    if config.backend_id not in supported:
        raise ValueError(
            "Unsupported local simulator or hardware ID; "
            "QB remote execution requires explicit access_path='remote'"
        )


def _import_qristal_core() -> Any:
    """Import ``qristal.core`` with an actionable error if it isn't installed.

    ``qristal`` is not published on PyPI — there is no ``pip install
    marqov[...]`` remedy. It must be built from source or run via Quantum
    Brilliance's Docker image; see https://qristal.readthedocs.io/.
    """
    import importlib

    try:
        return importlib.import_module("qristal.core")
    except ImportError as exc:
        raise ImportError(
            "qristal is required for Quantum Brilliance simulation backends "
            "but is not installed. qristal is not distributed on PyPI, so "
            "there is no 'pip install marqov[...]' extra for it — build it "
            "from source or use the official Docker image. See Quantum "
            "Brilliance's install instructions: "
            "https://qristal.readthedocs.io/en/latest/rst/getting_started.html"
        ) from exc


def _run_simulation(
    qasm_str: str, shots: int, config: SimulationConfig
) -> ExecutionResult:
    """Execute simulation in a thread (blocking C++ call)."""
    qristal_core = _import_qristal_core()
    _require_local_backend(config)
    session = qristal_core.session()
    # Current Qristal initializes on construction; older releases expose init().
    if hasattr(session, "init"):
        session.init()
    if config.remote_backend_database_path is not None:
        session.remote_backend_database_path = config.remote_backend_database_path
    session.acc = config.backend_id
    session.qn = config.num_qubits
    session.sn = shots
    session.instring = qasm_str

    if config.max_bond_dimension is not None:
        session.max_bond_dimension = config.max_bond_dimension
    if config.svd_cutoff is not None:
        session.svd_cutoffs[0][0][0] = config.svd_cutoff
    if config.rel_svd_cutoff is not None:
        session.rel_svd_cutoffs[0][0][0] = config.rel_svd_cutoff
    if config.seed is not None:
        session.seed = config.seed

    # Override backend to aer if noise model is present
    if config.noise_model is not None:
        if not hasattr(qristal_core, "NoiseModel"):
            raise RuntimeError(
                "Noise simulation requires the 'aer' backend, which is not "
                "available in this qristal build."
            )
        session.acc = "aer"
        qb_noise = qristal_core.NoiseModel()
        qb_noise.name = "marqov_noise"

        for channel, qubits in config.noise_model.entries():
            if isinstance(channel, Depolarizing):
                for qubit in qubits:
                    qb_noise.add_gate_error(
                        qristal_core.DepolarizingChannel.Create(qubit, channel.probability),
                        "u3", [qubit],
                    )
            elif isinstance(channel, AmplitudeDamping):
                for qubit in qubits:
                    qb_noise.add_gate_error(
                        qristal_core.AmplitudeDampingChannel.Create(qubit, channel.gamma),
                        "u3", [qubit],
                    )
            elif isinstance(channel, PhaseDamping):
                for qubit in qubits:
                    qb_noise.add_gate_error(
                        qristal_core.PhaseDampingChannel.Create(qubit, channel.gamma),
                        "u3", [qubit],
                    )
            elif isinstance(channel, ReadoutError):
                for qubit in qubits:
                    qb_noise.set_qubit_readout_error(
                        qubit, qristal_core.ReadoutError(qubit, channel.p0_given1, channel.p1_given0),
                    )

        session.noise = True
        session.noise_model = qb_noise

    # State vector extraction (qpp only, no noise)
    if config.extract_state_vector:
        if config.noise_model is not None:
            raise ValueError(
                "State vector extraction is incompatible with noise models. "
                "Noise simulation uses density matrix / qasm mode."
            )
        if config.backend_id != "qpp":
            raise ValueError(
                f"State vector extraction not supported for backend '{config.backend_id}'. "
                f"Supported: qpp."
            )
        if not hasattr(session, "get_state_vec") or not hasattr(session, "get_state_vec_raw"):
            raise NotImplementedError("This Qristal build does not expose state-vector extraction")
        session.get_state_vec = True

    start = time.monotonic()
    session.run()
    elapsed_ms = (time.monotonic() - start) * 1000

    # Current C++ API returns a single MapVectorBoolInt; older APIs nest the map.
    results = session.results
    direct_map = isinstance(results, Mapping) or type(results).__name__ == "MapVectorBoolInt"
    counts = convert_counts(results if direct_map else results[0][0])
    if any(len(bits) != config.num_qubits or set(bits) - {"0", "1"}
           or type(count) is not int or count < 0 for bits, count in counts.items()) or sum(counts.values()) != shots:
        raise RuntimeError("Qristal returned invalid counts or shot accounting")

    metadata = {
        "vendor": "Quantum Brilliance",
        "framework": "Qristal",
        "engine": "aer" if config.noise_model is not None else config.backend_id,
        "access_path": "local",
        "compute_provider": "local",
        "simulator": "aer" if config.noise_model is not None else config.backend_id,
        "reproducibility": {
            "record_version": 1,
            "qristal_version": getattr(qristal_core, "__version__", None),
            "seed": config.seed,
            "shots": shots,
            "program_sha256": hashlib.sha256(qasm_str.encode()).hexdigest(),
            "counts_sha256": hashlib.sha256(json.dumps(counts, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "bit_order": "qubit_0_first",
        },
        "num_qubits": config.num_qubits,
        "max_bond_dimension": config.max_bond_dimension,
        "svd_cutoff": config.svd_cutoff,
    }

    if config.extract_state_vector:
        metadata["state_vector"] = list(session.get_state_vec_raw)

    return ExecutionResult(
        counts=counts,
        backend=f"qb-sim-{config.backend_type}",
        execution_time_ms=elapsed_ms,
        shots=shots,
        metadata=metadata,
    )


def _validate_qubit_limit(config: SimulationConfig) -> None:
    """Raise ValueError if qubit count exceeds backend limit."""
    limits = {
        "qpp": 28,
        "cudaq:custatevec_fp64": 28,
        "cudaq:custatevec_fp32": 28,
        "cudaq:dm": 14,
        "tnqvm": 100,
        "cudaq:qb_mps": 100,
        "aer": 28,
    }
    max_q = limits.get(config.backend_id, 28)
    if config.num_qubits > max_q:
        raise ValueError(
            f"Backend {config.backend_id} supports up to {max_q} qubits, "
            f"got {config.num_qubits}"
        )
