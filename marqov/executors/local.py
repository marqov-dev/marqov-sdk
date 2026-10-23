"""Local executor using QuantumFlow simulator.

This executor runs circuits on QuantumFlow's built-in state vector simulator.
No cloud credentials required - perfect for development and testing.
"""

import hashlib
import json
import time
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np

from marqov.circuits import Circuit
from marqov.executors.base import BaseExecutor, ExecutionResult, reject_unsupported_options


@lru_cache(maxsize=1)
def _package_versions() -> dict[str, str | None]:
    """Installed versions of the simulator's packages (cached per process).

    A package whose distribution metadata is absent (a frozen app or vendored
    copy) records None rather than failing the run.
    """
    versions: dict[str, str | None] = {}
    for package in ("marqov-quantumflow", "numpy"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


@dataclass
class LocalExecutorConfig:
    """Configuration for local simulation.

    Attributes:
        seed: Random seed for reproducibility. None for random results.
    """

    seed: int | None = None


class LocalExecutor(BaseExecutor):
    """Execute circuits on QuantumFlow's local simulator.

    This executor uses QuantumFlow's state vector simulation for fast,
    local execution without any cloud dependencies.

    Example:
        >>> executor = LocalExecutor()
        >>> circuit = Circuit().h(0).cnot(0, 1)
        >>> result = await executor.execute(circuit, shots=1000)
        >>> print(result.counts)  # {"00": ~500, "11": ~500}

    For reproducible results:
        >>> executor = LocalExecutor(LocalExecutorConfig(seed=42))
    """

    def __init__(self, config: LocalExecutorConfig | None = None) -> None:
        """Initialize the local executor.

        Args:
            config: Configuration options. Uses defaults if not provided.
        """
        self.config = config or LocalExecutorConfig()
        self._rng = np.random.default_rng(self.config.seed)

    async def execute(
        self,
        circuit: Circuit,
        shots: int = 1000,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Run circuit on local simulator.

        Args:
            circuit: The circuit to execute.
            shots: Number of measurement shots.
            **kwargs: Unsupported; any option raises TypeError.

        Returns:
            ExecutionResult with measurement counts. ``metadata["reproducibility"]``
            records the configured ``seed`` (None when unseeded), shots, package
            versions (None for a package without distribution metadata), bit
            order and a counts hash. The executor draws from one
            generator seeded at construction, so the seed replays a run only as
            the same sequence of calls on a fresh executor.

        Raises:
            TypeError: If any keyword option is supplied.
        """
        reject_unsupported_options("LocalExecutor.execute()", kwargs)
        circuit = self._validate_circuit(circuit)

        start_time = time.perf_counter()

        # Run simulation to get state vector
        state = circuit.simulate()

        # Sample measurements from state
        counts = self._sample_measurements(state, shots)

        execution_time_ms = (time.perf_counter() - start_time) * 1000

        return ExecutionResult(
            counts=counts,
            backend="local",
            execution_time_ms=execution_time_ms,
            shots=shots,
            raw_result=state,
            metadata={
                "simulator": "quantumflow",
                "reproducibility": {
                    "record_version": 1,
                    "packages": dict(_package_versions()),
                    "seed": self.config.seed,
                    "rng_policy": "one generator seeded at construction; draws continue across calls",
                    "shots": shots,
                    "counts_sha256": hashlib.sha256(
                        json.dumps(counts, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "bit_order": "qubit_0_first",
                },
            },
        )

    def _sample_measurements(self, state: Any, shots: int) -> dict[str, int]:
        """Sample measurement outcomes from state vector.

        Args:
            state: QuantumFlow State object.
            shots: Number of samples to take.

        Returns:
            Dictionary mapping bitstrings to counts.
        """
        # Get probabilities from state
        probabilities = np.abs(state.tensor.flatten()) ** 2

        # Number of qubits
        num_qubits = int(np.log2(len(probabilities)))

        # Sample outcomes
        outcomes = self._rng.choice(
            len(probabilities),
            size=shots,
            p=probabilities,
        )

        # Convert to bitstrings and count
        counts: dict[str, int] = {}
        for outcome in outcomes:
            bitstring = format(outcome, f"0{num_qubits}b")
            counts[bitstring] = counts.get(bitstring, 0) + 1

        return counts
