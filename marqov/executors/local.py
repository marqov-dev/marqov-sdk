"""Local executor using QuantumFlow simulator.

This executor runs circuits on QuantumFlow's built-in state vector simulator.
No cloud credentials required - perfect for development and testing.
"""

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from marqov.circuits import Circuit
from marqov.executors.base import BaseExecutor, ExecutionResult


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
            **kwargs: Additional options (ignored for local).

        Returns:
            ExecutionResult with measurement counts.
        """
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
            metadata={"simulator": "quantumflow"},
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
