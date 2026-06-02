"""Base executor interface for quantum backends.

All executors must inherit from BaseExecutor and implement the execute method.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from marqov.circuits import Circuit


@dataclass
class ExecutionResult:
    """Result from executing a quantum circuit.

    Attributes:
        counts: Measurement outcome counts (e.g., {"00": 512, "11": 488}).
        backend: Name of the backend that executed the circuit.
        execution_time_ms: Execution time in milliseconds.
        shots: Number of shots executed.
        raw_result: Backend-specific raw result object.
        metadata: Additional metadata from execution.
    """

    counts: dict[str, int]
    backend: str
    execution_time_ms: float
    shots: int = 0
    raw_result: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def probabilities(self) -> dict[str, float]:
        """Calculate probabilities from counts.

        Returns:
            Dictionary mapping bitstrings to probabilities.
        """
        total = sum(self.counts.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in self.counts.items()}


@dataclass
class DeviceStatus:
    """Live status of a quantum device."""

    status: str  # "online" | "offline" | "maintenance"
    queue_depth: int | None
    queue_time_seconds: int | None

    @staticmethod
    def always_online() -> "DeviceStatus":
        return DeviceStatus(status="online", queue_depth=0, queue_time_seconds=0)


class BaseExecutor(ABC):
    """Base class for all quantum executors.

    Executors handle the submission and result retrieval for quantum circuits.
    Each executor targets a specific backend (local simulator, Braket, IBM, etc.).

    Example:
        >>> class MyExecutor(BaseExecutor):
        ...     async def execute(self, circuit, shots=1000, **kwargs):
        ...         # Submit to backend
        ...         result = await self._run_on_backend(circuit, shots)
        ...         return ExecutionResult(...)
    """

    # Framework module prefixes → (display name, extra name, conversion method).
    _FRAMEWORK_HINTS: dict[str, tuple[str, str, str]] = {
        "qiskit": ("Qiskit QuantumCircuit", "qiskit", "Circuit.from_qiskit(your_circuit)"),
        "cirq": ("Cirq Circuit", "cirq", "Circuit.from_cirq(your_circuit)"),
        "pennylane": ("PennyLane tape", "pennylane", "Circuit.from_pennylane(your_tape)"),
    }

    def _validate_circuit(self, circuit: Any) -> Circuit:
        """Validate that the input is a Marqov Circuit, or give a helpful error.

        If the input is already a Marqov Circuit, returns it unchanged.
        If it's a recognized framework type (Qiskit, Cirq, PennyLane),
        raises TypeError with a specific conversion hint.
        Otherwise raises a generic TypeError listing supported types.

        Args:
            circuit: The circuit to validate.

        Returns:
            The validated Marqov Circuit.

        Raises:
            TypeError: If the input is not a Marqov Circuit.
        """
        if isinstance(circuit, Circuit):
            return circuit

        module = type(circuit).__module__ or ""
        for prefix, (display_name, extra, conversion) in self._FRAMEWORK_HINTS.items():
            if module.startswith(prefix):
                raise TypeError(
                    f"You passed a {display_name} to execute(). "
                    f"Convert it first:\n\n"
                    f"    from marqov.circuits import Circuit\n"
                    f"    circuit = {conversion}\n\n"
                    f"Install the extra with: pip install marqov[{extra}]"
                )

        raise TypeError(
            f"Expected a Marqov Circuit, got {type(circuit).__qualname__}. "
            f"Supported types: Marqov Circuit, Qiskit QuantumCircuit "
            f"(via Circuit.from_qiskit()), Cirq Circuit (via Circuit.from_cirq()), "
            f"PennyLane tape (via Circuit.from_pennylane())."
        )

    @abstractmethod
    async def execute(
        self,
        circuit: Circuit,
        shots: int = 1000,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute a circuit and return results.

        Args:
            circuit: The circuit to execute.
            shots: Number of measurement shots.
            **kwargs: Backend-specific options.

        Returns:
            ExecutionResult with measurement counts and metadata.
        """
        pass

    async def cancel(self, job_id: str) -> bool:
        """Cancel a running job.

        Args:
            job_id: The ID of the job to cancel.

        Returns:
            True if cancellation was successful, False otherwise.
        """
        return False

    async def get_status(self) -> DeviceStatus:
        """Get live device status. Override for cloud backends."""
        return DeviceStatus.always_online()

    @property
    def name(self) -> str:
        """Return the executor name."""
        return self.__class__.__name__
