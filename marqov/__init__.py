"""Marqov - Orchestration engine for hybrid quantum-classical workflows.

Marqov provides:
- Backend-agnostic quantum circuit abstraction
- @task/@workflow decorators for defining workflows
- Automatic parallelization of independent tasks
- Temporal-backed workflow durability and scheduling
- Multi-vendor execution (Braket, IBM, IonQ, Rigetti)

Quick Start with @task/@workflow:
    >>> from marqov import task, workflow, Circuit
    >>>
    >>> @task
    ... def measure(circuit, pauli):
    ...     return run_circuit(circuit, pauli)
    >>>
    >>> @workflow
    ... def vqe_step(theta):
    ...     circuit = build(theta)
    ...     z0 = measure(circuit, "ZI")  # Runs in parallel
    ...     z1 = measure(circuit, "IZ")  # Runs in parallel
    ...     return compute(z0, z1)
    >>>
    >>> dispatch = vqe_step(0.5)
    >>> result = await dispatch.run(client)

Simple circuit execution:
    >>> from marqov import Circuit, bell_state
    >>> from marqov.executors import LocalExecutor
    >>>
    >>> circuit = Circuit().h(0).cnot(0, 1)
    >>> executor = LocalExecutor()
    >>> result = await executor.execute(circuit, shots=1000)
"""

__version__ = "0.2.0"

# Re-export commonly used items for convenience
from marqov.circuits import Circuit, bell_state, ghz_state
from marqov.device import MarqovDevice, get_device
from marqov.executors import BaseExecutor, ExecutionResult, LocalExecutor
from marqov.workflows import (
    task,
    workflow,
    TemporalConfig,
    create_worker,
)

__all__ = [
    "__version__",
    "task",
    "workflow",
    "Circuit",
    "bell_state",
    "ghz_state",
    "MarqovDevice",
    "get_device",
    "BaseExecutor",
    "ExecutionResult",
    "LocalExecutor",
    "TemporalConfig",
    "create_worker",
]
