"""Workflow orchestration with Temporal.

This module provides workflow primitives for Marqov, including
@task/@workflow decorators for defining quantum-classical workflows
with automatic parallelization.

Example:
    >>> from marqov.workflows import task, workflow
    >>>
    >>> @task
    ... def measure(circuit, pauli):
    ...     return executor.run(circuit, pauli)
    >>>
    >>> @workflow
    ... def vqe_step(theta):
    ...     circuit = build(theta)
    ...     z0 = measure(circuit, "ZI")  # Parallel
    ...     z1 = measure(circuit, "IZ")  # Parallel
    ...     return compute(z0, z1)
    >>>
    >>> dispatch = vqe_step(0.5)
    >>> result = await dispatch.run(client)

For native Temporal access, you can also use the underlying
workflow and activity classes directly:
    >>> from marqov.workflows import JobWorkflow, create_worker

Note:
    `electron` and `lattice` are deprecated aliases for `task` and `workflow`.
    `LatticeWorkflow` is a deprecated alias for `JobWorkflow`.
"""

from dataclasses import dataclass

from marqov.workflows.decorators import (
    task,
    workflow,
    WorkflowDispatch,
    # Deprecated aliases
    electron,
    lattice,
    LatticeDispatch,
)
from marqov.workflows.graph import TransportGraph, TaskProxy, ElectronProxy
from marqov.workflows.temporal_workflow import JobWorkflow
from marqov.workflows.runner import create_worker, LatticeWorkflow, execute_electron
from marqov.workflows.activity import execute_task, prepare_node_inputs


@dataclass
class TemporalConfig:
    """Configuration for Temporal connection.

    Attributes:
        host: Temporal server host.
        port: Temporal server port.
        namespace: Temporal namespace.
        task_queue: Default task queue name.
    """

    host: str = "localhost"
    port: int = 7233
    namespace: str = "default"
    task_queue: str = "marqov-workflows"

    @property
    def address(self) -> str:
        """Get Temporal server address."""
        return f"{self.host}:{self.port}"


__all__ = [
    # Primary decorators (new names)
    "task",
    "workflow",
    # Dispatch
    "WorkflowDispatch",
    # Graph
    "TransportGraph",
    "TaskProxy",
    # Temporal
    "JobWorkflow",
    "create_worker",
    "execute_task",
    "prepare_node_inputs",
    "TemporalConfig",
    # Deprecated aliases (will be removed in v0.3.0)
    "electron",
    "lattice",
    "LatticeDispatch",
    "ElectronProxy",
    "LatticeWorkflow",
    "execute_electron",
]
