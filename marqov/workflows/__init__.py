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

Provider credentials: every task body runs in a child process whose
environment is built from an allowlist, so host secrets are not inherited.
The only way a credential enters it is the ``provider_env`` argument of the
``execute_task`` activity, and ``JobWorkflow`` never passes it, so a task
dispatched through ``WorkflowDispatch`` (``run``, ``run_with_ids``,
``start_with_ids``, ``dispatch``) runs with no provider credentials. A host
runner that needs them must invoke ``execute_task`` itself.
``@task(executor=...)`` is reserved: it is recorded on the graph and not
consulted when dispatching, so it neither selects a backend nor supplies
credentials. The task's display ``name`` is likewise graph-local.

For native Temporal access, you can also use the underlying
workflow and activity classes directly:
    >>> from marqov.workflows import JobWorkflow, create_worker
"""

from dataclasses import dataclass

from marqov.workflows.decorators import (
    task,
    workflow,
    WorkflowDispatch,
)
from marqov.workflows.graph import TransportGraph, TaskProxy
from marqov.workflows.capture import WorkflowCapture, TaskResultReference
from marqov.workflows.temporal_workflow import JobWorkflow
from marqov.workflows.runner import create_worker
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
    "task",
    "workflow",
    "WorkflowDispatch",
    "WorkflowCapture",
    "TaskResultReference",
    "TransportGraph",
    "TaskProxy",
    "JobWorkflow",
    "create_worker",
    "execute_task",
    "prepare_node_inputs",
    "TemporalConfig",
]
