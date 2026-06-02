"""Temporal worker creation and workflow/activity exports.

This module provides the create_worker function and re-exports the
workflow and activity components needed for Temporal execution.

IMPORTANT: Imports are structured carefully to avoid sandbox issues.
The workflow module has NO marqov imports - it's safe to import.
The activity module has marqov imports but activities aren't sandboxed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.worker.workflow_sandbox import (
    SandboxedWorkflowRunner,
    SandboxRestrictions,
)

# These imports are safe:
# - temporal_workflow.py has NO marqov imports (uses string activity names)
# - activity.py imports cloudpickle but that's fine for activities
from marqov.workflows.temporal_workflow import JobWorkflow
from marqov.workflows.activity import execute_task, prepare_node_inputs


def create_worker(
    client: Client,
    task_queue: str = "marqov-workflows",
) -> Worker:
    """Create a Temporal worker for Marqov workflows.

    Args:
        client: Temporal client connection.
        task_queue: Temporal task queue name.

    Returns:
        Worker instance that can be started with `async with worker:`.

    Example:
        >>> client = await Client.connect("localhost:7233")
        >>> worker = create_worker(client)
        >>> async with worker:
        ...     # Worker is running, can execute workflows
        ...     result = await dispatch.run(client)
    """
    # temporal_workflow.py has zero marqov imports, but because it lives
    # under the marqov package, the sandbox walks up and validates the
    # parent — pulling in numpy/sympy which use subprocess. Mark marqov
    # as a passthrough module so the sandbox skips it.
    sandbox_runner = SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules(
            "marqov",
        )
    )

    return Worker(
        client,
        task_queue=task_queue,
        workflows=[JobWorkflow],
        activities=[execute_task, prepare_node_inputs],
        workflow_runner=sandbox_runner,
    )


# Backward compatibility aliases
LatticeWorkflow = JobWorkflow
execute_electron = execute_task

# Re-export for backwards compatibility
__all__ = [
    "JobWorkflow",
    "create_worker",
    "execute_task",
    "prepare_node_inputs",
    # Backward compatibility
    "LatticeWorkflow",
    "execute_electron",
]
