"""Task and workflow decorators for quantum workflow definition.

These decorators provide a simple API for defining quantum-classical
workflows with automatic parallelization.

Example:
    >>> from marqov import task, workflow
    >>>
    >>> @task
    ... def measure(circuit, pauli):
    ...     return executor.run(circuit, pauli)
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

Note:
    `electron` and `lattice` are deprecated aliases for `task` and `workflow`.
    They will be removed in a future version.
"""

from __future__ import annotations

import base64
from functools import wraps
from typing import Any, Callable, TypeVar, overload

import cloudpickle

import warnings

from marqov.workflows.graph import (
    TaskConfig,
    TaskNode,
    TaskProxy,
    TransportGraph,
    extract_dependencies,
    generate_node_id,
    get_active_graph,
    set_active_graph,
)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


def _serialize_arg(arg: Any) -> Any:
    """Serialize an argument for transport.

    Converts TaskProxy objects to placeholder dicts.
    """
    if isinstance(arg, TaskProxy):
        return {"__proxy__": True, "node_id": arg.node_id}
    elif isinstance(arg, (list, tuple)):
        return type(arg)(_serialize_arg(item) for item in arg)
    elif isinstance(arg, dict):
        return {k: _serialize_arg(v) for k, v in arg.items()}
    else:
        return arg


@overload
def task(func: F) -> F: ...


@overload
def task(
    *,
    name: str | None = None,
    executor: str = "local",
    retries: int = 0,
    timeout: float = 300.0,
) -> Callable[[F], F]: ...


def task(
    func: F | None = None,
    *,
    name: str | None = None,
    executor: str = "local",
    retries: int = 0,
    timeout: float = 300.0,
) -> F | Callable[[F], F]:
    """Decorator to mark a function as a task.

    Tasks are the basic units of work in a Marqov workflow. When called
    inside a @workflow function, they don't execute immediately - instead they
    return a proxy that registers the call in the transport graph.

    When called outside a workflow, tasks execute normally.

    Args:
        func: The function to decorate.
        name: Display name for the task. Defaults to function name.
        executor: Which executor to use ("local", "braket").
        retries: Number of retry attempts on failure.
        timeout: Maximum execution time in seconds.

    Returns:
        Decorated function that behaves as a task.

    Example:
        >>> @task
        ... def add(x, y):
        ...     return x + y
        >>>
        >>> # Outside workflow: executes normally
        >>> result = add(1, 2)  # Returns 3
        >>>
        >>> @task(executor="braket", timeout=600)
        ... async def measure(circuit):
        ...     return await executor.run(circuit)
    """

    def decorator(fn: F) -> F:
        config = TaskConfig(
            name=name or fn.__name__,
            executor=executor,
            retries=retries,
            timeout_seconds=timeout,
        )

        # Serialize the function once
        func_bytes = cloudpickle.dumps(fn)
        func_ref = base64.b64encode(func_bytes).decode("utf-8")

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check if we're inside an active lattice
            graph = get_active_graph()

            if graph is not None:
                # Inside lattice: create graph node, don't execute
                dependencies = extract_dependencies(args, kwargs)

                node = TaskNode(
                    id=generate_node_id(),
                    func_name=fn.__name__,
                    func_ref=func_ref,
                    args=list(_serialize_arg(arg) for arg in args),
                    kwargs={k: _serialize_arg(v) for k, v in kwargs.items()},
                    config=config,
                    dependencies=dependencies,
                )

                return TaskProxy(node, graph)
            else:
                # Outside lattice: execute directly
                return fn(*args, **kwargs)

        # Mark as task for introspection
        wrapper._is_task = True  # type: ignore
        wrapper._task_config = config  # type: ignore
        wrapper._task_func_ref = func_ref  # type: ignore

        return wrapper  # type: ignore

    if func is not None:
        return decorator(func)
    return decorator


class WorkflowDispatch:
    """Handle for dispatching a workflow.

    This object is returned when a @workflow function is called. It contains
    the transport graph and can be used to:
    - Visualize the workflow
    - Dispatch to Temporal for durable execution
    - Get the result
    """

    def __init__(
        self,
        graph: TransportGraph,
        name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        """Initialize the dispatch handle.

        Args:
            graph: The transport graph for this workflow.
            name: Name of the workflow.
            args: Original arguments passed to the workflow.
            kwargs: Original keyword arguments passed to the workflow.
        """
        self.graph = graph
        self.name = name
        self._args = args
        self._kwargs = kwargs

    def visualize(self) -> str:
        """Return DOT format graph visualization.

        Returns:
            DOT format string that can be rendered with graphviz.
        """
        return self.graph.to_dot()

    def get_parallel_groups(self) -> list[list[str]]:
        """Get groups of tasks that can run in parallel.

        Returns:
            List of parallel execution groups (node IDs).
        """
        return self.graph.get_parallel_groups()

    def _prepare_workflow_input(self) -> dict[str, Any]:
        """Prepare the workflow input as pure JSON-serializable data.

        This pre-computes execution levels and serializes all node data
        so the workflow only receives primitive types (no marqov imports needed).

        Returns:
            Dictionary with nodes, execution_levels, and output_nodes.
        """
        import json

        # Get execution levels (parallel groups)
        execution_levels = self.graph.get_execution_order()

        # Serialize all nodes as dicts
        nodes: dict[str, dict[str, Any]] = {}
        for node_id, node in self.graph.nodes.items():
            nodes[node_id] = {
                "node_id": node.id,
                "func_name": node.func_name,
                "func_ref": node.func_ref,
                "args": node.args,
                "kwargs": node.kwargs,
                "retries": node.config.retries,
                "timeout_seconds": node.config.timeout_seconds,
                "dependencies": node.dependencies,
            }

        return {
            "nodes": nodes,
            "execution_levels": execution_levels,
            "output_nodes": self.graph.output_nodes,
        }

    async def run(self, client: Any, task_queue: str = "marqov-workflows") -> Any:
        """Execute the lattice workflow on Temporal.

        Args:
            client: Temporal client connection.
            task_queue: Temporal task queue name.

        Returns:
            The result of the workflow execution.
        """
        import json
        import uuid

        from marqov.workflows.temporal_workflow import JobWorkflow

        # Prepare pure JSON input (no marqov objects)
        workflow_input = self._prepare_workflow_input()

        # Generate workflow ID
        workflow_id = f"{self.name}-{uuid.uuid4().hex[:8]}"

        # Start workflow
        await client.start_workflow(
            JobWorkflow.run,
            args=[workflow_input],
            id=workflow_id,
            task_queue=task_queue,
        )

        # Wait for result
        handle = client.get_workflow_handle(workflow_id)
        result_json = await handle.result()

        # Deserialize result — unwrap enriched format for backward compat
        parsed = json.loads(result_json)
        if isinstance(parsed, dict) and "result" in parsed and "_workflow_metadata" in parsed:
            return parsed["result"]
        return parsed

    async def start_with_ids(self, client: Any, task_queue: str = "marqov-workflows") -> tuple[Any, str, str]:
        """Start workflow and return handle with IDs, without waiting for result.

        Use this when you need the workflow_id before the workflow completes
        (e.g., to store it in the database for cancellation support).

        Args:
            client: Temporal client connection.
            task_queue: Temporal task queue name.

        Returns:
            Tuple of (handle, workflow_id, run_id). Call `await handle.result()`
            to get the workflow result when ready.
        """
        import uuid

        from marqov.workflows.temporal_workflow import JobWorkflow

        # Prepare pure JSON input (no marqov objects)
        workflow_input = self._prepare_workflow_input()

        # Generate workflow ID
        workflow_id = f"{self.name}-{uuid.uuid4().hex[:8]}"

        # Start workflow — returns handle immediately
        handle = await client.start_workflow(
            JobWorkflow.run,
            args=[workflow_input],
            id=workflow_id,
            task_queue=task_queue,
        )

        run_id = handle.first_execution_run_id

        return handle, workflow_id, run_id

    async def run_with_ids(self, client: Any, task_queue: str = "marqov-workflows") -> tuple[Any, str, str]:
        """Execute workflow and return result with workflow IDs.

        Args:
            client: Temporal client connection.
            task_queue: Temporal task queue name.

        Returns:
            Tuple of (result, workflow_id, run_id) for tracking and debugging.
        """
        import json

        handle, workflow_id, run_id = await self.start_with_ids(client, task_queue)

        # Wait for result
        result_json = await handle.result()

        # Deserialize result — return full enriched dict (metadata included)
        parsed = json.loads(result_json)

        return parsed, workflow_id, run_id

    async def dispatch(self, client: Any, task_queue: str = "marqov-workflows") -> str:
        """Submit the workflow to Temporal without waiting.

        Args:
            client: Temporal client connection.
            task_queue: Temporal task queue name.

        Returns:
            Workflow ID that can be used to check status/get result later.
        """
        import uuid

        from marqov.workflows.temporal_workflow import JobWorkflow

        # Prepare pure JSON input (no marqov objects)
        workflow_input = self._prepare_workflow_input()

        # Generate workflow ID
        workflow_id = f"{self.name}-{uuid.uuid4().hex[:8]}"

        # Start workflow (don't wait)
        await client.start_workflow(
            JobWorkflow.run,
            args=[workflow_input],
            id=workflow_id,
            task_queue=task_queue,
        )

        return workflow_id

    def __repr__(self) -> str:
        """Return string representation."""
        return f"WorkflowDispatch({self.name}, nodes={len(self.graph)}, parallel_groups={len(self.get_parallel_groups())})"


@overload
def workflow(func: F) -> Callable[..., WorkflowDispatch]: ...


@overload
def workflow(
    *,
    name: str | None = None,
) -> Callable[[F], Callable[..., WorkflowDispatch]]: ...


def workflow(
    func: F | None = None,
    *,
    name: str | None = None,
) -> Callable[..., WorkflowDispatch] | Callable[[F], Callable[..., WorkflowDispatch]]:
    """Decorator to mark a function as a workflow.

    Workflows are functions that compose tasks. When a workflow
    is called, it doesn't execute the tasks immediately. Instead, it:
    1. Creates a new transport graph
    2. Executes the function in "graph building mode"
    3. Captures all task calls and their dependencies
    4. Returns a WorkflowDispatch object for execution

    Args:
        func: The function to decorate.
        name: Display name for the workflow. Defaults to function name.

    Returns:
        Decorated function that returns a WorkflowDispatch.

    Example:
        >>> @workflow(name="VQE-H2")
        ... def vqe_step(theta):
        ...     circuit = build(theta)
        ...     z0 = measure(circuit, "ZI")  # Parallel
        ...     z1 = measure(circuit, "IZ")  # Parallel
        ...     return compute(z0, z1)
        >>>
        >>> dispatch = vqe_step(0.5)
        >>> print(dispatch.visualize())  # Show graph
        >>> result = await dispatch.run(client)  # Execute
    """

    def decorator(fn: F) -> Callable[..., WorkflowDispatch]:
        workflow_name = name or fn.__name__

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> WorkflowDispatch:
            # Create new transport graph
            graph = TransportGraph()

            # Enter workflow context
            previous_graph = get_active_graph()
            set_active_graph(graph)

            try:
                # Execute function - tasks will register nodes
                result = fn(*args, **kwargs)

                # Track output node(s)
                if isinstance(result, TaskProxy):
                    graph.set_output_node(result.node_id)
                elif isinstance(result, (list, tuple)):
                    output_ids = [
                        item.node_id for item in result if isinstance(item, TaskProxy)
                    ]
                    if output_ids:
                        graph.set_output_nodes(output_ids)
                elif isinstance(result, dict):
                    output_ids = [
                        v.node_id for v in result.values() if isinstance(v, TaskProxy)
                    ]
                    if output_ids:
                        graph.set_output_nodes(output_ids)

            finally:
                # Restore previous graph (for nested workflows)
                set_active_graph(previous_graph)

            return WorkflowDispatch(
                graph=graph,
                name=workflow_name,
                args=args,
                kwargs=kwargs,
            )

        # Mark as workflow for introspection
        wrapper._is_workflow = True  # type: ignore
        wrapper._workflow_name = workflow_name  # type: ignore

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


# =============================================================================
# Deprecated aliases for backwards compatibility
# =============================================================================

def electron(
    func: F | None = None,
    *,
    name: str | None = None,
    executor: str = "local",
    retries: int = 0,
    timeout: float = 300.0,
) -> F | Callable[[F], F]:
    """Deprecated alias for @task. Use @task instead.

    .. deprecated::
        `electron` is deprecated and will be removed in v0.3.0.
        Use `task` instead.
    """
    warnings.warn(
        "@electron is deprecated, use @task instead. "
        "@electron will be removed in v0.3.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return task(func, name=name, executor=executor, retries=retries, timeout=timeout)


def lattice(
    func: F | None = None,
    *,
    name: str | None = None,
) -> Callable[..., WorkflowDispatch] | Callable[[F], Callable[..., WorkflowDispatch]]:
    """Deprecated alias for @workflow. Use @workflow instead.

    .. deprecated::
        `lattice` is deprecated and will be removed in v0.3.0.
        Use `workflow` instead.
    """
    warnings.warn(
        "@lattice is deprecated, use @workflow instead. "
        "@lattice will be removed in v0.3.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return workflow(func, name=name)


# Backwards compatibility alias
LatticeDispatch = WorkflowDispatch
