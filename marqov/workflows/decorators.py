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
"""

from __future__ import annotations

import base64
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, TypeVar, overload

import cloudpickle

if TYPE_CHECKING:
    from marqov.workflows.capture import WorkflowCapture

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
_NO_CAPTURE_RESULT = object()


def _serialize_arg(arg: Any) -> Any:
    """Serialize an argument for the transport graph.

    Converts TaskProxy objects to placeholder dicts inside lists, tuples
    (namedtuples keep their type) and dict values. Other objects are kept
    as-is in the graph; ``_transport_arg`` encodes them for Temporal.
    """
    if isinstance(arg, TaskProxy):
        return {"__proxy__": True, "node_id": arg.node_id}
    elif isinstance(arg, list):
        return [_serialize_arg(item) for item in arg]
    elif isinstance(arg, tuple):
        items = [_serialize_arg(item) for item in arg]
        if type(arg) is not tuple and hasattr(type(arg), "_make"):
            return type(arg)._make(items)
        return tuple(items)
    elif isinstance(arg, dict):
        return {k: _serialize_arg(v) for k, v in arg.items()}
    else:
        return arg


def _is_marker(value: Any) -> bool:
    """Return True for a ``__proxy__`` placeholder or a ``__cloudpickle__`` envelope."""
    return isinstance(value, dict) and bool(value.get("__proxy__") or value.get("__cloudpickle__"))


def _contains_placeholder(value: Any) -> bool:
    """Return True if a ``__proxy__`` placeholder sits anywhere inside ``value``."""
    if isinstance(value, dict):
        return bool(value.get("__proxy__")) or any(
            _contains_placeholder(item) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_placeholder(item) for item in value)
    return False


def _transport_arg(value: Any) -> Any:
    """Encode a graph argument as JSON data for the Temporal workflow input.

    The same rule as task results (``activity._serialize_value``): JSON values
    pass through, NumPy bool/integer/float scalars are unboxed, lists and
    tuples become lists, and any other object travels in the
    ``__cloudpickle__`` envelope, which only the task child decodes. A
    ``__proxy__`` placeholder or an existing envelope passes through unchanged
    and is never pickled, because ``prepare_node_inputs`` resolves placeholders
    by that marker. A namedtuple (or other tuple subclass) is pickled whole, so
    it arrives as the same type, unless it holds a placeholder, in which case
    it arrives as a list like a plain tuple.

    Raises:
        TypeError: If a value cannot be cloudpickled.
    """
    from marqov.workflows.activity import _cloudpickle_envelope, _unbox_numpy_scalar

    value = _unbox_numpy_scalar(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    elif isinstance(value, list) or (
        isinstance(value, tuple) and (type(value) is tuple or _contains_placeholder(value))
    ):
        return [_transport_arg(item) for item in value]
    elif isinstance(value, dict):
        if _is_marker(value):
            return value
        return {k: _transport_arg(v) for k, v in value.items()}
    try:
        return _cloudpickle_envelope(value)
    except Exception as exc:
        raise TypeError(
            f"value of type {type(value).__module__}.{type(value).__qualname__} cannot "
            f"be serialized with cloudpickle: {type(exc).__name__}: {exc}"
        ) from exc


def _transport_inputs(
    func_name: str, args: list[Any], kwargs: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    """Encode one node's arguments, naming the task and argument on failure."""
    encoded_args: list[Any] = []
    for position, arg in enumerate(args):
        try:
            encoded_args.append(_transport_arg(arg))
        except TypeError as exc:
            raise TypeError(
                f"Task {func_name!r}: positional argument {position} cannot be sent "
                f"to the task: {exc}"
            ) from exc
    encoded_kwargs: dict[str, Any] = {}
    for key, value in kwargs.items():
        try:
            encoded_kwargs[key] = _transport_arg(value)
        except TypeError as exc:
            raise TypeError(
                f"Task {func_name!r}: keyword argument {key!r} cannot be sent "
                f"to the task: {exc}"
            ) from exc
    return encoded_args, encoded_kwargs


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

    When called outside a workflow, the wrapper just calls the decorated
    function directly (`fn(*args, **kwargs)`) — it does not special-case
    coroutine functions. For a sync function that means "executes normally
    and returns its result." For an `async def` function it does NOT await
    it: calling it bare returns an unawaited coroutine and nothing runs. If
    you need to call an async @task outside a workflow, drive it yourself
    with `asyncio.run(...)` (see the second example below).

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
        >>>
        >>> # Outside workflow, an async @task must be driven explicitly —
        >>> # measure(circuit) alone would return an unawaited coroutine.
        >>> import asyncio
        >>> result = asyncio.run(measure(circuit))
    """

    def decorator(fn: F) -> F:
        config = TaskConfig(
            name=name or fn.__name__,
            executor=executor,
            retries=retries,
            timeout_seconds=timeout,
        )

        # Serialize the function LAZILY — only when a task actually builds a graph
        # node, and cache the result so it is computed at most once per task.
        #
        # Serializing eagerly here (at decoration time) was an import-time landmine:
        # cloudpickle serializes local/closure functions BY VALUE, walking the
        # function's referenced globals and closure by hand. In a heavy-import
        # environment that deep traversal can overflow the recursion limit — a
        # catchable RecursionError on Linux, a hard C-stack segfault on macOS / musl
        # — and it fired for EVERY @task the moment its module was imported, whether
        # or not the task was ever dispatched. Deferring to graph-build means
        # importing a module full of @task functions pickles nothing; the cost (and
        # the residual by-value risk for genuinely local functions) moves to the point
        # a workflow graph is actually built.
        _func_ref_cache: list[str] = []

        def _func_ref() -> str:
            if not _func_ref_cache:
                _func_ref_cache.append(
                    base64.b64encode(cloudpickle.dumps(fn)).decode("utf-8")
                )
            return _func_ref_cache[0]

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
                    func_ref=_func_ref(),
                    args=list(_serialize_arg(arg) for arg in args),
                    kwargs={k: _serialize_arg(v) for k, v in kwargs.items()},
                    config=config,
                    dependencies=dependencies,
                )

                return TaskProxy(node, graph)
            else:
                # Outside lattice: execute directly
                return fn(*args, **kwargs)

        # Mark as task for introspection.
        # NB: no eager `_task_func_ref` attribute — it was write-only (zero readers
        # in the SDK or the platform) and computing it here would defeat the lazy
        # serialization above (cloudpickle by-value recursion at graph-build for
        # local @task fns under heavy imports).
        wrapper._is_task = True  # type: ignore
        wrapper._task_config = config  # type: ignore

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
        *,
        captured_result: Any = _NO_CAPTURE_RESULT,
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
        self._captured_result = captured_result

    def capture(self) -> WorkflowCapture:
        """Export this already-built graph and its complete return structure.

        No workflow body is rerun and no task is submitted. The returned object
        contains executable material; an adapter must select an appropriate
        isolated runtime and artifact codec. This is not the legacy Temporal
        JSON transport and does not change run()/start() result semantics.
        """
        from marqov.workflows.capture import capture_output

        if self._captured_result is _NO_CAPTURE_RESULT:
            raise ValueError("Dispatch has no captured return value")
        return capture_output(self.graph, self._captured_result)

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
        Task arguments are encoded by ``_transport_arg``: non-JSON values (a
        ``Circuit``, an array) travel as ``__cloudpickle__`` envelopes that
        only the task child unpickles.

        Returns:
            Dictionary with nodes, execution_levels, and output_nodes.

        Raises:
            TypeError: If a task argument cannot be cloudpickled; the message
                names the task and the argument position or keyword.
        """
        # Get execution levels (parallel groups)
        execution_levels = self.graph.get_execution_order()

        # Serialize all nodes as dicts
        nodes: dict[str, dict[str, Any]] = {}
        for node_id, node in self.graph.nodes.items():
            args, kwargs = _transport_inputs(node.func_name, node.args, node.kwargs)
            nodes[node_id] = {
                "node_id": node.id,
                "func_name": node.func_name,
                "func_ref": node.func_ref,
                "args": args,
                "kwargs": kwargs,
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
            The workflow's output, decoded in this process: a single output
            node yields that task's return value, several output nodes yield a
            dict keyed by node ID, and a workflow with no output nodes yields
            a dict of every node's result. JSON values (None, bool, int,
            float, str, list, dict) come back as-is, with tuples as lists and
            NumPy bool/integer/float scalars as Python bool/int/float. Any
            other return value (a dataclass, an ``ExecutionResult``, an
            array) travels as cloudpickle bytes and is unpickled here, so the
            classes it needs must be importable by the caller.
        """
        import json
        import uuid

        from marqov.workflows.activity import _deserialize_value
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
            return _deserialize_value(parsed["result"])
        return _deserialize_value(parsed)

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
            ``result`` is the enriched dict, ``{"result": ..., "_workflow_metadata": ...}``,
            with ``result["result"]`` decoded as described in :meth:`run`.
        """
        import json

        from marqov.workflows.activity import _deserialize_value

        handle, workflow_id, run_id = await self.start_with_ids(client, task_queue)

        # Wait for result
        result_json = await handle.result()

        # Deserialize result — return full enriched dict (metadata included)
        parsed = json.loads(result_json)
        if isinstance(parsed, dict) and "result" in parsed and "_workflow_metadata" in parsed:
            parsed["result"] = _deserialize_value(parsed["result"])

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

                # Track output node(s), rejecting proxies from another invocation.
                if isinstance(result, TaskProxy):
                    outputs = [result]
                elif isinstance(result, (list, tuple)):
                    outputs = [item for item in result if isinstance(item, TaskProxy)]
                elif isinstance(result, dict):
                    outputs = [item for item in result.values() if isinstance(item, TaskProxy)]
                else:
                    outputs = []
                for output in outputs:
                    if output._graph is not graph or output.node_id not in graph.nodes:
                        raise ValueError("Workflow output references a different graph")
                if outputs:
                    graph.set_output_nodes([output.node_id for output in outputs])

            finally:
                # Restore previous graph (for nested workflows)
                set_active_graph(previous_graph)

            return WorkflowDispatch(
                graph=graph,
                name=workflow_name,
                args=args,
                kwargs=kwargs,
                captured_result=result,
            )

        # Mark as workflow for introspection
        wrapper._is_workflow = True  # type: ignore
        wrapper._workflow_name = workflow_name  # type: ignore

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator
