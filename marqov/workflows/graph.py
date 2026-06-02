"""Transport graph for capturing task dependencies.

The transport graph is a directed acyclic graph (DAG) that captures the
dependencies between task calls in a workflow. It enables:
- Automatic parallelization of independent tasks
- Visualization of workflow structure
- Serialization for Temporal transport

Example:
    >>> @task
    ... def add(x, y): return x + y
    >>>
    >>> @workflow
    ... def compute():
    ...     a = add(1, 2)
    ...     b = add(3, 4)  # Independent of a
    ...     c = add(a, b)  # Depends on a and b
    ...     return c
    >>>
    >>> dispatch = compute()
    >>> levels = dispatch.graph.get_execution_order()
    >>> # levels == [['node_a', 'node_b'], ['node_c']]
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable

# Context variable for tracking active workflow graph
_active_graph: ContextVar[TransportGraph | None] = ContextVar("active_graph", default=None)


def get_active_graph() -> TransportGraph | None:
    """Get the currently active transport graph, if any."""
    return _active_graph.get()


def set_active_graph(graph: TransportGraph | None) -> None:
    """Set the active transport graph."""
    _active_graph.set(graph)


@dataclass
class TaskConfig:
    """Configuration for a task.

    Attributes:
        name: Display name for the task.
        executor: Which executor to use ("local", "braket").
        retries: Number of retry attempts on failure.
        timeout_seconds: Maximum execution time.
    """

    name: str
    executor: str = "local"
    retries: int = 0
    timeout_seconds: float = 300.0


@dataclass
class TaskNode:
    """Represents a task call in the transport graph.

    Attributes:
        id: Unique identifier for this node.
        func_name: Name of the function being called.
        func_ref: Serialized function reference (base64 cloudpickle).
        args: Positional arguments (may contain serialized proxies).
        kwargs: Keyword arguments (may contain serialized proxies).
        config: Task configuration.
        dependencies: IDs of nodes this node depends on.
    """

    id: str
    func_name: str
    func_ref: str  # base64 encoded cloudpickle
    args: list[Any]
    kwargs: dict[str, Any]
    config: TaskConfig
    dependencies: list[str] = field(default_factory=list)


class TaskProxy:
    """Proxy returned when a task is called inside a workflow.

    Instead of executing the function, calling a task inside a workflow
    returns this proxy object. The proxy registers itself in the transport
    graph and tracks dependencies.

    When used as an argument to another task, the dependency is automatically
    captured in the transport graph.
    """

    def __init__(self, node: TaskNode, graph: TransportGraph) -> None:
        """Initialize the proxy.

        Args:
            node: The task node this proxy represents.
            graph: The transport graph to register with.
        """
        self._node = node
        self._graph = graph
        graph.add_node(node)

    @property
    def node_id(self) -> str:
        """Return the node ID for this proxy."""
        return self._node.id

    def __repr__(self) -> str:
        """Return string representation."""
        return f"TaskProxy({self._node.func_name}, id={self._node.id})"


class TransportGraph:
    """Directed acyclic graph of task dependencies.

    The graph captures which tasks depend on which other tasks,
    enabling automatic parallelization of independent tasks.
    """

    def __init__(self) -> None:
        """Initialize an empty transport graph."""
        self.nodes: dict[str, TaskNode] = {}
        self.edges: list[tuple[str, str]] = []  # (from_id, to_id)
        self.output_nodes: list[str] = []

    def add_node(self, node: TaskNode) -> None:
        """Add a node to the graph.

        Args:
            node: The task node to add.
        """
        self.nodes[node.id] = node
        # Add edges from dependencies to this node
        for dep_id in node.dependencies:
            self.edges.append((dep_id, node.id))

    def set_output_node(self, node_id: str) -> None:
        """Set the output node of the workflow.

        Args:
            node_id: ID of the node that produces the final output.
        """
        self.output_nodes = [node_id]

    def set_output_nodes(self, node_ids: list[str]) -> None:
        """Set multiple output nodes.

        Args:
            node_ids: IDs of nodes that produce final outputs.
        """
        self.output_nodes = node_ids

    def get_execution_order(self) -> list[list[str]]:
        """Return nodes grouped by execution level.

        Nodes within the same level can run in parallel since they have
        no dependencies on each other. Levels must be executed in order.

        Returns:
            List of levels, where each level is a list of node IDs that
            can execute in parallel.

        Raises:
            ValueError: If a cycle is detected in the graph.
        """
        if not self.nodes:
            return []

        # Build adjacency list for predecessors
        predecessors: dict[str, set[str]] = {nid: set() for nid in self.nodes}
        for from_id, to_id in self.edges:
            predecessors[to_id].add(from_id)

        levels: list[list[str]] = []
        remaining = set(self.nodes.keys())

        while remaining:
            # Nodes with no remaining dependencies can run in parallel
            ready = [
                nid
                for nid in remaining
                if all(pred not in remaining for pred in predecessors[nid])
            ]

            if not ready:
                raise ValueError("Cycle detected in transport graph")

            levels.append(ready)
            remaining -= set(ready)

        return levels

    def get_parallel_groups(self) -> list[list[str]]:
        """Alias for get_execution_order.

        Returns:
            List of parallel execution groups.
        """
        return self.get_execution_order()

    def to_dict(self) -> dict[str, Any]:
        """Serialize graph for Temporal workflow input.

        Returns:
            Dictionary representation of the graph.
        """
        return {
            "nodes": {
                nid: {
                    "id": node.id,
                    "func_name": node.func_name,
                    "func_ref": node.func_ref,
                    "args": node.args,
                    "kwargs": node.kwargs,
                    "config": {
                        "name": node.config.name,
                        "executor": node.config.executor,
                        "retries": node.config.retries,
                        "timeout_seconds": node.config.timeout_seconds,
                    },
                    "dependencies": node.dependencies,
                }
                for nid, node in self.nodes.items()
            },
            "edges": self.edges,
            "output_nodes": self.output_nodes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransportGraph:
        """Deserialize graph from dictionary.

        Args:
            data: Dictionary from to_dict().

        Returns:
            Reconstructed TransportGraph instance.
        """
        graph = cls()

        for nid, ndata in data["nodes"].items():
            config = ElectronConfig(
                name=ndata["config"]["name"],
                executor=ndata["config"]["executor"],
                retries=ndata["config"]["retries"],
                timeout_seconds=ndata["config"]["timeout_seconds"],
            )
            node = ElectronNode(
                id=ndata["id"],
                func_name=ndata["func_name"],
                func_ref=ndata["func_ref"],
                args=ndata["args"],
                kwargs=ndata["kwargs"],
                config=config,
                dependencies=ndata["dependencies"],
            )
            graph.nodes[nid] = node

        graph.edges = [(e[0], e[1]) for e in data["edges"]]
        graph.output_nodes = data["output_nodes"]

        return graph

    def to_dot(self) -> str:
        """Generate DOT format for visualization.

        Returns:
            DOT format string for graphviz.
        """
        lines = ["digraph lattice {", "  rankdir=TB;"]

        for nid, node in self.nodes.items():
            label = node.func_name
            if nid in self.output_nodes:
                lines.append(f'  "{nid}" [label="{label}" style=filled fillcolor=lightblue];')
            else:
                lines.append(f'  "{nid}" [label="{label}"];')

        for src, dst in self.edges:
            lines.append(f'  "{src}" -> "{dst}";')

        lines.append("}")
        return "\n".join(lines)

    def __len__(self) -> int:
        """Return number of nodes in the graph."""
        return len(self.nodes)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"TransportGraph(nodes={len(self.nodes)}, edges={len(self.edges)})"


def generate_node_id() -> str:
    """Generate a unique node ID.

    Returns:
        Short UUID-based identifier.
    """
    return uuid.uuid4().hex[:8]


def extract_dependencies(args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[str]:
    """Extract node IDs from TaskProxy arguments.

    Args:
        args: Positional arguments that may contain proxies.
        kwargs: Keyword arguments that may contain proxies.

    Returns:
        List of node IDs that this task depends on.
    """
    deps: list[str] = []

    for arg in args:
        if isinstance(arg, TaskProxy):
            deps.append(arg.node_id)
        elif isinstance(arg, (list, tuple)):
            for item in arg:
                if isinstance(item, TaskProxy):
                    deps.append(item.node_id)

    for value in kwargs.values():
        if isinstance(value, TaskProxy):
            deps.append(value.node_id)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, TaskProxy):
                    deps.append(item.node_id)

    return deps


# Backward compatibility aliases
ElectronConfig = TaskConfig
ElectronNode = TaskNode
ElectronProxy = TaskProxy
