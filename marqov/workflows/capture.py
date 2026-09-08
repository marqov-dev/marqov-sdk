"""Process-local workflow capture for execution adapters.

Capture is executable material, not a JSON transport or permission to execute.
Adapters choose their own artifact codecs, runtime isolation and policy checks.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Callable, Mapping

from marqov.workflows.graph import TaskProxy, TransportGraph


@dataclass(frozen=True)
class TaskResultReference:
    """Reference to a task result within one captured graph."""

    node_id: str


@dataclass(frozen=True)
class WorkflowCapture:
    """Graph plus return template, without submitting work or choosing a codec.

    The graph includes callable bytes, arguments and requested task configuration.
    It remains mutable; this object is not a durable or authoritative snapshot.
    Lists, tuples and dictionary values in output preserve their structure;
    non-container values are opaque leaves and are not copied or inspected.
    """

    graph: TransportGraph
    output: Any

    def resolve_output(self, results: Mapping[str, Any]) -> Any:
        """Reconstruct the declared return value from completed node results.

        Missing results raise KeyError. This performs no task execution. Only
        use capture/material from code you trust or within its execution sandbox.
        """
        return _map_output(
            self.output,
            lambda value: (
                results[value.node_id] if isinstance(value, TaskResultReference) else value
            ),
        )


def _map_output(value: Any, leaf: Callable[[Any], Any], active: set[int] | None = None) -> Any:
    if active is None:
        active = set()
    if type(value) not in (list, tuple, dict):
        if isinstance(value, (list, tuple, dict)):
            raise TypeError("Capture supports plain list, tuple and dict containers only")
        return leaf(value)
    identity = id(value)
    if identity in active:
        raise ValueError("Cyclic workflow return containers are not supported")
    active.add(identity)
    try:
        if type(value) is dict:
            if any(isinstance(key, (TaskProxy, TaskResultReference)) for key in value):
                raise TypeError("Task references cannot be workflow return dictionary keys")
            return {key: _map_output(item, leaf, active) for key, item in value.items()}
        return type(value)(_map_output(item, leaf, active) for item in value)
    finally:
        active.remove(identity)


def capture_output(graph: TransportGraph, result: Any) -> WorkflowCapture:
    def reference(value: Any) -> Any:
        if isinstance(value, TaskProxy):
            if value._graph is not graph or value.node_id not in graph.nodes:
                raise ValueError("Workflow output references a different graph")
            return TaskResultReference(value.node_id)
        # A nested dispatch is not a completed task result. Import lazily to keep
        # this module independent of decorator initialization order.
        from marqov.workflows.decorators import WorkflowDispatch

        if isinstance(value, WorkflowDispatch):
            raise TypeError("Nested workflow dispatch output is not supported by capture")
        if inspect.isawaitable(value):
            if inspect.iscoroutine(value):
                value.close()
            raise TypeError("Awaitable workflow output is not supported by capture")
        return value

    return WorkflowCapture(graph, _map_output(result, reference))
