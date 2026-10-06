"""Regression tests for transport graph integrity before execution."""

import re
from uuid import UUID

import pytest

from marqov.workflows.graph import TaskConfig, TaskNode, TransportGraph, generate_node_id


def node(node_id, dependencies=(), func_name="task"):
    return TaskNode(
        id=node_id,
        func_name=func_name,
        func_ref="",
        args=[],
        kwargs={},
        config=TaskConfig(name=func_name),
        dependencies=list(dependencies),
    )


@pytest.mark.parametrize("method", ["get_execution_order", "get_parallel_groups"])
@pytest.mark.parametrize("source", ["dependencies", "edge_source", "edge_target"])
def test_missing_node_is_rejected(method, source):
    graph = TransportGraph()
    graph.add_node(node("dependent"))
    if source == "dependencies":
        # Dependencies can be edited independently of the captured edges.
        graph.nodes["dependent"].dependencies.append("missing")
    elif source == "edge_source":
        graph.edges.append(("missing", "dependent"))
    else:
        graph.edges.append(("dependent", "missing"))

    with pytest.raises(ValueError) as error:
        getattr(graph, method)()
    assert "dependent" in str(error.value)
    assert "missing" in str(error.value)
    assert "not in the graph" in str(error.value)


def test_captured_dangling_dependency_is_rejected_before_cycle_detection():
    graph = TransportGraph()
    graph.add_node(node("dependent", ["dependent", "missing"]))
    with pytest.raises(ValueError, match="missing.*not in the graph"):
        graph.get_execution_order()


def test_duplicate_id_leaves_graph_unchanged():
    graph = TransportGraph()
    graph.add_node(node("parent"))
    first = node("duplicate", ["parent"], "first")
    graph.add_node(first)
    before = graph.to_dict()
    with pytest.raises(ValueError) as error:
        graph.add_node(node("duplicate", ["extra"], "second"))
    for text in ("duplicate", "first", "second"):
        assert text in str(error.value)
    assert graph.nodes["duplicate"] is first
    assert graph.to_dict() == before


def test_valid_diamond_and_empty_graph():
    graph = TransportGraph()
    assert graph.get_execution_order() == []
    assert graph.get_parallel_groups() == []
    # Forward references are valid while the graph is being built.
    graph.add_node(node("join", ["left", "right"]))
    graph.add_node(node("left", ["root"]))
    graph.add_node(node("right", ["root"]))
    graph.add_node(node("root"))
    assert [set(level) for level in graph.get_execution_order()] == [
        {"root"},
        {"left", "right"},
        {"join"},
    ]


def test_cycle_message_is_preserved():
    graph = TransportGraph()
    graph.add_node(node("a", ["b"]))
    graph.add_node(node("b", ["a"]))
    with pytest.raises(ValueError, match="^Cycle detected in transport graph$"):
        graph.get_execution_order()


def test_node_ids_use_full_uuid_hex_and_are_unique():
    ids = [generate_node_id() for _ in range(100_000)]
    assert all(re.fullmatch(r"[0-9a-f]{32}", value) for value in ids)
    assert all(UUID(hex=value).version == 4 for value in ids)
    assert len(set(ids)) == len(ids)
