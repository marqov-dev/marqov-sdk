"""Capture semantics using module-level fixtures and real SDK serialization."""

import base64

import cloudpickle
import numpy as np
import pytest

from marqov import task, workflow
from marqov.workflows import TaskResultReference, WorkflowDispatch
from marqov.workflows.graph import TransportGraph, extract_dependencies


def _identity(value=None, **kwargs):
    return value


identity = task(_identity)


def test_recursive_dependencies_match_serialized_arguments_and_do_not_duplicate():
    @workflow
    def program():
        a = identity(1)
        b = identity(2)
        return identity({"nested": [(a, {"again": b})]}, direct=a, deep={"items": [b, a]})

    dispatch = program()
    a, b, join = list(dispatch.graph.nodes.values())
    assert join.dependencies == [a.id, b.id]
    levels = dispatch.get_parallel_groups()
    assert len(levels) == 2
    assert set(levels[0]) == {a.id, b.id}
    assert levels[1] == [join.id]
    assert join.args[0]["nested"][0][0] == {"__proxy__": True, "node_id": a.id}
    assert join.kwargs["deep"]["items"][0] == {"__proxy__": True, "node_id": b.id}
    assert cloudpickle.loads(base64.b64decode(join.func_ref))(3) == 3


def test_named_nested_mixed_and_duplicate_outputs_preserve_shape_without_rerun():
    calls = []

    @workflow
    def program():
        calls.append("capture")
        result = identity(7)
        return {"named": result, "nested": [result, ("constant", result)], "none": None}

    dispatch = program()
    before = dispatch._prepare_workflow_input()
    capture = dispatch.capture()
    node_id = next(iter(capture.graph.nodes))
    assert capture.output["named"] == TaskResultReference(node_id)
    assert capture.resolve_output({node_id: 7}) == {
        "named": 7,
        "nested": [7, ("constant", 7)],
        "none": None,
    }
    assert calls == ["capture"]
    assert dispatch.capture().output == capture.output
    assert dispatch._prepare_workflow_input() == before
    with pytest.raises(KeyError):
        capture.resolve_output({})


@pytest.mark.parametrize("value", [None, 3.5, [], (), {}, {"literal": [1, 2]}])
def test_constant_only_output_is_preserved(value):
    @workflow
    def program():
        return value

    capture = program().capture()
    assert capture.resolve_output({}) == value


def test_arrays_are_opaque_leaves_and_no_platform_credentials_are_needed(monkeypatch):
    for key in list(__import__("os").environ):
        if key.startswith(("MARQOV_", "AWS_", "SUPABASE_", "TEMPORAL_")):
            monkeypatch.delenv(key)
    array = np.array([0.25, 1.5])

    @workflow
    def program():
        return {"constant": array, "task": identity(array)}

    capture = program().capture()
    node = next(iter(capture.graph.nodes.values()))
    assert node.args[0] is array
    result = capture.resolve_output({node.id: array * 2})
    assert result["constant"] is array
    np.testing.assert_array_equal(result["task"], [0.5, 3.0])


def test_container_cycles_rejected_but_shared_containers_are_allowed():
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="Cyclic task argument"):
        extract_dependencies((cyclic,), {})

    @workflow
    def program(value):
        return value

    with pytest.raises(ValueError, match="Cyclic workflow return"):
        program(cyclic).capture()
    shared = [1, 2]
    assert program([shared, shared]).capture().resolve_output({}) == [[1, 2], [1, 2]]


def test_unresolved_boolean_control_flow_rejected():
    @workflow
    def program():
        if identity(True):
            return identity(1)
        return identity(2)

    with pytest.raises(TypeError, match="unresolved task result"):
        program()


def test_async_and_nested_workflow_returns_rejected_by_capture():
    @workflow
    async def asynchronous():
        return 1

    with pytest.raises(TypeError, match="Awaitable workflow output"):
        asynchronous().capture()

    @workflow
    def inner():
        return identity(1)

    @workflow
    def outer():
        return {"inner": inner()}

    with pytest.raises(TypeError, match="Nested workflow dispatch"):
        outer().capture()


def test_foreign_graph_output_and_missing_capture_fail_explicitly():
    @workflow
    def first():
        return identity(1)

    foreign = first()._captured_result

    @workflow
    def second():
        return foreign

    with pytest.raises(ValueError, match="different graph"):
        second().capture()
    dispatch = WorkflowDispatch(TransportGraph(), "manual", (), {})
    with pytest.raises(ValueError, match="no captured return"):
        dispatch.capture()


def test_container_subclasses_and_proxy_keys_do_not_silently_change_shape():
    class CustomList(list):
        pass

    @workflow
    def custom():
        return CustomList([1])

    with pytest.raises(TypeError, match="plain list"):
        custom().capture()

    @workflow
    def keyed():
        return {identity(1): "value"}

    with pytest.raises(TypeError, match="dictionary keys"):
        keyed().capture()
