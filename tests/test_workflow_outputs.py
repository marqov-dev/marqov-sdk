"""Workflow outputs must refer to nodes owned by the current graph."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.exceptions import ApplicationError, FailureError

from marqov import task, workflow
from marqov.workflows.decorators import WorkflowDispatch
from marqov.workflows.graph import TransportGraph
from marqov.workflows.temporal_workflow import JobWorkflow


@task
def identity(value):
    return value


@pytest.mark.parametrize("shape", ["single", "list", "tuple", "dict"])
@pytest.mark.parametrize("same_id", [False, True])
def test_foreign_outputs_rejected_at_graph_build(shape, same_id):
    @workflow
    def original():
        return identity(1)

    foreign = original()._captured_result

    @workflow
    def reuse():
        local = identity(2)
        if same_id:
            foreign._node.id = local.node_id
        if shape == "single":
            return foreign
        if shape == "list":
            return [local, foreign]
        if shape == "tuple":
            return (local, foreign)
        return {"local": local, "foreign": foreign}

    with pytest.raises(ValueError, match="different graph"):
        reuse()


@pytest.mark.parametrize("has_nodes", [False, True])
def test_missing_output_id_rejected_before_planning(has_nodes):
    @workflow
    def program():
        if has_nodes:
            return identity(1)

    dispatch = program()
    dispatch.graph.set_output_nodes(["missing"])
    with pytest.raises(ValueError, match="Output node 'missing'.*not in the graph"):
        dispatch.graph.get_execution_order()
    with pytest.raises(ValueError, match="Output node 'missing'.*not in the graph"):
        dispatch._prepare_workflow_input()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["run", "start_with_ids", "run_with_ids", "dispatch"])
async def test_missing_output_never_submitted(method):
    graph = TransportGraph()
    graph.set_output_node("missing")
    dispatch = WorkflowDispatch(graph, "invalid", (), {})
    client = MagicMock()
    client.start_workflow = AsyncMock()
    with pytest.raises(ValueError, match="Output node 'missing'"):
        await getattr(dispatch, method)(client)
    client.start_workflow.assert_not_called()
    client.get_workflow_handle.assert_not_called()


@pytest.mark.parametrize("shape", ["single", "list", "tuple", "dict", "none", "empty"])
def test_valid_output_selection_preserved(shape):
    @workflow
    def program():
        a, b = identity(1), identity(2)
        if shape == "single":
            return a
        if shape == "list":
            return [a, b, "literal"]
        if shape == "tuple":
            return (a, b)
        if shape == "dict":
            return {"a": a, "b": b, "literal": 3}
        if shape == "empty":
            return []

    dispatch = program()
    node_ids = list(dispatch.graph.nodes)
    expected = node_ids[:1] if shape == "single" else node_ids
    if shape in ("none", "empty"):
        expected = []
    assert dispatch._prepare_workflow_input()["output_nodes"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("outputs", [["missing"], ["a", "missing"]])
async def test_missing_completed_output_is_terminal_application_error(outputs):
    payload = {
        "nodes": {"a": {"func_name": "identity"}},
        "execution_levels": [],
        "output_nodes": outputs,
    }
    with pytest.raises(FailureError) as excinfo:
        await JobWorkflow().run(payload)
    assert isinstance(excinfo.value, ApplicationError)
    assert excinfo.value.non_retryable is True
    assert "completed result" in str(excinfo.value)


@pytest.mark.asyncio
async def test_empty_graph_without_outputs_preserved():
    import json

    result = await JobWorkflow().run({"nodes": {}, "execution_levels": [], "output_nodes": []})
    assert json.loads(result)["result"] == {}


def test_removed_output_rejected_and_graph_context_restored():
    from marqov.workflows.graph import get_active_graph

    previous = get_active_graph()

    @workflow
    def invalid():
        output = identity(1)
        del output._graph.nodes[output.node_id]
        return output

    with pytest.raises(ValueError, match="different graph"):
        invalid()
    assert get_active_graph() is previous
