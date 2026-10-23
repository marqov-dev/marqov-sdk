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


# ---------------------------------------------------------------------------
# Result decoding through a real Temporal worker (time-skipping environment)
# ---------------------------------------------------------------------------


@task
def produce_numpy_scalar():
    import numpy as np

    return np.int64(3)


@task
def produce_nested_numpy():
    import numpy as np

    return {
        "mean": np.float64(0.5),
        "stats": {"n": np.int64(4), "ok": np.bool_(True), "seq": [np.int32(1), 2]},
        "label": "ok",
    }


@task
def produce_dataclass():
    from dataclasses import dataclass

    @dataclass
    class Summary:
        name: str
        shots: int

    return Summary("bell", 100)


@workflow
def numpy_scalar_workflow():
    return produce_numpy_scalar()


@workflow
def nested_numpy_workflow():
    return produce_nested_numpy()


@workflow
def dataclass_workflow():
    return produce_dataclass()


@pytest.mark.asyncio
async def test_run_returns_decoded_task_outputs():
    """run() and run_with_ids() hand back Python values, never a cloudpickle envelope."""
    import dataclasses

    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import UnsandboxedWorkflowRunner, Worker

    from marqov.workflows.activity import execute_task, prepare_node_inputs

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="marqov-workflows",
            workflows=[JobWorkflow],
            activities=[execute_task, prepare_node_inputs],
            workflow_runner=UnsandboxedWorkflowRunner(),
        ):
            scalar = await numpy_scalar_workflow().run(env.client)
            nested = await nested_numpy_workflow().run(env.client)
            record = await dataclass_workflow().run(env.client)
            enriched, _, _ = await dataclass_workflow().run_with_ids(env.client)

    assert scalar == 3 and type(scalar) is int

    assert nested == {
        "mean": 0.5,
        "stats": {"n": 4, "ok": True, "seq": [1, 2]},
        "label": "ok",
    }
    assert type(nested["mean"]) is float
    assert type(nested["stats"]["n"]) is int
    assert type(nested["stats"]["ok"]) is bool
    assert type(nested["stats"]["seq"][0]) is int

    assert dataclasses.is_dataclass(record)
    assert (record.name, record.shots) == ("bell", 100)

    assert "_workflow_metadata" in enriched
    assert dataclasses.is_dataclass(enriched["result"])
    assert enriched["result"].shots == 100


@task
def produce_int_key_dict():
    return {0: 0.3, 1: 0.7}


@workflow
def int_key_dict_workflow():
    return produce_int_key_dict()


@pytest.mark.asyncio
async def test_run_returns_non_str_key_dict_with_its_keys():
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import UnsandboxedWorkflowRunner, Worker

    from marqov.workflows.activity import execute_task, prepare_node_inputs

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="marqov-workflows",
            workflows=[JobWorkflow],
            activities=[execute_task, prepare_node_inputs],
            workflow_runner=UnsandboxedWorkflowRunner(),
        ):
            result = await int_key_dict_workflow().run(env.client)

    assert result == {0: 0.3, 1: 0.7}
    assert all(type(k) is int for k in result)
