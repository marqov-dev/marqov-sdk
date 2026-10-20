"""``@task(executor=...)`` is reserved and ``provider_env`` is not wired (marqov-sdk#145).

The ``task`` and ``TaskConfig`` docstrings, the ``execute_task`` docstring,
the ``marqov.workflows`` package docstring and ``docs/workflow-capture.md``
say that ``executor`` and the display ``name`` are recorded on the graph but
never reach Temporal, and that ``JobWorkflow`` never passes ``provider_env``.
These tests pin that behaviour, so wiring either one fails here and forces the
documentation to change in the same PR. The child's error channel described in
``_task_child``'s docstring is pinned by
``tests/test_activity_subprocess_isolation.py::test_error_json_exists_result_json_absent_on_failure``.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Sequence

import pytest
from temporalio import activity
from temporalio.common import RawValue

from marqov import task, workflow
from marqov.workflows.activity import execute_task, prepare_node_inputs
from marqov.workflows.temporal_workflow import JobWorkflow


@task(name="pretty-name", executor="braket", timeout=600)
def provider_task(value):
    return value


@workflow
def provider_workflow():
    return provider_task(1)


def test_executor_and_name_are_recorded_but_not_sent():
    dispatch = provider_workflow()
    node = next(iter(dispatch.graph.nodes.values()))
    assert node.config.executor == "braket"
    assert node.config.name == "pretty-name"

    sent = next(iter(dispatch._prepare_workflow_input()["nodes"].values()))
    assert sent["timeout_seconds"] == 600
    assert not {"executor", "config", "name"} & set(sent)
    assert "braket" not in json.dumps(sent)
    assert "pretty-name" not in json.dumps(sent)


def test_execute_task_has_no_executor_parameter():
    params = inspect.signature(execute_task).parameters
    assert "executor" not in params
    assert list(params) == ["node_id", "func_ref", "args_json", "kwargs_json", "provider_env"]
    assert params["provider_env"].default is None


@pytest.mark.asyncio
async def test_job_workflow_never_supplies_provider_env():
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import UnsandboxedWorkflowRunner, Worker

    calls: list[list[object]] = []

    @activity.defn(dynamic=True)
    async def recording_activity(args: Sequence[RawValue]) -> str:
        assert activity.info().activity_type == "execute_task"
        converter = activity.payload_converter()
        values = [converter.from_payload(raw.payload) for raw in args]
        calls.append(values)
        return json.dumps({"node_id": values[0], "result": 1})

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="marqov-workflows",
            workflows=[JobWorkflow],
            activities=[prepare_node_inputs, recording_activity],
            workflow_runner=UnsandboxedWorkflowRunner(),
        ):
            assert await provider_workflow().run(env.client) == 1

    assert len(calls) == 1
    node_id, func_ref, args_json, kwargs_json = calls[0]
    assert isinstance(func_ref, str)
    assert json.loads(args_json) == [1]
    assert json.loads(kwargs_json) == {}
