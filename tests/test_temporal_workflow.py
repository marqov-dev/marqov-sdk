"""Tests for marqov.workflows.temporal_workflow module.

Tests for JobWorkflow class that orchestrates task execution
in Temporal workflows.
"""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from marqov.workflows.temporal_workflow import JobWorkflow


def _setup_workflow_mock(mock_workflow: MagicMock) -> None:
    """Stub workflow.now() so task_timeline timestamps are JSON-serializable."""
    mock_workflow.now.return_value.isoformat.return_value = "2026-01-01T00:00:00"


class TestJobWorkflow:
    """Tests for JobWorkflow class."""

    @pytest.fixture
    def mock_workflow(self) -> MagicMock:
        """Create mock workflow module."""
        mock = MagicMock()
        mock.execute_activity = AsyncMock()
        return mock

    def _create_workflow_input(
        self,
        nodes: dict[str, dict[str, Any]],
        execution_levels: list[list[str]],
        output_nodes: list[str],
    ) -> dict[str, Any]:
        """Helper to create workflow input structure."""
        return {
            "nodes": nodes,
            "execution_levels": execution_levels,
            "output_nodes": output_nodes,
        }

    @pytest.mark.asyncio
    async def test_single_node_execution(self) -> None:
        """Execute workflow with single node."""
        nodes = {
            "node1": {
                "node_id": "node1",
                "func_ref": "base64func",
                "args": [1, 2],
                "kwargs": {},
            }
        }
        workflow_input = self._create_workflow_input(
            nodes=nodes,
            execution_levels=[["node1"]],
            output_nodes=["node1"],
        )

        prepare_response = json.dumps({
            "node_id": "node1",
            "func_ref": "base64func",
            "args": [1, 2],
            "kwargs": {},
        })
        execute_response = json.dumps({
            "node_id": "node1",
            "result": 3,
        })

        with patch("marqov.workflows.temporal_workflow.workflow") as mock_workflow:
            _setup_workflow_mock(mock_workflow)
            mock_workflow.execute_activity = AsyncMock(
                side_effect=[prepare_response, execute_response]
            )

            workflow = JobWorkflow()
            result = await workflow.run(workflow_input)

        assert json.loads(result)["result"] == 3

    @pytest.mark.asyncio
    async def test_multiple_output_nodes(self) -> None:
        """Execute workflow with multiple output nodes."""
        nodes = {
            "a": {"node_id": "a", "func_ref": "f1", "args": [], "kwargs": {}},
            "b": {"node_id": "b", "func_ref": "f2", "args": [], "kwargs": {}},
        }
        workflow_input = self._create_workflow_input(
            nodes=nodes,
            execution_levels=[["a", "b"]],
            output_nodes=["a", "b"],
        )

        prepare_responses = [
            json.dumps({"node_id": "a", "func_ref": "f1", "args": [], "kwargs": {}}),
            json.dumps({"node_id": "b", "func_ref": "f2", "args": [], "kwargs": {}}),
        ]
        execute_responses = [
            json.dumps({"node_id": "a", "result": 10}),
            json.dumps({"node_id": "b", "result": 20}),
        ]

        with patch("marqov.workflows.temporal_workflow.workflow") as mock_workflow:
            _setup_workflow_mock(mock_workflow)
            mock_workflow.execute_activity = AsyncMock(
                side_effect=[
                    prepare_responses[0],
                    prepare_responses[1],
                    execute_responses[0],
                    execute_responses[1],
                ]
            )

            workflow = JobWorkflow()
            result = await workflow.run(workflow_input)

        assert json.loads(result)["result"] == {"a": 10, "b": 20}

    @pytest.mark.asyncio
    async def test_no_output_nodes_returns_all(self) -> None:
        """When no output nodes specified, return all results."""
        nodes = {
            "x": {"node_id": "x", "func_ref": "fx", "args": [], "kwargs": {}},
        }
        workflow_input = self._create_workflow_input(
            nodes=nodes,
            execution_levels=[["x"]],
            output_nodes=[],
        )

        prepare_response = json.dumps({
            "node_id": "x", "func_ref": "fx", "args": [], "kwargs": {}
        })
        execute_response = json.dumps({"node_id": "x", "result": "value"})

        with patch("marqov.workflows.temporal_workflow.workflow") as mock_workflow:
            _setup_workflow_mock(mock_workflow)
            mock_workflow.execute_activity = AsyncMock(
                side_effect=[prepare_response, execute_response]
            )

            workflow = JobWorkflow()
            result = await workflow.run(workflow_input)

        assert json.loads(result)["result"] == {"x": "value"}

    @pytest.mark.asyncio
    async def test_execution_levels_sequential(self) -> None:
        """Execution levels are processed sequentially."""
        nodes = {
            "level0_task": {"node_id": "level0_task", "func_ref": "f0", "args": [], "kwargs": {}},
            "level1_task": {"node_id": "level1_task", "func_ref": "f1", "args": [], "kwargs": {}},
        }
        workflow_input = self._create_workflow_input(
            nodes=nodes,
            execution_levels=[["level0_task"], ["level1_task"]],
            output_nodes=["level1_task"],
        )

        call_order = []

        async def mock_activity(name: str, **kwargs) -> str:
            call_order.append((name, kwargs.get("args", [])[0] if kwargs.get("args") else None))
            if name == "prepare_node_inputs":
                node_data = json.loads(kwargs["args"][0])
                return json.dumps({
                    "node_id": node_data["node_id"],
                    "func_ref": node_data["func_ref"],
                    "args": node_data["args"],
                    "kwargs": node_data["kwargs"],
                })
            else:
                node_id = kwargs["args"][0]
                return json.dumps({"node_id": node_id, "result": f"{node_id}_result"})

        with patch("marqov.workflows.temporal_workflow.workflow") as mock_workflow:
            _setup_workflow_mock(mock_workflow)
            mock_workflow.execute_activity = AsyncMock(side_effect=mock_activity)

            workflow = JobWorkflow()
            await workflow.run(workflow_input)

        activity_sequence = [c[0] for c in call_order]
        assert activity_sequence == [
            "prepare_node_inputs",
            "execute_task",
            "prepare_node_inputs",
            "execute_task",
        ]

    @pytest.mark.asyncio
    async def test_results_passed_between_levels(self) -> None:
        """Results from earlier levels are available in later levels."""
        nodes = {
            "first": {"node_id": "first", "func_ref": "f1", "args": [], "kwargs": {}},
            "second": {"node_id": "second", "func_ref": "f2", "args": [], "kwargs": {},
                       "dependencies": ["first"]},
        }
        workflow_input = self._create_workflow_input(
            nodes=nodes,
            execution_levels=[["first"], ["second"]],
            output_nodes=["second"],
        )

        captured_completed = []

        async def mock_activity(name: str, **kwargs) -> str:
            if name == "prepare_node_inputs":
                completed_json = kwargs["args"][1]
                captured_completed.append(json.loads(completed_json))
                node_data = json.loads(kwargs["args"][0])
                return json.dumps({
                    "node_id": node_data["node_id"],
                    "func_ref": node_data["func_ref"],
                    "args": [],
                    "kwargs": {},
                })
            else:
                node_id = kwargs["args"][0]
                return json.dumps({"node_id": node_id, "result": f"{node_id}_result"})

        with patch("marqov.workflows.temporal_workflow.workflow") as mock_workflow:
            _setup_workflow_mock(mock_workflow)
            mock_workflow.execute_activity = AsyncMock(side_effect=mock_activity)

            workflow = JobWorkflow()
            await workflow.run(workflow_input)

        assert captured_completed[0] == {}
        assert captured_completed[1] == {"first": "first_result"}

    @pytest.mark.asyncio
    async def test_retry_policy_from_node_data(self) -> None:
        """Retry policy uses retries from node data."""
        nodes = {
            "retry_task": {
                "node_id": "retry_task",
                "func_ref": "f",
                "args": [],
                "kwargs": {},
                "retries": 3,
            }
        }
        workflow_input = self._create_workflow_input(
            nodes=nodes,
            execution_levels=[["retry_task"]],
            output_nodes=["retry_task"],
        )

        captured_retry_policy = None

        async def mock_activity(name: str, **kwargs) -> str:
            nonlocal captured_retry_policy
            if name == "prepare_node_inputs":
                return json.dumps({
                    "node_id": "retry_task",
                    "func_ref": "f",
                    "args": [],
                    "kwargs": {},
                })
            else:
                captured_retry_policy = kwargs.get("retry_policy")
                return json.dumps({"node_id": "retry_task", "result": "done"})

        with patch("marqov.workflows.temporal_workflow.workflow") as mock_workflow:
            _setup_workflow_mock(mock_workflow)
            mock_workflow.execute_activity = AsyncMock(side_effect=mock_activity)

            workflow = JobWorkflow()
            await workflow.run(workflow_input)

        assert captured_retry_policy is not None
        assert captured_retry_policy.maximum_attempts == 4  # retries + 1

    @pytest.mark.asyncio
    async def test_timeout_from_node_data(self) -> None:
        """Timeout uses timeout_seconds from node data."""
        nodes = {
            "slow_task": {
                "node_id": "slow_task",
                "func_ref": "f",
                "args": [],
                "kwargs": {},
                "timeout_seconds": 600,
            }
        }
        workflow_input = self._create_workflow_input(
            nodes=nodes,
            execution_levels=[["slow_task"]],
            output_nodes=["slow_task"],
        )

        captured_timeout = None

        async def mock_activity(name: str, **kwargs) -> str:
            nonlocal captured_timeout
            if name == "prepare_node_inputs":
                return json.dumps({
                    "node_id": "slow_task",
                    "func_ref": "f",
                    "args": [],
                    "kwargs": {},
                })
            else:
                captured_timeout = kwargs.get("start_to_close_timeout")
                return json.dumps({"node_id": "slow_task", "result": "done"})

        with patch("marqov.workflows.temporal_workflow.workflow") as mock_workflow:
            _setup_workflow_mock(mock_workflow)
            mock_workflow.execute_activity = AsyncMock(side_effect=mock_activity)

            workflow = JobWorkflow()
            await workflow.run(workflow_input)

        assert captured_timeout == timedelta(seconds=600)

    @pytest.mark.asyncio
    async def test_default_timeout(self) -> None:
        """Default timeout is 300 seconds when not specified."""
        nodes = {
            "task": {
                "node_id": "task",
                "func_ref": "f",
                "args": [],
                "kwargs": {},
            }
        }
        workflow_input = self._create_workflow_input(
            nodes=nodes,
            execution_levels=[["task"]],
            output_nodes=["task"],
        )

        captured_timeout = None

        async def mock_activity(name: str, **kwargs) -> str:
            nonlocal captured_timeout
            if name == "prepare_node_inputs":
                return json.dumps({
                    "node_id": "task",
                    "func_ref": "f",
                    "args": [],
                    "kwargs": {},
                })
            else:
                captured_timeout = kwargs.get("start_to_close_timeout")
                return json.dumps({"node_id": "task", "result": "done"})

        with patch("marqov.workflows.temporal_workflow.workflow") as mock_workflow:
            _setup_workflow_mock(mock_workflow)
            mock_workflow.execute_activity = AsyncMock(side_effect=mock_activity)

            workflow = JobWorkflow()
            await workflow.run(workflow_input)

        assert captured_timeout == timedelta(seconds=300)

    @pytest.mark.asyncio
    async def test_prepare_timeout_is_30_seconds(self) -> None:
        """prepare_node_inputs activity has 30 second timeout."""
        nodes = {
            "task": {"node_id": "task", "func_ref": "f", "args": [], "kwargs": {}}
        }
        workflow_input = self._create_workflow_input(
            nodes=nodes,
            execution_levels=[["task"]],
            output_nodes=["task"],
        )

        captured_prepare_timeout = None

        async def mock_activity(name: str, **kwargs) -> str:
            nonlocal captured_prepare_timeout
            if name == "prepare_node_inputs":
                captured_prepare_timeout = kwargs.get("start_to_close_timeout")
                return json.dumps({
                    "node_id": "task",
                    "func_ref": "f",
                    "args": [],
                    "kwargs": {},
                })
            else:
                return json.dumps({"node_id": "task", "result": "done"})

        with patch("marqov.workflows.temporal_workflow.workflow") as mock_workflow:
            _setup_workflow_mock(mock_workflow)
            mock_workflow.execute_activity = AsyncMock(side_effect=mock_activity)

            workflow = JobWorkflow()
            await workflow.run(workflow_input)

        assert captured_prepare_timeout == timedelta(seconds=30)


class TestJobWorkflowParallelExecution:
    """Tests for parallel execution within levels."""

    def _create_workflow_input(
        self,
        nodes: dict[str, dict[str, Any]],
        execution_levels: list[list[str]],
        output_nodes: list[str],
    ) -> dict[str, Any]:
        return {
            "nodes": nodes,
            "execution_levels": execution_levels,
            "output_nodes": output_nodes,
        }

    @pytest.mark.asyncio
    async def test_parallel_nodes_in_same_level(self) -> None:
        """Nodes in the same level are executed in parallel."""
        nodes = {
            "a": {"node_id": "a", "func_ref": "fa", "args": [], "kwargs": {}},
            "b": {"node_id": "b", "func_ref": "fb", "args": [], "kwargs": {}},
            "c": {"node_id": "c", "func_ref": "fc", "args": [], "kwargs": {}},
        }
        workflow_input = self._create_workflow_input(
            nodes=nodes,
            execution_levels=[["a", "b", "c"]],
            output_nodes=["a", "b", "c"],
        )

        execution_tasks = []

        async def mock_activity(name: str, **kwargs) -> str:
            if name == "prepare_node_inputs":
                node_data = json.loads(kwargs["args"][0])
                return json.dumps({
                    "node_id": node_data["node_id"],
                    "func_ref": node_data["func_ref"],
                    "args": [],
                    "kwargs": {},
                })
            else:
                node_id = kwargs["args"][0]
                execution_tasks.append(node_id)
                return json.dumps({"node_id": node_id, "result": node_id})

        with patch("marqov.workflows.temporal_workflow.workflow") as mock_workflow:
            _setup_workflow_mock(mock_workflow)
            mock_workflow.execute_activity = AsyncMock(side_effect=mock_activity)

            workflow = JobWorkflow()
            result = await workflow.run(workflow_input)

        assert set(execution_tasks) == {"a", "b", "c"}
        assert json.loads(result)["result"] == {"a": "a", "b": "b", "c": "c"}


class TestJobWorkflowActivityReferences:
    """Tests verifying activities are called by string name."""

    def _create_workflow_input(
        self,
        nodes: dict[str, dict[str, Any]],
        execution_levels: list[list[str]],
        output_nodes: list[str],
    ) -> dict[str, Any]:
        return {
            "nodes": nodes,
            "execution_levels": execution_levels,
            "output_nodes": output_nodes,
        }

    @pytest.mark.asyncio
    async def test_activities_called_by_string_name(self) -> None:
        """Activities are referenced by string, not imported functions."""
        nodes = {
            "task": {"node_id": "task", "func_ref": "f", "args": [], "kwargs": {}}
        }
        workflow_input = self._create_workflow_input(
            nodes=nodes,
            execution_levels=[["task"]],
            output_nodes=["task"],
        )

        activity_names_called = []

        async def mock_activity(name: str, **kwargs) -> str:
            activity_names_called.append(name)
            if name == "prepare_node_inputs":
                return json.dumps({
                    "node_id": "task",
                    "func_ref": "f",
                    "args": [],
                    "kwargs": {},
                })
            else:
                return json.dumps({"node_id": "task", "result": "done"})

        with patch("marqov.workflows.temporal_workflow.workflow") as mock_workflow:
            _setup_workflow_mock(mock_workflow)
            mock_workflow.execute_activity = AsyncMock(side_effect=mock_activity)

            workflow = JobWorkflow()
            await workflow.run(workflow_input)

        assert "prepare_node_inputs" in activity_names_called
        assert "execute_task" in activity_names_called


class TestJobWorkflowMalformedPayloads:
    """A malformed activity payload must fail the workflow, not its workflow task.

    A bare json.JSONDecodeError is not a temporalio FailureError, so Temporal
    would retry the workflow task forever and the job would never reach a
    terminal state (marqov-sdk#141).
    """

    def _create_workflow_input(
        self,
        nodes: dict[str, dict[str, Any]],
        execution_levels: list[list[str]],
        output_nodes: list[str],
    ) -> dict[str, Any]:
        return {
            "nodes": nodes,
            "execution_levels": execution_levels,
            "output_nodes": output_nodes,
        }

    @pytest.mark.asyncio
    async def test_non_json_activity_result_raises_failure_error(self) -> None:
        """A non-JSON execute_task return becomes a FailureError."""
        from temporalio.exceptions import ApplicationError, FailureError

        nodes = {
            "task": {"node_id": "task", "func_ref": "f", "args": [], "kwargs": {}}
        }
        workflow_input = self._create_workflow_input(
            nodes=nodes,
            execution_levels=[["task"]],
            output_nodes=["task"],
        )

        prepare_response = json.dumps({
            "node_id": "task",
            "func_ref": "f",
            "args": [],
            "kwargs": {},
        })

        with patch("marqov.workflows.temporal_workflow.workflow") as mock_workflow:
            _setup_workflow_mock(mock_workflow)
            mock_workflow.execute_activity = AsyncMock(
                side_effect=[prepare_response, "not json at all"]
            )

            workflow = JobWorkflow()
            with pytest.raises(FailureError) as excinfo:
                await workflow.run(workflow_input)

        assert not isinstance(excinfo.value, json.JSONDecodeError)
        assert isinstance(excinfo.value, ApplicationError)
        assert excinfo.value.non_retryable is True
        assert "task" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_activity_result_missing_keys_raises_failure_error(self) -> None:
        """An envelope without the expected keys becomes a FailureError."""
        from temporalio.exceptions import FailureError

        nodes = {
            "task": {"node_id": "task", "func_ref": "f", "args": [], "kwargs": {}}
        }
        workflow_input = self._create_workflow_input(
            nodes=nodes,
            execution_levels=[["task"]],
            output_nodes=["task"],
        )

        prepare_response = json.dumps({
            "node_id": "task",
            "func_ref": "f",
            "args": [],
            "kwargs": {},
        })

        with patch("marqov.workflows.temporal_workflow.workflow") as mock_workflow:
            _setup_workflow_mock(mock_workflow)
            mock_workflow.execute_activity = AsyncMock(
                side_effect=[prepare_response, json.dumps({"oops": 1})]
            )

            workflow = JobWorkflow()
            with pytest.raises(FailureError):
                await workflow.run(workflow_input)

    @pytest.mark.asyncio
    async def test_non_json_prepare_result_raises_failure_error(self) -> None:
        """A malformed prepare_node_inputs payload becomes a FailureError."""
        from temporalio.exceptions import FailureError

        nodes = {
            "task": {"node_id": "task", "func_ref": "f", "args": [], "kwargs": {}}
        }
        workflow_input = self._create_workflow_input(
            nodes=nodes,
            execution_levels=[["task"]],
            output_nodes=["task"],
        )

        with patch("marqov.workflows.temporal_workflow.workflow") as mock_workflow:
            _setup_workflow_mock(mock_workflow)
            mock_workflow.execute_activity = AsyncMock(
                side_effect=["definitely not json"]
            )

            workflow = JobWorkflow()
            with pytest.raises(FailureError) as excinfo:
                await workflow.run(workflow_input)

        assert not isinstance(excinfo.value, json.JSONDecodeError)


def _proxy(node_id: str) -> dict[str, Any]:
    """The placeholder ``decorators._serialize_arg`` writes for a TaskProxy."""
    return {"__proxy__": True, "node_id": node_id}


def _diamond_input() -> dict[str, Any]:
    """Two roots, two middle nodes with different fan-in, one sink.

    ``d`` depends only on ``a``, so ``b``'s result is a completed sibling
    result that ``d`` must not receive.
    """
    nodes = {
        "a": {"node_id": "a", "func_name": "load", "func_ref": "fa",
              "args": [1], "kwargs": {}, "dependencies": []},
        "b": {"node_id": "b", "func_name": "load", "func_ref": "fb",
              "args": [2], "kwargs": {}, "dependencies": []},
        "c": {"node_id": "c", "func_name": "combine", "func_ref": "fc",
              "args": [_proxy("a"), _proxy("b")], "kwargs": {},
              "dependencies": ["a", "b"]},
        "d": {"node_id": "d", "func_name": "scale", "func_ref": "fd",
              "args": [_proxy("a")], "kwargs": {"k": 3}, "dependencies": ["a"]},
        "e": {"node_id": "e", "func_name": "report", "func_ref": "fe",
              "args": [], "kwargs": {"x": _proxy("c"), "y": [_proxy("d")]},
              "dependencies": ["c", "d"]},
    }
    return {
        "nodes": nodes,
        "execution_levels": [["a", "b"], ["c", "d"], ["e"]],
        "output_nodes": ["e"],
    }


def _echo_result(node_id: str, args: Any, kwargs: Any) -> dict[str, Any]:
    return {"node": node_id, "args": args, "kwargs": kwargs}


class _RecordingActivities:
    """Fake activities: the real ``prepare_node_inputs`` and an echoing ``execute_task``.

    Records each ``prepare_node_inputs`` call's node id and the completed
    results it was sent (its second argument), raw and decoded.
    """

    def __init__(self, results: dict[str, Any] | None = None) -> None:
        self.prepare_calls: list[tuple[str, dict[str, Any]]] = []
        self.prepare_raw: list[tuple[str, str]] = []
        self._results = results

    async def activity(self, name: str, **kwargs: Any) -> str:
        from marqov.workflows.activity import prepare_node_inputs

        if name == "prepare_node_inputs":
            node_json, completed_json = kwargs["args"]
            node_id = json.loads(node_json)["node_id"]
            self.prepare_calls.append((node_id, json.loads(completed_json)))
            self.prepare_raw.append((node_id, completed_json))
            return await prepare_node_inputs(node_json, completed_json)
        node_id, _func_ref, args_json, kwargs_json = kwargs["args"]
        if self._results is not None:
            result = self._results[node_id]
        else:
            result = _echo_result(node_id, json.loads(args_json), json.loads(kwargs_json))
        return json.dumps({"node_id": node_id, "result": result})


class TestJobWorkflowDependencySubset:
    """prepare_node_inputs receives only the node's own dependencies (marqov-sdk#147)."""

    @pytest.mark.asyncio
    async def test_prepare_receives_only_declared_dependencies(self) -> None:
        fake = _RecordingActivities()
        with patch("marqov.workflows.temporal_workflow.workflow") as mock_workflow:
            _setup_workflow_mock(mock_workflow)
            mock_workflow.execute_activity = AsyncMock(side_effect=fake.activity)
            await JobWorkflow().run(_diamond_input())

        sent = {node_id: list(completed) for node_id, completed in fake.prepare_calls}
        assert sent == {
            "a": [],
            "b": [],
            "c": ["a", "b"],
            "d": ["a"],
            "e": ["c", "d"],
        }

    @pytest.mark.asyncio
    async def test_resolve_payload_does_not_grow_with_unrelated_results(self) -> None:
        """A chain of four ~1.5 MB results sends each node one result, not all earlier ones.

        Before the fix the fourth node's argument carried three results (about
        4.5 MB); now every argument carries one.
        """
        from marqov.workflows.activity import (
            TEMPORAL_PAYLOAD_LIMIT_BYTES,
            _serialize_value,
        )

        # 1,125,000 raw bytes base64-encode to 1,500,000 characters, so each
        # result is a real __cloudpickle__ envelope of about 1.5 MB.
        results = {
            nid: _serialize_value(bytes([i]) * 1_125_000)
            for i, nid in enumerate(["n0", "n1", "n2", "n3"])
        }
        for nid, envelope in results.items():
            # Each result on its own passes execute_task's payload check.
            returned = json.dumps({"node_id": nid, "result": envelope})
            assert len(json.dumps(returned).encode("utf-8")) <= TEMPORAL_PAYLOAD_LIMIT_BYTES

        nodes: dict[str, dict[str, Any]] = {
            "n0": {"node_id": "n0", "func_ref": "f", "args": [], "kwargs": {},
                   "dependencies": []},
        }
        for prev, nid in [("n0", "n1"), ("n1", "n2"), ("n2", "n3")]:
            nodes[nid] = {"node_id": nid, "func_ref": "f", "args": [_proxy(prev)],
                          "kwargs": {}, "dependencies": [prev]}
        workflow_input = {
            "nodes": nodes,
            "execution_levels": [["n0"], ["n1"], ["n2"], ["n3"]],
            "output_nodes": ["n3"],
        }

        fake = _RecordingActivities(results=results)
        with patch("marqov.workflows.temporal_workflow.workflow") as mock_workflow:
            _setup_workflow_mock(mock_workflow)
            mock_workflow.execute_activity = AsyncMock(side_effect=fake.activity)
            await JobWorkflow().run(workflow_input)

        assert [nid for nid, _ in fake.prepare_raw] == ["n0", "n1", "n2", "n3"]
        for _nid, completed_json in fake.prepare_raw:
            # Temporal encodes the string argument as JSON again.
            payload_bytes = len(json.dumps(completed_json).encode("utf-8"))
            assert payload_bytes < 2 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_missing_dependency_result_is_non_retryable_and_names_both(self) -> None:
        from temporalio.exceptions import ApplicationError

        workflow_input = {
            "nodes": {
                "a": {"node_id": "a", "func_ref": "f", "args": [], "kwargs": {},
                      "dependencies": []},
                "b": {"node_id": "b", "func_ref": "f", "args": [_proxy("ghost")],
                      "kwargs": {}, "dependencies": ["ghost"]},
            },
            "execution_levels": [["a"], ["b"]],
            "output_nodes": ["b"],
        }
        fake = _RecordingActivities()
        with patch("marqov.workflows.temporal_workflow.workflow") as mock_workflow:
            _setup_workflow_mock(mock_workflow)
            mock_workflow.execute_activity = AsyncMock(side_effect=fake.activity)
            with pytest.raises(ApplicationError) as excinfo:
                await JobWorkflow().run(workflow_input)

        assert excinfo.value.non_retryable is True
        assert "'b'" in str(excinfo.value)
        assert "'ghost'" in str(excinfo.value)
        # The failing level's resolve was never scheduled.
        assert [nid for nid, _ in fake.prepare_calls] == ["a"]


class TestJobWorkflowConcurrentResolve:
    """The resolve calls for one level run concurrently (marqov-sdk#147)."""

    @staticmethod
    def _flat_level_input() -> dict[str, Any]:
        return {
            "nodes": {
                nid: {"node_id": nid, "func_ref": "f", "args": [], "kwargs": {},
                      "dependencies": []}
                for nid in ["a", "b", "c"]
            },
            "execution_levels": [["a", "b", "c"]],
            "output_nodes": ["a", "b", "c"],
        }

    @pytest.mark.asyncio
    async def test_resolve_calls_in_a_level_overlap(self) -> None:
        events: list[tuple[str, str]] = []

        async def mock_activity(name: str, **kwargs: Any) -> str:
            if name == "prepare_node_inputs":
                node_data = json.loads(kwargs["args"][0])
                events.append(("enter", node_data["node_id"]))
                # Yield so any other already-scheduled resolve can start.
                await asyncio.sleep(0)
                events.append(("exit", node_data["node_id"]))
                return json.dumps({
                    "node_id": node_data["node_id"],
                    "func_ref": node_data["func_ref"],
                    "args": [],
                    "kwargs": {},
                })
            node_id = kwargs["args"][0]
            return json.dumps({"node_id": node_id, "result": node_id})

        with patch("marqov.workflows.temporal_workflow.workflow") as mock_workflow:
            _setup_workflow_mock(mock_workflow)
            mock_workflow.execute_activity = AsyncMock(side_effect=mock_activity)
            result = await JobWorkflow().run(self._flat_level_input())

        first_exit = next(i for i, (kind, _) in enumerate(events) if kind == "exit")
        assert [nid for _kind, nid in events[:first_exit]] == ["a", "b", "c"]
        assert json.loads(result)["result"] == {"a": "a", "b": "b", "c": "c"}

    @pytest.mark.asyncio
    async def test_resolve_failure_raises_first_failing_node_in_level_order(self) -> None:
        """When several resolves in a level fail, the first in level order is raised."""

        async def mock_activity(name: str, **kwargs: Any) -> str:
            node_id = json.loads(kwargs["args"][0])["node_id"]
            if node_id == "a":
                await asyncio.sleep(0)  # finish after "c" has failed
                raise RuntimeError("resolve failed for a")
            if node_id == "c":
                raise RuntimeError("resolve failed for c")
            return json.dumps({"node_id": node_id, "func_ref": "f", "args": [],
                               "kwargs": {}})

        with patch("marqov.workflows.temporal_workflow.workflow") as mock_workflow:
            _setup_workflow_mock(mock_workflow)
            mock_workflow.execute_activity = AsyncMock(side_effect=mock_activity)
            with pytest.raises(RuntimeError, match="resolve failed for a"):
                await JobWorkflow().run(self._flat_level_input())


class TestJobWorkflowEnvelopeUnchanged:
    """The returned envelope for an existing graph is unchanged by marqov-sdk#147."""

    @pytest.mark.asyncio
    async def test_diamond_graph_envelope(self) -> None:
        fake = _RecordingActivities()
        with patch("marqov.workflows.temporal_workflow.workflow") as mock_workflow:
            _setup_workflow_mock(mock_workflow)
            mock_workflow.execute_activity = AsyncMock(side_effect=fake.activity)
            envelope = json.loads(await JobWorkflow().run(_diamond_input()))

        a = _echo_result("a", [1], {})
        b = _echo_result("b", [2], {})
        c = _echo_result("c", [a, b], {})
        d = _echo_result("d", [a], {"k": 3})
        e = _echo_result("e", [], {"x": c, "y": [d]})
        stamp = "2026-01-01T00:00:00"

        def meta(node_id: str, func_name: str, level: int) -> dict[str, Any]:
            return {"node_id": node_id, "func_name": func_name, "level": level,
                    "started_at": stamp, "completed_at": stamp, "status": "completed"}

        assert envelope == {
            "result": e,
            "_workflow_metadata": {
                "execution_graph": {
                    "nodes": {
                        "a": {"func_name": "load", "dependencies": []},
                        "b": {"func_name": "load", "dependencies": []},
                        "c": {"func_name": "combine", "dependencies": ["a", "b"]},
                        "d": {"func_name": "scale", "dependencies": ["a"]},
                        "e": {"func_name": "report", "dependencies": ["c", "d"]},
                    },
                    "execution_levels": [["a", "b"], ["c", "d"], ["e"]],
                    "output_nodes": ["e"],
                },
                "task_timeline": [
                    meta("a", "load", 0),
                    meta("b", "load", 0),
                    meta("c", "combine", 1),
                    meta("d", "scale", 1),
                    meta("e", "report", 2),
                ],
                "total_tasks": 5,
                "total_levels": 3,
            },
        }
        # Dict equality ignores key order; the serialised order is part of the
        # returned string too.
        assert list(envelope) == ["result", "_workflow_metadata"]
        assert list(envelope["_workflow_metadata"]["task_timeline"][0]) == [
            "node_id", "func_name", "level", "started_at", "completed_at", "status",
        ]
