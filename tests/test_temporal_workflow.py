"""Tests for marqov.workflows.temporal_workflow module.

Tests for JobWorkflow class that orchestrates task execution
in Temporal workflows.
"""

from __future__ import annotations

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
            "second": {"node_id": "second", "func_ref": "f2", "args": [], "kwargs": {}},
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
