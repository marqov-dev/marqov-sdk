"""Tests for WorkflowDispatch.start_with_ids() — non-blocking workflow start."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestStartWithIds:
    """Tests for start_with_ids() returning handle before result."""

    @pytest.mark.asyncio
    async def test_start_with_ids_returns_handle_and_ids(self):
        """start_with_ids() should return (handle, workflow_id, run_id) without blocking."""
        from marqov.workflows.decorators import WorkflowDispatch

        dispatch = WorkflowDispatch.__new__(WorkflowDispatch)
        dispatch.name = "test-workflow"
        dispatch._prepare_workflow_input = MagicMock(return_value={"key": "value"})

        mock_handle = MagicMock()
        mock_handle.first_execution_run_id = "run-abc123"
        mock_handle.result = AsyncMock(return_value='{"result": "done"}')

        mock_client = MagicMock()
        mock_client.start_workflow = AsyncMock(return_value=mock_handle)

        with patch("marqov.workflows.temporal_workflow.JobWorkflow"):
            handle, workflow_id, run_id = await dispatch.start_with_ids(mock_client)

        assert handle is mock_handle
        assert workflow_id.startswith("test-workflow-")
        assert run_id == "run-abc123"
        # Crucially, result() was never called
        mock_handle.result.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_with_ids_delegates_to_start_with_ids(self):
        """run_with_ids() should call start_with_ids() then await handle.result()."""
        from marqov.workflows.decorators import WorkflowDispatch

        dispatch = WorkflowDispatch.__new__(WorkflowDispatch)
        dispatch.name = "test-workflow"
        dispatch._prepare_workflow_input = MagicMock(return_value={})

        mock_handle = MagicMock()
        mock_handle.first_execution_run_id = "run-xyz"
        mock_handle.result = AsyncMock(return_value='{"result": 42}')

        mock_client = MagicMock()
        mock_client.start_workflow = AsyncMock(return_value=mock_handle)

        with patch("marqov.workflows.temporal_workflow.JobWorkflow"):
            result, wf_id, run_id = await dispatch.run_with_ids(mock_client)

        assert result == {"result": 42}
        assert wf_id.startswith("test-workflow-")
        assert run_id == "run-xyz"
        mock_handle.result.assert_awaited_once()
