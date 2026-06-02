"""Tests for marqov.workflows.runner module.

Tests for worker creation and component exports.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from marqov.workflows.runner import (
    create_worker,
    JobWorkflow,
    execute_task,
    prepare_node_inputs,
    LatticeWorkflow,
    execute_electron,
)


class TestCreateWorker:
    """Tests for create_worker function."""

    def test_create_worker_default_task_queue(self) -> None:
        """create_worker uses default task queue."""
        mock_client = MagicMock()

        with patch("marqov.workflows.runner.Worker") as MockWorker:
            create_worker(mock_client)

            MockWorker.assert_called_once()
            call_kwargs = MockWorker.call_args
            assert call_kwargs[1]["task_queue"] == "marqov-workflows"

    def test_create_worker_custom_task_queue(self) -> None:
        """create_worker accepts custom task queue."""
        mock_client = MagicMock()

        with patch("marqov.workflows.runner.Worker") as MockWorker:
            create_worker(mock_client, task_queue="custom-queue")

            call_kwargs = MockWorker.call_args
            assert call_kwargs[1]["task_queue"] == "custom-queue"

    def test_create_worker_registers_workflow(self) -> None:
        """create_worker registers JobWorkflow."""
        mock_client = MagicMock()

        with patch("marqov.workflows.runner.Worker") as MockWorker:
            create_worker(mock_client)

            call_kwargs = MockWorker.call_args
            assert JobWorkflow in call_kwargs[1]["workflows"]

    def test_create_worker_registers_activities(self) -> None:
        """create_worker registers both activities."""
        mock_client = MagicMock()

        with patch("marqov.workflows.runner.Worker") as MockWorker:
            create_worker(mock_client)

            call_kwargs = MockWorker.call_args
            activities = call_kwargs[1]["activities"]
            assert execute_task in activities
            assert prepare_node_inputs in activities

    def test_create_worker_passes_client(self) -> None:
        """create_worker passes client to Worker."""
        mock_client = MagicMock()

        with patch("marqov.workflows.runner.Worker") as MockWorker:
            create_worker(mock_client)

            call_args = MockWorker.call_args
            assert call_args[0][0] is mock_client

    def test_create_worker_returns_worker(self) -> None:
        """create_worker returns Worker instance."""
        mock_client = MagicMock()
        mock_worker = MagicMock()

        with patch("marqov.workflows.runner.Worker", return_value=mock_worker):
            result = create_worker(mock_client)

            assert result is mock_worker


class TestBackwardCompatibility:
    """Tests for backward compatibility aliases."""

    def test_lattice_workflow_alias(self) -> None:
        """LatticeWorkflow is alias for JobWorkflow."""
        assert LatticeWorkflow is JobWorkflow

    def test_execute_electron_alias(self) -> None:
        """execute_electron is alias for execute_task."""
        assert execute_electron is execute_task


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_jobworkflow_exported(self) -> None:
        """JobWorkflow is exported."""
        from marqov.workflows import runner
        assert hasattr(runner, "JobWorkflow")

    def test_create_worker_exported(self) -> None:
        """create_worker is exported."""
        from marqov.workflows import runner
        assert hasattr(runner, "create_worker")

    def test_execute_task_exported(self) -> None:
        """execute_task is exported."""
        from marqov.workflows import runner
        assert hasattr(runner, "execute_task")

    def test_prepare_node_inputs_exported(self) -> None:
        """prepare_node_inputs is exported."""
        from marqov.workflows import runner
        assert hasattr(runner, "prepare_node_inputs")

    def test_backward_compat_aliases_exported(self) -> None:
        """Backward compatibility aliases are exported."""
        from marqov.workflows import runner
        assert hasattr(runner, "LatticeWorkflow")
        assert hasattr(runner, "execute_electron")


class TestWorkerConfiguration:
    """Tests for worker configuration options."""

    def test_worker_has_correct_workflow_list(self) -> None:
        """Worker is configured with exactly one workflow."""
        mock_client = MagicMock()

        with patch("marqov.workflows.runner.Worker") as MockWorker:
            create_worker(mock_client)

            call_kwargs = MockWorker.call_args
            workflows = call_kwargs[1]["workflows"]
            assert len(workflows) == 1
            assert workflows[0] == JobWorkflow

    def test_worker_has_correct_activity_count(self) -> None:
        """Worker is configured with exactly two activities."""
        mock_client = MagicMock()

        with patch("marqov.workflows.runner.Worker") as MockWorker:
            create_worker(mock_client)

            call_kwargs = MockWorker.call_args
            activities = call_kwargs[1]["activities"]
            assert len(activities) == 2
