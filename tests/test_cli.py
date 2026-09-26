"""Tests for the marqov CLI.

These tests drive the real ``marqov.cli.main`` group through
``click.testing.CliRunner``. ``marqov.cli.Client.connect`` is patched to a
fake Temporal client so no network access or running Temporal server is
required.
"""

from __future__ import annotations

import builtins
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import click
import pytest
from click.testing import CliRunner

import marqov.cli as cli


class FakeWorkflowHandle:
    """Minimal stand-in for a Temporal ``WorkflowHandle``."""

    def __init__(self, workflow_id: str) -> None:
        self.id = workflow_id

    async def result(self) -> dict[str, Any]:
        return {"ok": True}


class FakeClient:
    """Minimal stand-in for a Temporal ``Client``.

    Records the arguments passed to ``start_workflow`` so tests can assert on
    them.
    """

    def __init__(self) -> None:
        self.start_workflow_calls: list[dict[str, Any]] = []

    async def start_workflow(self, *args: Any, **kwargs: Any) -> FakeWorkflowHandle:
        self.start_workflow_calls.append(kwargs)
        return FakeWorkflowHandle(kwargs["id"])

    async def list_workflows(self, query: str = ""):
        # Empty async generator: no workflows to report.
        return
        yield  # pragma: no cover - makes this an async generator


@pytest.fixture()
def workflow_module(tmp_path: Path) -> Path:
    """Write a minimal workflow module usable by ``marqov run``."""
    module_path = tmp_path / "wf.py"
    module_path.write_text(
        textwrap.dedent(
            """
            class Workflow:
                async def run(self, *args):
                    return {"ok": True}
            """
        )
    )
    return module_path


@pytest.fixture()
def fake_client() -> FakeClient:
    return FakeClient()


@pytest.fixture()
def patched_connect(fake_client: FakeClient):
    with patch.object(cli.Client, "connect", new=AsyncMock(return_value=fake_client)):
        yield fake_client


def test_no_command_shadows_a_builtin() -> None:
    """Planted-regression check: no module attribute named after a builtin
    may be a click.Command.

    This fails on the unfixed code because ``marqov.cli.list`` is a
    ``click.Command`` instead of the builtin ``list``.
    """
    for name in dir(cli):
        if hasattr(builtins, name):
            assert not isinstance(
                getattr(cli, name), click.Command
            ), f"marqov.cli.{name} shadows the builtin {name!r} with a click.Command"


def test_run_with_numeric_arg_passes_positional_args(
    workflow_module: Path, patched_connect: FakeClient
) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        ["run", f"{workflow_module}::Workflow", "--arg", "shots=1000", "--no-wait"],
    )

    assert result.exit_code == 0, result.output
    assert len(patched_connect.start_workflow_calls) == 1
    assert patched_connect.start_workflow_calls[0]["args"] == [1000]


def test_run_with_string_arg_passes_positional_args(
    workflow_module: Path, patched_connect: FakeClient
) -> None:
    """Planted-regression case: this used to exit 2 because ``list`` was
    shadowed by the click command, so click re-parsed the string argument as
    a command line."""
    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        ["run", f"{workflow_module}::Workflow", "--arg", "label=hello", "--no-wait"],
    )

    assert result.exit_code == 0, result.output
    assert len(patched_connect.start_workflow_calls) == 1
    assert patched_connect.start_workflow_calls[0]["args"] == ["hello"]


def test_run_without_arg_passes_none(
    workflow_module: Path, patched_connect: FakeClient
) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        ["run", f"{workflow_module}::Workflow", "--no-wait"],
    )

    assert result.exit_code == 0, result.output
    assert len(patched_connect.start_workflow_calls) == 1
    assert patched_connect.start_workflow_calls[0]["args"] is None


def test_run_with_malformed_spec_exits_1() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.main, ["run", "not-a-valid-spec"])

    assert result.exit_code == 1
    assert "Invalid workflow spec" in result.output


def test_run_with_arg_missing_equals_exits_1(workflow_module: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        ["run", f"{workflow_module}::Workflow", "--arg", "shots"],
    )

    assert result.exit_code == 1
    assert "Invalid argument format" in result.output


def test_list_invokes_temporal_client_and_exits_0(patched_connect: FakeClient) -> None:
    runner = CliRunner()
    result = runner.invoke(cli.main, ["list"])

    assert result.exit_code == 0, result.output
    assert "No workflows found." in result.output


def test_help_lists_a_command_literally_named_list() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.main, ["--help"])

    assert result.exit_code == 0
    assert "list" in result.output


def test_list_help_still_shows_options() -> None:
    runner = CliRunner()
    result = runner.invoke(cli.main, ["list", "--help"])

    assert result.exit_code == 0
    assert "--host" in result.output
    assert "--port" in result.output
    assert "--limit" in result.output
