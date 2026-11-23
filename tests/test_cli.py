"""Tests for the marqov CLI.

These tests drive the real ``marqov.cli.main`` group through
``click.testing.CliRunner``. ``marqov.cli.Client.connect`` is patched to a
fake Temporal client so no network access or running Temporal server is
required.
"""

from __future__ import annotations

import asyncio
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
                async def run(self, shots=None, label=None):
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


def test_run_with_numeric_arg_binds_by_name(
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


def test_run_with_string_arg_binds_by_name(
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
    assert patched_connect.start_workflow_calls[0]["args"] == [None, "hello"]


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


@pytest.fixture()
def two_param_module(tmp_path: Path) -> Path:
    module_path = tmp_path / "wf2.py"
    module_path.write_text(
        textwrap.dedent(
            """
            class Workflow:
                async def run(self, a, b, c=3):
                    return {"ok": True}
            """
        )
    )
    return module_path


def test_run_binds_out_of_order_args_by_name(
    two_param_module: Path, patched_connect: FakeClient
) -> None:
    result = CliRunner().invoke(
        cli.main,
        ["run", f"{two_param_module}::Workflow", "--arg", "b=2", "--arg", "a=1", "--no-wait"],
    )

    assert result.exit_code == 0, result.output
    assert patched_connect.start_workflow_calls[0]["args"] == [1, 2]


def test_run_fills_skipped_defaults_in_parameter_order(
    two_param_module: Path, patched_connect: FakeClient
) -> None:
    result = CliRunner().invoke(
        cli.main,
        ["run", f"{two_param_module}::Workflow", "-a", "a=1", "-a", "b=2", "-a", "c=9", "--no-wait"],
    )

    assert result.exit_code == 0, result.output
    assert patched_connect.start_workflow_calls[0]["args"] == [1, 2, 9]


def test_run_unknown_arg_exits_nonzero_naming_it_and_valid_parameters(
    two_param_module: Path, patched_connect: FakeClient
) -> None:
    result = CliRunner().invoke(
        cli.main,
        ["run", f"{two_param_module}::Workflow", "--arg", "a=1", "--arg", "b=2", "--arg", "zzz=5", "--no-wait"],
    )

    assert result.exit_code != 0
    assert "zzz" in result.output
    assert "a, b, c" in result.output
    assert patched_connect.start_workflow_calls == []


def test_run_missing_required_parameter_exits_nonzero(
    two_param_module: Path, patched_connect: FakeClient
) -> None:
    result = CliRunner().invoke(
        cli.main,
        ["run", f"{two_param_module}::Workflow", "--arg", "a=1", "--no-wait"],
    )

    assert result.exit_code != 0
    assert "b" in result.output
    assert "missing" in result.output.lower()
    assert patched_connect.start_workflow_calls == []


def test_run_repeated_arg_name_is_refused(
    two_param_module: Path, patched_connect: FakeClient
) -> None:
    result = CliRunner().invoke(
        cli.main,
        ["run", f"{two_param_module}::Workflow", "--arg", "a=1", "--arg", "a=2", "--arg", "b=2", "--no-wait"],
    )

    assert result.exit_code != 0
    assert "more than once" in result.output


def _module(tmp_path: Path, sig: str, pre: str = "") -> Path:
    path = tmp_path / "wfx.py"
    path.write_text(f"{pre}\nclass Workflow:\n    async def run(self, {sig}):\n        return 1\n")
    return path


def test_run_skipped_non_json_default_exits_1_without_traceback(
    tmp_path: Path, patched_connect: FakeClient
) -> None:
    mod = _module(tmp_path, "a=_S, b=2", "_S = object()")
    result = CliRunner().invoke(cli.main, ["run", f"{mod}::Workflow", "--arg", "b=2", "--no-wait"])

    assert result.exit_code == 1
    assert "--arg a=..." in result.output
    assert not isinstance(result.exception, TypeError)
    assert patched_connect.start_workflow_calls == []


def test_run_naming_keyword_only_parameter_exits_1(
    tmp_path: Path, patched_connect: FakeClient
) -> None:
    mod = _module(tmp_path, "a=1, *, k=2")
    result = CliRunner().invoke(cli.main, ["run", f"{mod}::Workflow", "--arg", "k=3", "--no-wait"])

    assert result.exit_code == 1
    assert "keyword-only" in result.output
    assert "unknown" not in result.output
    assert patched_connect.start_workflow_calls == []


def test_run_required_keyword_only_parameter_is_an_error_up_front(
    tmp_path: Path, patched_connect: FakeClient
) -> None:
    mod = _module(tmp_path, "a=1, *, k")
    result = CliRunner().invoke(cli.main, ["run", f"{mod}::Workflow", "--arg", "a=1", "--no-wait"])

    assert result.exit_code == 1
    assert "keyword-only" in result.output
    assert "k" in result.output
    assert patched_connect.start_workflow_calls == []


@pytest.mark.parametrize("bad", ["0", "-5", "inf", "nan", "abc"])
def test_run_rejects_invalid_timeout(workflow_module: Path, bad: str) -> None:
    result = CliRunner().invoke(
        cli.main, ["run", f"{workflow_module}::Workflow", "--timeout", bad, "--no-wait"]
    )

    assert result.exit_code != 0
    assert "timeout" in result.output.lower()


def test_run_hung_connect_exits_within_timeout(workflow_module: Path) -> None:
    async def hang(*_a: Any, **_k: Any) -> None:
        await asyncio.sleep(60)

    with patch.object(cli.Client, "connect", new=hang):
        result = CliRunner().invoke(
            cli.main, ["run", f"{workflow_module}::Workflow", "--timeout", "0.2", "--no-wait"]
        )

    assert result.exit_code == 1
    assert "timed out" in result.output.lower()
    assert "connect" in result.output.lower()


def test_run_hung_wait_exits_within_timeout(workflow_module: Path, fake_client: FakeClient) -> None:
    async def hang() -> dict[str, Any]:
        await asyncio.sleep(60)
        return {}

    handle_result = FakeWorkflowHandle.result
    FakeWorkflowHandle.result = lambda self: hang()  # type: ignore[method-assign]
    try:
        with patch.object(cli.Client, "connect", new=AsyncMock(return_value=fake_client)):
            result = CliRunner().invoke(
                cli.main, ["run", f"{workflow_module}::Workflow", "--timeout", "0.2"]
            )
    finally:
        FakeWorkflowHandle.result = handle_result  # type: ignore[method-assign]

    assert result.exit_code == 1
    assert "timed out" in result.output.lower()
    assert "may still be running" in result.output.lower()


@pytest.mark.parametrize("wait_flag", ["--wait", "--no-wait"])
def test_run_hung_start_exits_within_timeout(workflow_module: Path, wait_flag: str) -> None:
    class HangingStart(FakeClient):
        async def start_workflow(self, *args: Any, **kwargs: Any) -> FakeWorkflowHandle:
            self.start_workflow_calls.append(kwargs)
            await asyncio.sleep(60)
            return FakeWorkflowHandle(kwargs["id"])

    client = HangingStart()
    with patch.object(cli.Client, "connect", new=AsyncMock(return_value=client)):
        result = CliRunner().invoke(
            cli.main, ["run", f"{workflow_module}::Workflow", "--timeout", "0.2", wait_flag]
        )

    assert result.exit_code == 1
    assert "timed out" in result.output.lower()
    workflow_id = client.start_workflow_calls[0]["id"]
    assert f"marqov status {workflow_id}" in result.output


def _class_module(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "wfc.py"
    path.write_text("class Workflow:\n" + textwrap.indent(textwrap.dedent(body), "    "))
    return path


@pytest.mark.parametrize("first", ["this", "_"])
def test_run_instance_method_first_parameter_is_dropped_whatever_its_name(
    tmp_path: Path, patched_connect: FakeClient, first: str
) -> None:
    mod = _class_module(tmp_path, f"async def run({first}, a):\n    return 1\n")
    result = CliRunner().invoke(cli.main, ["run", f"{mod}::Workflow", "--arg", "a=1", "--no-wait"])

    assert result.exit_code == 0, result.output
    assert patched_connect.start_workflow_calls[0]["args"] == [1]


def test_run_staticmethod_keeps_a_first_parameter_named_self(
    tmp_path: Path, patched_connect: FakeClient
) -> None:
    mod = _class_module(tmp_path, "@staticmethod\nasync def run(self, b):\n    return 1\n")
    result = CliRunner().invoke(
        cli.main, ["run", f"{mod}::Workflow", "--arg", "b=2", "--arg", "self=1", "--no-wait"]
    )

    assert result.exit_code == 0, result.output
    assert patched_connect.start_workflow_calls[0]["args"] == [1, 2]


def test_run_classmethod_binds_after_cls(tmp_path: Path, patched_connect: FakeClient) -> None:
    mod = _class_module(tmp_path, "@classmethod\nasync def run(klass, self, b):\n    return 1\n")
    result = CliRunner().invoke(
        cli.main, ["run", f"{mod}::Workflow", "--arg", "b=2", "--arg", "self=1", "--no-wait"]
    )

    assert result.exit_code == 0, result.output
    assert patched_connect.start_workflow_calls[0]["args"] == [1, 2]


def test_run_callable_object_keeps_its_first_parameter(
    tmp_path: Path, patched_connect: FakeClient
) -> None:
    path = tmp_path / "wfo.py"
    path.write_text(
        "class _Run:\n"
        "    def __call__(self, a):\n"
        "        return 1\n"
        "class Workflow:\n"
        "    run = _Run()\n"
    )
    result = CliRunner().invoke(cli.main, ["run", f"{path}::Workflow", "--arg", "a=1", "--no-wait"])

    assert result.exit_code == 0, result.output
    assert patched_connect.start_workflow_calls[0]["args"] == [1]


@pytest.mark.parametrize("extra", [[], ["--timeout", "5"]])
@pytest.mark.parametrize("where", ["start", "result"])
def test_run_inner_timeout_error_is_not_reported_as_our_timeout(
    workflow_module: Path, extra: list[str], where: str
) -> None:
    class Handle(FakeWorkflowHandle):
        async def result(self) -> dict[str, Any]:
            raise TimeoutError("inner")

    class Raising(FakeClient):
        async def start_workflow(self, *args: Any, **kwargs: Any) -> FakeWorkflowHandle:
            if where == "start":
                raise TimeoutError("inner")
            return Handle(kwargs["id"])

    with patch.object(cli.Client, "connect", new=AsyncMock(return_value=Raising())):
        result = CliRunner().invoke(cli.main, ["run", f"{workflow_module}::Workflow", *extra])

    assert result.exit_code != 0
    assert isinstance(result.exception, TimeoutError)
    assert str(result.exception) == "inner"
    assert "timed out after" not in result.output


def test_connect_inner_timeout_error_is_not_reported_as_our_timeout(workflow_module: Path) -> None:
    async def raising(*_a: Any, **_k: Any) -> None:
        raise TimeoutError("inner")

    with patch.object(cli.Client, "connect", new=raising):
        result = CliRunner().invoke(cli.main, ["run", f"{workflow_module}::Workflow", "--no-wait"])

    assert isinstance(result.exception, TimeoutError)
    assert "timed out after" not in result.output


def test_run_partialmethod_binds_like_a_method(tmp_path: Path, patched_connect: FakeClient) -> None:
    path = tmp_path / "wfp.py"
    path.write_text(
        "import functools\n"
        "async def _r(self, tag, a):\n"
        "    return 1\n"
        "class Workflow:\n"
        "    run = functools.partialmethod(_r, 'x')\n"
    )
    result = CliRunner().invoke(cli.main, ["run", f"{path}::Workflow", "--arg", "a=1", "--no-wait"])

    assert result.exit_code == 0, result.output
    assert patched_connect.start_workflow_calls[0]["args"] == [1]


def test_run_descriptor_wrapper_binds_like_a_method(tmp_path: Path, patched_connect: FakeClient) -> None:
    path = tmp_path / "wfd.py"
    path.write_text(
        "import functools, types\n"
        "class wrap:\n"
        "    def __init__(self, f):\n"
        "        functools.update_wrapper(self, f)\n"
        "        self.f = f\n"
        "    def __call__(self, *a, **k):\n"
        "        return self.f(*a, **k)\n"
        "    def __get__(self, obj, cls=None):\n"
        "        return self if obj is None else types.MethodType(self, obj)\n"
        "class Workflow:\n"
        "    @wrap\n"
        "    async def run(self, a):\n"
        "        return 1\n"
    )
    result = CliRunner().invoke(cli.main, ["run", f"{path}::Workflow", "--arg", "a=1", "--no-wait"])

    assert result.exit_code == 0, result.output
    assert patched_connect.start_workflow_calls[0]["args"] == [1]


def test_status_hung_connect_exits_within_timeout() -> None:
    async def hang(*_a: Any, **_k: Any) -> None:
        await asyncio.sleep(60)

    with patch.object(cli.Client, "connect", new=hang):
        result = CliRunner().invoke(cli.main, ["status", "wf-1", "--timeout", "0.2"])

    assert result.exit_code == 1


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
