"""Marqov CLI for workflow management.

This module provides the command-line interface for:
- Running workflows locally or with Temporal
- Managing workflow status
- Starting Temporal workers

Other installed packages can add commands through the ``marqov.cli``
entry-point group; the Platform's ``marqov watch`` arrives that way, so this
module never imports ``marqov.platform``.

Usage:
    $ marqov run workflow.py::my_workflow --arg shots=1000
    $ marqov status <workflow-id>
    $ marqov worker start --task-queue marqov
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import math
import sys
import time
import uuid
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from typing import Any

import click
from temporalio.client import Client

from marqov import __version__


def parse_workflow_spec(spec: str) -> tuple[str, str]:
    """Parse workflow specification into module path and class name.

    Args:
        spec: Workflow spec in format 'path/to/module.py::ClassName'

    Returns:
        Tuple of (module_path, class_name)

    Raises:
        ValueError: If spec format is invalid
    """
    if "::" not in spec:
        raise ValueError(f"Invalid workflow spec: {spec}. Expected format: path/to/module.py::ClassName")

    module_path, class_name = spec.split("::", 1)
    return module_path, class_name


def load_workflow_class(module_path: str, class_name: str) -> type:
    """Dynamically load a workflow class from a module.

    Args:
        module_path: Path to the Python module
        class_name: Name of the workflow class

    Returns:
        The workflow class

    Raises:
        FileNotFoundError: If module doesn't exist
        AttributeError: If class doesn't exist in module
    """
    path = Path(module_path)
    if not path.exists():
        raise FileNotFoundError(f"Module not found: {module_path}")

    # Load module dynamically
    spec = importlib.util.spec_from_file_location("workflow_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["workflow_module"] = module
    spec.loader.exec_module(module)

    # Get the workflow class
    if not hasattr(module, class_name):
        raise AttributeError(f"Class '{class_name}' not found in {module_path}")

    return getattr(module, class_name)


def parse_args(args: tuple[str, ...]) -> dict[str, Any]:
    """Parse workflow arguments from CLI.

    Args:
        args: Tuple of 'key=value' strings

    Returns:
        Dictionary of parsed arguments
    """
    result: dict[str, Any] = {}
    for arg in args:
        if "=" not in arg:
            raise ValueError(f"Invalid argument format: {arg}. Expected: key=value")

        key, value = arg.split("=", 1)
        if key in result:
            raise ValueError(f"Argument '{key}' given more than once")

        # Try to parse as JSON for complex types
        try:
            result[key] = json.loads(value)
        except json.JSONDecodeError:
            # Keep as string
            result[key] = value

    return result


DEFAULT_CONNECT_TIMEOUT = 30.0


def _validate_timeout(ctx: click.Context, param: click.Parameter, value: float | None) -> float | None:
    """Reject a --timeout that is not a positive, finite number of seconds."""
    if value is not None and not (math.isfinite(value) and value > 0):
        raise click.BadParameter("must be a positive, finite number of seconds", param=param)
    return value


def _timeout_option(help_text: str) -> Any:
    return click.option(
        "--timeout",
        type=float,
        default=None,
        callback=_validate_timeout,
        metavar="SECONDS",
        help=help_text,
    )


async def _connect(host: str, port: int, timeout: float | None) -> Client:
    """Connect to Temporal, giving up after ``timeout`` seconds (default 30)."""
    limit = DEFAULT_CONNECT_TIMEOUT if timeout is None else timeout
    deadline = asyncio.timeout(limit)
    try:
        async with deadline:
            return await Client.connect(f"{host}:{port}")
    except TimeoutError:
        if not deadline.expired():
            raise
        click.echo(f"Error: timed out after {limit:g}s connecting to Temporal at {host}:{port}", err=True)
        raise SystemExit(1)


_AN_INSTANCE = object()


def _run_parameters(run_method: Any, owner: type | None) -> list[inspect.Parameter]:
    """The parameters ``run`` takes when Temporal calls it on a workflow instance.

    With the owning class, ``run`` is bound the way Python binds it on an
    instance (its descriptor's ``__get__``), so the instance parameter is
    dropped for a plain function, a ``functools.partialmethod`` or a wrapper
    with ``__get__``, whatever it is named, and kept for a ``staticmethod``.
    Without the class, a bound method is taken as it is, and otherwise a first
    parameter named ``self`` or ``cls`` is taken to be the instance.
    """
    if inspect.ismethod(run_method):
        return list(inspect.signature(run_method).parameters.values())
    if owner is not None:
        static = inspect.getattr_static(owner, "run", None)
        bind = getattr(type(static), "__get__", None)
        if static is not None and bind is not None:
            try:
                bound = bind(static, _AN_INSTANCE, owner)
                return list(inspect.signature(bound).parameters.values())
            except (TypeError, ValueError):
                pass
        elif static is not None:
            return list(inspect.signature(run_method).parameters.values())
    params = list(inspect.signature(run_method).parameters.values())
    if (
        params
        and params[0].kind in (params[0].POSITIONAL_ONLY, params[0].POSITIONAL_OR_KEYWORD)
        and params[0].name in ("self", "cls")
    ):
        params = params[1:]
    return params


def bind_workflow_args(
    run_method: Any, named: dict[str, Any], *, owner: type | None = None
) -> list[Any]:
    """Bind ``--arg`` values to a workflow's ``run`` signature by name.

    Temporal's client takes positional arguments, so the values are returned
    in parameter order. A parameter that is skipped between two supplied ones
    gets its default. Pass the workflow class as ``owner`` so the instance
    parameter is recognised by how ``run`` is defined, not by its name.

    Raises:
        ValueError: On an unknown name, a keyword-only or variadic target
            that cannot be passed by name, or a missing required parameter.
    """
    params = _run_parameters(run_method, owner)
    passable = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    kw_only = [p for p in params if p.kind is p.KEYWORD_ONLY]
    names = [p.name for p in passable]
    valid = ", ".join(names) if names else "(none)"
    kw_note = "--arg cannot pass keyword-only parameters (Temporal passes positional arguments only)"

    for key in named:
        if any(p.name == key for p in kw_only):
            raise ValueError(f"'{key}' is a keyword-only parameter; {kw_note}")
        if key not in names:
            raise ValueError(f"unknown argument '{key}'. Valid parameters: {valid}")

    required_kw = [p.name for p in kw_only if p.default is p.empty]
    if required_kw:
        raise ValueError(f"required keyword-only parameter(s) {', '.join(required_kw)}: {kw_note}")

    values: list[Any] = []
    last = max((i for i, p in enumerate(passable) if p.name in named), default=-1)
    for param in passable[: last + 1]:
        if param.name in named:
            values.append(named[param.name])
        elif param.default is not param.empty:
            try:
                json.dumps(param.default)
            except (TypeError, ValueError):
                raise ValueError(
                    f"parameter '{param.name}' is skipped and its default cannot be sent as JSON; "
                    f"pass --arg {param.name}=... explicitly"
                )
            values.append(param.default)
    # Any required parameter not supplied is an error, even beyond `last`.
    missing = [p.name for p in passable if p.default is p.empty and p.name not in named]
    if missing:
        raise ValueError(f"missing required argument(s): {', '.join(missing)}. Valid parameters: {valid}")
    return values


class _MarqovGroup(click.Group):
    """A group that also offers commands registered under the ``marqov.cli`` entry points.

    A registered command is imported only when it is invoked or listed by
    ``--help``. A built-in command of the same name wins.
    """

    @staticmethod
    def _registered() -> dict[str, EntryPoint]:
        return {ep.name: ep for ep in entry_points(group="marqov.cli")}

    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted({*super().list_commands(ctx), *self._registered()})

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        command = super().get_command(ctx, cmd_name)
        if command is None and (entry := self._registered().get(cmd_name)) is not None:
            command = entry.load()
        return command


@click.group(cls=_MarqovGroup)
@click.version_option(version=__version__)
def main() -> None:
    """Marqov - Quantum workflow orchestration."""
    pass


@main.command()
@click.argument("workflow")
@click.option("--arg", "-a", multiple=True, help="Workflow argument in key=value format")
@click.option("--host", default="localhost", help="Temporal server host")
@click.option("--port", default=7233, help="Temporal server port")
@click.option("--task-queue", default="marqov", help="Temporal task queue")
@click.option("--wait/--no-wait", default=True, help="Wait for workflow completion")
@_timeout_option(
    "Give up after SECONDS. Covers connecting (default 30s), starting the workflow "
    "and, with --wait, waiting for the result; without it only connecting is limited. "
    "The workflow is not cancelled when the limit expires."
)
def run(
    workflow: str,
    arg: tuple[str, ...],
    host: str,
    port: int,
    task_queue: str,
    wait: bool,
    timeout: float | None,
) -> None:
    """Run a quantum workflow.

    WORKFLOW should be in the format: module.py::ClassName

    --arg values bind to the workflow's run() parameters by name, in any
    order. An unknown name or a missing required parameter is an error.

    Examples:
        marqov run examples/temporal_bell_state.py::BellStateWorkflow --arg shots=1000
        marqov run my_workflow.py::VQEWorkflow --arg s3_bucket=my-bucket --no-wait
    """
    try:
        module_path, class_name = parse_workflow_spec(workflow)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    try:
        workflow_class = load_workflow_class(module_path, class_name)
    except (FileNotFoundError, AttributeError, ImportError) as e:
        click.echo(f"Error loading workflow: {e}", err=True)
        raise SystemExit(1)

    try:
        workflow_args = parse_args(arg)
    except ValueError as e:
        click.echo(f"Error parsing arguments: {e}", err=True)
        raise SystemExit(1)

    try:
        args_list = bind_workflow_args(workflow_class.run, workflow_args, owner=workflow_class)
    except (ValueError, TypeError, AttributeError) as e:
        click.echo(f"Error binding arguments: {e}", err=True)
        raise SystemExit(1)

    click.echo(f"Workflow: {class_name}")
    click.echo(f"Module: {module_path}")
    if workflow_args:
        click.echo(f"Arguments: {workflow_args}")

    async def execute():
        # Connect to Temporal
        click.echo(f"\nConnecting to Temporal at {host}:{port}...")
        started = time.monotonic()
        client = await _connect(host, port, timeout)

        # Generate workflow ID
        workflow_id = f"{class_name.lower()}-{uuid.uuid4().hex[:8]}"

        # Start the workflow
        click.echo(f"Starting workflow: {workflow_id}")

        def remaining() -> float | None:
            if timeout is None:
                return None
            return max(timeout - (time.monotonic() - started), 0.001)

        limit = asyncio.timeout(remaining())
        try:
            async with limit:
                handle = await client.start_workflow(
                    workflow_class.run,
                    args=args_list if args_list else None,
                    id=workflow_id,
                    task_queue=task_queue,
                )
        except TimeoutError:
            if not limit.expired():
                raise
            click.echo(
                f"Error: timed out after {timeout:g}s starting workflow {workflow_id}. "
                f"It may or may not have started; check it with: marqov status {workflow_id}",
                err=True,
            )
            raise SystemExit(1)

        click.echo(f"Workflow started: {handle.id}")
        click.echo(f"View in Temporal UI: http://{host}:8088/namespaces/default/workflows/{handle.id}")

        if wait:
            click.echo("\nWaiting for result...")
            limit = asyncio.timeout(remaining())
            try:
                async with limit:
                    result = await handle.result()
            except TimeoutError:
                if not limit.expired():
                    raise
                click.echo(
                    f"Error: timed out after {timeout:g}s waiting for the result. "
                    f"Workflow {handle.id} may still be running; check it with: marqov status {handle.id}",
                    err=True,
                )
                raise SystemExit(1)
            click.echo(f"\nResult: {json.dumps(result, indent=2, default=str)}")
        else:
            click.echo("\nWorkflow started in background (--no-wait)")

    asyncio.run(execute())


@main.command()
@click.argument("workflow_id")
@click.option("--host", default="localhost", help="Temporal server host")
@click.option("--port", default=7233, help="Temporal server port")
@_timeout_option("Connection timeout in seconds (default 30)")
def status(workflow_id: str, host: str, port: int, timeout: float | None) -> None:
    """Check status of a running workflow.

    Examples:
        marqov status bellstateworkflow-abc12345
        marqov status my-workflow-id --host temporal.example.com
    """

    async def check_status():
        click.echo(f"Connecting to Temporal at {host}:{port}...")
        client = await _connect(host, port, timeout)

        try:
            handle = client.get_workflow_handle(workflow_id)
            desc = await handle.describe()

            click.echo(f"\nWorkflow: {workflow_id}")
            click.echo(f"Status: {desc.status.name}")
            click.echo(f"Run ID: {desc.run_id}")
            click.echo(f"Started: {desc.start_time}")

            if desc.close_time:
                click.echo(f"Completed: {desc.close_time}")

            if desc.status.name == "COMPLETED":
                result = await handle.result()
                click.echo(f"\nResult: {json.dumps(result, indent=2, default=str)}")

        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            raise SystemExit(1)

    asyncio.run(check_status())


@main.group()
def worker() -> None:
    """Manage Temporal workers."""
    pass


@worker.command(name="start")
@click.option("--host", default="localhost", help="Temporal server host")
@click.option("--port", default=7233, help="Temporal server port")
@click.option("--task-queue", default="marqov", help="Temporal task queue name")
@click.option(
    "--workflow",
    "-w",
    multiple=True,
    help="Workflow module to load (format: path.py::ClassName)",
)
def worker_start(
    host: str,
    port: int,
    task_queue: str,
    workflow: tuple[str, ...],
) -> None:
    """Start a Temporal worker.

    The worker will process workflows and activities from the specified task queue.

    Examples:
        marqov worker start --task-queue marqov
        marqov worker start -w examples/temporal_bell_state.py::BellStateWorkflow
    """
    from temporalio.worker import Worker

    # Load workflow classes if specified
    workflows = []
    activities = []

    for spec in workflow:
        try:
            module_path, class_name = parse_workflow_spec(spec)
            workflow_class = load_workflow_class(module_path, class_name)
            workflows.append(workflow_class)
            click.echo(f"Loaded workflow: {class_name}")

            # Try to find activities in the same module
            path = Path(module_path)
            spec_obj = importlib.util.spec_from_file_location("workflow_module", path)
            if spec_obj and spec_obj.loader:
                module = importlib.util.module_from_spec(spec_obj)
                spec_obj.loader.exec_module(module)

                # Look for activity functions (decorated with @activity.defn)
                for name in dir(module):
                    obj = getattr(module, name)
                    if callable(obj) and hasattr(obj, "__temporal_activity_definition"):
                        activities.append(obj)
                        click.echo(f"Loaded activity: {name}")

        except Exception as e:
            click.echo(f"Warning: Could not load {spec}: {e}", err=True)

    async def run_worker():
        click.echo(f"\nConnecting to Temporal at {host}:{port}...")
        client = await Client.connect(f"{host}:{port}")

        click.echo(f"Starting worker on task queue: {task_queue}")
        if workflows:
            click.echo(f"Workflows: {[w.__name__ for w in workflows]}")
        if activities:
            click.echo(f"Activities: {[a.__name__ for a in activities]}")

        if not workflows and not activities:
            click.echo("Warning: No workflows or activities loaded. Worker will idle.")

        worker = Worker(
            client,
            task_queue=task_queue,
            workflows=workflows if workflows else [],
            activities=activities if activities else [],
        )

        click.echo("\nWorker running. Press Ctrl+C to stop.")
        try:
            await worker.run()
        except KeyboardInterrupt:
            click.echo("\nShutting down worker...")

    asyncio.run(run_worker())


@main.command(name="list")
@click.option("--host", default="localhost", help="Temporal server host")
@click.option("--port", default=7233, help="Temporal server port")
@click.option("--limit", default=10, help="Maximum number of workflows to list")
@_timeout_option("Connection timeout in seconds (default 30)")
def list_workflows(host: str, port: int, limit: int, timeout: float | None) -> None:
    """List recent workflows.

    Examples:
        marqov list
        marqov list --limit 20
    """

    async def _run():
        click.echo(f"Connecting to Temporal at {host}:{port}...")
        client = await _connect(host, port, timeout)

        click.echo(f"\nRecent workflows (limit {limit}):")
        click.echo("-" * 60)

        count = 0
        async for workflow in client.list_workflows(query=""):
            if count >= limit:
                break

            status = workflow.status.name if workflow.status else "UNKNOWN"
            click.echo(f"{workflow.id}")
            click.echo(f"  Status: {status}")
            click.echo(f"  Type: {workflow.workflow_type}")
            click.echo(f"  Started: {workflow.start_time}")
            click.echo()
            count += 1

        if count == 0:
            click.echo("No workflows found.")

    asyncio.run(_run())


if __name__ == "__main__":
    main()
