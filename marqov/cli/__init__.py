"""Marqov CLI for workflow management.

This module provides the command-line interface for:
- Running workflows locally or with Temporal
- Managing workflow status
- Starting Temporal workers

Usage:
    $ marqov run workflow.py::my_workflow --arg shots=1000
    $ marqov status <workflow-id>
    $ marqov worker start --task-queue marqov
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import uuid
from datetime import timedelta
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

        # Try to parse as JSON for complex types
        try:
            result[key] = json.loads(value)
        except json.JSONDecodeError:
            # Keep as string
            result[key] = value

    return result


@click.group()
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
def run(
    workflow: str,
    arg: tuple[str, ...],
    host: str,
    port: int,
    task_queue: str,
    wait: bool,
) -> None:
    """Run a quantum workflow.

    WORKFLOW should be in the format: module.py::ClassName

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

    click.echo(f"Workflow: {class_name}")
    click.echo(f"Module: {module_path}")
    if workflow_args:
        click.echo(f"Arguments: {workflow_args}")

    async def execute():
        # Connect to Temporal
        click.echo(f"\nConnecting to Temporal at {host}:{port}...")
        client = await Client.connect(f"{host}:{port}")

        # Generate workflow ID
        workflow_id = f"{class_name.lower()}-{uuid.uuid4().hex[:8]}"

        # Start the workflow
        click.echo(f"Starting workflow: {workflow_id}")

        # Convert args dict to positional args for workflow.run
        # Workflows typically take positional args, so we pass as list
        args_list = list(workflow_args.values()) if workflow_args else []

        handle = await client.start_workflow(
            workflow_class.run,
            args=args_list if args_list else None,
            id=workflow_id,
            task_queue=task_queue,
        )

        click.echo(f"Workflow started: {handle.id}")
        click.echo(f"View in Temporal UI: http://{host}:8088/namespaces/default/workflows/{handle.id}")

        if wait:
            click.echo("\nWaiting for result...")
            result = await handle.result()
            click.echo(f"\nResult: {json.dumps(result, indent=2, default=str)}")
        else:
            click.echo("\nWorkflow started in background (--no-wait)")

    asyncio.run(execute())


@main.command()
@click.argument("workflow_id")
@click.option("--host", default="localhost", help="Temporal server host")
@click.option("--port", default=7233, help="Temporal server port")
def status(workflow_id: str, host: str, port: int) -> None:
    """Check status of a running workflow.

    Examples:
        marqov status bellstateworkflow-abc12345
        marqov status my-workflow-id --host temporal.example.com
    """

    async def check_status():
        click.echo(f"Connecting to Temporal at {host}:{port}...")
        client = await Client.connect(f"{host}:{port}")

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


@main.command()
@click.option("--host", default="localhost", help="Temporal server host")
@click.option("--port", default=7233, help="Temporal server port")
@click.option("--limit", default=10, help="Maximum number of workflows to list")
def list(host: str, port: int, limit: int) -> None:
    """List recent workflows.

    Examples:
        marqov list
        marqov list --limit 20
    """

    async def list_workflows():
        click.echo(f"Connecting to Temporal at {host}:{port}...")
        client = await Client.connect(f"{host}:{port}")

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

    asyncio.run(list_workflows())


if __name__ == "__main__":
    main()
