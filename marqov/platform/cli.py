"""Compose local workflow commands with Platform commands at the CLI boundary."""

import click

from marqov import __version__
from marqov.cli import main as workflow_commands
from marqov.platform.watch import watch


@click.group()
def platform_commands() -> None:
    """Platform commands."""


platform_commands.add_command(watch)


@click.command(cls=click.CommandCollection, sources=[workflow_commands, platform_commands])
@click.version_option(version=__version__)
def main() -> None:
    """Marqov — local workflows and the managed Platform."""


if __name__ == "__main__":
    main()
