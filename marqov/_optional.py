"""Actionable dependency checks at optional provider boundaries."""

from importlib import import_module


def require_braket() -> None:
    try:
        import_module("braket")
    except ModuleNotFoundError as error:
        if error.name != "braket":
            raise
        raise ImportError('AWS Braket requires pip install "marqov[braket]"') from error
