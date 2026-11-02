"""Actionable dependency checks at optional provider boundaries."""

from importlib import import_module


def _require(module: str, message: str) -> None:
    try:
        import_module(module)
    except ModuleNotFoundError as error:
        if error.name != module:
            raise
        raise ImportError(message) from error


def require_braket() -> None:
    _require("braket", 'AWS Braket requires pip install "marqov[braket]"')


def require_qiskit() -> None:
    _require("qiskit", 'Qiskit interop requires pip install "marqov[qiskit]"')


def require_cirq() -> None:
    _require("cirq", 'Cirq interop requires pip install "marqov[cirq]"')
