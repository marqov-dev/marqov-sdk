"""Backend type detection utilities."""

SIMULATOR_BACKENDS = {"local", "marqov-sim", "sv1", "dm1", "tn1", "rigetti-qvm", "ionq-sim"}


def is_simulator(backend: str) -> bool:
    """Return True if the backend is a simulator."""
    return backend in SIMULATOR_BACKENDS


def is_azure(params: dict) -> bool:
    """Return True if params indicate an Azure Quantum target."""
    return bool(params.get("azure_subscription_id"))


def is_ibm(params: dict) -> bool:
    """Return True if params indicate an IBM Quantum target."""
    return bool(params.get("ibm_token") or params.get("ibm_channel"))
