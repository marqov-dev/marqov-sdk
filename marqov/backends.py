"""Backend type detection utilities."""

SIMULATOR_BACKENDS = {"local", "marqov-sim", "sv1", "dm1", "tn1", "rigetti-qvm", "ionq-sim"}


def is_simulator(backend: str) -> bool:
    """Return True if the backend is a simulator."""
    return backend in SIMULATOR_BACKENDS


def is_azure(params: dict) -> bool:
    """Return True if params indicate an Azure Quantum target.

    ``subscription_id`` is canonical and ``azure_subscription_id`` is its
    deprecated spelling (see ``marqov.executors._params``). The canonical key
    selects Azure only when no ``device_arn`` names a Braket device, so a dict
    that routed somewhere before canonical keys existed still routes there.
    """
    if params.get("azure_subscription_id"):
        return True
    return bool(params.get("subscription_id")) and not is_braket(params)


def is_ibm(params: dict) -> bool:
    """Return True if params indicate an IBM Quantum target.

    ``channel`` and ``instance`` are canonical IBM keys; ``ibm_channel`` and
    ``ibm_token`` are deprecated spellings that also select IBM (see
    ``marqov.executors._params``). A canonical ``token`` never selects IBM on
    its own, because IonQ, Quantum Brilliance and others take a ``token`` too:
    supply ``channel`` or ``instance`` with it. The canonical keys select IBM
    only when nothing else names a provider, so a dict that routed somewhere
    before canonical keys existed still routes there.
    """
    if params.get("ibm_token") or params.get("ibm_channel"):
        return True
    return (
        bool(params.get("channel") or params.get("instance"))
        and not is_azure(params)
        and not is_braket(params)
    )


def is_braket(params: dict) -> bool:
    """Return True if params indicate an AWS Braket target."""
    return bool(params.get("device_arn"))
