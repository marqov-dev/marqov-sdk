"""Tests for the IBM Cloud channel defaults in MarqovDevice._get_provider_device.

IBM retired the legacy `ibm_quantum` channel. The current IBM Quantum Platform
uses `ibm_cloud`, where `instance` is an account CRN and `token` is an IBM Cloud
API key. These tests pin the new defaults.

`qiskit_ibm_runtime` is not installed in this environment, so we inject a fake
module into sys.modules; the SDK imports `QiskitRuntimeService` *inside* the IBM
branch of `_get_provider_device`, so this fake resolves at that import site.
"""

import sys
import types

import pytest

from marqov.device import MarqovDevice


@pytest.fixture
def fake_runtime(monkeypatch):
    """Install a fake qiskit_ibm_runtime module and capture constructor kwargs."""
    captured = {}
    sentinel_backend = object()

    class FakeQiskitRuntimeService:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def backend(self, name):
            captured["backend_name"] = name
            return sentinel_backend

    module = types.ModuleType("qiskit_ibm_runtime")
    module.QiskitRuntimeService = FakeQiskitRuntimeService
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", module)

    return captured, sentinel_backend

def test_ibm_defaults_to_ibm_quantum_platform_channel(fake_runtime):
    captured, sentinel = fake_runtime
    device = MarqovDevice(
        "ibm_brisbane",
        {
            "backend": "ibm_brisbane",
            "ibm_token": "tok",
            "ibm_instance": "crn:v1:bluemix:public:quantum-computing:us-east:a/abc::",
        },
    )

    result = device._get_provider_device()

    assert result is sentinel
    kwargs = captured["kwargs"]
    assert kwargs["channel"] == "ibm_quantum_platform"
    assert kwargs["instance"] == "crn:v1:bluemix:public:quantum-computing:us-east:a/abc::"
    assert kwargs["token"] == "tok"
    assert captured["backend_name"] == "ibm_brisbane"


def test_ibm_honors_explicit_channel(fake_runtime):
    """ibm_cloud remains valid and must not be overridden by the default."""
    captured, _ = fake_runtime
    device = MarqovDevice(
        "ibm_brisbane",
        {
            "backend": "ibm_brisbane",
            "ibm_token": "tok",
            "ibm_channel": "ibm_cloud",
            "ibm_instance": "crn:v1:...",
        },
    )

    device._get_provider_device()

    assert captured["kwargs"]["channel"] == "ibm_cloud"


def test_ibm_omits_legacy_instance_when_absent(fake_runtime):
    captured, _ = fake_runtime
    device = MarqovDevice(
        "ibm_brisbane",
        {"backend": "ibm_brisbane", "ibm_token": "tok"},
    )

    device._get_provider_device()

    kwargs = captured["kwargs"]
    # Must NOT force the dead legacy instance default.
    assert kwargs.get("instance") != "ibm-q/open/main"
    # When no CRN is provided, instance should be omitted entirely so the
    # service auto-discovers it from the API key.
    assert "instance" not in kwargs
    assert kwargs["channel"] == "ibm_quantum_platform"
    assert kwargs["token"] == "tok"


def test_ibm_empty_instance_string_is_omitted(fake_runtime):
    """A blank stored ibm_instance must not be forwarded — it fails auth."""
    captured, _ = fake_runtime
    device = MarqovDevice(
        "ibm_brisbane",
        {"backend": "ibm_brisbane", "ibm_token": "tok", "ibm_instance": ""},
    )

    device._get_provider_device()

    assert "instance" not in captured["kwargs"]


# ---------------------------------------------------------------------------
# ExecutorFactory / IBMExecutorConfig — the async execution path
# ---------------------------------------------------------------------------


def test_ibm_executor_config_defaults():
    from marqov.executors.ibm import IBMExecutorConfig

    cfg = IBMExecutorConfig(backend_name="ibm_kingston")

    assert cfg.channel == "ibm_quantum_platform"
    assert cfg.instance is None


def test_ibm_executor_omits_instance_when_unset(fake_runtime):
    from marqov.executors.ibm import IBMExecutor, IBMExecutorConfig

    captured, sentinel = fake_runtime
    executor = IBMExecutor(IBMExecutorConfig(backend_name="ibm_kingston", token="tok"))

    assert executor._get_backend_sync() is sentinel
    kwargs = captured["kwargs"]
    assert kwargs["channel"] == "ibm_quantum_platform"
    assert "instance" not in kwargs
    assert captured["backend_name"] == "ibm_kingston"


def test_factory_does_not_inject_legacy_ibm_defaults():
    from marqov.executors.factory import ExecutorFactory

    executor = ExecutorFactory.create_executor(
        "ibm-kingston", {"provider": "IBM Quantum", "backend_name": "ibm_kingston"}
    )

    assert executor.config.channel == "ibm_quantum_platform"
    assert executor.config.instance is None
