"""Tests for the IBM channel and instance defaults across all three call sites.

IBM retired the legacy ``ibm_quantum`` channel and the ``hub/group/project``
instance form. The current default channel is ``ibm_quantum_platform``;
``ibm_cloud`` remains accepted. ``instance`` is optional: when omitted the
service auto-discovers the instance from the token, and a CRN is only needed
when a token maps to more than one instance. These tests pin those defaults and
the shared normaliser that translates the retired values.

A fake ``qiskit_ibm_runtime`` module is injected into ``sys.modules`` so the
tests capture the exact constructor kwargs without a network call and without
depending on the ``ibm`` extra being installed. The SDK imports
``QiskitRuntimeService`` inside the IBM branches, so the fake resolves at that
import site.
"""

from __future__ import annotations

import sys
import types
import warnings

import pytest

from marqov.device import MarqovDevice
from marqov.executors.factory import ExecutorFactory
from marqov.executors.ibm import (
    DEFAULT_IBM_CHANNEL,
    IBMExecutor,
    IBMExecutorConfig,
    normalize_ibm_connection,
)

CRN = "crn:v1:bluemix:public:quantum-computing:us-east:a/abc::"


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


# ---------------------------------------------------------------------------
# The shared normaliser
# ---------------------------------------------------------------------------


class TestNormalizeIBMConnection:
    def test_defaults_when_both_absent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert normalize_ibm_connection(None, None) == (DEFAULT_IBM_CHANNEL, None)

    @pytest.mark.parametrize("channel", [None, ""])
    @pytest.mark.parametrize("instance", [None, ""])
    def test_none_and_empty_string_are_equivalent(self, channel, instance):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert normalize_ibm_connection(channel, instance) == (DEFAULT_IBM_CHANNEL, None)

    def test_retired_channel_is_translated_with_warning(self):
        with pytest.warns(DeprecationWarning, match="ibm_quantum_platform"):
            channel, instance = normalize_ibm_connection("ibm_quantum", CRN)
        assert channel == DEFAULT_IBM_CHANNEL
        assert instance == CRN

    def test_legacy_instance_is_dropped_with_warning(self):
        with pytest.warns(DeprecationWarning, match="auto-discovers"):
            channel, instance = normalize_ibm_connection(DEFAULT_IBM_CHANNEL, "ibm-q/open/main")
        assert channel == DEFAULT_IBM_CHANNEL
        assert instance is None

    @pytest.mark.parametrize("channel", ["ibm_cloud", "ibm_quantum_platform", "some_future_channel"])
    def test_other_channels_pass_through(self, channel):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert normalize_ibm_connection(channel, None) == (channel, None)

    @pytest.mark.parametrize("instance", [CRN, "crn:v1:a/b/c", "my-instance"])
    def test_non_legacy_instances_pass_through(self, instance):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert normalize_ibm_connection(None, instance) == (DEFAULT_IBM_CHANNEL, instance)


# ---------------------------------------------------------------------------
# MarqovDevice._get_provider_device (the synchronous user-script path)
# ---------------------------------------------------------------------------


def test_ibm_defaults_to_ibm_quantum_platform_channel(fake_runtime):
    captured, sentinel = fake_runtime
    device = MarqovDevice(
        "ibm_brisbane",
        {"backend": "ibm_brisbane", "ibm_token": "tok", "ibm_instance": CRN},
    )

    result = device._get_provider_device()

    assert result is sentinel
    kwargs = captured["kwargs"]
    assert kwargs["channel"] == "ibm_quantum_platform"
    assert kwargs["instance"] == CRN
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
            "ibm_instance": CRN,
        },
    )

    device._get_provider_device()

    assert captured["kwargs"]["channel"] == "ibm_cloud"


def test_ibm_omits_legacy_instance_when_absent(fake_runtime):
    captured, _ = fake_runtime
    device = MarqovDevice("ibm_brisbane", {"backend": "ibm_brisbane", "ibm_token": "tok"})

    device._get_provider_device()

    kwargs = captured["kwargs"]
    # When no CRN is provided, instance is omitted entirely so the service
    # auto-discovers it from the API key.
    assert "instance" not in kwargs
    assert kwargs["channel"] == "ibm_quantum_platform"
    assert kwargs["token"] == "tok"


def test_ibm_empty_instance_string_is_omitted(fake_runtime):
    """A blank stored ibm_instance must not be forwarded: it fails auth."""
    captured, _ = fake_runtime
    device = MarqovDevice(
        "ibm_brisbane",
        {"backend": "ibm_brisbane", "ibm_token": "tok", "ibm_instance": ""},
    )

    device._get_provider_device()

    assert "instance" not in captured["kwargs"]


def test_device_translates_stored_legacy_values(fake_runtime):
    """A params dict saved under the old defaults must still connect."""
    captured, _ = fake_runtime
    device = MarqovDevice(
        "ibm_brisbane",
        {
            "backend": "ibm_brisbane",
            "ibm_token": "tok",
            "ibm_channel": "ibm_quantum",
            "ibm_instance": "ibm-q/open/main",
        },
    )

    with pytest.warns(DeprecationWarning):
        device._get_provider_device()

    kwargs = captured["kwargs"]
    assert kwargs["channel"] == "ibm_quantum_platform"
    assert "instance" not in kwargs


# ---------------------------------------------------------------------------
# IBMExecutorConfig / IBMExecutor / ExecutorFactory (the async execution path)
# ---------------------------------------------------------------------------


def test_ibm_executor_config_defaults():
    cfg = IBMExecutorConfig(backend_name="ibm_kingston")

    assert cfg.channel == "ibm_quantum_platform"
    assert cfg.instance is None


def test_ibm_executor_config_translates_retired_values_on_construction():
    with pytest.warns(DeprecationWarning) as record:
        cfg = IBMExecutorConfig(
            backend_name="ibm_kingston", channel="ibm_quantum", instance="ibm-q/open/main"
        )

    assert cfg.channel == "ibm_quantum_platform"
    assert cfg.instance is None
    messages = " ".join(str(w.message) for w in record)
    assert "ibm_quantum_platform" in messages
    assert "auto-discovers" in messages


def test_ibm_executor_omits_instance_when_unset(fake_runtime):
    captured, sentinel = fake_runtime
    executor = IBMExecutor(IBMExecutorConfig(backend_name="ibm_kingston", token="tok"))

    assert executor._get_backend_sync() is sentinel
    kwargs = captured["kwargs"]
    assert kwargs["channel"] == "ibm_quantum_platform"
    assert "instance" not in kwargs
    assert captured["backend_name"] == "ibm_kingston"


def test_ibm_executor_forwards_crn_instance(fake_runtime):
    captured, _ = fake_runtime
    executor = IBMExecutor(IBMExecutorConfig(backend_name="ibm_kingston", instance=CRN))

    executor._get_backend_sync()

    assert captured["kwargs"]["instance"] == CRN


def test_factory_does_not_inject_legacy_ibm_defaults():
    executor = ExecutorFactory.create_executor(
        "ibm-kingston", {"provider": "IBM Quantum", "backend_name": "ibm_kingston"}
    )

    assert executor.config.channel == "ibm_quantum_platform"
    assert executor.config.instance is None


def test_factory_translates_stored_legacy_values():
    """A backend config row saved under the old defaults must still build."""
    with pytest.warns(DeprecationWarning):
        executor = ExecutorFactory.create_executor(
            "ibm-kingston",
            {
                "provider": "IBM Quantum",
                "backend_name": "ibm_kingston",
                "channel": "ibm_quantum",
                "instance": "ibm-q/open/main",
            },
        )

    assert executor.config.channel == "ibm_quantum_platform"
    assert executor.config.instance is None


# ---------------------------------------------------------------------------
# All three call sites agree on None and "" (the regression from the
# abandoned branch, where device.py used `or` and factory.py used `.get`)
# ---------------------------------------------------------------------------


def _device_kwargs(fake_runtime, channel, instance):
    captured, _ = fake_runtime
    params = {"backend": "ibm_brisbane", "ibm_token": "tok"}
    if channel is not None:
        params["ibm_channel"] = channel
    if instance is not None:
        params["ibm_instance"] = instance
    MarqovDevice("ibm_brisbane", params)._get_provider_device()
    return captured["kwargs"]


def _factory_kwargs(fake_runtime, channel, instance):
    captured, _ = fake_runtime
    backend_config = {"provider": "IBM Quantum", "backend_name": "ibm_brisbane", "token": "tok"}
    if channel is not None:
        backend_config["channel"] = channel
    if instance is not None:
        backend_config["instance"] = instance
    ExecutorFactory.create_executor("ibm-brisbane", backend_config)._get_backend_sync()
    return captured["kwargs"]


def _config_kwargs(fake_runtime, channel, instance):
    captured, _ = fake_runtime
    extra = {}
    if channel is not None:
        extra["channel"] = channel
    if instance is not None:
        extra["instance"] = instance
    IBMExecutor(
        IBMExecutorConfig(backend_name="ibm_brisbane", token="tok", **extra)
    )._get_backend_sync()
    return captured["kwargs"]


@pytest.mark.parametrize("channel", [None, ""])
@pytest.mark.parametrize("instance", [None, ""])
def test_all_call_sites_agree_on_absent_and_empty_values(fake_runtime, channel, instance):
    expected = {"channel": "ibm_quantum_platform", "token": "tok"}

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _device_kwargs(fake_runtime, channel, instance) == expected
        assert _factory_kwargs(fake_runtime, channel, instance) == expected
        assert _config_kwargs(fake_runtime, channel, instance) == expected
